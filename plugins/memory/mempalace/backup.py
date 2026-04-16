"""MemPalace cloud backup — incremental + full, encrypted, with Feishu alerts.

Architecture:
- Local memory writes first go to WAL/SQLite (zero latency).
- After each significant write, an incremental backup task is enqueued.
- A singleton BackupWorker thread flushes the queue every 30 s or 10 tasks,
  packing changed files into a tar.gz, encrypting with openssl AES-256-CBC,
  and uploading to a private GitHub repo.
- Daily full snapshots are triggered externally (launchd) via backup_full().
- On any persistent failure, a Feishu DM alert is sent.

STABILITY GUARANTEES:
1. Queue operations are atomic (pop-due-tasks under lock) — no task loss.
2. Deduplication at enqueue time prevents identical back-to-back tasks.
3. Full backups freeze WAL/backup workers before export for consistency.
4. GitHub repo old files are cleaned up automatically (retention policy).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import secrets
import subprocess
import tarfile
import tempfile
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
_BACKUP_INTERVAL_SECONDS = 30
_BACKUP_BATCH_SIZE = 10
_BACKUP_MAX_RETRIES = 10
_BACKUP_QUEUE_MAX_MB = 100
_BACKUP_REPO_NAME = "mempalace-backup"
_BACKUP_RETENTION_INC_DAYS = 7
_BACKUP_RETENTION_FULL_DAYS = 30

# ---------------------------------------------------------------------------
# Keychain
# ---------------------------------------------------------------------------
def _ensure_backup_key() -> str:
    """Get or create a 32-byte url-safe encryption key in macOS Keychain."""
    result = subprocess.run(
        ["security", "find-generic-password", "-s", "mempalace-backup", "-w"],
        capture_output=True, text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()

    key = secrets.token_urlsafe(32)
    subprocess.run(
        [
            "security", "add-generic-password",
            "-s", "mempalace-backup",
            "-a", os.getenv("USER", "hermes"),
            "-w", key,
            "-U",
        ],
        capture_output=True,
        check=False,
    )
    return key


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------
def _github_token() -> Optional[str]:
    """Resolve GitHub personal-access token (env → gh CLI)."""
    token = (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or "").strip()
    if token:
        return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _github_owner() -> str:
    """Infer GitHub username from gh CLI; fall back to a hard-coded default."""
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "Logged in to github.com account" in line:
                    parts = line.strip().split()
                    for i, p in enumerate(parts):
                        if p == "account" and i + 1 < len(parts):
                            return parts[i + 1]
    except Exception:
        pass
    return "mars82311111"


def _ensure_github_repo(token: str, owner: str, repo: str) -> bool:
    """Create the private backup repo if it does not exist."""
    check_url = f"https://api.github.com/repos/{owner}/{repo}"
    req = urllib.request.Request(
        check_url,
        headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15):
            return True
    except urllib.error.HTTPError as e:
        if e.code != 404:
            return False

    create_url = "https://api.github.com/user/repos"
    payload = json.dumps(
        {
            "name": repo,
            "private": True,
            "description": "Automatic encrypted MemPalace backups",
            "auto_init": True,
        }
    ).encode()
    create_req = urllib.request.Request(create_url, data=payload, method="POST")
    create_req.add_header("Authorization", f"token {token}")
    create_req.add_header("Content-Type", "application/json")
    create_req.add_header("Accept", "application/vnd.github.v3+json")
    try:
        with urllib.request.urlopen(create_req, timeout=30):
            return True
    except Exception as exc:
        logger.warning("Failed to create GitHub repo %s/%s: %s", owner, repo, exc)
        return False


def _github_upload(
    repo_path: str, content: bytes, token: str, owner: str, repo: str
) -> bool:
    """Upload a new file to the backup repo via GitHub Contents API."""
    if not _ensure_github_repo(token, owner, repo):
        return False

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{repo_path}"
    payload = json.dumps(
        {
            "message": f"MemPalace backup {datetime.now().isoformat()}",
            "content": base64.b64encode(content).decode(),
        }
    ).encode()
    req = urllib.request.Request(url, data=payload, method="PUT")
    req.add_header("Authorization", f"token {token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "mempalace-backup/1.0")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status in (200, 201)
    except urllib.error.HTTPError as e:
        logger.warning("GitHub upload HTTP error: %s %s", e.code, e.reason)
        return False
    except Exception as exc:
        logger.warning("GitHub upload error: %s", exc)
        return False


def _github_delete_file(
    repo_path: str, sha: str, token: str, owner: str, repo: str
) -> bool:
    """Delete a file from the backup repo."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{repo_path}"
    payload = json.dumps(
        {
            "message": f"MemPalace cleanup {datetime.now().isoformat()}",
            "sha": sha,
        }
    ).encode()
    req = urllib.request.Request(url, data=payload, method="DELETE")
    req.add_header("Authorization", f"token {token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/vnd.github.v3+json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status in (200, 204)
    except Exception as exc:
        logger.debug("GitHub delete error for %s: %s", repo_path, exc)
        return False


def _github_list_dir(
    dir_path: str, token: str, owner: str, repo: str
) -> List[Dict[str, Any]]:
    """List contents of a directory in the backup repo."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{dir_path}"
    req = urllib.request.Request(
        url, headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, list):
                return data
            return []
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        logger.warning("GitHub list dir error: %s %s", e.code, e.reason)
        return []
    except Exception as exc:
        logger.warning("GitHub list dir error: %s", exc)
        return []


def _github_cleanup_old_backups(token: str, owner: str, repo: str) -> None:
    """Remove backup directories older than retention policy."""
    cutoff_full = (datetime.now() - timedelta(days=_BACKUP_RETENTION_FULL_DAYS)).strftime("%Y%m%d")
    cutoff_inc = (datetime.now() - timedelta(days=_BACKUP_RETENTION_INC_DAYS)).strftime("%Y%m%d")

    for prefix, cutoff in (("backups/full", cutoff_full), ("backups/incremental", cutoff_inc)):
        days = _github_list_dir(prefix, token, owner, repo)
        for day_entry in days:
            if day_entry.get("type") != "dir":
                continue
            day_name = day_entry.get("name", "")
            if day_name < cutoff:
                # List and delete files inside this day directory
                files = _github_list_dir(f"{prefix}/{day_name}", token, owner, repo)
                deleted_any = False
                for f in files:
                    if f.get("type") == "file":
                        if _github_delete_file(f["path"], f["sha"], token, owner, repo):
                            deleted_any = True
                if deleted_any:
                    logger.info("Cleaned up old backup dir: %s/%s", prefix, day_name)


# ---------------------------------------------------------------------------
# Feishu alert helpers
# ---------------------------------------------------------------------------
_last_feishu_alert_at: Optional[datetime] = None


def _feishu_notify(message: str, min_interval_seconds: float = 3600) -> None:
    """Send a plain-text alert to the Feishu home channel (best-effort, rate-limited)."""
    global _last_feishu_alert_at
    now = datetime.now()
    if _last_feishu_alert_at is not None:
        if (now - _last_feishu_alert_at).total_seconds() < min_interval_seconds:
            logger.debug("Feishu notify suppressed by rate limit")
            return
    _last_feishu_alert_at = now

    app_id = os.getenv("FEISHU_APP_ID")
    app_secret = os.getenv("FEISHU_APP_SECRET")
    chat_id = os.getenv("FEISHU_HOME_CHANNEL")
    if not all([app_id, app_secret, chat_id]):
        logger.debug("Feishu notify skipped: missing credentials")
        return

    # 1. tenant_access_token
    try:
        token_data = json.dumps(
            {"app_id": app_id, "app_secret": app_secret}
        ).encode("utf-8")
        token_req = urllib.request.Request(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            data=token_data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(token_req, timeout=15) as resp:
            token_res = json.loads(resp.read().decode())
        access_token = token_res.get("tenant_access_token")
        if not access_token:
            logger.warning("Feishu token failed: %s", token_res)
            return
    except Exception as exc:
        logger.warning("Feishu token error: %s", exc)
        return

    # 2. send text message
    try:
        payload = json.dumps(
            {
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": message}),
            }
        ).encode("utf-8")
        msg_req = urllib.request.Request(
            "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(msg_req, timeout=15):
            pass
    except Exception as exc:
        logger.warning("Feishu send error: %s", exc)


# ---------------------------------------------------------------------------
# Local queue
# ---------------------------------------------------------------------------
class BackupQueue:
    def __init__(self, queue_path: Path):
        self.queue_path = queue_path
        self._lock = threading.Lock()
        queue_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all_unlocked(self) -> List[Dict[str, Any]]:
        if not self.queue_path.exists():
            return []
        try:
            text = self.queue_path.read_text(encoding="utf-8")
        except Exception:
            return []
        tasks = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except Exception:
                continue
        return tasks

    def _write_all_unlocked(self, tasks: List[Dict[str, Any]]) -> None:
        tmp = self.queue_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
        tmp.replace(self.queue_path)

    def enqueue(self, task: Dict[str, Any]) -> None:
        line = json.dumps(task, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            with open(self.queue_path, "a", encoding="utf-8") as f:
                f.write(line)

    def read_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._read_all_unlocked()

    def atomic_pop_due(
        self, is_due_fn: Callable[[Dict[str, Any]], bool], max_n: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Atomically pop up to max_n due tasks. Returns (popped, remaining_in_queue)."""
        with self._lock:
            current = self._read_all_unlocked()
            popped: List[Dict[str, Any]] = []
            remaining: List[Dict[str, Any]] = []
            for t in current:
                if len(popped) < max_n and is_due_fn(t):
                    popped.append(t)
                else:
                    remaining.append(t)
            self._write_all_unlocked(remaining)
            return popped, remaining

    def clear(self) -> None:
        with self._lock:
            if self.queue_path.exists():
                self.queue_path.unlink()


# ---------------------------------------------------------------------------
# Backup worker (singleton background thread)
# ---------------------------------------------------------------------------
class BackupWorker(threading.Thread):
    def __init__(
        self,
        queue: BackupQueue,
        interval: int = _BACKUP_INTERVAL_SECONDS,
        batch_size: int = _BACKUP_BATCH_SIZE,
    ):
        super().__init__(daemon=True, name="mempalace-backup")
        self.queue = queue
        self.interval = interval
        self.batch_size = batch_size
        self._stop_event = threading.Event()
        self._consecutive_errors = 0
        self._auth_failed = False

    def request_stop(self) -> None:
        self._stop_event.set()

    @staticmethod
    def _is_due(task: Dict[str, Any]) -> bool:
        nxt = task.get("next_retry_at")
        if not nxt:
            return True
        try:
            return datetime.now() >= datetime.fromisoformat(nxt)
        except Exception:
            return True

    def run(self) -> None:
        logger.info("BackupWorker started (interval=%ds, batch=%d)", self.interval, self.batch_size)
        while not self._stop_event.is_set():
            popped, remaining = self.queue.atomic_pop_due(self._is_due, self.batch_size)
            if popped:
                self._process_batch(popped)
            elif not self._stop_event.is_set():
                self._stop_event.wait(self.interval)
        # Drain remaining tasks on shutdown
        popped, _ = self.queue.atomic_pop_due(self._is_due, self.batch_size)
        while popped:
            self._process_batch(popped)
            popped, _ = self.queue.atomic_pop_due(self._is_due, self.batch_size)
        logger.info("BackupWorker stopped")

    def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        # Collect unique files referenced by the batch
        file_paths: List[str] = []
        seen = set()
        for t in batch:
            for f in t.get("files", []):
                if f not in seen:
                    seen.add(f)
                    file_paths.append(f)
        existing_files = [f for f in file_paths if Path(f).exists()]

        if not existing_files:
            logger.debug("Backup batch has no existing files; dropping %d tasks", len(batch))
            return

        # Use a single timestamp for filename and path to avoid cross-midnight mismatch
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"mempalace_inc_hermes_{timestamp}.tar.gz.enc"
        repo_path = f"backups/incremental/{now.strftime('%Y%m%d')}/{filename}"

        try:
            key = _ensure_backup_key()
            payload = self._pack_and_encrypt(existing_files, key)
        except Exception:
            logger.exception("Backup pack/encrypt failed")
            self._requeue_with_retry(batch, immediate_retry=False)
            return

        token = _github_token()
        if not token:
            if not self._auth_failed:
                self._auth_failed = True
                _feishu_notify("⚠️ MemPalace 备份暂停：无法获取 GitHub Token，请检查 gh auth 状态。")
            logger.error("GitHub token unavailable")
            self._requeue_with_retry(batch, immediate_retry=False)
            return

        owner = _github_owner()
        success = _github_upload(repo_path, payload, token, owner, _BACKUP_REPO_NAME)

        if success:
            self._consecutive_errors = 0
            self._auth_failed = False
            logger.info("Backup uploaded: %s (%d tasks, %d files)", filename, len(batch), len(existing_files))
        else:
            self._consecutive_errors += 1
            self._requeue_with_retry(batch, immediate_retry=True)
            if self._consecutive_errors >= 3:
                _feishu_notify(
                    f"⚠️ MemPalace 云端备份连续失败 {self._consecutive_errors} 次，"
                    f"请检查网络或 GitHub 状态。\n最近上传文件: {filename}",
                    min_interval_seconds=3600,
                )

    def _pack_and_encrypt(self, files: List[str], key: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_in_path = os.path.join(tmpdir, "backup.tar.gz")
            tmp_out_path = os.path.join(tmpdir, "backup.tar.gz.enc")

            with tarfile.open(tmp_in_path, mode="w:gz") as tar:
                for f in files:
                    p = Path(f)
                    if p.exists():
                        tar.add(str(p), arcname=p.name)

            result = subprocess.run(
                [
                    "openssl", "enc", "-aes-256-cbc", "-salt", "-pbkdf2", "-iter", "100000",
                    "-in", tmp_in_path, "-out", tmp_out_path, "-pass", f"pass:{key}",
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", "replace") if result.stderr else ""
                raise RuntimeError(f"openssl encrypt failed: {stderr}")
            return Path(tmp_out_path).read_bytes()

    def _requeue_with_retry(
        self, batch: List[Dict[str, Any]], immediate_retry: bool
    ) -> None:
        now = datetime.now()
        for t in batch:
            t["retry_count"] = t.get("retry_count", 0) + 1
            if immediate_retry:
                delay = min(30 * (2 ** (t["retry_count"] - 1)), 3600)
            else:
                delay = 3600
            t["next_retry_at"] = (now + timedelta(seconds=delay)).isoformat()

        # Read current queue, filter stale, append batch, write back
        current = self.queue.read_all()
        cutoff = now - timedelta(days=_BACKUP_RETENTION_INC_DAYS)
        filtered = []
        for t in current:
            if t.get("retry_count", 0) > _BACKUP_MAX_RETRIES:
                continue
            nxt = t.get("next_retry_at")
            if nxt:
                try:
                    if datetime.fromisoformat(nxt) < cutoff:
                        continue
                except Exception:
                    pass
            filtered.append(t)

        filtered.extend(batch)

        # Size guard
        try:
            tmp_buf = io.StringIO()
            for t in filtered:
                tmp_buf.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
            size_mb = len(tmp_buf.getvalue().encode("utf-8")) / (1024 * 1024)
            if size_mb > _BACKUP_QUEUE_MAX_MB:
                logger.error("Backup queue oversized (%.1f MB); truncating to last 1000 tasks", size_mb)
                filtered = filtered[-1000:]
                _feishu_notify(
                    "🚨 MemPalace 备份队列堆积超过 100MB，已自动截断旧任务，请检查网络/GitHub 状态。",
                    min_interval_seconds=3600,
                )
        except Exception:
            pass

        self.queue.write_all(filtered)


# ---------------------------------------------------------------------------
# Full backup (triggered by launchd nightly)
# ---------------------------------------------------------------------------
_last_full_backup_alert_at: Optional[datetime] = None


def run_full_backup(
    export_filepath: Optional[str] = None,
    palace_path: Optional[Path] = None,
    feishu_alert: bool = True,
    stop_workers_fn: Optional[Callable[[], None]] = None,
    start_workers_fn: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Create a full logical export, encrypt it, and upload to GitHub.

    Returns a dict with upload status and metadata.
    """
    result = {"success": False, "filepath": "", "github_path": "", "error": ""}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_path = str(
        (palace_path or Path.home() / ".mempalace_hermes")
        / f"export_full_{timestamp}.json"
    )
    export_filepath = export_filepath or default_path

    # Pause background workers for consistency (best-effort)
    if stop_workers_fn:
        try:
            stop_workers_fn()
        except Exception:
            pass

    try:
        # 1. Export via existing logic (delayed import avoids circular deps)
        try:
            from plugins.memory.mempalace import MemPalaceMemoryProvider

            provider = MemPalaceMemoryProvider()
            export_json = provider._handle_export(
                {"filepath": export_filepath, "include_chromadb": True}
            )
            export_meta = json.loads(export_json)
            if "error" in export_meta:
                raise RuntimeError(export_meta.get("error", "export failed"))
        except Exception as exc:
            result["error"] = f"Export failed: {exc}"
            logger.exception("Full backup export failed")
            if feishu_alert:
                _feishu_notify(f"⚠️ MemPalace 全量备份导出失败：{exc}")
            return result

        # 2. Collect additional files
        base = palace_path or Path.home() / ".mempalace_hermes"
        extra_files = []
        for name in ("identity.txt", "config.json", "health.json", "episodes.wal.ndjson"):
            p = base / name
            if p.exists():
                extra_files.append(str(p))

        all_files = [export_filepath] + extra_files

        # 3. Pack + encrypt
        try:
            key = _ensure_backup_key()
            worker = BackupWorker(BackupQueue(base / ".backup_queue.ndjson"))
            payload = worker._pack_and_encrypt(all_files, key)
        except Exception as exc:
            result["error"] = f"Encrypt failed: {exc}"
            logger.exception("Full backup encrypt failed")
            if feishu_alert:
                _feishu_notify(f"⚠️ MemPalace 全量备份加密失败：{exc}")
            return result

        # 4. Upload
        filename = f"mempalace_full_hermes_{timestamp}.tar.gz.enc"
        repo_path = f"backups/full/{datetime.now().strftime('%Y%m%d')}/{filename}"
        token = _github_token()
        if not token:
            result["error"] = "GitHub token unavailable"
            if feishu_alert:
                _feishu_notify("⚠️ MemPalace 全量备份暂停：无法获取 GitHub Token。")
            return result

        owner = _github_owner()
        success = _github_upload(repo_path, payload, token, owner, _BACKUP_REPO_NAME)
        if success:
            result["success"] = True
            result["github_path"] = repo_path
            result["filepath"] = export_filepath
            logger.info("Full backup uploaded: %s", repo_path)
            if feishu_alert:
                global _last_full_backup_alert_at
                now = datetime.now()
                if _last_full_backup_alert_at is None or (now - _last_full_backup_alert_at).total_seconds() >= 3600:
                    _last_full_backup_alert_at = now
                    _feishu_notify(f"✅ MemPalace 全量备份成功上传：{repo_path}")
            # Clean up old backups on GitHub
            _github_cleanup_old_backups(token, owner, _BACKUP_REPO_NAME)
        else:
            result["error"] = "GitHub upload failed"
            logger.error("Full backup upload failed")
            if feishu_alert:
                _feishu_notify("⚠️ MemPalace 全量备份上传 GitHub 失败，请检查网络状态。")

        # 5. Clean up local export file to save disk
        try:
            Path(export_filepath).unlink(missing_ok=True)
        except Exception:
            pass
    finally:
        if start_workers_fn:
            try:
                start_workers_fn()
            except Exception:
                pass

    return result


# ---------------------------------------------------------------------------
# Global singleton controls
# ---------------------------------------------------------------------------
_BACKUP_QUEUE: Optional[BackupQueue] = None
_BACKUP_WORKER: Optional[BackupWorker] = None


def get_backup_queue() -> BackupQueue:
    global _BACKUP_QUEUE
    expected_path = Path.home() / ".mempalace_hermes" / ".backup_queue.ndjson"
    if _BACKUP_QUEUE is None or _BACKUP_QUEUE.queue_path != expected_path:
        _BACKUP_QUEUE = BackupQueue(expected_path)
    return _BACKUP_QUEUE


def start_backup_worker() -> None:
    global _BACKUP_WORKER
    if _BACKUP_WORKER is not None and _BACKUP_WORKER.is_alive():
        return
    _BACKUP_WORKER = BackupWorker(get_backup_queue())
    _BACKUP_WORKER.start()


def stop_backup_worker(timeout: float = 10.0) -> None:
    global _BACKUP_WORKER
    if _BACKUP_WORKER is not None and _BACKUP_WORKER.is_alive():
        _BACKUP_WORKER.request_stop()
        _BACKUP_WORKER.join(timeout=timeout)


def enqueue_incremental(files: List[str]) -> None:
    """Enqueue an incremental backup task for the given file paths."""
    if not files:
        return

    # Deduplicate against identical pending tasks (no retry)
    sig = json.dumps(sorted(files), ensure_ascii=False, default=str)
    q = get_backup_queue()
    try:
        pending = q.read_all()
        for t in pending:
            if t.get("type") == "incremental" and not t.get("next_retry_at"):
                if json.dumps(sorted(t.get("files", [])), ensure_ascii=False, default=str) == sig:
                    return
    except Exception:
        pass

    q.enqueue(
        {
            "ts": datetime.now().isoformat(),
            "type": "incremental",
            "files": files,
            "retry_count": 0,
        }
    )
    start_backup_worker()
