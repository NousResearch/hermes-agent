"""Orchestration for enabling, disabling, and inspecting encryption-at-rest.

Every file conversion follows the same safety contract:

1. Back up the current contents under ``<HERMES_HOME>/.encryption/backup/``.
2. Convert (encrypt or decrypt).
3. Read the result back and verify it round-trips to the original bytes.
4. On any failure, restore from the backup and abort — credentials are never
   left in an unreadable state.

The config flag ``security.encryption.enabled`` is flipped *last*, only after
every target has converted and verified.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

_log = logging.getLogger(__name__)

from hermes_constants import get_config_path, get_hermes_home

from . import audit, dbcrypt, detect, envelope, keystore, session_writer
from .errors import HermesCryptoError
from .fileio import atomic_copy, atomic_write_private, harden_dir

ProgressFn = Callable[[str], None]


def _noop(_message: str) -> None:
    pass


# Long-lived Hermes runtimes that may hold ``state.db`` / credential files open.
# The ``encrypt …`` migration CLI is deliberately excluded.
_RE_ENCRYPT_CLI = re.compile(r"\sencrypt\s", re.IGNORECASE)
# tighten ``hermes-agent`` and ``hermes`` matchers with a
# negative lookahead so a checkout path like ``hermes-agent-main`` or
# ``hermes-encryption-layer`` (substring match) doesn't trip the detector.
# ``(?![-\w])`` requires the next character to be neither a hyphen nor a
# word character, so the token must stand alone (followed by EOL, whitespace,
# ``.exe``, ``/``, ``\``, etc.).
_RE_HERMES_AGENT = re.compile(r"\bhermes-agent(?![-\w])", re.IGNORECASE)
_RE_SPECIAL_RUNTIME = re.compile(r"(?:tui_gateway|acp_adapter)", re.IGNORECASE)
_RE_MODULE_RUNTIME = re.compile(
    r"-m\s+(?:[\w.]+\.)?(?:gateway|agent)(?:[.\s]|$)",
    re.IGNORECASE,
)
_RE_HERMES = re.compile(r"\bhermes(?![-\w])", re.IGNORECASE)
_RE_RUNTIME_ROLE = re.compile(r"\b(?:gateway|agent)\b", re.IGNORECASE)

# explicit test-runner / one-liner-helper denylist.
# pytest workers and ``python -c "..."`` test helpers are never a real
# long-lived Hermes runtime; without this filter, a pytest run from a
# checkout whose path contains ``hermes-agent`` (or any sibling marker)
# would be detected as a "concurrent Hermes process" and block
# ``migrate.enable`` / ``migrate.disable`` without ``force=True`` —
# the documented ISSUES #16 false positive.
_RE_TEST_RUNNER = re.compile(
    r"\bpytest\b|\b_pytest\b|_jb_pytest_runner",
    re.IGNORECASE,
)
_RE_PYTHON_INLINE = re.compile(
    r"\bpython(?:\.exe|3)?\b\s+(?:-[^c\s]+\s+)*-c\b",
    re.IGNORECASE,
)
_RE_ROTATED_LOG_ARCHIVE = re.compile(r"\.log\.\d+$", re.IGNORECASE)


def _is_hermes_runtime_cmdline(parts: List[str]) -> bool:
    joined = " ".join(parts)
    if _RE_ENCRYPT_CLI.search(f" {joined.strip()} "):
        return False
    # see denylist regexes above.
    if _RE_TEST_RUNNER.search(joined):
        return False
    if _RE_PYTHON_INLINE.search(joined):
        return False
    if _RE_HERMES_AGENT.search(joined):
        return True
    if _RE_SPECIAL_RUNTIME.search(joined):
        return True
    if _RE_MODULE_RUNTIME.search(joined):
        return True
    return bool(_RE_HERMES.search(joined) and _RE_RUNTIME_ROLE.search(joined))


def _enumeration_unavailable(method: str) -> List[Dict[str, object]]:
    """Return a sentinel requiring explicit ``--force`` when listing is unavailable."""
    audit.log_event(
        audit.MIGRATION_ENUMERATION_UNAVAILABLE,
        audit.INFO,
        reason="enumeration_unavailable",
        method=method,
    )
    return [
        {
            "pid": None,
            "cmdline": f"{method} process enumeration unavailable",
            "enumeration_unavailable": True,
            "method": method,
        }
    ]


def _detect_concurrent_hermes_instances() -> List[Dict[str, object]]:
    """Return other running Hermes gateway/agent processes (not this PID).

    Each dict has ``pid`` (int) and ``cmdline`` (str). Returns ``[]`` when
    process enumeration is unavailable — callers treat that as none detected.
    """
    my_pid = os.getpid()
    try:
        import psutil
    except ImportError:
        return _detect_concurrent_hermes_instances_fallback(my_pid)

    found: List[Dict[str, object]] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = proc.info.get("pid")
            if not isinstance(pid, int) or pid == my_pid:
                continue
            cmdline = proc.info.get("cmdline") or []
            if not cmdline or not _is_hermes_runtime_cmdline(cmdline):
                continue
            found.append({"pid": pid, "cmdline": " ".join(cmdline)})
        except Exception:
            continue
    return found


def _detect_concurrent_hermes_instances_fallback(my_pid: int) -> List[Dict[str, object]]:
    if sys.platform == "win32":
        return _detect_concurrent_hermes_windows(my_pid)
    if sys.platform == "darwin":
        return _detect_concurrent_hermes_ps(my_pid)
    return _detect_concurrent_hermes_procfs(my_pid)


def _detect_concurrent_hermes_procfs(my_pid: int) -> List[Dict[str, object]]:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return _enumeration_unavailable("procfs")
    found: List[Dict[str, object]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == my_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        parts = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
        if parts and _is_hermes_runtime_cmdline(parts):
            found.append({"pid": pid, "cmdline": " ".join(parts)})
    return found


def _detect_concurrent_hermes_ps(my_pid: int) -> List[Dict[str, object]]:
    try:
        completed = subprocess.run(
            ["ps", "-ax", "-o", "pid=", "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return _enumeration_unavailable("ps")
    if completed.returncode != 0:
        return _enumeration_unavailable("ps")
    return _parse_pid_command_lines(completed.stdout, my_pid)


def _detect_concurrent_hermes_windows(my_pid: int) -> List[Dict[str, object]]:
    script = (
        "Get-CimInstance Win32_Process | "
        "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return _enumeration_unavailable("windows")
    if completed.returncode != 0:
        return _enumeration_unavailable("windows")
    return _parse_pid_command_lines(completed.stdout, my_pid)


def _parse_pid_command_lines(text: str, my_pid: int) -> List[Dict[str, object]]:
    found: List[Dict[str, object]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, sep, cmdline = line.partition("\t")
        if not sep:
            continue
        try:
            pid = int(pid_text.strip())
        except ValueError:
            continue
        if pid == my_pid or not cmdline.strip():
            continue
        parts = cmdline.split()
        if parts and _is_hermes_runtime_cmdline(parts):
            found.append({"pid": pid, "cmdline": cmdline.strip()})
    return found


def _require_no_concurrent_hermes(*, force: bool, operation: str) -> None:
    """Abort *operation* when another Hermes runtime is still running."""
    if force:
        return
    instances = _detect_concurrent_hermes_instances()
    if not instances:
        return
    unavailable = [
        item for item in instances if item.get("enumeration_unavailable")
    ]
    if unavailable:
        methods = sorted(
            {
                str(item.get("method") or "unknown")
                for item in unavailable
            }
        )
        audit.log_event(
            audit.MIGRATION_BLOCKED,
            audit.FAILURE,
            operation=operation,
            reason="enumeration_unavailable",
            methods=methods,
        )
        method_list = ", ".join(methods)
        raise HermesCryptoError(
            f"Cannot {operation}: process enumeration is unavailable "
            f"({method_list}). Re-run with --force only after stopping any "
            "running Hermes gateway/agent."
        )
    pids = [int(item["pid"]) for item in instances if isinstance(item.get("pid"), int)]
    audit.log_event(
        audit.MIGRATION_BLOCKED,
        audit.FAILURE,
        operation=operation,
        concurrent_count=len(instances),
        concurrent_pids=pids,
    )
    pid_list = ", ".join(str(pid) for pid in pids) or "unknown"
    raise HermesCryptoError(
        f"Cannot {operation}: another Hermes process is still running "
        f"(PIDs: {pid_list}). Stop the gateway/agent first, or pass --force "
        "to override."
    )


@dataclass
class FileTarget:
    """A credential file that participates in encryption.

    ``kind`` tags the target's origin bucket ("credential" / "session" /
    "log") so :func:`full_rekey` can categorise its result without sniffing
    the path. Path sniffing is fragile — a HERMES_HOME whose ancestor
    directory is literally named ``logs`` would mis-bucket every credential
    file under it.
    """

    path: Path
    env_framed: bool = False
    kind: str = "credential"

    @property
    def label(self) -> str:
        try:
            return str(self.path.relative_to(get_hermes_home()))
        except ValueError:
            return self.path.name


def _shared_nous_store_path() -> Path:
    """Resolve the cross-profile shared Nous OAuth store path.

    Mirrors ``hermes_cli.auth._nous_shared_store_path`` without importing it
    (auth.py imports ``hermes_crypto``, so importing it back here would risk a
    circular import). The store lives at
    ``${HERMES_SHARED_AUTH_DIR}/nous_auth.json`` when that env var is set, else
    ``<hermes-root>/shared/nous_auth.json`` — OUTSIDE any named profile's
    HERMES_HOME so every profile under the same root shares one file. Keep this
    in sync with the canonical resolver in ``hermes_cli.auth``.
    """
    from hermes_constants import get_default_hermes_root

    filename = "nous_auth.json"
    override = os.getenv("HERMES_SHARED_AUTH_DIR", "").strip()
    if override:
        return Path(override).expanduser() / filename
    return get_default_hermes_root() / "shared" / filename


def credential_targets() -> List[FileTarget]:
    """Return the credential files the layer encrypts.

    Fixed files are returned whether or not they exist yet; the MCP token
    store (one JSON file per server under ``mcp-tokens/``) is globbed.
    """
    home = get_hermes_home()
    targets = [
        FileTarget(home / ".env", env_framed=True, kind="credential"),
        FileTarget(home / "auth.json", kind="credential"),
        FileTarget(home / ".anthropic_oauth.json", kind="credential"),
        FileTarget(home / "auth" / "google_oauth.json", kind="credential"),
        # Cross-profile shared Nous OAuth store. Holds a live refresh_token,
        # so it must seal under the DEK like every other credential file
        # instead of sitting in cleartext (blocker #4). Lives outside the
        # profile HERMES_HOME, hence resolved separately.
        FileTarget(_shared_nous_store_path(), kind="credential"),
    ]
    mcp_dir = home / "mcp-tokens"
    if mcp_dir.is_dir():
        targets.extend(
            FileTarget(path, kind="credential") for path in sorted(mcp_dir.glob("*.json"))
        )
    return targets


def session_targets() -> List[FileTarget]:
    """Return the gateway session transcripts the layer encrypts.

    JSONL files under ``~/.hermes/sessions/``. Returns ``[]`` when the
    directory does not exist (no sessions written yet). Distinct from
    :func:`credential_targets` because the gate is the ``encrypt_logs`` flag,
    not the master ``enabled`` flag — sessions follow the log-encryption
    rollout, not the credential rollout.
    """
    home = get_hermes_home()
    sessions = home / "sessions"
    if not sessions.is_dir():
        return []
    return [FileTarget(p, kind="session") for p in sorted(sessions.glob("*.jsonl"))]


def log_archive_targets() -> List[FileTarget]:
    """Return rotated log segments that are already HRMSENC-encrypted.

    Detection-driven — only files whose on-disk prefix matches the envelope
    magic are returned. Live plaintext logs are skipped.
    """
    home = get_hermes_home()
    logs = home / "logs"
    if not logs.is_dir():
        return []
    targets: List[FileTarget] = []
    # walk the tree without following symlinks so a
    # symlink loop under ``~/.hermes/logs/`` can't hang the migration.
    # ``Path.rglob("*")`` follows symlinks by default on POSIX
    # (``follow_symlinks=False`` is Python 3.13+ only). ``os.walk(followlinks
    # =False)`` is the portable equivalent.
    discovered: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(logs, followlinks=False):
        for name in filenames:
            discovered.append(Path(dirpath) / name)
    for path in sorted(discovered):
        if not path.is_file():
            continue
        try:
            head = path.read_bytes()[:32]
        except OSError:
            continue
        if detect.is_encrypted(head):
            targets.append(FileTarget(path, kind="log"))
    return targets


def plaintext_log_archive_targets() -> List[FileTarget]:
    """Return plaintext rotated ``*.log.N`` segments that can be sealed at enable."""
    home = get_hermes_home()
    logs = home / "logs"
    if not logs.is_dir():
        return []
    targets: List[FileTarget] = []
    discovered: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(logs, followlinks=False):
        for name in filenames:
            if _RE_ROTATED_LOG_ARCHIVE.search(name):
                discovered.append(Path(dirpath) / name)
    for path in sorted(discovered):
        if not path.is_file():
            continue
        try:
            head = path.read_bytes()[:32]
        except OSError:
            continue
        if not detect.is_encrypted(head):
            targets.append(FileTarget(path, kind="log"))
    return targets


def envelope_encrypted_targets() -> List[FileTarget]:
    """Return every on-disk artifact sealed with the DEK envelope."""
    seen: set[Path] = set()
    ordered: List[FileTarget] = []
    for target in (
        *credential_targets(),
        *session_targets(),
        *log_archive_targets(),
    ):
        if target.path in seen:
            continue
        seen.add(target.path)
        ordered.append(target)
    return ordered


def database_targets() -> List[Path]:
    """Return the SQLite databases the layer encrypts.

    Covers the session store, the default kanban board, and any additional
    per-project kanban boards under ``kanban/boards/<slug>/``.
    """
    home = get_hermes_home()
    targets = [home / "state.db", home / "kanban.db"]
    boards = home / "kanban" / "boards"
    if boards.is_dir():
        targets.extend(sorted(boards.glob("*/kanban.db")))
    return targets


# ─── Backups ──────────────────────────────────────────────────────────────────


def _backup(path: Path) -> Path:
    """Copy *path* (and SQLCipher sidecars if present) into the backup directory.

    Returns the backup path for *path*. Sidecars ``<name>-wal`` and
    ``<name>-shm`` are snapshotted alongside as ``<dest>-wal`` /
    ``<dest>-shm`` when present, so a later rollback can restore the
    consistent (main, ``-wal``, ``-shm``) triple. No-op for envelope
    files (no sidecars exist). Best-effort: a live writer may hold a
    sidecar exclusively on Windows; the ``rotate-key`` CLI ``--force``
    help and ``full_rekey`` docstring warn operators to stop Hermes
    first.
    """
    harden_dir(keystore.backup_dir())
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    flat = path.name
    try:
        rel = path.relative_to(get_hermes_home())
        flat = str(rel).replace("/", "_").replace("\\", "_")
    except ValueError:
        pass
    dest = keystore.backup_dir() / f"{flat}.{stamp}.{secrets.token_hex(4)}.bak"
    # atomic_copy (copy-to-tmp + fsync + os.replace) so a
    # crash mid-copy never leaves a truncated .bak that a later
    # ``_encrypt_file`` post-write-verify failure would silently restore from.
    atomic_copy(path, dest)
    # snapshot SQLCipher sidecars so a later rollback can restore
    # the main file + uncheckpointed WAL as a consistent triple.
    for suffix in ("-wal", "-shm"):
        sidecar = path.with_name(path.name + suffix)
        if sidecar.is_file():
            sidecar_dest = dest.with_name(dest.name + suffix)
            try:
                atomic_copy(sidecar, sidecar_dest)
            except OSError:
                pass
    return dest


def _iter_backup_files() -> List[Path]:
    """Return every ``*.bak`` migration backup under ``.encryption/backup/``."""
    root = keystore.backup_dir()
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.bak") if p.is_file())


def _is_ciphertext_backup(path: Path) -> bool:
    """Return True when *path* starts with the HRMSENC envelope magic.

    ``_iter_backup_files`` returns every ``*.bak``,
    including the rekey-run backups produced by ``full_rekey --keep-backups``
    (which are sealed under the OLD DEK). Without classification,
    ``clean-backups`` would delete those alongside plaintext credential
    backups, even though operators may have kept them intentionally for
    forensic recovery. Reading the first envelope-header bytes
    disambiguates the two kinds cheaply.
    """
    try:
        head = path.read_bytes()[:32]
    except OSError:
        return False
    return detect.is_encrypted(head)


def _parse_backup_stamp(name: str) -> Optional[datetime]:
    """Parse the ``YYYYMMDDTHHMMSS`` stamp embedded in a backup filename."""
    if not name.endswith(".bak"):
        return None
    stem = name[:-4]
    match = re.search(r"\.(\d{8}T\d{6})(?:\.[0-9a-f]{8})?$", stem, re.IGNORECASE)
    if match is None:
        return None
    stamp_str = match.group(1)
    try:
        return datetime.strptime(stamp_str, "%Y%m%dT%H%M%S")
    except ValueError:
        return None


def _backup_timestamp(path: Path) -> datetime:
    """Return the timestamp for a backup — filename stamp, else mtime."""
    parsed = _parse_backup_stamp(path.name)
    if parsed is not None:
        return parsed
    # filename does not match the ``*.YYYYMMDDTHHMMSS.bak``
    # pattern, so we fall back to ``st_mtime``. On a backup that was copied
    # across filesystems or restored from tape, mtime reflects the *copy*
    # date, not the original backup date — so ``clean_backups --older-than N``
    # may keep a 6-month-old backup or delete a 1-day-old one. Log the
    # fallback so the operator can correlate odd age-filter results with
    # the unrecognised filename.
    _log.warning(
        "backup filename does not match expected '*.YYYYMMDDTHHMMSS.bak' "
        "pattern; using st_mtime for age calculation: %s",
        path.name,
    )
    return datetime.fromtimestamp(path.stat().st_mtime)


def backup_summary() -> Dict[str, object]:
    """Return count, total size, oldest age, and plaintext/ciphertext split.

    the split lets the CLI warn operators that a
    ``clean-backups`` run would touch rekey-run ciphertext backups (sealed
    under the old DEK, often kept intentionally for forensic recovery) in
    addition to the plaintext credential backups the help text suggests.
    """
    files = _iter_backup_files()
    total_bytes = sum(f.stat().st_size for f in files)
    oldest_days: Optional[int] = None
    if files:
        oldest = min(_backup_timestamp(f) for f in files)
        oldest_days = (datetime.now() - oldest).days
    ciphertext_count = sum(1 for f in files if _is_ciphertext_backup(f))
    return {
        "count": len(files),
        "total_bytes": total_bytes,
        "oldest_days": oldest_days,
        "plaintext_count": len(files) - ciphertext_count,
        "ciphertext_count": ciphertext_count,
    }


def clean_backups(
    *, older_than_days: Optional[int] = None, include_ciphertext: bool = False
) -> Dict[str, object]:
    """Delete migration ``*.bak`` files, optionally keeping recent ones.

    Returns a summary dict for the CLI — deleted filenames, counts, any
    unlink failures, and a ``skipped_ciphertext`` count (no secret file
    contents).

    When ``include_ciphertext`` is False (the default — hardening work),
    ``*.bak`` files whose envelope header detects as HRMSENC are skipped
    and reported separately. These are rekey-run backups sealed under the
    old DEK; operators who ran ``full_rekey --keep-backups`` intentionally
    keep them for forensic recovery, and they're not plaintext credentials
    even though they share the ``.bak`` suffix.
    """
    files = _iter_backup_files()
    cutoff: Optional[datetime] = None
    if older_than_days is not None:
        cutoff = datetime.now() - timedelta(days=older_than_days)

    deleted: List[str] = []
    kept = 0
    skipped_ciphertext = 0
    errors: List[Dict[str, str]] = []

    for path in files:
        if cutoff is not None and _backup_timestamp(path) > cutoff:
            kept += 1
            continue
        if not include_ciphertext and _is_ciphertext_backup(path):
            skipped_ciphertext += 1
            continue
        try:
            path.unlink()
            deleted.append(path.name)
        except OSError as exc:
            errors.append({"name": path.name, "error": type(exc).__name__})

    audit.log_event(
        audit.BACKUPS_CLEANED,
        audit.FAILURE if errors else audit.SUCCESS,
        deleted=len(deleted),
        kept=kept,
        skipped_ciphertext=skipped_ciphertext,
        include_ciphertext=include_ciphertext,
        older_than_days=older_than_days,
        errors=len(errors),
    )
    return {
        "deleted": deleted,
        "deleted_count": len(deleted),
        "kept_count": kept,
        "skipped_ciphertext": skipped_ciphertext,
        "errors": errors,
    }


def _remove_rekey_run_backups(backup_paths: List[Path]) -> int:
    """Unlink ciphertext backups produced during a :func:`full_rekey` run.

    Only paths collected for this run are removed — older migration backups
    (e.g. plaintext copies from ``enable``) are left intact.
    """
    deleted = 0
    errors: List[Dict[str, str]] = []
    for path in backup_paths:
        try:
            path.unlink()
            deleted += 1
        except OSError as exc:
            errors.append({"name": path.name, "error": type(exc).__name__})
            continue
        # also drop any sidecar snapshots saved alongside this
        # backup. Silent cleanup — sidecars do not count toward the
        # caller-visible deleted total (they shadow the main backup).
        for suffix in ("-wal", "-shm"):
            sidecar = path.with_name(path.name + suffix)
            try:
                sidecar.unlink(missing_ok=True)
            except OSError:
                pass

    audit.log_event(
        audit.BACKUPS_REMOVED_POST_REKEY,
        audit.FAILURE if errors else audit.SUCCESS,
        deleted=deleted,
        errors=len(errors),
    )
    return deleted


# ─── Enable ───────────────────────────────────────────────────────────────────


@dataclass
class EnableResult:
    encrypted_files: List[str] = field(default_factory=list)
    encrypted_databases: List[str] = field(default_factory=list)
    encrypted_sessions: List[str] = field(default_factory=list)
    encrypted_logs: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)


def enable(
    key_source: str,
    *,
    passphrase: Optional[str] = None,
    encrypt_databases: bool = False,
    encrypt_logs: bool = False,
    argon2_params: Optional[Dict[str, int]] = None,
    progress: ProgressFn = _noop,
    force: bool = False,
) -> EnableResult:
    """Create the keystore and encrypt every credential file (and, optionally,
    the databases). Flips ``security.encryption.*`` config keys on success.
    """
    if keystore.keystore_exists():
        from . import is_encryption_enabled

        if not is_encryption_enabled():
            # A keystore without the config flag means a previous enable()
            # crashed mid-migration (the flag is flipped last, so the
            # half-migrated state is inert). disable() handles this state
            # cleanly: it decrypts whatever was already encrypted, skips the
            # rest, and removes the keystore.
            raise HermesCryptoError(
                "A keystore exists but encryption is not enabled in config — "
                "a previous 'hermes encrypt enable' was likely interrupted. "
                "Run 'hermes encrypt disable' to roll everything back to "
                "plaintext, then re-run 'hermes encrypt enable'."
            )
        raise HermesCryptoError(
            "A keystore already exists — encryption looks enabled already. "
            "Use 'hermes encrypt status' / 'rotate-key', or 'disable' first."
        )
    if encrypt_databases and not dbcrypt.sqlcipher_available():
        raise HermesCryptoError(
            "Database encryption requested but 'sqlcipher3-wheels' is not "
            "installed. Install it with:  pip install 'hermes-agent[encryption]'"
        )

    _require_no_concurrent_hermes(force=force, operation="enable encryption")

    progress("Setting up keystore…")
    dek = keystore.init_keystore(key_source, passphrase=passphrase, argon2_params=argon2_params)
    result = EnableResult()

    try:
        for target in credential_targets():
            if not target.path.is_file():
                continue
            raw = target.path.read_bytes()
            if detect.is_encrypted(raw):
                result.skipped.append(target.label)
                continue
            progress(f"Encrypting {target.label}…")
            _encrypt_file(target, raw, dek)
            result.encrypted_files.append(target.label)

        if encrypt_databases:
            for db_path in database_targets():
                if not db_path.is_file():
                    continue
                if dbcrypt.is_not_plaintext_sqlite(db_path):
                    result.skipped.append(db_path.name)
                    continue
                progress(f"Encrypting database {db_path.name}…")
                _backup(db_path)
                dbcrypt.encrypt_database(db_path, dek)
                # Verify the encrypted database opens with the key.
                dbcrypt.connect_encrypted(db_path, dek).close()
                result.encrypted_databases.append(db_path.name)

        if encrypt_logs:
            for target in session_targets():
                lock_fd = session_writer._try_lock_session(target.path)
                if lock_fd is None:
                    result.skipped.append(target.label)
                    continue
                try:
                    if not target.path.is_file():
                        continue
                    raw = target.path.read_bytes()
                    if detect.is_encrypted(raw):
                        result.skipped.append(target.label)
                        continue
                    progress(f"Encrypting session {target.label}…")
                    _encrypt_file(target, raw, dek)
                    result.encrypted_sessions.append(target.label)
                finally:
                    session_writer._drop_session_lock(lock_fd, target.path)
            for target in plaintext_log_archive_targets():
                if not target.path.is_file():
                    continue
                raw = target.path.read_bytes()
                if detect.is_encrypted(raw):
                    result.skipped.append(target.label)
                    continue
                progress(f"Encrypting log archive {target.label}…")
                _encrypt_file(target, raw, dek)
                result.encrypted_logs.append(target.label)
    except Exception:
        # Keystore + any partial work is left in place for inspection, but the
        # config flag is NOT set, so the half-migrated state is inert.
        progress("Migration failed — see backups in .encryption/backup/.")
        raise

    progress("Updating config…")
    # wrap the multi-key config writes so a mid-sequence
    # failure (disk full, YAML corruption) still emits an audit event with the
    # partial-write breakdown before re-raising. `enabled` flip stays LAST per
    # AGENTS.md §3.7 — "fully migrated" is defined by that flip landing.
    _apply_config_writes_or_audit_failure(
        [
            ("security.encryption.key_source", key_source),
            ("security.encryption.encrypt_credentials", True),
            ("security.encryption.encrypt_databases", bool(encrypt_databases)),
            ("security.encryption.encrypt_logs", bool(encrypt_logs)),
            ("security.encryption.enabled", True),  # flipped last per §3.7
        ],
        activity=audit.ENCRYPTION_ENABLED,
        key_source=key_source,
    )
    audit.log_event(
        audit.ENCRYPTION_ENABLED,
        audit.SUCCESS,
        key_source=key_source,
        files=len(result.encrypted_files),
        databases=len(result.encrypted_databases),
        sessions=len(result.encrypted_sessions),
        logs=len(result.encrypted_logs),
    )
    return result


def _encrypt_file(target: FileTarget, raw: bytes, dek: bytes) -> None:
    """Encrypt one credential file with backup + round-trip verification."""
    backup_path = _backup(target.path)
    if target.env_framed:
        blob = envelope.encrypt_env(raw, dek)
        check = envelope.decrypt_env(blob, dek)
    else:
        blob = envelope.encrypt(raw, dek)
        check = envelope.decrypt(blob, dek)
    if check != raw:
        raise HermesCryptoError(f"in-memory verification failed for {target.label}")

    atomic_write_private(target.path, blob)

    # Read back from disk and verify before trusting the migration.
    written = target.path.read_bytes()
    roundtrip = (
        envelope.decrypt_env(written, dek)
        if target.env_framed
        else envelope.decrypt(written, dek)
    )
    if roundtrip != raw:
        # restore the envelope atomically (copy-to-tmp + os.replace) so a
        # crash mid-copy can never leave a half-written ciphertext at the
        # live path.
        # switched from ``atomic_write_private(read_bytes())``
        # (which loads the entire backup into memory) to ``atomic_copy``
        # (streaming + atomic-replace). Still atomic at the destination,
        # no longer allocates the whole file.
        atomic_copy(backup_path, target.path)
        raise HermesCryptoError(
            f"post-write verification failed for {target.label} — restored from backup"
        )


def _decrypt_file(target: FileTarget, raw: bytes, dek: bytes) -> None:
    """Decrypt one envelope file with backup + post-write verification."""
    backup_path = _backup(target.path)
    plaintext = (
        envelope.decrypt_env(raw, dek)
        if target.env_framed
        else envelope.decrypt(raw, dek)
    )
    atomic_write_private(target.path, plaintext)
    written = target.path.read_bytes()
    if written != plaintext:
        atomic_copy(backup_path, target.path)
        raise HermesCryptoError(
            f"post-write verification failed for {target.label} — restored from backup"
        )


# ─── Full DEK re-key ─────────────────────────────────────────────────────────


@dataclass
class FullRekeyResult:
    rekeyed_files: List[str] = field(default_factory=list)
    rekeyed_databases: List[str] = field(default_factory=list)
    rekeyed_sessions: List[str] = field(default_factory=list)
    rekeyed_logs: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    recovery_slots_dropped: int = 0
    backups_removed: int = 0


def full_rekey(
    *,
    passphrase: Optional[str] = None,
    progress: ProgressFn = _noop,
    force: bool = False,
    keep_backups: bool = False,
) -> FullRekeyResult:
    """Generate a fresh DEK and re-encrypt every encrypted artifact.

    Follows the same backup → convert → verify discipline as :func:`enable`.
    The keystore is updated *last* so a failure leaves the old DEK able to
    read everything. Recovery slots are dropped — run ``add-recovery`` after.

    By default, ciphertext backups produced during this run are removed on
    success so the retired DEK cannot decrypt them. Pass ``keep_backups=True``
    to retain them (e.g. for forensic recovery).

    Note: SQLCipher ``-wal``/``-shm`` sidecars are best-effort snapshotted
    alongside the main ``.db`` file so a rollback can restore the consistent
    triple. A live writer may hold the sidecars exclusively — stop Hermes
    before ``--force`` for guaranteed rollback safety on databases.
    """
    if not keystore.keystore_exists():
        raise HermesCryptoError("No keystore found — encryption is not enabled.")

    _require_no_concurrent_hermes(force=force, operation="re-key encryption")

    old_dek = keystore.get_cached_dek()
    if old_dek is None:
        raise HermesCryptoError(
            "Keystore is locked — unlock it before running a full re-key."
        )

    new_dek = secrets.token_bytes(32)
    result = FullRekeyResult()
    # Each entry: (live_path, backup_path, kind) where kind is "envelope"
    # or "database". Kind drives (envelope = atomic restore) vs the
    # (database = main + sidecar copy) rollback paths.
    restored: List[tuple[Path, Path, str]] = []

    def _rollback() -> int:
        """Restore every backed-up artifact, counting per-file OSErrors.

        returns the number of files whose restore raised ``OSError``
        (disk full, target locked on Windows, backup unreadable, …). The
        outer ``except`` branch surfaces this count in the
        ``DATA_KEY_REKEY_FAILED`` audit event so an operator who ran
        ``--full --force`` after suspected DEK compromise can tell whether
        rollback actually completed.
        """
        errors = 0
        for path, backup, kind in reversed(restored):
            try:
                if kind == "envelope":
                    # atomic restore — no half-written ciphertext can be
                    # left at the live path if the process dies mid-restore.
                    # ``atomic_copy`` streams the backup
                    # instead of loading it whole into memory (matters when
                    # rollback iterates hundreds of log segments / sessions).
                    atomic_copy(backup, path)
                else:
                    # restore the SQLCipher main file plus any
                    # sidecars together so SQLCipher sees a consistent
                    # (main, -wal, -shm) triple, not a mix of new main
                    # plus stale WAL. Atomic-replace for the main file
                    # via atomic_copy; sidecars below.
                    atomic_copy(backup, path)
                    for suffix in ("-wal", "-shm"):
                        live_sidecar = path.with_name(path.name + suffix)
                        backup_sidecar = backup.with_name(backup.name + suffix)
                        try:
                            live_sidecar.unlink(missing_ok=True)
                        except OSError:
                            pass
                        if backup_sidecar.is_file():
                            try:
                                # atomic + streaming
                                atomic_copy(backup_sidecar, live_sidecar)
                            except OSError:
                                pass
            except OSError:
                errors += 1
        return errors

    try:
        # Phase ordering invariant: envelopes -> databases -> keystore flip.
        # Tests that need to exercise rollback over already-rekeyed DBs must
        # inject the failure at or after replace_data_key (the only work
        # after the DB loop). There is no envelope phase after DBs.
        #
        # Restore-list invariant: append before rekey so a failure after
        # atomic_write_private / atomic_replace still leaves the path in
        # the rollback set (idempotent restore if rekey never ran).
        for target in envelope_encrypted_targets():
            if not target.path.is_file():
                continue
            raw = target.path.read_bytes()
            if not detect.is_encrypted(raw):
                result.skipped.append(target.label)
                continue
            progress(f"Re-keying {target.label}…")
            backup_path = _backup(target.path)
            restored.append((target.path, backup_path, "envelope"))
            _rekey_file(target, raw, old_dek, new_dek)
            # route by the target's explicit ``kind`` tag, not by
            # sniffing the path. A HERMES_HOME whose ancestor is literally
            # named ``logs`` would otherwise mis-bucket every credential
            # file under it.
            if target.kind == "session":
                result.rekeyed_sessions.append(target.label)
            elif target.kind == "log":
                result.rekeyed_logs.append(target.label)
            else:
                result.rekeyed_files.append(target.label)

        for db_path in database_targets():
            if not db_path.is_file() or dbcrypt.is_plaintext_sqlite(db_path):
                continue
            progress(f"Re-keying database {db_path.name}…")
            backup_path = _backup(db_path)
            restored.append((db_path, backup_path, "database"))
            dbcrypt.rekey_database(db_path, old_dek, new_dek)
            result.rekeyed_databases.append(db_path.name)

        progress("Updating keystore…")
        result.recovery_slots_dropped = keystore.replace_data_key(
            new_dek, passphrase=passphrase
        )
    except Exception as exc:
        progress("Re-key failed — restoring from backups…")
        # run rollback first, then surface its OSError count in the
        # audit event so an operator can tell whether the restore actually
        # completed. ``finally`` guarantees the event fires even if
        # ``_rollback()`` itself raises something other than ``OSError``.
        rollback_errors = 0
        try:
            rollback_errors = _rollback()
        finally:
            audit.log_event(
                audit.DATA_KEY_REKEY_FAILED,
                audit.FAILURE,
                reason=type(exc).__name__,
                restored_count=len(restored),
                rollback_errors=rollback_errors,
            )
        raise

    if not keep_backups and restored:
        backup_paths = [backup for _, backup, _ in restored]
        result.backups_removed = _remove_rekey_run_backups(backup_paths)

    return result


def _rekey_file(
    target: FileTarget,
    raw: bytes,
    old_dek: bytes,
    new_dek: bytes,
) -> None:
    """Decrypt *raw* with *old_dek*, re-seal with *new_dek*, verify on disk."""
    if target.env_framed:
        plaintext = envelope.decrypt_env(raw, old_dek)
        blob = envelope.encrypt_env(plaintext, new_dek)
        check = envelope.decrypt_env(blob, new_dek)
    else:
        plaintext = envelope.decrypt(raw, old_dek)
        blob = envelope.encrypt(plaintext, new_dek)
        check = envelope.decrypt(blob, new_dek)
    if check != plaintext:
        raise HermesCryptoError(f"in-memory verification failed for {target.label}")

    atomic_write_private(target.path, blob)

    written = target.path.read_bytes()
    roundtrip = (
        envelope.decrypt_env(written, new_dek)
        if target.env_framed
        else envelope.decrypt(written, new_dek)
    )
    if roundtrip != plaintext:
        raise HermesCryptoError(
            f"post-write verification failed for {target.label}"
        )


# ─── Disable ──────────────────────────────────────────────────────────────────


def disable(
    *,
    passphrase: Optional[str] = None,
    recovery_code: Optional[str] = None,
    progress: ProgressFn = _noop,
    force: bool = False,
) -> EnableResult:
    """Decrypt every encrypted file/database back to plaintext and remove the
    keystore. Clears the ``security.encryption.enabled`` config flag.
    """
    if not keystore.keystore_exists():
        raise HermesCryptoError("No keystore found — encryption is not enabled.")

    _require_no_concurrent_hermes(force=force, operation="disable encryption")

    dek = keystore.get_cached_dek()
    if dek is None:
        progress("Unlocking keystore…")
        dek = keystore.unlock(passphrase=passphrase, recovery_code=recovery_code)

    result = EnableResult()

    for target in credential_targets():
        if not target.path.is_file():
            continue
        raw = target.path.read_bytes()
        if not detect.is_encrypted(raw):
            continue
        progress(f"Decrypting {target.label}…")
        _decrypt_file(target, raw, dek)
        result.encrypted_files.append(target.label)

    for db_path in database_targets():
        if not db_path.is_file() or dbcrypt.is_plaintext_sqlite(db_path):
            continue
        progress(f"Decrypting database {db_path.name}…")
        _backup(db_path)
        dbcrypt.decrypt_database(db_path, dek)
        result.encrypted_databases.append(db_path.name)

    # Sessions are always inspected, never gated on the config flag, so a
    # half-finished migration cannot strand an encrypted session file as
    # unreadable when the operator turns encryption back off. AGENTS.md §3.3.
    for target in session_targets():
        if not target.path.is_file():
            continue
        raw = target.path.read_bytes()
        if not detect.is_encrypted(raw):
            continue
        progress(f"Decrypting session {target.label}…")
        _decrypt_file(target, raw, dek)
        result.encrypted_sessions.append(target.label)

    progress("Removing keystore…")
    keystore.destroy_keystore()
    progress("Updating config…")
    # flip `enabled` FIRST in disable so a downstream
    # `encrypt_logs` write failure leaves the runtime in a safe "off" state —
    # `logs_encryption_active()` short-circuits on `enabled=False`, so a
    # stranded `encrypt_logs=True` flag becomes inert until the operator
    # reconciles. This is the inverse of `enable`'s flipped-last pattern
    # (§3.7): for disable, "fully disabled" is defined by `enabled=False`
    # landing first. Do NOT re-order without re-reading this rationale.
    _apply_config_writes_or_audit_failure(
        [
            ("security.encryption.enabled", False),  # flipped first
            ("security.encryption.encrypt_logs", False),
        ],
        activity=audit.ENCRYPTION_DISABLED,
    )
    audit.log_event(
        audit.ENCRYPTION_DISABLED,
        audit.SUCCESS,
        files=len(result.encrypted_files),
        databases=len(result.encrypted_databases),
        sessions=len(result.encrypted_sessions),
    )
    progress("Encryption disabled. Plaintext backups remain in .encryption/backup/.")
    return result


# ─── Status ───────────────────────────────────────────────────────────────────


def status() -> Dict[str, object]:
    """Return a structured snapshot for ``hermes encrypt status``."""
    from . import encryption_settings, is_encryption_enabled
    from utils import is_truthy_value

    settings = encryption_settings()
    files: List[Dict[str, str]] = []
    for target in credential_targets():
        if not target.path.is_file():
            state = "missing"
        elif detect.is_encrypted(target.path.read_bytes()):
            state = "encrypted"
        else:
            state = "plaintext"
        files.append({"name": target.label, "state": state})

    databases: List[Dict[str, str]] = []
    for db_path in database_targets():
        if not db_path.is_file():
            state = "missing"
        elif dbcrypt.is_plaintext_sqlite(db_path):
            state = "plaintext"
        else:
            state = "encrypted"
        databases.append({"name": db_path.name, "state": state})

    return {
        "enabled": is_encryption_enabled(),
        "encrypt_logs": is_truthy_value(settings.get("encrypt_logs", False)),
        "config_key_source": settings.get("key_source", ""),
        "keystore_exists": keystore.keystore_exists(),
        "primary_slot": keystore.primary_slot_type(),
        "has_recovery": keystore.has_recovery_slot(),
        "keyring_secure": keystore.keyring_is_secure(),
        "sqlcipher_available": dbcrypt.sqlcipher_available(),
        "files": files,
        "databases": databases,
        "sessions": session_summary(),
        "backups": backup_summary(),
    }


def session_summary() -> Dict[str, int]:
    """Return counts of session transcripts by state — encrypted / plaintext / locked.

    Aggregated rather than per-file because the sessions directory can grow
    to thousands of files on a long-lived install. ``locked`` is the count of
    sessions whose lockfile is currently held by another live writer.
    """
    encrypted = plaintext = locked = 0
    for target in session_targets():
        if not target.path.is_file():
            continue
        try:
            head = target.path.read_bytes()[:32]
        except OSError:
            continue
        if detect.is_encrypted(head):
            encrypted += 1
        else:
            plaintext += 1
        # Read-only probe — must not delete a stale lockfile as a side effect.
        if session_writer._probe_session_lock_held(target.path):
            locked += 1
    return {
        "count": encrypted + plaintext,
        "encrypted": encrypted,
        "plaintext": plaintext,
        "locked": locked,
    }


# ─── Sweep — close the crash-window plaintext leak ───────────────────────────


def sweep_sessions(*, progress: ProgressFn = _noop) -> Dict[str, object]:
    """Encrypt plaintext sessions/*.jsonl whose writer is no longer live.

    Skips any session whose lockfile is currently held by another process —
    that file is still being written. For each abandoned plaintext session
    (no lockfile, or lockfile present but acquirable because the previous
    writer is dead), takes the lock, encrypts the file, releases+removes
    the lock.

    Best-effort: per-file failures emit ``audit.SESSION_ENCRYPT_FAILED``
    (critical severity) and the function continues. Never raises out — this
    runs during startup; a crash here would break the whole gateway.

    Returns a summary dict with ``swept`` / ``skipped_locked`` / ``errors``.
    """
    swept: List[str] = []
    skipped_locked: List[str] = []
    errors: List[Dict[str, str]] = []

    try:
        targets = session_targets()
    except Exception:
        return {"swept": swept, "skipped_locked": skipped_locked, "errors": errors}

    if not targets:
        return {"swept": swept, "skipped_locked": skipped_locked, "errors": errors}

    dek: Optional[bytes] = None

    for target in targets:
        path = target.path
        try:
            if not path.is_file():
                continue
            lock_fd = session_writer._try_lock_session(path)
            if lock_fd is None:
                skipped_locked.append(target.label)
                continue

            try:
                if not path.is_file():
                    continue
                raw = path.read_bytes()
                if not raw or detect.is_encrypted(raw):
                    continue
                if dek is None:
                    from . import get_data_key

                    dek = get_data_key()
                progress(f"Sweeping abandoned session {target.label}…")
                _encrypt_file(target, raw, dek)
                swept.append(target.label)
            finally:
                session_writer._drop_session_lock(lock_fd, path)
        except Exception as exc:
            audit.log_event(
                audit.SESSION_ENCRYPT_FAILED,
                audit.FAILURE,
                path=str(path),
                reason=type(exc).__name__,
            )
            errors.append({"name": target.label, "error": type(exc).__name__})

    return {"swept": swept, "skipped_locked": skipped_locked, "errors": errors}


# ─── Config writes ────────────────────────────────────────────────────────────


def _set_config(dotted_key: str, value: object) -> None:
    from utils import atomic_roundtrip_yaml_update

    atomic_roundtrip_yaml_update(get_config_path(), dotted_key, value)


def _apply_config_writes_or_audit_failure(
    writes,
    *,
    activity: str,
    **failure_detail: object,
) -> None:
    """Apply *writes* sequentially; on the first failure emit a FAILURE-outcome
    audit event with the partial-write breakdown, then re-raise.

    *writes* is a list of ``(dotted_key, value)`` pairs. The audit detail
    carries ``config_written`` (the dotted-key names that landed) and
    ``config_not_written`` (the rest) plus any ``**failure_detail`` passed in.
    Detail values are non-secret per AGENTS.md §3.5 — dotted config paths
    only, never values.

    Closes a hardening gap: a mid-sequence ``_set_config`` failure (disk
    full, YAML corruption) used to leave runtime in a half-migrated state
    with no audit trail. Now operators get a paging signal (severity
    ``warning`` via ``_severity_for``) plus the per-key breakdown that
    tells them which flag is stuck.
    """
    written: list[str] = []
    try:
        for dotted_key, value in writes:
            _set_config(dotted_key, value)
            written.append(dotted_key)
    except Exception:
        not_written = [k for k, _ in writes if k not in written]
        audit.log_event(
            activity,
            audit.FAILURE,
            config_written=written,
            config_not_written=not_written,
            **failure_detail,
        )
        raise
