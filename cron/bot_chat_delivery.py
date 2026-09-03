"""Durable cron delivery into canonical Bot Chat sessions.

Bot Chat turns cannot start a second process while CLI/TUI/Desktop owns the
same session: the active-session lease correctly rejects stale transcript
writers. This module durably admits each completed cron output, attempts the
canonical one-shot chat lane immediately when unowned, and retries retained
records on later scheduler ticks (#99956).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None

from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_cli.config import load_config

logger = logging.getLogger(__name__)

QUEUE_DIR_NAME = "bot_chat_delivery_queue"


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


@contextlib.contextmanager
def _file_lock(queue_dir: Path, name: str):
    """Take one blocking cross-process lock inside *queue_dir*."""
    _ensure_private_dir(queue_dir)
    fh = open(queue_dir / name, "a+b")
    try:
        if os.name == "nt" and msvcrt:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                fh.write(b"\0")
                fh.flush()
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        elif fcntl:
            fcntl.flock(fh, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt" and msvcrt:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            elif fcntl:
                fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()


def _queue_dir(source_home: Path) -> Path:
    return Path(source_home) / "cron" / QUEUE_DIR_NAME


def _queue_delivery(job: dict, message: str, profile: str, source_home: Path) -> str:
    """Atomically persist one turn before the scheduler acknowledges it."""
    delivery_id = uuid.uuid4().hex
    record = {
        "id": delivery_id,
        "job_id": str(job.get("id") or "?"),
        "job_name": str(job.get("name") or job.get("id") or "?"),
        "profile": str(profile or ""),
        "message": message,
        "created_at": time.time(),
    }
    queue_dir = _queue_dir(source_home)
    with _file_lock(queue_dir, ".queue.lock"):
        path = queue_dir / f"{time.time_ns():020d}-{delivery_id}.json"
        tmp = path.with_suffix(f".json.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            if os.name != "nt":
                dir_fd = os.open(queue_dir, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    return delivery_id


def _target_home(profile: str, source_home: Path) -> Path:
    if not profile:
        return Path(source_home)
    from hermes_cli.profiles import get_profile_dir

    return get_profile_dir(profile)


def _target_has_live_owner(profile: str, source_home: Path) -> Optional[bool]:
    """Return whether the target Bot Chat is leased, or None if unknown."""
    try:
        target_home = _target_home(profile, source_home)
        db_path = target_home / "state.db"
        if not db_path.exists():
            return False

        from hermes_state import SessionDB

        db = SessionDB(db_path=db_path, read_only=True)
        try:
            row = db.get_session_by_title("Bot Chat")
            if not row:
                return False
            session_id = str(row.get("id") or "")
            try:
                session_id = str(db.get_compression_tip(session_id) or session_id)
            except Exception:
                pass
        finally:
            db.close()

        from hermes_cli.active_sessions import active_session_registry_snapshot

        return any(
            str(entry.get("session_id") or "") == session_id
            for entry in active_session_registry_snapshot(registry_home=target_home)
        )
    except Exception:
        logger.warning(
            "Could not prove Bot Chat ownership for profile '%s'; retaining queued delivery",
            profile or "(own)",
            exc_info=True,
        )
        return None


def _delivery_timeout() -> int:
    try:
        cfg = load_config()
        value = int(cfg.get("cron", {}).get("bot_chat_delivery_timeout_seconds", 600))
        return value if value > 0 else 600
    except Exception:
        return 600


def _run_delivery(record: dict) -> Optional[str]:
    """Run one already-durable record through the canonical one-shot CLI lane."""
    profile = str(record.get("profile") or "")
    job_id = str(record.get("job_id") or "?")
    message = str(record.get("message") or "")

    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        argv = [hermes_bin]
    else:
        try:
            import importlib.util

            if importlib.util.find_spec("hermes_cli") is not None:
                argv = [sys.executable, "-m", "hermes_cli.main"]
            else:
                return "bot-chat delivery failed: hermes CLI not resolvable"
        except Exception:
            return "bot-chat delivery failed: hermes CLI not resolvable"

    env = os.environ.copy()
    if profile:
        argv += ["-p", profile]
        env.pop("HERMES_HOME", None)

    query_file = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=".txt",
            prefix="hermes-cron-botchat-",
            delete=False,
        ) as fh:
            fh.write(message)
            query_file = fh.name

        argv += [
            "chat",
            "--in",
            "~",
            "-c",
            "Bot Chat",
            "--create-if-missing",
            "-Q",
            "--query-file",
            query_file,
        ]
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_delivery_timeout(),
            env=env,
            creationflags=windows_hide_flags(),
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-500:]
            msg = (
                f"bot-chat delivery to profile '{profile or '(own)'}' failed "
                f"(exit {result.returncode})" + (f": {tail}" if tail else "")
            )
            logger.warning("Job '%s': %s", job_id, msg)
            return msg
        logger.info(
            "Job '%s': delivered to Bot Chat of profile '%s'",
            job_id,
            profile or "(own)",
        )
        return None
    except subprocess.TimeoutExpired:
        msg = (
            f"bot-chat delivery to profile '{profile or '(own)'}' timed out "
            f"after {_delivery_timeout()}s (the bot's turn may still complete; "
            "the durable record is retained for retry)"
        )
        logger.warning("Job '%s': %s", job_id, msg)
        return msg
    except Exception as exc:
        msg = f"bot-chat delivery failed: {str(exc) or type(exc).__name__}"
        logger.warning("Job '%s': %s", job_id, msg, exc_info=True)
        return msg
    finally:
        if query_file:
            try:
                os.unlink(query_file)
            except OSError:
                pass


def drain_queue(source_home: Path) -> tuple[int, Optional[str]]:
    """Retry retained turns in order per target profile.

    One blocked profile does not head-of-line block independent profiles.
    """
    delivered = 0
    first_error: Optional[str] = None
    blocked_profiles: set[str] = set()
    queue_dir = _queue_dir(source_home)
    if not queue_dir.exists():
        return 0, None
    with _file_lock(queue_dir, ".drain.lock"):
        for path in sorted(queue_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(record, dict) or not record.get("message"):
                    raise ValueError("invalid queued bot-chat delivery")
            except Exception as exc:
                msg = f"bot-chat delivery queue record unreadable: {path.name}: {exc}"
                logger.error(msg)
                first_error = first_error or msg
                continue

            profile = str(record.get("profile") or "")
            if profile in blocked_profiles:
                continue
            owner = _target_has_live_owner(profile, source_home)
            if owner is not False:
                reason = (
                    "target session has a live owner"
                    if owner
                    else "target ownership is unknown"
                )
                first_error = first_error or reason
                blocked_profiles.add(profile)
                continue

            error = _run_delivery(record)
            if error:
                first_error = first_error or error
                blocked_profiles.add(profile)
                continue
            try:
                path.unlink()
            except OSError as exc:
                msg = f"delivered bot-chat queue record could not be acknowledged: {exc}"
                logger.warning(msg)
                first_error = first_error or msg
                blocked_profiles.add(profile)
                continue
            delivered += 1
    return delivered, first_error


def deliver(job: dict, content: str, profile: str, source_home: Path) -> Optional[str]:
    """Durably admit one cron output and attempt immediate ordered delivery.

    Returns None once the output is on disk. A live owner does not fail the
    job: the turn retries on later scheduler ticks after the owner releases.
    """
    job_id = job.get("id", "?")
    job_name = job.get("name", job_id)
    message = (
        f'[Cronjob "{job_name}" output — scheduled job, not the user. '
        f"Review it, act on anything that needs action, and summarize "
        f"for the chat.]\n\n{content}"
    )
    try:
        delivery_id = _queue_delivery(job, message, profile, Path(source_home))
    except Exception as exc:
        msg = (
            f"bot-chat delivery could not be durably queued: "
            f"{str(exc) or type(exc).__name__}"
        )
        logger.warning("Job '%s': %s", job_id, msg, exc_info=True)
        return msg

    delivered, deferred_reason = drain_queue(Path(source_home))
    if deferred_reason:
        logger.info(
            "Job '%s': Bot Chat delivery %s admitted; queue remains pending (%s)",
            job_id,
            delivery_id,
            deferred_reason,
        )
    elif delivered:
        logger.info("Job '%s': durably queued Bot Chat delivery completed", job_id)
    return None
