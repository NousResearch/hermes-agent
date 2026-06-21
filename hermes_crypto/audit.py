"""Structured security-audit log for encryption-at-rest events.

Inspired by NVIDIA OpenShell's OCSF event auditing — adapted to a single,
dependency-free JSONL file rather than the full OCSF schema. One JSON object
per line is appended to ``<HERMES_HOME>/logs/security-audit.jsonl``. When the
active file reaches 10 MiB it rolls over into numbered segments (``.1`` …
``.5``); older segments are dropped once the retention cap is reached.

What this log is for: knowing *when* the keystore was unlocked, enabled,
disabled, or rotated — and, importantly, when an unlock **failed**, which is
the signal that someone is trying keys against your encrypted data.

Hard rule: this log records **events, never key material**. Callers pass only
non-secret descriptive detail (slot type, key source, file count). Mirrors the
discipline of ``agent/redact.py``.

Every function here is best-effort and never raises — an audit-write failure
must not break an encryption operation.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from ._lockfile import _acquire_exclusive, _open_lockfile, _release_exclusive

# ── Activity identifiers (stable strings; safe to grep / alert on) ──────────
KEYSTORE_CREATED = "keystore_created"
KEYSTORE_UNLOCKED = "keystore_unlocked"
KEYSTORE_UNLOCK_FAILED = "keystore_unlock_failed"
KEYSTORE_ROTATED = "keystore_rotated"
DATA_KEY_REKEYED = "data_key_rekeyed"
DATA_KEY_REKEY_FAILED = "data_key_rekey_failed"
KEYSTORE_DESTROYED = "keystore_destroyed"
RECOVERY_CODE_ADDED = "recovery_code_added"
ENCRYPTION_ENABLED = "encryption_enabled"
ENCRYPTION_DISABLED = "encryption_disabled"
MIGRATION_BLOCKED = "migration_blocked"
MIGRATION_ENUMERATION_UNAVAILABLE = "migration_enumeration_unavailable"
BACKUPS_CLEANED = "backups_cleaned"
BACKUPS_REMOVED_POST_REKEY = "backups_removed_post_rekey"
DATA_KEY_UNAVAILABLE = "data_key_unavailable"
LOG_ENCRYPT_FAILED = "log_encrypt_failed"
SESSION_ENCRYPT_FAILED = "session_encrypt_failed"

# Outcomes
SUCCESS = "success"
FAILURE = "failure"
INFO = "info"

# Size-based rotation — same defaults as ``log_handler.build_rotating_handler``.
MAX_SEGMENT_BYTES = 10 * 1024 * 1024
MAX_ROTATED_SEGMENTS = 5
_ROTATE_LOCK_SUFFIX = ".rotate-lock"


def audit_log_path() -> Path:
    """Return the path to the security-audit log."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "logs" / "security-audit.jsonl"


# Failures that should page, not just warn — a failed unlock means someone is
# trying keys against your data; a failed log/session encrypt means current
# prompts and responses are hitting disk in cleartext while the operator
# believes otherwise. A failed ``--full`` rekey is the mitigation itself
# failing after suspected DEK compromise; in scripted/automated contexts the
# operator may never see the CLI's red banner, so paging is the only reliable
# signal that the compromised DEK is still in force on disk.
CRITICAL_FAILURES = frozenset(
    {
        KEYSTORE_UNLOCK_FAILED,
        LOG_ENCRYPT_FAILED,
        SESSION_ENCRYPT_FAILED,
        DATA_KEY_REKEY_FAILED,
    }
)


def _severity_for(outcome: str, activity: str) -> str:
    """Derive a severity from the outcome."""
    if outcome == FAILURE:
        return "critical" if activity in CRITICAL_FAILURES else "warning"
    return "info"


def _rotated_segment_path(path: Path, index: int) -> Path:
    """Return the path for rotated segment *index* (1 = newest archive)."""
    return path.with_name(f"{path.name}.{index}")


def _maybe_rotate(path: Path) -> None:
    """Roll *path* over when it reaches :data:`MAX_SEGMENT_BYTES`.

    Mirrors :class:`logging.handlers.RotatingFileHandler` naming
    (``security-audit.jsonl.1`` … ``.5``). Best-effort; never raises.
    """
    if not path.is_file():
        return
    try:
        if path.stat().st_size < MAX_SEGMENT_BYTES:
            return
    except OSError:
        return

    lock_path = path.with_name(path.name + _ROTATE_LOCK_SUFFIX)
    lock_fd: int | None = None
    try:
        # Kernel-released advisory lock (fcntl.flock / msvcrt.locking) — the
        # lock evaporates on process death even if .rotate-lock stays on disk,
        # so a SIGKILL'd previous rotator never permanently blocks future
        # rotation. (Older versions of this function used O_EXCL on the lockfile
        # itself, which sticks across crashes and broke rotation indefinitely —
        # the exact defect ISSUES #12 was supposed to close.)
        try:
            lock_fd = _open_lockfile(lock_path)
            _acquire_exclusive(lock_fd)
        except OSError:
            # Either the open itself failed (permissions, full disk) or the
            # acquire conflicted with a *live* rotator. The append below is
            # still safe; drop the fd if we got that far.
            if lock_fd is not None:
                try:
                    os.close(lock_fd)
                except OSError:
                    pass
                lock_fd = None
            return

        try:
            if not path.is_file() or path.stat().st_size < MAX_SEGMENT_BYTES:
                return
        except OSError:
            return

        oldest = _rotated_segment_path(path, MAX_ROTATED_SEGMENTS)
        try:
            oldest.unlink(missing_ok=True)
        except (OSError, TypeError):
            try:
                if oldest.is_file():
                    oldest.unlink()
            except OSError:
                pass

        for index in range(MAX_ROTATED_SEGMENTS - 1, 0, -1):
            src = _rotated_segment_path(path, index)
            dst = _rotated_segment_path(path, index + 1)
            if not src.is_file():
                continue
            try:
                os.replace(src, dst)
            except OSError:
                try:
                    if dst.is_file():
                        dst.unlink()
                    os.replace(src, dst)
                except OSError:
                    pass

        archive = _rotated_segment_path(path, 1)
        try:
            if archive.is_file():
                archive.unlink()
            os.replace(path, archive)
        except OSError:
            return

        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        os.close(fd)
        # Best-effort directory fsync so the rename chain above is durable —
        # a crash right after rotation must not lose the archived segments.
        # Opening a directory fails on Windows; the renames still land.
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except Exception:
        pass
    finally:
        if lock_fd is not None:
            _release_exclusive(lock_fd)
            try:
                os.close(lock_fd)
            except OSError:
                pass
            try:
                lock_path.unlink()
            except OSError:
                pass


def log_event(activity: str, outcome: str = INFO, **detail: Any) -> None:
    """Append one structured event to the security-audit log.

    *detail* must contain only non-secret descriptive values. Never raises.
    """
    try:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "activity": activity,
            "outcome": outcome,
            "severity": _severity_for(outcome, activity),
            "pid": os.getpid(),
            "detail": {k: v for k, v in detail.items() if v is not None},
        }
        # a caller passing a non-JSON-serialisable detail
        # value would silently lose the entire event under the outer
        # try/except. Retry with ``default=repr`` so the event still lands
        # (with the offending value as its repr) instead of vanishing —
        # the audit log's job is to survive caller mistakes.
        try:
            line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        except TypeError:
            line = (
                json.dumps(
                    record, ensure_ascii=False, sort_keys=True, default=repr
                ) + "\n"
            ).encode("utf-8")
        path = audit_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # harden the audit-log dir to 0o700 on POSIX so the
        # JSONL stream (which carries non-secret but operator-sensitive
        # security event metadata) isn't world-readable. Best-effort —
        # Windows + cross-FS mounts are silent no-ops.
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
        _maybe_rotate(path)
        # O_APPEND keeps concurrent writers from clobbering each other; the
        # 0o600 mode is applied only when the file is first created.
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except Exception:
        # Audit logging is best-effort — never break the caller.
        pass


def read_recent(limit: int = 10) -> List[Dict[str, Any]]:
    """Return the most recent audit events (newest last), or [] on any error."""
    try:
        path = audit_log_path()
        candidates = [
            _rotated_segment_path(path, index)
            for index in range(MAX_ROTATED_SEGMENTS, 0, -1)
        ] + [path]
        lines: List[str] = []
        for candidate in candidates:
            if not candidate.is_file():
                continue
            lines.extend(
                candidate.read_text(encoding="utf-8", errors="replace").splitlines()
            )
        if not lines:
            return []
        events: List[Dict[str, Any]] = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        return events
    except Exception:
        return []
