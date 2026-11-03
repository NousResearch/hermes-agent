"""Preserve WhatsApp authentication state before dashboard re-pairing."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from fastapi import HTTPException


def _assert_whatsapp_bridge_stopped(session_path: Path) -> None:
    from gateway.status import _pid_exists, get_process_start_time

    pid_file = session_path / "bridge.pid"
    if not pid_file.exists():
        return
    unknown = "Cannot verify the WhatsApp bridge is stopped. Check its bridge.pid before re-pairing."
    try:
        lines = pid_file.read_text(encoding="utf-8-sig").splitlines()
        pid = int(lines[0])
        if pid <= 0:
            raise ValueError("invalid bridge PID")
    except (OSError, ValueError, IndexError) as exc:
        raise HTTPException(status_code=409, detail=unknown) from exc
    if not _pid_exists(pid):
        return
    try:
        recorded_start = int(lines[1])
        if recorded_start <= 0:
            raise ValueError("invalid bridge start time")
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=409, detail=unknown) from exc
    current_start = get_process_start_time(pid)
    if current_start is None:
        raise HTTPException(status_code=409, detail=unknown)
    if current_start == recorded_start:
        raise HTTPException(status_code=409, detail="The WhatsApp bridge is still running. Stop it before re-pairing.")
    # Both fingerprints are available and differ. This is a recycled PID,
    # not evidence that the old bridge still owns the session.


def _backup_whatsapp_session(session_path: Path) -> None:
    """Called while onboarding owns the session reservation. Never delete keys."""
    _assert_whatsapp_bridge_stopped(session_path)
    if session_path.is_symlink():
        raise HTTPException(status_code=409, detail="The WhatsApp session is a symbolic link. Back it up manually before re-pairing.")
    backup = None
    try:
        if session_path.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup = Path(tempfile.mkdtemp(prefix=f"session-backup-{stamp}-", dir=session_path.parent)) / "session"
            session_path.rename(backup)
            session_path.mkdir(mode=0o700)
    except OSError as exc:
        detail = f"Could not back up the WhatsApp session: {exc}. Pairing was not started."
        if backup is not None and backup.exists():
            detail += f" The previous session is preserved at {backup}."
        raise HTTPException(status_code=500, detail=detail) from exc
