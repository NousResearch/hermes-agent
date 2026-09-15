"""Authenticated browser-to-host attachment staging for Hermes Webapp."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import re
import secrets
import stat
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from hermes_constants import WEBAPP_ATTACHMENT_MAX_BYTES
from hermes_cli.install_identity import get_install_id
from hermes_cli.profile_incarnation import (
    ensure_profile_incarnation,
    profile_incarnation_lease,
)
from hermes_cli.web_deps import late


router = APIRouter()
_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")
_MAX_UPLOAD_BYTES = WEBAPP_ATTACHMENT_MAX_BYTES
_CHUNK_BYTES = 1024 * 1024
_UPLOAD_RETENTION_SECONDS = 7 * 24 * 60 * 60
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(value: str | None) -> str:
    name = Path(str(value or "attachment")).name
    clean = _SAFE_FILENAME.sub("-", name).strip(".-")
    return (clean or "attachment")[-120:]


def _prune_stale_uploads(root: Path, *, now: float | None = None) -> None:
    """Bound abandoned browser-picker staging without following symlinks."""
    cutoff = (time.time() if now is None else now) - _UPLOAD_RETENTION_SECONDS
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for entry in entries:
        if not entry.name.startswith("web-"):
            continue
        try:
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISREG(metadata.st_mode) and metadata.st_mtime < cutoff:
                entry.unlink()
        except OSError:
            continue


def _resolve_upload_generation(profile: str | None) -> tuple[Path, str | None]:
    """Resolve one profile home and capture the named generation it denotes."""
    from hermes_constants import get_hermes_home, named_profile_home_is_unavailable

    with _profile_scope(profile) as scoped_home:
        home = Path(scoped_home or get_hermes_home())
        if named_profile_home_is_unavailable(home):
            raise HTTPException(status_code=404, detail="Profile home is unavailable")
        try:
            incarnation = ensure_profile_incarnation(home)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="Profile home is unavailable",
            ) from exc
        return home, incarnation


def _publish_staged_upload(
    staged,
    profile_home: Path,
    expected_incarnation: str | None,
    filename: str,
) -> Path:
    """Publish staged bytes while the captured profile generation is leased."""
    from hermes_constants import named_profile_home_is_unavailable

    target: Path | None = None
    try:
        with profile_incarnation_lease(
            profile_home,
            expected_incarnation,
            require_incarnation=expected_incarnation is not None,
        ):
            if named_profile_home_is_unavailable(profile_home):
                raise FileNotFoundError(profile_home)
            upload_root = profile_home / "uploads"
            try:
                upload_root.mkdir(parents=False, exist_ok=True, mode=0o700)
                upload_root.chmod(0o700)
            except FileNotFoundError:
                raise
            except PermissionError as exc:
                raise HTTPException(
                    status_code=403,
                    detail="Upload directory is not writable",
                ) from exc
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not create upload directory: {exc}",
                ) from exc

            _prune_stale_uploads(upload_root)
            target = upload_root / (
                f"web-{secrets.token_hex(8)}-{_safe_filename(filename)}"
            )
            completed = False
            try:
                fd = os.open(
                    target,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with os.fdopen(fd, "wb") as handle:
                    staged.seek(0)
                    while chunk := staged.read(_CHUNK_BYTES):
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                completed = True
            finally:
                if not completed and target is not None:
                    try:
                        target.unlink(missing_ok=True)
                    except OSError:
                        pass
            return target
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Profile was deleted or replaced during upload",
        ) from exc
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not stage file: {exc}") from exc


@router.post("/api/chat/file-upload")
async def upload_chat_file(
    file: UploadFile = File(...),
    profile: str | None = None,
):
    """Stage one user-selected browser file under the active Hermes profile.

    The client never chooses a server path. A unique 0600 file under
    ``$HERMES_HOME/uploads`` is returned for the existing ``file.attach`` flow.
    """
    profile_home, expected_incarnation = await asyncio.to_thread(
        _resolve_upload_generation,
        profile,
    )
    try:
        # FastAPI has already spooled the complete upload outside the profile.
        # Check the actual bytes, then publish that spool under the captured lease.
        total = await asyncio.to_thread(file.file.seek, 0, os.SEEK_END)
        if total > _MAX_UPLOAD_BYTES:
            cap_mib = _MAX_UPLOAD_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"File is too large; cap is {cap_mib} MiB",
            )
        target = await asyncio.to_thread(
            _publish_staged_upload,
            file.file,
            profile_home,
            expected_incarnation,
            file.filename or "attachment",
        )
    finally:
        await file.close()

    result = {"ok": True, "path": str(target), "size": total}
    install_id = await asyncio.to_thread(get_install_id)
    if install_id:
        # Identity is already persisted by Hermes. If it is unavailable, keep
        # the legacy path/byte-upload flow rather than invent an ephemeral id.
        result["staged_upload"] = {
            "install_id": install_id,
            "path": str(target),
            "profile_home": str(profile_home),
            "profile_incarnation": expected_incarnation,
        }
    return result
