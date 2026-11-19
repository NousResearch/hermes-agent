"""Browser uploads staged in a profile home, and the receipt ``file.attach`` redeems.

``POST /api/chat/file-upload`` publishes a picked file at ``<profile home>/uploads/web-<hex>-<name>``
and returns a ``staged_upload`` descriptor: that path, the profile home and this installation's id.
The gateway's ``file.attach`` trusts the descriptor only after resolving it back to exactly that
layout, so the writer and the reader of the layout live here together. The file routes read a staged
file back from this host by the same layout when the profile's workspace is an SSH machine.

A named profile that is missing or deleted is refused on both sides by the rule every named-home
write follows (``hermes_constants.assert_named_profile_home_live`` / ``mkdir_under_hermes_home``):
an upload never recreates a deleted home, and a receipt from one is never redeemed.
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
import os
from pathlib import Path
import re
import secrets
import stat
import time

from fastapi import HTTPException

from hermes_constants import WEBAPP_ATTACHMENT_MAX_BYTES
from hermes_cli.install_identity import get_install_id
from hermes_cli.web_deps import late


_is_current_profile = late("_is_current_profile", "hermes_cli.web_server_profiles")
_resolve_profile_dir = late("_resolve_profile_dir", "hermes_cli.web_server_profiles")
_UPLOADS_DIR = "uploads"
_STAGED_PREFIX = "web-"
_CHUNK_BYTES = 1024 * 1024
_UPLOAD_RETENTION_SECONDS = 7 * 24 * 60 * 60
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(value: str | None) -> str:
    name = Path(str(value or "attachment"))
    # Sanitizing the whole name would turn an all-non-ASCII stem into a
    # leading dot, then strip the extension separator needed by read_file.
    stem = _SAFE_FILENAME.sub("-", name.stem).strip(".-") or "attachment"
    suffix = _SAFE_FILENAME.sub("-", name.suffix).rstrip(".-")[:119]
    return stem[:120 - len(suffix)] + suffix


def _prune_stale_uploads(root: Path, *, now: float | None = None) -> None:
    """Bound abandoned browser-picker staging without following symlinks."""
    cutoff = (time.time() if now is None else now) - _UPLOAD_RETENTION_SECONDS
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for entry in entries:
        if not entry.name.startswith(_STAGED_PREFIX):
            continue
        try:
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISREG(metadata.st_mode) and metadata.st_mtime < cutoff:
                entry.unlink()
        except OSError:
            continue


@contextmanager
def removed_on_failure(target: Path):
    """Unlink a partially published file; a cleanup error never masks the cause."""
    try:
        yield
    except BaseException:
        with suppress(OSError):
            target.unlink(missing_ok=True)
        raise


def resolve_upload_home(profile: str | None) -> Path:
    """Resolve the profile home an upload is published into, with ``_profile_scope``'s 400/404s.

    Without entering that scope: nothing here reads config or secrets, so an upload must not queue
    on its process-wide skills lock or wait on its secret-source hydration. A named home that is
    missing or deleted is a 404.
    """
    from hermes_constants import assert_named_profile_home_live, get_hermes_home

    if _is_current_profile(profile):
        home = get_hermes_home()
    else:
        home = _resolve_profile_dir(profile.strip())
    try:
        assert_named_profile_home_live(home)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Profile home is unavailable") from exc
    return home


def ensure_profile_subdir(
    home: Path,
    name: str,
    *,
    denied: str,
    failed: str,
    mode: int | None = None,
) -> Path:
    """``<home>/<name>``, created without ever recreating a missing or deleted named home.

    That case is a 404 (``mkdir_under_hermes_home``). Other directory errors are 403 *denied* and
    500 ``"<failed>: <exc>"``; *mode* is applied inside that mapping.
    """
    from hermes_constants import mkdir_under_hermes_home

    directory = home / name
    try:
        mkdir_under_hermes_home(directory)
        if mode is not None:
            directory.chmod(mode)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Profile was deleted during upload") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=denied) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"{failed}: {exc}") from exc
    return directory


def publish_staged_upload(staged, profile_home: Path, filename: str) -> Path:
    """Publish staged bytes as a new owner-only file under ``<profile_home>/uploads``."""
    from hermes_constants import assert_named_profile_home_live

    upload_root = ensure_profile_subdir(
        profile_home,
        _UPLOADS_DIR,
        denied="Upload directory is not writable",
        failed="Could not create upload directory",
        mode=0o700,
    )
    _prune_stale_uploads(upload_root)
    target = upload_root / f"{_STAGED_PREFIX}{secrets.token_hex(8)}-{_safe_filename(filename)}"
    try:
        with removed_on_failure(target):
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
            # A deletion recorded while the bytes were written wins: never report a path in it.
            assert_named_profile_home_live(profile_home)
        return target
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Profile was deleted during upload") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not stage file: {exc}") from exc


def staged_upload_descriptor(path: Path, profile_home: Path) -> dict | None:
    """The ``staged_upload`` receipt for a published file, or None without an install identity."""
    install_id = get_install_id()
    if not install_id:
        # Identity is already persisted by Hermes. If it is unavailable, keep
        # the legacy path/byte-upload flow rather than invent an ephemeral id.
        return None
    return {
        "install_id": install_id,
        "path": str(path),
        "profile_home": str(profile_home),
    }


def parse_staged_upload(descriptor: object) -> tuple[str, str]:
    """``(profile_home, path)`` of a descriptor this installation issued.

    Checked before the source is touched. A replacement backend at the same URL has a different
    install id, and the source bytes alone never authorize reuse.
    """
    if not isinstance(descriptor, dict):
        raise ValueError("invalid staged upload")
    install_id = get_install_id()
    if not install_id or descriptor.get("install_id") != install_id:
        raise ValueError("Staged attachment belongs to another Hermes backend; select the file again")
    raw_home = descriptor.get("profile_home")
    raw_path = descriptor.get("path")
    if not isinstance(raw_home, str) or not raw_home or not isinstance(raw_path, str) or not raw_path:
        raise ValueError("invalid staged upload source")
    return raw_home, raw_path


def resolve_staged_upload(home: Path, raw_path: str) -> Path:
    """The staged regular file *raw_path* names under *home*, within the browser size cap.

    The check is anchored on *home*, never on where the descriptor's path claims to point. A named
    *home* that is missing or deleted raises ``FileNotFoundError``: its receipts are void.
    """
    from hermes_constants import assert_named_profile_home_live

    assert_named_profile_home_live(home)
    source = Path(raw_path)
    if source.is_symlink():
        raise ValueError("staged upload is no longer a regular file")
    source = source.resolve(strict=True)
    if (
        source.parent != (home / _UPLOADS_DIR).resolve(strict=True)
        or not source.name.startswith(_STAGED_PREFIX)
        or not source.is_file()
    ):
        raise ValueError("staged upload is outside its source profile")
    if source.stat().st_size > WEBAPP_ATTACHMENT_MAX_BYTES:
        raise ValueError("staged upload exceeds the browser attachment size limit")
    return source


def is_staged_upload(home: Path, target: Path) -> bool:
    """Whether the resolved *target* is a file the upload route staged for *home*: a regular file
    directly in its uploads folder, named with the staged prefix."""
    try:
        uploads = (home / _UPLOADS_DIR).resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    return target.parent == uploads and target.name.startswith(_STAGED_PREFIX) and target.is_file()
