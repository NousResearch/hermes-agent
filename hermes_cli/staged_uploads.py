"""Browser uploads staged in a profile home, and the receipt ``file.attach`` redeems.

``POST /api/chat/file-upload`` publishes a picked file at ``<profile home>/uploads/web-<hex>-<name>``
and returns a ``staged_upload`` descriptor: that path, the profile home, the generation it was
written under, and this installation's id. The gateway's ``file.attach`` trusts the descriptor only
after resolving it back to exactly that layout under its own leases, so the writer and the reader of
the layout live here together. ``POST /api/chat/image-upload`` shares the leased publish into a
profile subdirectory (``images/``).
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
import os
from pathlib import Path
import re
import secrets
import stat
import time
from typing import Iterator

from fastapi import HTTPException

from hermes_constants import WEBAPP_ATTACHMENT_MAX_BYTES
from hermes_cli.install_identity import get_install_id
from hermes_cli.profile_incarnation import (
    ensure_profile_incarnation,
    profile_incarnation_lease,
)
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


def resolve_upload_generation(profile: str | None) -> tuple[Path, str | None]:
    """Resolve one profile home and capture the named generation it denotes.

    The home ``_profile_scope`` would bind, with its 400/404s, but without entering it: nothing
    here reads config or secrets, so an upload must not queue on that scope's process-wide skills
    lock or wait on its secret-source hydration.
    """
    from hermes_constants import get_hermes_home, named_profile_home_is_unavailable

    if _is_current_profile(profile):
        home = get_hermes_home()
    else:
        home = _resolve_profile_dir(profile.strip())
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


@contextmanager
def leased_profile_dir(
    home: Path,
    expected_incarnation: str | None,
    name: str,
    *,
    denied: str,
    failed: str,
    mode: int | None = None,
) -> Iterator[Path]:
    """``<home>/<name>``, created and held under the captured generation's lease.

    A generation deleted or replaced before or during the block is a 404: neither the lease nor
    ``mkdir_under_hermes_home`` recreates a retired home. Other directory errors are 403 *denied*
    and 500 ``"<failed>: <exc>"``; *mode* is applied inside that mapping.
    """
    from hermes_constants import mkdir_under_hermes_home

    try:
        with profile_incarnation_lease(
            home,
            expected_incarnation,
            require_incarnation=expected_incarnation is not None,
        ):
            directory = home / name
            try:
                mkdir_under_hermes_home(directory)
                if mode is not None:
                    directory.chmod(mode)
            except FileNotFoundError:
                raise
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=denied) from exc
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"{failed}: {exc}") from exc
            yield directory
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Profile was deleted or replaced during upload",
        ) from exc


def publish_staged_upload(
    staged,
    profile_home: Path,
    expected_incarnation: str | None,
    filename: str,
) -> Path:
    """Publish staged bytes while the captured profile generation is leased."""
    try:
        with leased_profile_dir(
            profile_home,
            expected_incarnation,
            _UPLOADS_DIR,
            denied="Upload directory is not writable",
            failed="Could not create upload directory",
            mode=0o700,
        ) as upload_root:
            _prune_stale_uploads(upload_root)
            target = upload_root / (
                f"{_STAGED_PREFIX}{secrets.token_hex(8)}-{_safe_filename(filename)}"
            )
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
            return target
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not stage file: {exc}") from exc


def staged_upload_descriptor(
    path: Path,
    profile_home: Path,
    profile_incarnation: str | None,
) -> dict | None:
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
        "profile_incarnation": profile_incarnation,
    }


def parse_staged_upload(descriptor: object) -> tuple[str, str, str | None]:
    """``(profile_home, path, profile_incarnation)`` of a descriptor this installation issued.

    Checked before any lease is taken. A replacement backend at the same URL has a different
    install id, and the source bytes alone never authorize reuse.
    """
    if not isinstance(descriptor, dict):
        raise ValueError("invalid staged upload")
    install_id = get_install_id()
    if not install_id or descriptor.get("install_id") != install_id:
        raise ValueError("Staged attachment belongs to another Hermes backend; select the file again")
    raw_home = descriptor.get("profile_home")
    raw_path = descriptor.get("path")
    incarnation = descriptor.get("profile_incarnation")
    if (
        not isinstance(raw_home, str) or not raw_home
        or not isinstance(raw_path, str) or not raw_path
        or (incarnation is not None and not isinstance(incarnation, str))
    ):
        raise ValueError("invalid staged upload source")
    return raw_home, raw_path, incarnation


def resolve_staged_upload(home: Path, raw_path: str) -> Path:
    """The staged regular file *raw_path* names under *home*, within the browser size cap.

    Call under a lease on the descriptor's generation, passing the leased *home*: the check is
    anchored on it, never on where the descriptor's path claims to point.
    """
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
