"""Managed-files, chat image upload, /api/media and /api/fs dashboard routes.

Helpers/state that tests monkeypatch on ``web_server`` stay there and are
reached through the late-binding seam (cycle-safe).
"""

import asyncio
import base64
import binascii
import contextlib
import mimetypes
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_cli.web_deps import late
from hermes_cli.web_server_files import (
    _fs_path, _managed_file_entry, _managed_response_meta, _resolve_managed_path,
)
from hermes_cli.web_models import (
    ChatImageUpload, FsWriteText, ManagedDirectoryCreate, ManagedFileDelete, ManagedFileUpload,
)

router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")
get_hermes_home = late("get_hermes_home", "hermes_cli.config")
load_config = late("load_config", "hermes_cli.config")
# Image types GET /api/media serves — extension-allowlisted so an authenticated
# caller can't pull non-image files through it.
_MEDIA_CONTENT_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
    ".webp": "image/webp", ".svg": "image/svg+xml", ".bmp": "image/bmp", ".ico": "image/x-icon",
}
_MEDIA_MAX_BYTES = 25 * 1024 * 1024

_STREAMABLE_MEDIA_EXTENSIONS = frozenset({
    ".avi", ".flac", ".m4a", ".mkv", ".mov", ".mp3", ".mp4", ".ogg", ".opus", ".wav", ".webm",
})

_FS_READDIR_HIDDEN = {
    ".git", ".hg", ".svn", ".cache", ".next", ".turbo", ".venv", "__pycache__",
    "build", "dist", "node_modules", "target", "venv",
}

# Basenames the managed-files API must never list, read or download: credential
# stores that become live secrets in the browsable tree the moment an operator
# points the managed root at HERMES_HOME. Mirrors the two canonical guards
# (agent.file_safety.get_read_block_error, gateway.platforms.base
# ._ROOT_CREDENTIAL_FILES) so the Files tab never lags behind them.
# These typically contain credentials (API keys, tokens) and exposing them through the dashboard file
# browser is a security leak — see issue #57505.
_SENSITIVE_MANAGED_FILE_BASENAMES = frozenset({
    "auth.json", "auth.lock", "credentials", "config.yaml", ".anthropic_oauth.json",
    "google_token.json", "google_oauth_pending.json", "google_oauth.json",
    "webhook_subscriptions.json", "bws_cache.json", "bws_cache.enc.json",
    ".git-credentials",  # git's credential-store cache (file_safety blocks it too)
})

# Directory names whose whole subtree is credential material (the canonical
# guards deny these as trees: _ROOT_CREDENTIAL_DIRS and the mcp-tokens/ prefix
# match). The browser can descend into subdirs, so a basename-only guard would
# still expose ``mcp-tokens/<server>.json``; match on ANY path component so the
# trees are blocked wherever they sit under the root, no HERMES_HOME resolution.
_SENSITIVE_MANAGED_DIR_NAMES = frozenset({"mcp-tokens", "pairing"})


def _is_sensitive_filename(name: str) -> bool:
    """Basename denylist: ``.env`` / ``.env.<suffix>`` / ``.envrc`` plus the
    credential-store basenames. Case-insensitive so ``.ENV`` / ``Auth.JSON``
    on case-insensitive mounts can't slip past. Basename-only — call sites use
    :func:`_is_sensitive_path`, which adds the credential-directory check."""
    lowered = name.lower()
    if lowered == ".env" or lowered.startswith(".env.") or lowered == ".envrc":
        return True
    return lowered in _SENSITIVE_MANAGED_FILE_BASENAMES


def _is_sensitive_path(path: Path) -> bool:
    """True when the basename is sensitive OR any path component (case-
    insensitive) is a credential directory. Read-side guard (list/read/
    download); the write endpoints are a separate threat class.

    Read-side only: this guards list/read/download (the #57505 exfil surface). The write endpoints
    (upload/mkdir/delete) are a separate threat class handled by the write-path checks; extending this guard
    to them is out of scope for this fix.
    """
    if _is_sensitive_filename(path.name):
        return True
    return any(part.lower() in _SENSITIVE_MANAGED_DIR_NAMES for part in path.parts)


_FS_TEXT_SOURCE_MAX_BYTES = 64 * 1024 * 1024
_FS_TEXT_PREVIEW_MAX_BYTES = 512 * 1024
# Spot-editor save ceiling: the editor only opens non-truncated text (<= the
# preview cap), so this guards against a pasted megablob, not expected payloads.
_FS_TEXT_WRITE_MAX_BYTES = 8 * 1024 * 1024

_FS_PREVIEW_LANGUAGE_BY_EXT = {
    ".c": "c", ".conf": "ini", ".cpp": "cpp", ".css": "css", ".csv": "csv", ".go": "go",
    ".graphql": "graphql", ".h": "c", ".hpp": "cpp", ".html": "html", ".java": "java",
    ".js": "javascript", ".json": "json", ".jsx": "jsx", ".kt": "kotlin", ".lua": "lua",
    ".md": "markdown", ".mjs": "javascript", ".py": "python", ".rb": "ruby", ".rs": "rust",
    ".sh": "shell", ".sql": "sql", ".svg": "xml", ".toml": "toml", ".ts": "typescript",
    ".tsx": "tsx", ".txt": "text", ".xml": "xml", ".yaml": "yaml", ".yml": "yaml", ".zsh": "shell",
}

_FS_MIME_TYPES = {
    ".avi": "video/x-msvideo", ".bmp": "image/bmp", ".flac": "audio/flac", ".gif": "image/gif",
    ".jpeg": "image/jpeg", ".jpg": "image/jpeg", ".m4a": "audio/mp4", ".mkv": "video/x-matroska",
    ".mov": "video/quicktime", ".mp3": "audio/mpeg", ".mp4": "video/mp4", ".ogg": "audio/ogg",
    ".opus": "audio/ogg; codecs=opus", ".png": "image/png", ".svg": "image/svg+xml",
    ".wav": "audio/wav", ".webm": "video/webm", ".webp": "image/webp",
}


def _fs_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _FS_MIME_TYPES:
        return _FS_MIME_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _fs_looks_binary(data: bytes) -> bool:
    if not data:
        return False
    if b"\0" in data:
        return True
    suspicious = sum(1 for byte in data if byte < 32 and byte not in {9, 10, 13})
    return suspicious / len(data) > 0.12


@contextlib.contextmanager
def _io_errors(denied: str, failed: str):
    """PermissionError -> 403 ``denied``; other OSError -> 500 ``"<failed>: <exc>"``."""
    try:
        yield
    except PermissionError:
        raise HTTPException(status_code=403, detail=denied)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"{failed}: {exc}")


def _fs_regular_file(path: Path) -> tuple[Path, os.stat_result]:
    target = _fs_path(str(path))
    try:
        st = target.stat()
    except (FileNotFoundError, NotADirectoryError):
        raise HTTPException(status_code=404, detail="File not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Invalid path")
    if stat.S_ISDIR(st.st_mode):
        raise HTTPException(status_code=400, detail="Path points to a directory")
    if not stat.S_ISREG(st.st_mode):
        raise HTTPException(status_code=400, detail="Only regular files can be read")
    return target, st


def _fs_read_bytes(target: Path, limit: Optional[int] = None) -> bytes:
    """Read (a prefix of) ``target``; 403/400 on failure."""
    try:
        if limit is None:
            return target.read_bytes()
        with target.open("rb") as handle:
            return handle.read(limit)
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "File read failed")


def _fs_find_git_root(start: Path) -> str | None:
    directory = start
    for _ in range(50):
        try:
            if (directory / ".git").exists():
                return str(directory)
        except OSError:
            return None
        parent = directory.parent
        if parent == directory:
            return None
        directory = parent
    return None


def _fs_default_cwd() -> str:
    cfg_terminal = load_config().get("terminal") or {}
    raw = str(cfg_terminal.get("cwd") or os.environ.get("TERMINAL_CWD") or "").strip()
    if raw and raw not in {".", "auto", "cwd"}:
        try:
            candidate = Path(raw).expanduser().resolve(strict=False)
            if candidate.is_dir():
                return str(candidate)
        except (OSError, RuntimeError):
            pass
    return str(Path.cwd())


def _fs_git_branch(cwd: str) -> str:
    try:
        run_kwargs: Dict[str, Any] = {"capture_output": True, "text": True, "timeout": 2, "check": False}
        if sys.platform == "win32":
            run_kwargs["creationflags"] = windows_hide_flags()
        result = subprocess.run(["git", "-C", cwd, "branch", "--show-current"], **run_kwargs)
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _media_serve_roots() -> list[Path]:
    """Directories GET /api/media may read: where the agent and attach pipeline
    actually write media (images, screenshots, cache). Stops an authenticated
    client reading image-suffixed files anywhere else on disk."""
    home = get_hermes_home()
    out: list[Path] = []
    for root in (home / "images", home / "screenshots", home / "cache"):
        try:
            out.append(root.resolve())
        except (OSError, RuntimeError):
            continue
    return out


@router.get("/api/media")
async def get_media(path: str):
    """Return a gateway-local image as a base64 data URL for remote clients
    that can't read this machine's disk. Auth-gated; restricted to the image
    allowlist, a size cap AND the resolved (symlink-safe) media roots."""
    try:
        target = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid path")

    if target.suffix.lower() not in _MEDIA_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported media type")
    roots = _media_serve_roots()
    if not any(target == root or root in target.parents for root in roots):
        raise HTTPException(status_code=403, detail="Path outside media roots")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if target.stat().st_size > _MEDIA_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    return {"data_url": f"data:{_MEDIA_CONTENT_TYPES[target.suffix.lower()]};base64,{encoded}"}


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    from hermes_cli.web_server import _MANAGED_FILE_MAX_BYTES
    text = (data_url or "").strip()
    if not text.startswith("data:") or "," not in text:
        raise HTTPException(status_code=400, detail="Upload payload must be a data URL")
    header, encoded = text.split(",", 1)
    mime_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    if ";base64" not in header:
        raise HTTPException(status_code=400, detail="Upload payload must be base64 encoded")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Upload payload is not valid base64")
    if len(data) > _MANAGED_FILE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is too large")
    return data, mime_type


_CHAT_IMAGE_UPLOAD_MAX_BYTES = 25 * 1024 * 1024
_CHAT_IMAGE_ALLOWED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
_CHAT_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"), (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"), (b"GIF89a", ".gif"), (b"BM", ".bmp"),
)


def _sanitize_chat_image_filename(filename: str | None) -> str:
    candidate = Path(str(filename or "").strip()).name
    candidate = re.sub(r"[\x00-\x1f]+", "_", candidate)
    return candidate.strip().strip(".") or "pasted-image"


def _chat_image_extension(data: bytes) -> str | None:
    head = data[:16]
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return ".webp"
    for sig, ext in _CHAT_IMAGE_MAGIC:
        if head.startswith(sig):
            return ext
    return None


def _decode_chat_image_upload(payload: ChatImageUpload) -> tuple[bytes, str, str]:
    data, mime_type = _decode_data_url(payload.data_url)
    if not mime_type.lower().startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload payload must be an image")
    if len(data) > _CHAT_IMAGE_UPLOAD_MAX_BYTES:
        mb = _CHAT_IMAGE_UPLOAD_MAX_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"Image is too large; cap is {mb} MB")
    ext = _chat_image_extension(data)
    if ext not in _CHAT_IMAGE_ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image type")
    return data, mime_type, ext


@router.post("/api/chat/image-upload")
async def upload_chat_image(payload: ChatImageUpload, profile: Optional[str] = None):
    """Persist a browser clipboard image where the embedded TUI can read it.

    Browser clipboard bytes aren't visible to the server-side clipboard, so the
    /chat page uploads them here and drives the TUI's ``/image <path>`` with
    the returned gateway-visible path under ``HERMES_HOME/images/`` (the same
    dir ``clipboard.paste`` / ``image.attach`` use).
    """
    def _run():
        data, mime_type, ext = _decode_chat_image_upload(payload)
        with _profile_scope(profile) as scoped_home:
            img_dir = Path(scoped_home or get_hermes_home()) / "images"
            with _io_errors("Image directory is not writable", "Could not create image directory"):
                img_dir.mkdir(parents=True, exist_ok=True)

            stem = Path(_sanitize_chat_image_filename(payload.filename)).stem or "pasted-image"
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-") or "pasted-image"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = img_dir / f"dashboard_{ts}_{secrets.token_hex(4)}_{stem}{ext}"
            with _io_errors("Image directory is not writable", "Could not write image"):
                target.write_bytes(data)

        return {
            "ok": True,
            "path": str(target),
            "name": target.name,
            "bytes": len(data),
            "mime_type": mime_type,
        }

    # _profile_scope takes _SKILLS_PROFILE_LOCK and the body does file I/O — both
    # off the loop; to_thread copies the contextvar context so the override
    # stays scoped to the worker thread.
    return await asyncio.to_thread(_run)


@router.get("/api/files")
async def list_managed_files(request: Request, path: Optional[str] = None):
    policy, target, display_path = _resolve_managed_path(path, request)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    with _io_errors("Directory is not readable", "Could not read directory"), os.scandir(target) as scan:
        entries = [
            _managed_file_entry(policy, Path(entry.path))
            for entry in scan
            if not _is_sensitive_path(Path(entry.path))
        ]

    entries.sort(key=lambda item: (not item["is_directory"], str(item["name"]).lower()))
    locked_root = policy.locked_root
    parent = None
    if target.parent != target and (locked_root is None or target != locked_root):
        parent = str(target.parent)
    return {"path": display_path, "parent": parent, "entries": entries, **_managed_response_meta(policy)}


def _managed_readable_file(request: Request, path: str) -> tuple[Any, Path, str, int, str]:
    """Resolve + guard a managed file for reading: existence, regular file,
    sensitive-path denylist, size cap. Returns (policy, target, display_path,
    size, mime_type)."""
    from hermes_cli.web_server import _MANAGED_FILE_MAX_BYTES
    policy, target, display_path = _resolve_managed_path(path, request)
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    if _is_sensitive_path(target):
        raise HTTPException(status_code=403, detail="Access to sensitive files is not allowed")
    mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return policy, target, display_path, _MANAGED_FILE_MAX_BYTES, mime_type


def _managed_file_size(target: Path, max_bytes: int) -> int:
    try:
        size = target.stat().st_size
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not stat file: {exc}")
    if size > max_bytes:
        raise HTTPException(status_code=413, detail="File is too large")
    return size


@router.get("/api/files/read")
async def read_managed_file(request: Request, path: str):
    policy, target, display_path, max_bytes, mime_type = _managed_readable_file(request, path)
    size = _managed_file_size(target, max_bytes)
    with _io_errors("File is not readable", "Could not read file"):
        encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    return {
        "name": target.name,
        "path": display_path,
        "size": size,
        "mime_type": mime_type,
        "data_url": f"data:{mime_type};base64,{encoded}",
        **_managed_response_meta(policy),
    }


def _managed_file_response(
    request: Request,
    path: str,
    *,
    content_disposition_type: str,
    media_only: bool = False,
) -> FileResponse:
    """Range-aware response after applying managed-file policy."""
    _policy, target, _display_path, max_bytes, mime_type = _managed_readable_file(request, path)
    if media_only and target.suffix.lower() not in _STREAMABLE_MEDIA_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported media type")
    _managed_file_size(target, max_bytes)
    return FileResponse(
        path=str(target),
        media_type=mime_type,
        filename=target.name,
        content_disposition_type=content_disposition_type,
        headers={"X-Content-Type-Options": "nosniff"} if media_only else None,
    )


@router.get("/api/files/download")
async def download_managed_file(request: Request, path: str):
    """Stream a managed file as an attachment download.

    ``auth_middleware`` also accepts the session token as ``?token=`` here so a
    shell/browser-opened download (no session header) still authenticates.
    Chromium marks ``<audio>``/``<video>`` subresource requests via
    ``Sec-Fetch-Dest``; those are served inline for Desktop builds that still
    use this route as their player source, attachment semantics otherwise.
    """
    fetch_destination = request.headers.get("sec-fetch-dest", "").lower()
    is_media_subresource = fetch_destination in {"audio", "video"}
    return _managed_file_response(
        request,
        path,
        content_disposition_type="inline" if is_media_subresource else "attachment",
        media_only=is_media_subresource,
    )


@router.get("/api/files/stream")
@router.head("/api/files/stream")
async def stream_managed_file(request: Request, path: str):
    """Stream managed audio/video inline with HTTP Range support — Electron's
    media pipeline may reject an attachment response as an ``<audio>``/
    ``<video>`` source. Same auth, size cap, sensitive guard and MIME detection
    as download."""
    return _managed_file_response(request, path, content_disposition_type="inline", media_only=True)


def _managed_write_target(path: str, request: Request, overwrite: bool):
    policy, target, display_path = _resolve_managed_path(path, request, for_write=True)
    if target.exists() and target.is_dir():
        raise HTTPException(status_code=409, detail="A directory already exists at that path")
    if target.exists() and not overwrite:
        raise HTTPException(status_code=409, detail="File already exists")
    return policy, target, display_path


def _managed_write_result(policy, target: Path, display_path: str) -> dict:
    return {
        "ok": True,
        "entry": _managed_file_entry(policy, target),
        "path": display_path,
        **_managed_response_meta(policy),
    }


@router.post("/api/files/upload")
async def upload_managed_file(payload: ManagedFileUpload, request: Request):
    policy, target, display_path = _managed_write_target(payload.path, request, payload.overwrite)
    data, _mime_type = _decode_data_url(payload.data_url)
    with _io_errors("File is not writable", "Could not write file"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return _managed_write_result(policy, target, display_path)


async def stream_upload_to_path(
    file: UploadFile,
    target: Path,
    *,
    too_large: str,
    not_writable: str,
    write_failed: str,
) -> int:
    """Stream a multipart upload to ``target`` in chunks; returns bytes written.

    Writes a sibling temp file first so a partial/aborted upload never clobbers
    an existing file, enforces ``_MANAGED_FILE_MAX_BYTES`` as it goes (413
    ``too_large``), then atomically renames into place. The temp file is
    removed on EVERY non-success exit — including asyncio.CancelledError when a
    browser aborts a large upload mid-stream.
    """
    from hermes_cli.web_server import _MANAGED_FILE_MAX_BYTES, _UPLOAD_CHUNK_BYTES
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".upload", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    total = 0
    renamed = False
    try:
        with os.fdopen(tmp_fd, "wb") as out:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MANAGED_FILE_MAX_BYTES:
                    raise HTTPException(status_code=413, detail=too_large)
                out.write(chunk)
        os.replace(tmp_path, target)
        renamed = True
    except PermissionError:
        raise HTTPException(status_code=403, detail=not_writable)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"{write_failed}: {exc}")
    finally:
        if not renamed:
            tmp_path.unlink(missing_ok=True)
        await file.close()
    return total


@router.post("/api/files/upload-stream")
async def upload_managed_file_stream(
    request: Request,
    file: UploadFile = File(...),
    path: str = Form(...),
    overwrite: bool = Form(True),
):
    """Chunked multipart upload: constant memory and no base64 inflation, unlike
    the JSON data-URL endpoint that trips proxy body-size limits on large archives."""
    policy, target, display_path = _managed_write_target(path, request, overwrite)
    with _io_errors("File is not writable", "Could not create parent directory"):
        target.parent.mkdir(parents=True, exist_ok=True)
    await stream_upload_to_path(
        file, target,
        too_large="File is too large",
        not_writable="File is not writable",
        write_failed="Could not write file",
    )
    return _managed_write_result(policy, target, display_path)


@router.post("/api/files/mkdir")
async def create_managed_directory(payload: ManagedDirectoryCreate, request: Request):
    policy, target, display_path = _resolve_managed_path(payload.path, request, for_write=True)
    if target.exists() and not target.is_dir():
        raise HTTPException(status_code=409, detail="A file already exists at that path")
    with _io_errors("Directory is not writable", "Could not create directory"):
        target.mkdir(parents=True, exist_ok=True)
    return _managed_write_result(policy, target, display_path)


@router.delete("/api/files")
async def delete_managed_file(payload: ManagedFileDelete, request: Request):
    policy, target, display_path = _resolve_managed_path(payload.path, request)
    if policy.locked_root is not None and target == policy.locked_root:
        raise HTTPException(status_code=400, detail="Cannot delete the managed files root")
    if target.parent == target:
        raise HTTPException(status_code=400, detail="Cannot delete the filesystem root")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    try:
        if target.is_dir():
            if payload.recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()
        else:
            target.unlink()
    except OSError as exc:
        status_code = 409 if target.is_dir() and not payload.recursive else 500
        raise HTTPException(status_code=status_code, detail=f"Could not delete path: {exc}")
    return {"ok": True, "path": display_path, **_managed_response_meta(policy)}


_FS_LIST_ERRNO = (
    (FileNotFoundError, "ENOENT"),
    (NotADirectoryError, "ENOTDIR"),
    (PermissionError, "EACCES"),
)


@router.get("/api/fs/list")
async def fs_list(path: str):
    target = _fs_path(path)
    try:
        entries = []
        with os.scandir(target) as scan:
            for entry in scan:
                if entry.name in _FS_READDIR_HIDDEN:
                    continue
                entries.append({
                    "name": entry.name,
                    "path": str(target / entry.name),
                    "isDirectory": entry.is_dir(follow_symlinks=False),
                })
        entries.sort(key=lambda item: (not item["isDirectory"], item["name"].lower(), item["name"]))
        return {"entries": entries}
    except OSError as exc:
        for exc_type, code in _FS_LIST_ERRNO:
            if isinstance(exc, exc_type):
                return {"entries": [], "error": code}
        return {"entries": [], "error": getattr(exc, "strerror", None) or "read-error"}


@router.get("/api/fs/read-text")
async def fs_read_text(path: str):
    target, st = _fs_regular_file(_fs_path(path))
    if st.st_size > _FS_TEXT_SOURCE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    data = _fs_read_bytes(target, min(st.st_size, _FS_TEXT_PREVIEW_MAX_BYTES))
    return {
        "binary": _fs_looks_binary(data[:4096]),
        "byteSize": st.st_size,
        "language": _FS_PREVIEW_LANGUAGE_BY_EXT.get(target.suffix.lower(), "text"),
        "mimeType": _fs_mime_type(target),
        "path": str(target),
        "text": data.decode("utf-8", errors="replace"),
        "truncated": st.st_size > _FS_TEXT_PREVIEW_MAX_BYTES,
    }


@router.post("/api/fs/write-text")
async def fs_write_text(payload: FsWriteText):
    """Overwrite (or create) a UTF-8 text file for the in-app spot editor.

    Mirrors the Electron ``hermes:fs:writeText`` hardening: path validated by
    ``_fs_path``, the parent must already exist (never build trees), only
    regular files may be replaced, payload size-capped, staged to a sibling
    temp file and ``os.replace``-d so a crash can't truncate the original.
    Stale-on-disk detection is the client's job (re-read before save).
    """
    target = _fs_path(payload.path)
    text = payload.content or ""
    if len(text.encode("utf-8")) > _FS_TEXT_WRITE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Content too large")

    try:
        st: Optional[os.stat_result] = target.stat()
    except FileNotFoundError:
        st = None
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Invalid path")

    if st is not None and stat.S_ISDIR(st.st_mode):
        raise HTTPException(status_code=400, detail="Path points to a directory")
    if st is not None and not stat.S_ISREG(st.st_mode):
        raise HTTPException(status_code=400, detail="Only regular files can be written")
    if not target.parent.is_dir():
        raise HTTPException(status_code=400, detail="Parent directory does not exist")

    tmp = target.with_name(f".{target.name}.hermes-tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not write file: {exc}")
    return {"ok": True, "path": str(target), "byteSize": len(text.encode("utf-8"))}


@router.get("/api/fs/read-data-url")
async def fs_read_data_url(path: str):
    from hermes_cli.web_server import _FS_DATA_URL_MAX_BYTES
    target, st = _fs_regular_file(_fs_path(path))
    if st.st_size > _FS_DATA_URL_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    encoded = base64.b64encode(_fs_read_bytes(target)).decode("ascii")
    return {"dataUrl": f"data:{_fs_mime_type(target)};base64,{encoded}"}


@router.get("/api/fs/download")
async def fs_download(path: str):
    target, _st = _fs_regular_file(_fs_path(path))
    if _is_sensitive_path(target):
        raise HTTPException(status_code=403, detail="Access to sensitive files is not allowed")
    return FileResponse(
        path=str(target),
        media_type=_fs_mime_type(target),
        filename=target.name,
        content_disposition_type="attachment",
    )


@router.get("/api/fs/git-root")
async def fs_git_root(path: str):
    target = _fs_path(path)
    try:
        st = target.stat()
        start = target if stat.S_ISDIR(st.st_mode) else target.parent
    except OSError:
        start = target
    return {"root": _fs_find_git_root(start)}


@router.get("/api/fs/default-cwd")
async def fs_default_cwd():
    cwd = _fs_default_cwd()
    return {"cwd": cwd, "branch": _fs_git_branch(cwd)}
