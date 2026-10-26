"""Managed-files, chat image upload, /api/media and /api/fs dashboard routes.

Helpers/state that tests monkeypatch on ``web_server`` stay there and are
reached through the late-binding seam (cycle-safe).
"""

import asyncio
import base64
import binascii
import contextlib
import logging
import mimetypes
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from hermes_cli import web_server_file_tickets as file_tickets
from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_cli.sqlite_safe_read import LiveConnectionError, offline_file_access
from hermes_cli.staged_uploads import (
    ensure_profile_subdir, is_staged_upload, removed_on_failure, resolve_upload_home,
)
from hermes_cli.web_deps import late
from hermes_cli.web_server_remote_media import remote_media_response
from hermes_cli.web_upload_limit import CHAT_IMAGE_MAX_BYTES
from hermes_cli.web_server_surface import route_on
from hermes_cli.web_server_files import (
    SANDBOX_NOT_FOUND, _agent_fs_path, _file_url_path, _fs_path, _hosted_fs_read_guard, _is_unmapped_container_path,
    _managed_file_entry, _managed_response_meta, _resolve_managed_path, _session_workspace_key,
)
from hermes_cli.web_models import (
    ChatImageUpload, FileTicketRequest, FsWriteText, ManagedDirectoryCreate, ManagedFileDelete,
    ManagedFileUpload,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")
get_hermes_home = late("get_hermes_home", "hermes_cli.config")
load_config = late("load_config", "hermes_cli.config")
load_env = late("load_env", "hermes_cli.config")
_require_token = late("_require_token")
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


def _fs_sandbox_not_found_detail(path: Path, profile_home: Optional[Path] = None) -> str:
    """Distinguish an untranslated agent-sandbox path from a plain missing file."""
    unmapped = _is_unmapped_container_path(path.as_posix(), profile_home=profile_home)
    return SANDBOX_NOT_FOUND if unmapped else "File not found"


def _fs_regular_file(path: Path, profile_home: Optional[Path] = None) -> tuple[Path, os.stat_result]:
    # The caller already resolved *path* (a file: URL decoded once); decoding it again here would
    # read the name report%2fsecret.txt as report/secret.txt.
    target = _fs_path(str(path), decode_fallback=False)
    try:
        st = target.stat()
    except (FileNotFoundError, NotADirectoryError):
        raise HTTPException(status_code=404, detail=_fs_sandbox_not_found_detail(target, profile_home))
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Invalid path")
    if stat.S_ISDIR(st.st_mode):
        raise HTTPException(status_code=400, detail="Path points to a directory")
    if not stat.S_ISREG(st.st_mode):
        raise HTTPException(status_code=400, detail="Only regular files can be read")
    if _is_sensitive_path(target):
        raise HTTPException(status_code=403, detail="Access to sensitive files is not allowed")
    return target, st


@contextlib.contextmanager
def _serve_offline(target: Path):
    """Hold ``offline_file_access`` for serving ``target``; 409 while a SQLite connection to it is live.

    Callers keep the block open through the file's close: a raw close cancels this process's
    SQLite POSIX locks on it."""
    try:
        with offline_file_access(target, what="serve"):
            yield
    except LiveConnectionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _refuse_live_database(target: Path) -> None:
    """Point-in-time 409 before streaming ``target`` while a SQLite connection to it is live.

    ``FileResponse`` opens and closes the file in-process, and that raw close cancels the
    connection's POSIX locks. The registry lock can't be held across a streamed response
    (never across an await/yield), so streamed routes only get this admission check. It
    takes the global registry lock, which other threads hold across whole-file reads, so
    call it via ``asyncio.to_thread``."""
    with _serve_offline(target):
        pass


def _fs_read_bytes(target: Path, limit: Optional[int] = None) -> bytes:
    """Read (a prefix of) ``target``; 403/400 on failure, 409 while a SQLite connection to it is live."""
    try:
        with _serve_offline(target):
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


def _fs_default_cwd(profile: Optional[str] = None) -> str:
    """The host folder a new session or browser terminal of *profile* starts in: its
    ``terminal.cwd`` when that names a folder, else the home folder."""
    from tools.terminal_scope import TerminalPolicyUnavailable, build_profile_terminal_scope

    with _profile_scope(profile) as scoped:
        if scoped is None:
            # The serving process's own profile: its startup bridged terminal.cwd into
            # TERMINAL_CWD. A placeholder terminal.cwd counts as unset, so TERMINAL_CWD answers.
            cfg_terminal = load_config().get("terminal") or {}
            candidates = (cfg_terminal.get("cwd"), os.environ.get("TERMINAL_CWD"))
        else:
            # Another profile served by this process: the TERMINAL_CWD its own turns run under
            # (config.yaml over .env), never this process's environment.
            try:
                candidates = (build_profile_terminal_scope(scoped).get("TERMINAL_CWD"),)
            except TerminalPolicyUnavailable:
                candidates = ()
    for raw in candidates:
        raw = str(raw or "").strip()
        if raw in {"", ".", "auto", "cwd"}:
            continue
        try:
            candidate = Path(raw).expanduser().resolve(strict=False)
            if candidate.is_dir():
                return str(candidate)
        except (OSError, RuntimeError):
            pass
    return str(Path.home())


def _fs_git_branch(cwd: str) -> str:
    try:
        # git emits UTF-8 (branch names, localized "not a git repository" stderr); the locale codec
        # (cp936 on zh-CN Windows) raised inside communicate()'s reader threads on every poll (#83851).
        run_kwargs: dict[str, Any] = {"capture_output": True, "text": True, "encoding": "utf-8",
                                      "errors": "replace", "timeout": 2}
        if sys.platform == "win32":
            run_kwargs["creationflags"] = windows_hide_flags()
        result = subprocess.run(["git", "-C", cwd, "branch", "--show-current"], **run_kwargs, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _fs_backend(profile: Optional[str] = None):
    """Return the profile's SSH workspace adapter, or None for host-local FS.

    Only an uncached adapter dials the host: the profile's first file request since the server
    started, or the first after a setting the adapter is built from changes. When that dial fails,
    or the settings cannot dial at all (no host or user, no ``ssh`` on PATH), the request answers
    503 ``SSH workspace is unavailable`` and the log names the cause the 503 hides. A cached
    adapter is not re-dialed here: a host that goes down after it connected fails the read itself,
    which each route answers on its own.
    """
    from hermes_cli.ssh_workspace_fs import SshWorkspaceFsError, get_ssh_workspace_fs

    with _profile_scope(profile):
        terminal = dict(load_config().get("terminal") or {})
        profile_env = load_env()
    for key, env_name in {
        "ssh_host": "TERMINAL_SSH_HOST",
        "ssh_user": "TERMINAL_SSH_USER",
        "ssh_port": "TERMINAL_SSH_PORT",
        "ssh_key": "TERMINAL_SSH_KEY",
    }.items():
        if not terminal.get(key) and profile_env.get(env_name):
            terminal[key] = profile_env[env_name]
    profile_key = (profile or "current").strip().lower() or "current"
    try:
        return get_ssh_workspace_fs(profile_key, terminal)
    except SshWorkspaceFsError as exc:
        logger.warning("SSH workspace for profile %s is unavailable: %s", profile_key, exc.__cause__ or exc)
        _raise_fs_backend_error(exc)


def _chat_image_dir(home: Path) -> Path:
    """Where browser image pastes land: the profile's canonical image staging dir on this host
    (``gateway.platforms.base.get_image_cache_dir``), the only one attachment admission accepts."""
    from hermes_constants import get_hermes_dir

    return get_hermes_dir("cache/images", "image_cache", home=home)


def _is_host_file(path: str, profile: Optional[str]) -> bool:
    """Whether *path* is a file the profile keeps on this host even when its workspace is an SSH
    machine, read back to be previewed and sent: a browser paste in its image cache, or a file the
    browser's picker staged in its uploads folder. Only a file directly in one of those two folders
    counts, after resolving symlinks and ``..``, so an SSH profile reads no other host file here.
    A ``file:`` URL counts by the native path it names, decoded once as ``_fs_path`` decodes it."""
    if path.lower().startswith("file:"):
        try:
            path = _file_url_path(path)
        except ValueError:
            return False
    if not Path(path).is_absolute():
        return False
    try:
        target = Path(path).resolve(strict=True)
    except (OSError, RuntimeError):
        return False  # not on this host: the workspace's to answer
    with _profile_scope(profile):
        home = get_hermes_home()
        cache = _chat_image_dir(home)
    try:
        if target.parent == cache.resolve(strict=True):
            return True
    except (OSError, RuntimeError):
        pass
    return is_staged_upload(home, target)


def _remote_fs_backend(path: str, profile: Optional[str]):
    """The SSH workspace reader for *path*, or None when this host owns it. The host's pastes and
    staged uploads are checked before the SSH machine is dialed, so they read back while that
    machine is down; every other path is the SSH machine's to answer (see ``_fs_backend``)."""
    if _is_host_file(path, profile):
        return None
    return _fs_backend(profile)


def _raise_fs_backend_error(exc: Exception) -> None:
    from hermes_cli.ssh_workspace_fs import SshWorkspaceFsError

    if not isinstance(exc, SshWorkspaceFsError):
        raise exc
    status = {
        "EACCES": 403,
        "ECONN": 503,
        "EFBIG": 413,
        "EINVAL": 400,
        "EIO": 502,
        "ENOENT": 404,
        "ENOTREG": 400,
    }.get(exc.code, 400)
    detail = {
        "EACCES": "File is not readable",
        "ECONN": "SSH workspace is unavailable",
        "EFBIG": "File too large",
        "EINVAL": "Invalid path",
        "EIO": "SSH workspace request failed",
        "ENOENT": "File not found",
        "ENOTREG": "Only regular files can be read",
    }.get(exc.code, str(exc) or "Filesystem request failed")
    raise HTTPException(status_code=status, detail=detail)


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


def _read_base64_file(path: Path) -> str:
    """Read and encode a bounded file from a worker thread; 409 while a SQLite connection to it is live."""
    with _serve_offline(path):
        data = path.read_bytes()
    # Encode after release: only the open/read/close needs the registry lock.
    return base64.b64encode(data).decode("ascii")


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

    encoded = await asyncio.to_thread(_read_base64_file, target)
    return {"data_url": f"data:{_MEDIA_CONTENT_TYPES[target.suffix.lower()]};base64,{encoded}"}


# Remote image URLs an authenticated proxy may fetch for a client that cannot
# reach the CDN itself (#74564: a Desktop on a restricted network renders an
# agent-generated FAL image inline; the direct link is blocked, but the GATEWAY
# hosts a working route to the CDN and already holds the generation credentials).
# Host-allowlisted + size-capped like the rest of the media surface.
_MEDIA_PROXY_ALLOWED_HOSTS = (
    "fal.media", "fal.run", "v3.fal.media", "storage.googleapis.com",
    "*.fal.media", "*.fal.run",
)
_MEDIA_PROXY_TIMEOUT_S = 20.0


def _media_proxy_host_allowed(host: str) -> bool:
    host = host.lower().rstrip(".")
    return any(
        host == allowed.lstrip("*.") or (allowed.startswith("*.") and host.endswith(allowed[1:]))
        for allowed in _MEDIA_PROXY_ALLOWED_HOSTS
    )


@router.get("/api/media/proxy")
async def proxy_remote_media(url: str, request: Request):
    """Fetch a remote image URL the gateway can reach but the client cannot
    (#74564), returning the same ``data_url`` shape as ``/api/media``. Only
    allowlisted image CDNs; the bytes stay behind the size cap."""
    _require_token(request)
    try:
        parsed = urllib.parse.urlparse((url or "").strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="A remote image URL is required") from exc
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(status_code=400, detail="A remote image URL is required")
    if not _media_proxy_host_allowed(parsed.hostname):
        raise HTTPException(status_code=403, detail="Image host not allowed")
    if not parsed.path or Path(parsed.path).suffix.lower() not in _MEDIA_CONTENT_TYPES:
        # Generated-image CDN URLs are content-hash paths with no extension;
        # a missing extension is expected, so only reject explicit non-image
        # extensions and let content type be sniffed from the response.
        if Path(parsed.path).suffix and Path(parsed.path).suffix.lower() not in _MEDIA_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported media type")

    import httpx

    try:
        async with httpx.AsyncClient(timeout=_MEDIA_PROXY_TIMEOUT_S, follow_redirects=True) as client:
            response = await client.get(url)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Image fetch failed: {exc}")

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Image fetch returned HTTP {response.status_code}")
    content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
    if content_type not in _MEDIA_CONTENT_TYPES.values():
        raise HTTPException(status_code=415, detail="Unsupported media type")
    data = response.content
    if len(data) > _MEDIA_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    encoded = base64.b64encode(data).decode("ascii")
    return {"data_url": f"data:{content_type};base64,{encoded}"}


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


_CHAT_IMAGE_UPLOAD_MAX_BYTES = CHAT_IMAGE_MAX_BYTES
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
        home = resolve_upload_home(profile)
        from hermes_constants import apply_secure_dir_policy, assert_named_profile_home_live

        img_dir = ensure_profile_subdir(
            home,
            str(_chat_image_dir(home).relative_to(home)),
            denied="Image directory is not writable",
            failed="Could not create image directory",
        )
        apply_secure_dir_policy(img_dir, home=home)
        stem = Path(_sanitize_chat_image_filename(payload.filename)).stem or "pasted-image"
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-") or "pasted-image"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = img_dir / f"dashboard_{ts}_{secrets.token_hex(4)}_{stem}{ext}"

        with removed_on_failure(target):
            with _io_errors("Image directory is not writable", "Could not write image"):
                target.write_bytes(data)
            try:
                # A deletion recorded while the bytes were written wins: never report a path in it.
                assert_named_profile_home_live(home)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Profile was deleted during upload") from exc

        return {
            "ok": True,
            "path": str(target),
            "name": target.name,
            "bytes": len(data),
            "mime_type": mime_type,
        }

    # The body does file I/O: keep it off the loop. It resolves the profile home without
    # entering _profile_scope, so a paste never queues behind its process-wide skills lock.
    return await asyncio.to_thread(_run)


@router.get("/api/files")
async def list_managed_files(request: Request, path: Optional[str] = None):
    policy, target, display_path = _resolve_managed_path(path, request)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    with _io_errors("Directory is not readable", "Could not read directory"), os.scandir(target) as scan:
        entries = []
        for entry in scan:
            entry_path = Path(entry.path)
            if _is_sensitive_path(entry_path):
                continue
            metadata = _managed_file_entry(policy, entry_path, skip_missing=True)
            if metadata is not None:
                entries.append(metadata)

    entries.sort(key=lambda item: (not item["is_directory"], str(item["name"]).lower()))
    locked_root = policy.locked_root
    parent = None
    if target.parent != target and (locked_root is None or target != locked_root):
        parent = str(target.parent)
    return {"path": display_path, "parent": parent, "entries": entries, **_managed_response_meta(policy)}


def _managed_readable_file(
    request: Request, path: str, *, create_root: bool = True,
) -> tuple[Any, Path, str, int, str]:
    """Resolve + guard a managed file for reading: existence, regular file,
    sensitive-path denylist. Returns (policy, target, display_path, max_bytes,
    mime_type). Callers own the live-SQLite 409 (held through the read for
    /api/files/read, point-in-time for streamed responses)."""
    from hermes_cli.web_server import _MANAGED_FILE_MAX_BYTES
    policy, target, display_path = _resolve_managed_path(
        path, request, create_root=create_root, for_agent_read=True,
    )
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
        encoded = await asyncio.to_thread(_read_base64_file, target)
    return {
        "name": target.name,
        "path": display_path,
        "size": size,
        "mime_type": mime_type,
        "data_url": f"data:{mime_type};base64,{encoded}",
        **_managed_response_meta(policy),
    }


async def _managed_file_response(
    request: Request,
    path: str,
    *,
    content_disposition_type: str,
    media_only: bool = False,
) -> FileResponse:
    """Range-aware response after applying managed-file policy."""
    authority = getattr(request.state, "file_ticket_authority", None)
    if authority is not None:
        from hermes_cli.web_server_file_response import OpenedFileResponse

        return OpenedFileResponse(
            path=str(authority.target),
            opener=lambda: _open_ticket_file(request, authority, media_only=media_only),
            media_type=_fs_mime_type(authority.target),
            filename=authority.target.name,
            content_disposition_type=content_disposition_type,
            headers={"X-Content-Type-Options": "nosniff"} if media_only else None,
        )
    _policy, target, _display_path, max_bytes, mime_type = _managed_readable_file(request, path)
    if media_only and target.suffix.lower() not in _STREAMABLE_MEDIA_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported media type")
    _managed_file_size(target, max_bytes)
    await asyncio.to_thread(_refuse_live_database, target)
    return FileResponse(
        path=str(target),
        media_type=mime_type,
        filename=target.name,
        content_disposition_type=content_disposition_type,
        headers={"X-Content-Type-Options": "nosniff"} if media_only else None,
    )


def _open_ticket_file(request: Request, authority: file_tickets.FileAuthority, *, media_only: bool):
    """Check the ticket's profile homes are still live and pin the file the ticket names."""
    def checked():
        _policy, target, _display, max_bytes, _mime = _managed_readable_file(
            request, str(authority.target), create_root=False,
        )
        if media_only and target.suffix.lower() not in _STREAMABLE_MEDIA_EXTENSIONS:
            raise HTTPException(status_code=415, detail="Unsupported media type")
        return target, max_bytes

    return _pin_ticket_file(authority, checked)


def _pin_ticket_file(authority: file_tickets.FileAuthority, checked):
    """Open the host file *authority* names once its profile homes are still live and ``checked()``
    (the route's own read rules) returns that same file and its size cap (None: no cap)."""
    try:
        authority.check_live()
        target, max_bytes = checked()
        if target != authority.target:
            raise HTTPException(status_code=403, detail="File ticket target changed")
        _refuse_live_database(target)
        handle = target.open("rb")
        try:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise HTTPException(status_code=400, detail="Only regular files can be read")
            if max_bytes is not None and opened.st_size > max_bytes:
                raise HTTPException(status_code=413, detail="File is too large")
            return handle
        except BaseException:
            handle.close()
            raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="File ticket profile or resource is unavailable") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="File is not readable") from exc


@router.post("/api/files/ticket", dependencies=[Depends(route_on("browser_files"))])
async def issue_file_ticket(payload: FileTicketRequest, request: Request):
    """Ticket a browser download/media URL instead of putting the session token in it."""
    from hermes_cli.web_routers.sessions import _history_profile_home
    from hermes_constants import named_profile_home

    def owner_home():
        return Path(_history_profile_home(payload.profile))

    def capture():
        _policy, target, _display, _max, _mime = _managed_readable_file(request, path, create_root=False)
        homes = [owner]
        target_home = named_profile_home(target)
        if target_home is not None and target_home not in homes:
            homes.append(target_home)
        authority = file_tickets.FileAuthority(target, tuple(homes))
        authority.check_live()
        return authority

    try:
        # Capture the session/profile owner BEFORE looking up its cwd. The file
        # may be outside that home, but the ticket must still die with its owner.
        owner = await asyncio.to_thread(owner_home)
        # Before the backend is chosen: an SSH profile's ticket checks its session too.
        await _fs_owned_session(payload.session_id, payload.profile)
        if payload.route == "download":
            authority = await _fs_download_authority(
                request, payload.path, payload.profile, payload.session_id, owner,
            )
        else:
            # An SSH profile's clip is captured on its SSH machine; a host-staged file on this host.
            authority = await _remote_file_authority(payload.path, payload.profile, owner, _remote_stream_target)
            path = payload.path
            if authority is None and (payload.session_id is not None or path.lower().startswith("file:")):
                path = str(await _fs_download_path(path, payload.profile, payload.session_id))
            if authority is None:
                authority = await asyncio.to_thread(capture)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="File ticket profile is unavailable") from exc
    query = {"path": payload.path, "profile": payload.profile, "session_id": payload.session_id}
    ticket = file_tickets.mint(
        payload.route, {key: value for key, value in query.items() if value is not None}, authority=authority,
        session_token=file_tickets.session_token_of(request),
    )
    return JSONResponse({"ticket": ticket}, headers={"Cache-Control": "no-store"})


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
    return await _managed_file_response(
        request,
        path,
        content_disposition_type="inline" if is_media_subresource else "attachment",
        media_only=is_media_subresource,
    )


@router.get("/api/files/stream")
@router.head("/api/files/stream")
async def stream_managed_file(
    request: Request, path: str, profile: Optional[str] = None, session_id: Optional[str] = None,
):
    """Stream managed audio/video inline with HTTP Range support — Electron's
    media pipeline may reject an attachment response as an ``<audio>``/
    ``<video>`` source. Browser media elements authenticate with a reusable
    path-bound ``?ticket=``; the session token is never accepted in this URL.
    Same size cap, sensitive guard and MIME detection as download. A
    ``session_id`` resolves ``/workspace`` in that session's own sandbox, as the
    ticket route does. An SSH profile's clip streams from its SSH machine
    (``_remote_fs_backend``, the rule download uses), one Range at a time.
    Every request checks its ``session_id`` first, a ticketed one included."""
    await _fs_owned_session(session_id, profile)
    authority = getattr(request.state, "file_ticket_authority", None)
    if authority is not None and authority.workspace is not None:
        target, size = await _ticketed_remote_target(authority, _remote_stream_target)
        return remote_media_response(request, authority.workspace, target, size, _fs_mime_type(Path(target)))
    if authority is None:
        backend = await asyncio.to_thread(_remote_fs_backend, path, profile)
        if backend is not None:
            target, size = await asyncio.to_thread(_remote_stream_target, backend, path)
            return remote_media_response(request, backend, target, size, _fs_mime_type(Path(target)))
        if session_id is not None or path.lower().startswith("file:"):
            path = str(await _fs_download_path(path, profile, session_id))
    return await _managed_file_response(request, path, content_disposition_type="inline", media_only=True)


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
async def fs_list(path: str, request: Request, profile: Optional[str] = None):
    backend = await asyncio.to_thread(_fs_backend, profile)
    if backend is not None:
        try:
            return await asyncio.to_thread(backend.list_dir, path, _FS_READDIR_HIDDEN)
        except Exception as exc:
            _raise_fs_backend_error(exc)
    target = _agent_fs_path(path, profile_home=_fs_profile_home(profile))
    _hosted_fs_read_guard(target, request)
    try:
        entries = []
        with os.scandir(target) as scan:
            for entry in scan:
                if entry.name in _FS_READDIR_HIDDEN or _is_sensitive_path(Path(entry.path)):
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
async def fs_read_text(
    path: str, request: Request = None, profile: Optional[str] = None, session_id: Optional[str] = None,
):
    await _fs_owned_session(session_id, profile)
    # A staged paste or pick previews from this host, as read-data-url reads it back to send.
    backend = await asyncio.to_thread(_remote_fs_backend, path, profile)
    if backend is not None:
        try:
            data, size, target = await asyncio.to_thread(
                backend.read_bytes,
                path,
                max_bytes=_FS_TEXT_SOURCE_MAX_BYTES,
                read_limit=_FS_TEXT_PREVIEW_MAX_BYTES,
            )
        except Exception as exc:
            _raise_fs_backend_error(exc)
        target_path = Path(target)
        return {
            "binary": _fs_looks_binary(data[:4096]),
            "byteSize": size,
            "language": _FS_PREVIEW_LANGUAGE_BY_EXT.get(target_path.suffix.lower(), "text"),
            "mimeType": _fs_mime_type(target_path),
            "path": target,
            "text": data.decode("utf-8", errors="replace"),
            "truncated": size > _FS_TEXT_PREVIEW_MAX_BYTES,
        }
    profile_home = _fs_profile_home(profile)
    # A session's /workspace is its own sandbox, resolved as download resolves it.
    target = await _fs_download_path(path, profile, session_id)
    # The same staged-file exception read-data-url makes under a locked files root.
    if not await asyncio.to_thread(_is_host_file, str(target), profile):
        _hosted_fs_read_guard(target, request)
    target, st = _fs_regular_file(target, profile_home)
    if st.st_size > _FS_TEXT_SOURCE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    data = await asyncio.to_thread(
        _fs_read_bytes, target, min(st.st_size, _FS_TEXT_PREVIEW_MAX_BYTES),
    )
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
async def fs_write_text(payload: FsWriteText, profile: Optional[str] = None):
    """Overwrite (or create) a UTF-8 text file for the in-app spot editor.

    Mirrors the Electron ``hermes:fs:writeText`` hardening: path validated by
    ``_fs_path``, the parent must already exist (never build trees), only
    regular files may be replaced, payload size-capped, staged to a sibling
    temp file and ``os.replace``-d so a crash can't truncate the original.
    Stale-on-disk detection is the client's job (re-read before save). The
    target resolves as ``read-text`` resolves it, ``session_id`` included, so a
    file saves to the same place it was read from.
    """
    text = payload.content or ""
    await _fs_owned_session(payload.session_id, profile)
    # Save where read-text read: a staged paste or pick on this host, all else on the SSH machine.
    backend = await asyncio.to_thread(_remote_fs_backend, payload.path, profile)
    if backend is not None:
        try:
            target, byte_size = await asyncio.to_thread(
                backend.write_text,
                payload.path,
                text,
                max_bytes=_FS_TEXT_WRITE_MAX_BYTES,
            )
        except Exception as exc:
            _raise_fs_backend_error(exc)
        return {"ok": True, "path": target, "byteSize": byte_size}

    target = await _fs_download_path(
        payload.path, profile, payload.session_id, decode_fallback=False, own_sandbox_only=True,
    )
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


async def _fs_download_path(
    path: str, profile: Optional[str], session_id: Optional[str], *, decode_fallback: bool = True,
    own_sandbox_only: bool = False,
) -> Path:
    # Resolve container paths in the requested profile's sandbox, not the active one's.
    # A write passes decode_fallback=False: it must never land on a decoded twin of its name.
    # A session's write also passes own_sandbox_only: reads fall back to the shared sandbox
    # when the session's own copy is gone, a Save must not (409 instead).
    profile_home = _fs_profile_home(profile)
    if session_id is not None:
        session = await _fs_owned_session(session_id, profile) or {}
        # Validate ownership even for absolute paths; never trust a client cwd.
        cwd = session.get("cwd") or ""
        return _agent_fs_path(
            path, cwd=cwd, profile_home=profile_home,
            workspace_key=_session_workspace_key(session_id, cwd), decode_fallback=decode_fallback,
            own_sandbox_only=own_sandbox_only,
        )
    return _agent_fs_path(path, profile_home=profile_home, decode_fallback=decode_fallback)


async def _fs_owned_session(session_id: Optional[str], profile: Optional[str]) -> Optional[dict]:
    """The session *session_id* names, looked up in *profile*'s own sessions, or None when the
    request names none. A blank id, or one that profile does not own, is 404. A route that takes a
    ``session_id`` calls this before it picks a backend, so an SSH profile refuses the same
    sessions a host or Docker one does."""
    if session_id is None:
        return None
    from hermes_cli.web_routers.sessions import get_session_detail

    if not session_id.strip():
        raise HTTPException(status_code=404, detail="Session not found")
    return await get_session_detail(session_id, profile)


def _fs_profile_home(profile: Optional[str]) -> Optional[Path]:
    """HERMES_HOME of a requested profile (404/400 on an unknown one), None for the request's
    own profile (omitted, ``""`` or ``current``, as ``_is_current_profile`` reads them)."""
    from hermes_cli.web_server_cron import _cron_profile_home
    from hermes_cli.web_server_profiles import _is_current_profile

    if _is_current_profile(profile):
        return None

    return _cron_profile_home(profile)[1]


@router.get("/api/fs/read-data-url")
async def fs_read_data_url(
    path: str, request: Request, profile: Optional[str] = None, session_id: Optional[str] = None,
):
    from hermes_cli.web_server import _FS_DATA_URL_MAX_BYTES
    await _fs_owned_session(session_id, profile)
    backend = await asyncio.to_thread(_remote_fs_backend, path, profile)
    if backend is not None:
        try:
            data, _size, target = await asyncio.to_thread(
                backend.read_bytes, path, max_bytes=_FS_DATA_URL_MAX_BYTES
            )
        except Exception as exc:
            _raise_fs_backend_error(exc)
        encoded = base64.b64encode(data).decode("ascii")
        return {"dataUrl": f"data:{_fs_mime_type(Path(target))};base64,{encoded}"}
    target = await _fs_download_path(path, profile, session_id)
    # A browser paste or pick reads back to be sent wherever a locked files root sits: the server
    # wrote it into the profile's own image cache or staged uploads, outside the managed tree.
    if not await asyncio.to_thread(_is_host_file, str(target), profile):
        _hosted_fs_read_guard(target, request)
    target, st = _fs_regular_file(target, _fs_profile_home(profile))
    if st.st_size > _FS_DATA_URL_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    encoded = await asyncio.to_thread(
        lambda: base64.b64encode(_fs_read_bytes(target)).decode("ascii"),
    )
    return {"dataUrl": f"data:{_fs_mime_type(target)};base64,{encoded}"}


def _remote_download_target(backend, path: str) -> tuple[str, int]:
    """*path* resolved on the SSH workspace to a regular file a download may read, and its size."""
    from hermes_cli.ssh_workspace_fs import SshWorkspaceFsError

    try:
        target, size = backend.inspect_file(path)
    except SshWorkspaceFsError as exc:
        _raise_fs_backend_error(exc)
    if _is_sensitive_path(Path(target)):
        raise HTTPException(status_code=403, detail="Access to sensitive files is not allowed")
    return target, size


def _remote_stream_target(backend, path: str) -> tuple[str, int]:
    """*path* on the SSH workspace as media the stream route may serve, and its size: download's
    checks, then the host stream's media-type and size caps."""
    from hermes_cli.web_server import _MANAGED_FILE_MAX_BYTES

    target, size = _remote_download_target(backend, path)
    if PurePosixPath(target).suffix.lower() not in _STREAMABLE_MEDIA_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported media type")
    if size > _MANAGED_FILE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is too large")
    return target, size


async def _remote_file_authority(
    path: str, profile: Optional[str], owner: Path, inspect,
) -> Optional[file_tickets.FileAuthority]:
    """A ticket's authority for *path* on the profile's SSH workspace, checked by *inspect*
    (download's or the stream's rules), or None when this host owns the path."""
    backend = await asyncio.to_thread(_remote_fs_backend, path, profile)
    if backend is None:
        return None
    target, _size = await asyncio.to_thread(inspect, backend, path)
    authority = file_tickets.FileAuthority(PurePosixPath(target), (owner,), workspace=backend)
    await asyncio.to_thread(authority.check_live)
    return authority


async def _ticketed_remote_target(authority: file_tickets.FileAuthority, inspect) -> tuple[str, int]:
    """The SSH file a ticket captured, re-checked by *inspect* while its profile is still live."""
    try:
        await asyncio.to_thread(authority.check_live)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="File ticket profile or resource is unavailable") from exc
    target, size = await asyncio.to_thread(inspect, authority.workspace, str(authority.target))
    if target != str(authority.target):
        raise HTTPException(status_code=403, detail="File ticket target changed")
    return target, size


def _remote_download_response(request: Optional[Request], backend, target: str, size: int) -> Response:
    """*target* streamed from the SSH workspace as an attachment; a HEAD reads none of it."""
    target_path = Path(target)
    filename = urllib.parse.quote(target_path.name)
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"}
    media_type = _fs_mime_type(target_path)
    if request is not None and request.method == "HEAD":
        return Response(media_type=media_type, headers={**headers, "Content-Length": str(size)})
    return StreamingResponse(backend.stream_file(target), media_type=media_type, headers=headers)


async def _fs_download_authority(
    request: Request, path: str, profile: Optional[str], session_id: Optional[str], owner: Path,
) -> file_tickets.FileAuthority:
    """The file ``/api/fs/download`` serves for this query, captured for a download ticket: on the
    profile's SSH workspace when it has one, else on this host under the same read rules."""
    from hermes_constants import named_profile_home

    remote = await _remote_file_authority(path, profile, owner, _remote_download_target)
    if remote is not None:
        return remote
    local = await _fs_download_path(path, profile, session_id)

    def capture():
        _hosted_fs_read_guard(local, request)
        target, _st = _fs_regular_file(local, _fs_profile_home(profile))
        homes = [owner]
        target_home = named_profile_home(target)
        if target_home is not None and target_home not in homes:
            homes.append(target_home)
        authority = file_tickets.FileAuthority(target, tuple(homes))
        authority.check_live()
        return authority

    return await asyncio.to_thread(capture)


async def _ticketed_fs_download(
    request: Request, authority: file_tickets.FileAuthority, profile: Optional[str],
) -> Response:
    """Serve the file a download ticket captured while its profile homes are still live."""
    if authority.workspace is not None:
        target, size = await _ticketed_remote_target(authority, _remote_download_target)
        return _remote_download_response(request, authority.workspace, target, size)

    from hermes_cli.web_server_file_response import OpenedFileResponse

    def checked():
        _hosted_fs_read_guard(authority.target, request)
        target, _st = _fs_regular_file(authority.target, _fs_profile_home(profile))
        return target, None

    return OpenedFileResponse(
        path=str(authority.target),
        opener=lambda: _pin_ticket_file(authority, checked),
        media_type=_fs_mime_type(authority.target),
        filename=authority.target.name,
        content_disposition_type="attachment",
    )


@router.get("/api/fs/download")
@router.head("/api/fs/download")
async def fs_download(
    path: str, request: Request = None, profile: Optional[str] = None, session_id: Optional[str] = None,
):
    """A workspace file as an attachment: from the profile's SSH workspace when it has one, else
    from this host. A browser download link authenticates with a path-bound ``?ticket=``."""
    await _fs_owned_session(session_id, profile)
    authority = getattr(getattr(request, "state", None), "file_ticket_authority", None)
    if authority is not None:
        return await _ticketed_fs_download(request, authority, profile)
    backend = await asyncio.to_thread(_remote_fs_backend, path, profile)
    if backend is not None:
        target, size = await asyncio.to_thread(_remote_download_target, backend, path)
        return _remote_download_response(request, backend, target, size)
    target = await _fs_download_path(path, profile, session_id)
    _hosted_fs_read_guard(target, request)
    target, _st = _fs_regular_file(target, _fs_profile_home(profile))
    await asyncio.to_thread(_refuse_live_database, target)
    return FileResponse(
        path=str(target),
        media_type=_fs_mime_type(target),
        filename=target.name,
        content_disposition_type="attachment",
    )


@router.get("/api/fs/git-root")
async def fs_git_root(path: str, profile: Optional[str] = None):
    backend = await asyncio.to_thread(_fs_backend, profile)
    if backend is not None:
        try:
            return {"root": await asyncio.to_thread(backend.git_root, path)}
        except Exception as exc:
            _raise_fs_backend_error(exc)
    target = _fs_path(path)
    try:
        st = target.stat()
        start = target if stat.S_ISDIR(st.st_mode) else target.parent
    except OSError:
        start = target
    return {"root": _fs_find_git_root(start)}


@router.get("/api/fs/default-cwd")
async def fs_default_cwd(profile: Optional[str] = None):
    backend = await asyncio.to_thread(_fs_backend, profile)
    if backend is not None:
        cwd = backend.cwd
        try:
            branch = await asyncio.to_thread(backend.git_branch, cwd)
        except Exception as exc:
            _raise_fs_backend_error(exc)
        return {"cwd": cwd, "branch": branch}
    cwd = await asyncio.to_thread(_fs_default_cwd, profile)
    return {"cwd": cwd, "branch": _fs_git_branch(cwd)}
