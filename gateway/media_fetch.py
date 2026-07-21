"""Fetch MEDIA files from remote terminal backends for platform delivery.

``MEDIA:/absolute/path`` directives only deliver when the file exists on the
machine running the gateway. Remote terminal backends (ssh, modal, daytona,
managed_modal — and docker/singularity paths outside their bind mounts) put
the agent's artifacts on a different filesystem, so ``validate_media_delivery_path``
rejects them and delivery silently fails (issue #466).

This module bridges the gap: when a MEDIA path fails local resolution and a
remote backend is active, the file is fetched through the backend's
``fetch_file`` transport into the canonical document cache
(``~/.hermes/cache/documents``, already in ``MEDIA_DELIVERY_SAFE_ROOTS``) and
delivered from there. The fetched copy still goes through the normal local
validation, and the *remote* path is checked against the same credential /
system-path denylist as local deliveries before any bytes move — a remote
fetch must not become a bypass of ``_path_under_denied_prefix``.
"""

import logging
import os
import posixpath
import uuid
from pathlib import Path, PurePosixPath
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum size fetched out of a backend for delivery. Telegram bot uploads
# cap at 50 MB; other platforms are in the same order of magnitude. Operators
# can raise/lower it via the env var.
MEDIA_FETCH_MAX_BYTES_ENV = "HERMES_MEDIA_FETCH_MAX_BYTES"
_MEDIA_FETCH_MAX_BYTES_DEFAULT = 50 * 1024 * 1024


def media_fetch_max_bytes() -> int:
    """Return the configured remote-fetch size cap in bytes."""
    raw = os.environ.get(MEDIA_FETCH_MAX_BYTES_ENV, "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return _MEDIA_FETCH_MAX_BYTES_DEFAULT


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} bytes"


def remote_path_is_denied(path: str, remote_home: Optional[str] = None) -> bool:
    """Return True when *path* on a backend filesystem must not be fetched.

    Pure string check (the remote filesystem can't be stat'd from here)
    applying the same denylist as local delivery: system prefixes
    (``/etc``, ``/proc``, ...), credential directories under the backend
    home (``~/.ssh``, ``~/.aws``, ...), and the Hermes credential stores
    (``~/.hermes/.env``, ``auth.json``, ``mcp-tokens/``, ...).

    When *remote_home* is unknown, home-relative entries are matched against
    ANY path component (conservative: ``/data/.ssh/key`` is denied too).
    Mirrors the ``/root``-is-home exception of ``_path_under_denied_prefix``:
    a denied system prefix that IS the backend's own home stays fetchable —
    its credential subpaths are separate, more-specific entries.
    """
    from gateway.platforms.base import (
        _MEDIA_DELIVERY_DENIED_HOME_SUBPATHS,
        _MEDIA_DELIVERY_DENIED_PREFIXES,
        _ROOT_CREDENTIAL_DIRS,
        _ROOT_CREDENTIAL_FILES,
    )

    normalized = PurePosixPath(posixpath.normpath(path))
    if not normalized.is_absolute():
        return True

    home = PurePosixPath(posixpath.normpath(remote_home)) if remote_home else None

    def _under(candidate: PurePosixPath, root: PurePosixPath) -> bool:
        return candidate == root or root in candidate.parents

    for prefix in _MEDIA_DELIVERY_DENIED_PREFIXES:
        root = PurePosixPath(prefix)
        if home is not None and root == home:
            continue
        if _under(normalized, root):
            return True

    home_relative = list(_MEDIA_DELIVERY_DENIED_HOME_SUBPATHS)
    home_relative.extend(
        posixpath.join(".hermes", *PurePosixPath(rel.replace(os.sep, "/")).parts)
        for rel in (*_ROOT_CREDENTIAL_FILES, *_ROOT_CREDENTIAL_DIRS)
    )

    if home is not None:
        for rel in home_relative:
            if _under(normalized, home / rel):
                return True
        return False

    # Unknown home: deny when the denied subpath appears anywhere in the path.
    parts = normalized.parts
    for rel in home_relative:
        rel_parts = PurePosixPath(rel.replace(os.sep, "/")).parts
        for start in range(len(parts) - len(rel_parts) + 1):
            if parts[start:start + len(rel_parts)] == rel_parts:
                return True
    return False


def _active_remote_environment(task_id: Optional[str] = None):
    """Return (backend_name, env) when a remote terminal backend is active.

    Returns ``(backend, None)`` when the backend is remote but no live
    environment exists yet (nothing to fetch from), and ``(backend, None)``
    with backend ``""`` when the configured backend is local.
    """
    backend = (os.getenv("TERMINAL_ENV") or "local").strip().lower()
    from agent.prompt_builder import _REMOTE_TERMINAL_BACKENDS

    if backend not in _REMOTE_TERMINAL_BACKENDS:
        return "", None
    try:
        from tools.terminal_tool import get_active_env

        return backend, get_active_env(task_id or "default")
    except Exception as exc:
        logger.debug("Remote media fetch: could not resolve active env: %s", exc)
        return backend, None


def _sanitize_basename(path: str) -> str:
    name = posixpath.basename(posixpath.normpath(path.replace("\\", "/")))
    name = "".join(c for c in name if c.isprintable() and c not in '/\\:*?"<>|')
    return name.strip() or "file"


def fetch_remote_media(
    path: str, task_id: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Fetch *path* from the active remote backend into the document cache.

    Returns ``(local_path, None)`` on success — *local_path* already passed
    ``validate_media_delivery_path`` — or ``(None, reason)`` with a short
    user-presentable reason on failure. Never raises.
    """
    from gateway.platforms.base import (
        get_document_cache_dir,
        validate_media_delivery_path,
    )
    from tools.environments.base import FileFetchError

    backend, env = _active_remote_environment(task_id)
    if not backend:
        return None, "no remote terminal backend is active"
    if env is None:
        return None, f"no active {backend} terminal session to fetch from"

    candidate = posixpath.normpath(str(path).strip())
    remote_home = env.remote_home
    if candidate == "~" or candidate.startswith("~/"):
        if not remote_home:
            return None, "cannot resolve ~ in the backend"
        candidate = posixpath.normpath(
            posixpath.join(remote_home, candidate[2:])
        )
    if not candidate.startswith("/"):
        return None, "only absolute paths can be fetched from the backend"

    if remote_path_is_denied(candidate, remote_home):
        return None, "the path is not allowed for delivery"

    try:
        # Best-effort symlink resolution so a link planted at an innocuous
        # path can't smuggle out a denied credential file.
        resolved = env.fetch_realpath(candidate)
        if resolved and remote_path_is_denied(resolved, remote_home):
            return None, "the path is not allowed for delivery"

        size = env.fetch_file_size(resolved or candidate)
        if size is None:
            return None, f"the file was not found in the {backend} backend"
        limit = media_fetch_max_bytes()
        if size > limit:
            return None, (
                f"the file is {_format_size(size)}, above the "
                f"{_format_size(limit)} delivery limit"
            )

        dest = get_document_cache_dir() / (
            f"doc_{uuid.uuid4().hex[:12]}_{_sanitize_basename(candidate)}"
        )
        env.fetch_file(resolved or candidate, str(dest))
    except FileFetchError as exc:
        return None, str(exc)
    except Exception as exc:
        logger.warning(
            "Remote media fetch failed for %s backend: %s", backend, exc,
            exc_info=True,
        )
        return None, f"fetching from the {backend} backend failed"

    validated = validate_media_delivery_path(str(dest))
    if not validated:
        try:
            Path(dest).unlink()
        except OSError:
            pass
        return None, "the fetched file failed delivery validation"

    logger.info(
        "Fetched remote media from %s backend: %s (%s)",
        backend, candidate, _format_size(size),
    )
    return validated, None
