"""Deliver ``MEDIA:<path>`` files that live inside a remote terminal sandbox (#466).

``validate_media_delivery_path`` only accepts files on the gateway host. When the terminal backend
is ssh / modal / daytona / singularity / vercel (or a docker path outside its bind mounts), the
agent's artifact is on another filesystem, so the tag was silently dropped. This module pulls the
file through the active environment's ``fetch_file`` into the document cache (already an
allowlisted delivery root) and hands back the host copy.

The remote path is screened against the same denylist as local deliveries BEFORE any bytes move,
and again after ``readlink -f`` — a remote fetch must never become a bypass of the host denylist.
"""

from __future__ import annotations

import logging
import os
import posixpath
import uuid
from pathlib import Path, PurePosixPath
from typing import Optional

logger = logging.getLogger(__name__)

# Telegram bot uploads cap at 50 MB; the other platforms are in the same range. Mirrors
# tools.image_source._MAX_INGEST_BYTES.
_FETCH_MAX_BYTES = 50 * 1024 * 1024


def remote_path_is_denied(path: str, remote_home: Optional[str]) -> bool:
    """Pure string check (the remote fs can't be stat'd from here) applying the host denylist to a
    sandbox path: system prefixes, credential dirs under the sandbox home, Hermes credential stores.
    Unknown home ⇒ home-relative entries match ANY path component (conservative)."""
    from gateway.platforms.base import (
        _MEDIA_DELIVERY_DENIED_HOME_SUBPATHS, _MEDIA_DELIVERY_DENIED_PREFIXES, _ROOT_CREDENTIAL_PATHS)

    target = PurePosixPath(posixpath.normpath(path))
    if not target.is_absolute():
        return True
    home = PurePosixPath(posixpath.normpath(remote_home)) if remote_home else None

    def _under(root: PurePosixPath) -> bool:
        return target == root or root in target.parents

    # The sandbox's own home may be a denied system prefix (/root); its credential subpaths are
    # separate, more specific entries — same exception as _path_under_denied_prefix.
    if any(_under(PurePosixPath(p)) for p in _MEDIA_DELIVERY_DENIED_PREFIXES if PurePosixPath(p) != home):
        return True
    home_relative = [PurePosixPath(s) for s in _MEDIA_DELIVERY_DENIED_HOME_SUBPATHS]
    home_relative += [PurePosixPath(".hermes", *PurePosixPath(rel.replace(os.sep, "/")).parts) for rel in _ROOT_CREDENTIAL_PATHS]
    if home is not None:
        return any(_under(home / rel) for rel in home_relative)
    parts = target.parts
    return any(parts[i:i + len(rel.parts)] == rel.parts
               for rel in home_relative for i in range(len(parts) - len(rel.parts) + 1))


def _active_remote_env():
    """The live remote BaseEnvironment for the current session, or None (local backend / no env yet)."""
    from agent.prompt_builder import _REMOTE_TERMINAL_BACKENDS, _plugin_backend_is_remote
    from gateway.platforms.base import _tenv
    backend = _tenv("TERMINAL_ENV", "local").strip().lower()
    if backend not in _REMOTE_TERMINAL_BACKENDS and not _plugin_backend_is_remote(backend):
        return None
    try:
        from tools.terminal_tool import _current_session_key
        from tools.terminal_tool_lifecycle import get_active_env
        return get_active_env(_current_session_key() or "default")
    except Exception as exc:
        logger.debug("Remote media fetch: no active env: %s", exc)
        return None


def fetch_remote_media(path: str) -> Optional[str]:
    """Host path of a validated copy of sandbox file ``path``, or None (never raises). Only fires
    when a remote backend is active; the caller has already failed local validation."""
    env = _active_remote_env()
    if env is None:
        return None
    from gateway.platforms.base import DOCUMENT_CACHE_DIR, _log_safe_path, validate_media_delivery_path
    from tools.environments.base import FileFetchError

    remote_home = getattr(env, "_remote_home", None)
    candidate = posixpath.normpath(str(path).strip())
    if candidate == "~" or candidate.startswith("~/"):
        if not remote_home:
            return None
        candidate = posixpath.normpath(posixpath.join(remote_home, candidate[2:]))
    if not candidate.startswith("/") or remote_path_is_denied(candidate, remote_home):
        return None
    try:
        resolved = env.fetch_realpath(candidate) or candidate
        if remote_path_is_denied(resolved, remote_home):
            return None
        basename = "".join(c for c in posixpath.basename(resolved) if c.isprintable() and c not in '/\\:*?"<>|') or "file"
        dest = Path(DOCUMENT_CACHE_DIR) / f"remote_{uuid.uuid4().hex[:12]}_{basename}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        env.fetch_file(resolved, dest, max_bytes=_FETCH_MAX_BYTES)
    except FileFetchError as exc:
        logger.warning("Remote media fetch of %s skipped: %s", _log_safe_path(candidate), exc)
        return None
    except Exception:
        logger.warning("Remote media fetch of %s failed", _log_safe_path(candidate), exc_info=True)
        return None
    validated = validate_media_delivery_path(str(dest))
    if not validated:
        dest.unlink(missing_ok=True)
        return None
    logger.info("Fetched remote media %s from the %s sandbox", _log_safe_path(candidate), type(env).__name__)
    return validated
