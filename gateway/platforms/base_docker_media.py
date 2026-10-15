"""Docker sandbox path translation for media delivery: container path -> host file.

The mount discovery helpers stay in ``base.py``; both functions here read them through the
facade at call time, so a patch on ``gateway.platforms.base`` still takes effect.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional


def _warn_unresolved_docker_media(candidate: Path, session_key: str, reason: str) -> None:
    """Name WHY a container-absolute MEDIA path failed translation (otherwise the only signal is
    the generic "Skipping unsafe MEDIA directive path" line). Docker-only; host rejections quiet.

    See #93950.
    """
    from gateway.platforms import base as _base

    if not _base._docker_env_active():
        return
    _base.logger.warning(
        "Docker MEDIA path %s did not resolve to a host sandbox file (%s%s); "
        "the producing container's sandbox directory may not exist yet or "
        "was pruned", _base._log_safe_path(str(candidate)), reason,
        f", session_key={session_key}" if session_key else "")


def _translate_docker_container_media_path(candidate: Path, session_key: str = "", *,
                                           warn: bool = True) -> Optional[Path]:
    """Container-absolute path -> host path via longest-prefix match over ``docker_volumes``, the
    auto-mounted cache dirs (``/root/.hermes/...``), persistent ``/workspace`` and ``/root``.
    ``warn=False`` silences the unresolved-path warning for callers that log the miss themselves
    (bare paths in reply prose are often not files at all)."""
    from gateway.platforms import base as _base

    if not candidate.is_absolute():
        return None
    report = _warn_unresolved_docker_media if warn else (lambda *_args: None)
    # In-process gateways (Desktop, `hermes serve`) may not have bridged terminal.* config into
    # TERMINAL_* env yet; the bridge is idempotent.
    with contextlib.suppress(Exception):
        from tools.terminal_tool import _ensure_terminal_env_bridged
        _ensure_terminal_env_bridged()
    mounts = [*_base._parse_docker_volume_mounts(), *_base._cache_dir_container_mounts()]
    mounted = {c.as_posix() for _, c in mounts}
    # Synthetic /workspace mounts: profile-scoped layout first, then legacy per-session.
    if "/workspace" not in mounted:
        mounts.extend((root, Path("/workspace")) for root in _base._default_docker_workspace_host_roots(session_key))
    # Synthetic /root mounts catch stray home writes (/root/out.png; cache mounts are longer
    # prefixes). /root/.hermes/* that missed a cache mount is the container's credential surface —
    # translating it via the home mount would dodge the host denylist.
    if "/root" not in mounted and not candidate.as_posix().startswith("/root/.hermes"):
        mounts.extend(
            (root, Path("/root")) for root in _base._docker_persistent_sandbox_roots(session_key, "home"))
    if not mounts:
        report(candidate, session_key, "no sandbox mounts resolved")
        return None
    # Longest container-prefix match; equal-length prefixes are tried in insertion order.
    candidate_posix = candidate.as_posix()
    matched = [(host_root, container_root, len(prefix)) for host_root, container_root in mounts
               for prefix in (container_root.as_posix().rstrip("/") or "/",)
               if candidate_posix == prefix or candidate_posix.startswith(prefix + "/")]
    if not matched:
        report(candidate, session_key, "no mounted prefix matches")
        return None
    for host_root, container_root, _score in sorted(matched, key=lambda m: -m[2]):
        translated = _base._resolve_path(host_root / candidate.relative_to(container_root), strict=True)
        if translated is not None and (
                translated == host_root or _base._path_is_within(translated, host_root)):
            return translated
    report(candidate, session_key, "host file missing from sandbox")
    return None
