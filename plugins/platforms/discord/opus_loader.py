"""Opus codec loading for Discord voice support.

Centralises libopus discovery so the Discord adapter and the
``scripts/discord-voice-doctor.py`` diagnostic share a single resolution
strategy. The loader is dependency-injectable so unit tests can drive
every branch (NixOS-style ``find_library`` returns ``None``, env-var
override, per-platform fallbacks) without touching the host system.

Resolution order (#30723):

    1. ``discord.opus.is_loaded()`` — short-circuit if already loaded.
    2. ``DISCORD_OPUS_LIBRARY`` environment variable — full path
       override. This is the recommended fix for NixOS / Nix-store
       layouts where ``ctypes.util.find_library("opus")`` returns
       ``None`` because libopus lives under a Nix store path that is
       not in the linker cache.
    3. ``ctypes.util.find_library("opus")`` — existing automatic
       discovery (works on most distros + macOS with the system
       linker, works on macOS-Homebrew via the Apple-Silicon linker
       cache, doesn't work on NixOS).
    4. Platform-specific candidate list — bare library names like
       ``libopus.so.0`` (resolved via the dynamic linker) plus a small
       set of absolute paths matching distro conventions (mirrors the
       doctor-script list so the two stay in sync).

Failures during individual candidate loads are logged at DEBUG and
swallowed so the next candidate can be tried. If every candidate
fails, a single WARNING with the full attempt list and an explicit
``DISCORD_OPUS_LIBRARY=...`` hint is emitted.
"""

from __future__ import annotations

import ctypes.util
import logging
import os
import os.path
import sys
from typing import Any, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


# Environment variable name — exported for tests and doctor script so
# the string lives in exactly one place.
OPUS_LIBRARY_ENV_VAR = "DISCORD_OPUS_LIBRARY"


# Bare library names tried via the dynamic linker on Linux-family
# platforms. ``libopus.so.0`` is the runtime SONAME shipped by the
# ``libopus0`` package on Debian/Ubuntu and most other distros;
# ``libopus.so`` is the unversioned symlink installed by the ``-dev``
# package.
_LINUX_LIB_NAMES: Tuple[str, ...] = (
    "libopus.so.0",
    "libopus.so",
)


# Absolute paths kept here as a last-resort fallback when both
# ``DISCORD_OPUS_LIBRARY`` and ``ctypes.util.find_library`` fail.
# Mirrors the list in ``scripts/discord-voice-doctor.py``.
_LINUX_FALLBACK_PATHS: Tuple[str, ...] = (
    "/usr/lib/x86_64-linux-gnu/libopus.so.0",   # Debian/Ubuntu x86_64
    "/usr/lib/aarch64-linux-gnu/libopus.so.0",  # Debian/Ubuntu arm64
    "/usr/lib/libopus.so",                       # Arch Linux
    "/usr/lib64/libopus.so",                     # RHEL / Fedora 64-bit
    "/usr/lib/libopus.so.0",                     # generic
)


_DARWIN_FALLBACK_PATHS: Tuple[str, ...] = (
    "/opt/homebrew/lib/libopus.dylib",  # Apple Silicon (default brew prefix)
    "/usr/local/lib/libopus.dylib",     # Intel Mac
)


def _candidates_for_platform(platform: str) -> List[str]:
    """Return the ordered list of platform-specific candidates to try
    AFTER the env-var override and ``ctypes.util.find_library`` have
    been consulted. The DISCORD_OPUS_LIBRARY override is handled in
    :func:`ensure_discord_opus_loaded` directly so it stays
    cross-platform."""
    if platform == "darwin":
        return list(_DARWIN_FALLBACK_PATHS)
    if platform == "win32":
        # discord.py bundles ``libopus-0.x64.dll`` / ``libopus-0.x86.dll``
        # on Windows and loads them transparently in ``discord.opus`` at
        # import time, so once ``is_loaded()`` returns True we're done.
        # Nothing useful to add here.
        return []
    # Linux + every other Unix-like platform (FreeBSD, NixOS-on-Linux, …).
    return [*_LINUX_LIB_NAMES, *_LINUX_FALLBACK_PATHS]


def _build_unavailable_hint(attempts: List[Tuple[str, str]]) -> str:
    """Build the user-facing 'libopus not loadable' diagnostic.

    Kept as a separate function so tests can assert the exact phrasing
    without having to fake out a logger.
    """
    base = (
        "Discord voice playback requires libopus. Hermes could not "
        f"locate or load it. Set {OPUS_LIBRARY_ENV_VAR} to the full "
        "path of libopus.so (Linux/NixOS) or libopus.dylib (macOS) and "
        "restart. See issue #30723 for the NixOS rationale."
    )
    if not attempts:
        return f"{base} No candidates were available to try."
    formatted = ", ".join(f"{candidate!r}: {reason}" for candidate, reason in attempts)
    return f"{base} Attempted candidates: {formatted}"


def ensure_discord_opus_loaded(
    *,
    discord_module: Any = None,
    platform: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    isfile=os.path.isfile,
    find_library=ctypes.util.find_library,
) -> bool:
    """Load libopus for discord.py voice support and return ``True`` on success.

    All external dependencies are injectable so tests can exercise every
    branch in isolation:

    :param discord_module: defaults to ``import discord`` lazily. Tests
        should pass a stub exposing ``opus.is_loaded()`` and
        ``opus.load_opus(name)``.
    :param platform: defaults to ``sys.platform``. Tests pass
        ``"linux"`` / ``"darwin"`` / ``"win32"`` directly.
    :param env: defaults to ``os.environ``. Tests pass a plain dict.
    :param isfile: defaults to ``os.path.isfile``. Tests stub it to
        simulate which absolute paths exist on disk.
    :param find_library: defaults to ``ctypes.util.find_library``. Tests
        return ``None`` (the NixOS case) or a fake path.

    The return value indicates whether ``discord.opus.is_loaded()`` ends
    up ``True``. The function never raises for individual candidate
    failures; it logs them at DEBUG and proceeds. A single WARNING is
    emitted at the end if no candidate succeeded.
    """
    if discord_module is None:
        try:
            import discord as discord_module  # type: ignore[no-redef]
        except ImportError:
            logger.warning(
                "Cannot load Opus codec — discord.py is not installed. "
                "Install with: pip install 'discord.py[voice]'"
            )
            return False

    # Already loaded — common in long-running bots where connect() runs
    # more than once (reconnects, multi-bot setups, etc.).
    if discord_module.opus.is_loaded():
        return True

    if platform is None:
        platform = sys.platform
    if env is None:
        env = os.environ

    candidates: List[str] = []

    # 1. User-provided override (#30723 — NixOS recommended path).
    override = env.get(OPUS_LIBRARY_ENV_VAR)
    if override:
        candidates.append(override)

    # 2. ctypes.util.find_library — the existing strategy. Wrapped in
    # try/except because some exotic ctypes shims raise instead of
    # returning ``None``.
    try:
        discovered = find_library("opus")
    except Exception as exc:
        logger.debug(
            "ctypes.util.find_library('opus') raised %s: %s",
            type(exc).__name__,
            exc,
        )
        discovered = None
    if discovered:
        candidates.append(discovered)

    # 3. Platform-specific candidates. Absolute paths are gated by an
    # existence check (so we don't log a misleading "load failed"
    # warning for paths that don't exist on this distro). Bare names
    # are always tried — the dynamic linker resolves them.
    for candidate in _candidates_for_platform(platform):
        if os.path.isabs(candidate):
            try:
                if not isfile(candidate):
                    continue
            except Exception:
                continue
        candidates.append(candidate)

    # Dedup while preserving order — the override and ``find_library``
    # may both return the same path.
    seen: set[str] = set()
    ordered: List[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)

    attempts: List[Tuple[str, str]] = []
    for candidate in ordered:
        try:
            discord_module.opus.load_opus(candidate)
        except Exception as exc:
            attempts.append((candidate, f"{type(exc).__name__}: {exc}"))
            logger.debug(
                "discord.opus.load_opus(%r) failed: %s: %s",
                candidate,
                type(exc).__name__,
                exc,
            )
            continue
        if discord_module.opus.is_loaded():
            logger.info("Loaded Discord Opus library from %s", candidate)
            return True
        attempts.append((candidate, "is_loaded() returned False after load_opus"))

    logger.warning("%s", _build_unavailable_hint(attempts))
    return False
