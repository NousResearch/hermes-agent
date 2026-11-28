"""Load the opus codec Discord voice playback needs, once per process.

``ctypes.util.find_library`` runs ``ldconfig``/``gcc``/``ld`` subprocesses on Linux.
On a host without libopus it never finds anything, so probing on every Discord
(re)connect repeated that work each time on the gateway event loop. Callers run
``load_opus_codec`` in a worker thread; it probes at most once per process, so a
libopus installed later needs a restart.
"""
from __future__ import annotations

import logging
import os
import struct
import sys
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_probe_lock = threading.Lock()
_probed = False

# Runtime soname locations for the opus Discord voice playback needs, per loader convention.
# ``ctypes.util.find_library("opus")`` resolves the linker name (``libopus.so``), which only the
# ``-dev`` package ships; the runtime package installs the versioned soname alone. Order mirrors
# scripts/discord-voice-doctor.py so the probe and the doctor never disagree.
_LINUX_OPUS_SONAME_CANDIDATES = (
    "/usr/lib/x86_64-linux-gnu/libopus.so.0",   # Debian/Ubuntu x86_64 (libopus0)
    "/usr/lib/aarch64-linux-gnu/libopus.so.0",  # Debian/Ubuntu ARM (libopus0)
    "/usr/lib64/libopus.so.0",                  # RHEL/Fedora/CentOS (opus)
    "/usr/lib64/libopus.so",                    # ditto, dev layout
    "/usr/lib/libopus.so.0",                    # Arch, non-multiarch layouts
    "/usr/lib/libopus.so",                      # Arch, dev layout
    "/usr/local/lib/libopus.so.0",              # built from source / container layers
)


def load_opus_codec() -> None:
    """Try bundled (Windows) opus, then ``ctypes.util.find_library``, then the versioned-soname
    paths each platform's package manager installs but ``find_library`` cannot resolve (Homebrew
    on macOS, ``libopus0`` / ``opus`` on Linux); warn once if none loads."""
    global _probed
    import ctypes.util

    import discord

    with _probe_lock:
        if _probed or discord.opus.is_loaded():
            return
        _probed = True
        opus_candidates = []
        bundled_opus = _find_discord_windows_bundled_opus(discord)
        if bundled_opus:
            opus_candidates.append(bundled_opus)
        opus_path = ctypes.util.find_library("opus")
        if opus_path:
            opus_candidates.append(opus_path)
        elif sys.platform == "darwin":
            for _hp in ("/opt/homebrew/lib/libopus.dylib", "/usr/local/lib/libopus.dylib"):  # Apple Silicon, Intel
                if os.path.isfile(_hp):
                    opus_candidates.append(_hp)
                    break
        else:
            # ``find_library`` resolves the linker name ``libopus.so``, which only ``libopus-dev``
            # ships; the runtime package ``libopus0`` installs ``libopus.so.0`` alone. So an opus
            # installed by the documented ``apt install libopus0`` is invisible to the probe above
            # on the very hosts that guide tells them to run it on. Same shape as the Homebrew
            # fallback above: probe the runtime soname where each distro's loader looks.
            for _lp in _LINUX_OPUS_SONAME_CANDIDATES:
                if os.path.isfile(_lp):
                    opus_candidates.append(_lp)
                    break
        for opus_path in opus_candidates:
            try:
                discord.opus.load_opus(opus_path)
                if discord.opus.is_loaded():
                    break
            except Exception:
                logger.warning("Opus codec found at %s but failed to load", opus_path, exc_info=True)
        if not discord.opus.is_loaded():
            logger.warning("Opus codec not found — voice channel playback disabled")


def _find_discord_windows_bundled_opus(discord_module: Any) -> Optional[str]:
    """Return discord.py's bundled Windows opus DLL path when present."""
    if sys.platform != "win32":
        return None
    opus_module = getattr(discord_module, "opus", None)
    opus_file = getattr(opus_module, "__file__", None)
    if not opus_file:
        return None
    target = "x64" if struct.calcsize("P") * 8 > 32 else "x86"
    bundled = Path(opus_file).resolve().parent / "bin" / f"libopus-0.{target}.dll"
    if bundled.is_file():
        return str(bundled)
    return None
