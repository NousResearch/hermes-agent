"""Warm the OS page cache with the Codex CLI's native binaries at host startup.

The Codex app-server runtime spawns ``codex app-server`` for every Codex turn. After a reboot (or
once memory pressure evicted it) the first spawn faults ~300 MB of native executable in from disk,
which on a spinning or network disk costs seconds before the JSON-RPC handshake even starts. A
long-lived host (the gateway, ``hermes serve``) reads those files once in a background thread
after boot so the first turn spawns from the cache. Nothing is kept in memory: the files are read
in fixed-size chunks and discarded.

Only profiles whose config selects ``model.openai_runtime: codex_app_server`` contribute a binary,
so hosts that never use the runtime read nothing. Every failure is logged at debug and skipped.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterable
from pathlib import Path

from agent.memory_provider import spawn_context_thread
from agent.runtime_scope import bind_profile_scope
from hermes_cli.codex_runtime_switch import get_configured_codex_binary, get_current_runtime
from hermes_cli.config_effective import load_user_config_effective
from hermes_cli.profiles import profiles_to_serve
from hermes_constants import get_hermes_home
from hermes_platform.resolver import locate_command

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20
_THREAD_NAME = "codex-binary-preread"
# ELF, PE, Mach-O (both endians, 32/64-bit) and universal (fat) Mach-O headers.
_NATIVE_MAGIC = (b"\x7fELF", b"MZ", b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe",
                 b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce", b"\xca\xfe\xba\xbe")
# npm's ``@openai/codex`` launcher (bin/codex.js, or a .cmd/.ps1 shim on Windows) execs a binary
# shipped in a per-platform package: nested under the package, hoisted beside it as
# ``@openai/codex-<platform>``, or (older releases) in the package's own ``vendor/``.
_VENDOR_GLOBS = ("vendor/*/bin/*", "vendor/*/codex/*", "node_modules/@openai/codex-*/vendor/*/bin/*",
                 "node_modules/@openai/codex-*/vendor/*/codex/*")
_HOISTED_GLOBS = ("codex-*/vendor/*/bin/*", "codex-*/vendor/*/codex/*")


def _is_native(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
    except OSError:
        return False
    return any(head.startswith(magic) for magic in _NATIVE_MAGIC)


def _vendor_searches(launcher: Path) -> list[tuple[Path, str]]:
    """``(root, glob)`` pairs that may hold the native binaries behind a non-native launcher:
    ``<pkg>/bin/codex.js`` (Unix symlink target) or ``<prefix>/codex.cmd`` (Windows npm shim)."""
    package_roots = [launcher.parent.parent, launcher.parent / "node_modules" / "@openai" / "codex"]
    searches = [(root, pattern) for root in package_roots for pattern in _VENDOR_GLOBS]
    searches += [(root.parent, pattern) for root in package_roots for pattern in _HOISTED_GLOBS]
    return [(root, pattern) for root, pattern in searches if root.is_dir()]


def native_binaries(codex_bin: str) -> list[Path]:
    """The native executables ``codex_bin`` runs: itself when native, else its npm vendor binaries."""
    found = locate_command(codex_bin).command
    if not found:
        return []
    launcher = Path(os.path.realpath(found[0]))
    if _is_native(launcher):
        return [launcher]
    binaries: dict[Path, None] = {}
    for root, pattern in _vendor_searches(launcher):
        for candidate in sorted(root.glob(pattern)):
            real = Path(os.path.realpath(candidate))
            if real.is_file() and _is_native(real):
                binaries[real] = None
    return list(binaries)


def configured_codex_bins(profile_homes: Iterable[Path]) -> list[str]:
    """``model.codex_bin`` of every profile home whose config selects the codex app-server runtime.

    Each config is read inside its own profile's scope, so ``${VAR}`` refs expand against that
    profile's ``.env``, not the launch profile's."""
    bins: dict[str, None] = {}
    for home in profile_homes:
        with bind_profile_scope(home, hydrate_secrets=False):
            config = load_user_config_effective(Path(home) / "config.yaml")
        if get_current_runtime(config) == "codex_app_server":
            bins[get_configured_codex_binary(config)] = None
    return list(bins)


def _read_through(path: Path) -> int:
    total = 0
    with open(path, "rb", buffering=0) as fh:
        while chunk := fh.read(_CHUNK):
            total += len(chunk)
    return total


def preread_configured_binaries(profile_homes: Iterable[Path]) -> list[tuple[Path, int]]:
    """Read every configured codex native binary once; ``(path, bytes_read)`` per file read."""
    done: list[tuple[Path, int]] = []
    seen: set[Path] = set()
    for codex_bin in configured_codex_bins(profile_homes):
        for binary in native_binaries(codex_bin):
            if binary in seen:
                continue
            seen.add(binary)
            try:
                done.append((binary, _read_through(binary)))
            except OSError:
                logger.debug("codex pre-read skipped %s", binary, exc_info=True)
    return done


def _run(launch_home: Path, multiplex: bool) -> None:
    try:
        for path, size in preread_configured_binaries(host_profile_homes(launch_home, multiplex=multiplex)):
            logger.debug("codex pre-read %s (%d bytes) into the page cache", path, size)
    except Exception:
        logger.debug("codex binary pre-read failed (non-fatal)", exc_info=True)


def host_profile_homes(launch_home: Path, *, multiplex: bool) -> list[Path]:
    """*launch_home* plus, on a multiplexing host, every profile it serves."""
    homes = [launch_home]
    if multiplex:
        homes += [home for _name, home in profiles_to_serve(multiplex=True)]
    return list(dict.fromkeys(Path(home) for home in homes))


def start_codex_preread(*, multiplex: bool) -> threading.Thread:
    """Start the background pre-read for this host's profiles; returns the daemon thread.

    The launch home and profile scope are captured here, on the caller's thread, where they are bound."""
    thread = spawn_context_thread(_run, args=(get_hermes_home(), multiplex), name=_THREAD_NAME)
    thread.start()
    return thread
