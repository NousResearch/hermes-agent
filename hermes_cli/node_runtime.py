"""Node-runtime resolution helpers — extracted from ``hermes_cli/main.py``.

Mechanical move (main.py decomposition): the three node-runtime leaf helpers
(``_is_termux_env``, ``_is_windows_npm_path``, ``_resolve_node_runtime_npm``)
are lifted verbatim. References to helpers that STAY in ``hermes_cli.main``
(``_is_termux_startup_environment``, ``_is_windows``) are routed through a
lazy ``_m()`` main reference so existing test monkeypatches on
``hermes_cli.main.<name>`` keep reaching this code path, and imports stay
one-way at import time (main.py imports this module, never the reverse).
``main.py`` re-exports all three names (``# noqa: F401``) so callers and test
patches on ``hermes_cli.main`` resolve unchanged.
"""

import os
import shutil


def _m():
    """Lazy ``hermes_cli.main`` reference (call-time; keeps patches working)."""
    from hermes_cli import main

    return main


def _is_termux_env(env: dict[str, str] | None = None) -> bool:
    return _m()._is_termux_startup_environment(env)


def _is_windows_npm_path(npm_path: str) -> bool:
    """Return True if ``npm_path`` points at a Windows npm shim.

    On WSL the Windows install dir is exposed through the ``/mnt/c`` drive
    mount and PATH interop, so ``shutil.which("npm")`` can hand back
    ``/mnt/c/Program Files/nodejs/npm`` (or the ``npm.cmd`` / ``npm.exe``
    shim). Those are detected here by their ``.exe``/``.cmd``/``.bat``
    suffix, a ``/mnt/`` drive-mount prefix, or an embedded backslash (a UNC
    path). Callers use this only on a POSIX host — on native Windows an
    ``npm.cmd`` shim is the correct executable.
    """
    low = npm_path.lower()
    return (
        low.endswith((".exe", ".cmd", ".bat"))
        or low.startswith("/mnt/")
        or "\\" in npm_path
    )


def _resolve_node_runtime_npm() -> str | None:
    """Resolve an npm executable that belongs to the host's Node runtime.

    On WSL/Linux ``shutil.which("npm")`` may resolve a Windows npm exposed
    through PATH interop. Running that Windows npm against the Linux checkout
    operates over ``\\wsl.localhost\\...`` UNC paths and fails with EISDIR /
    symlink errors in symlink-heavy trees like ``ui-tui`` (#30271). Refuse a
    Windows npm on a POSIX host and re-scan PATH (skipping ``/mnt/*`` interop
    entries) for a Linux-native npm. Returns the npm path, or ``None`` when
    no suitable npm is reachable.
    """
    from hermes_constants import find_node_executable

    npm = find_node_executable("npm")

    # On native Windows the platform npm (``npm.cmd``) is exactly what we
    # want — only reject Windows shims when we're a POSIX/WSL process.
    if _m()._is_windows():
        return npm

    if not npm:
        return None

    if not _is_windows_npm_path(npm):
        return npm

    # The first resolution was a Windows npm. Re-scan PATH skipping the
    # ``/mnt/*`` Windows drive mounts WSL injects, so a Linux-native npm that
    # came later on PATH is still found.
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory or directory.lower().startswith("/mnt/"):
            continue
        candidate = shutil.which("npm", path=directory)
        if candidate and not _is_windows_npm_path(candidate):
            return candidate
    return None
