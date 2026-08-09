"""Pre-import startup fast paths — THE canonical lightweight helpers.

This module is imported by ``hermes_cli/main.py`` BEFORE its heavy import
wall (config, argparse tree, logging, providers). Everything here must stay
**stdlib-only and cheap** (os/sys file probes; no yaml, no hermes_cli.config,
no argparse). A guard test (``test_startup_fast_import_weight``) subprocess-
imports this module and fails if any heavy module sneaks into sys.modules.

Why this module exists (the bug class it kills): version-printing kept being
reimplemented as ``*_fast()`` copies at the top of main.py (Termux first,
then globally), each duplicating canonical logic — project-root resolution,
container detection, profile detection. The copies drifted: eb4040242
changed the canonical output and referenced ``PROJECT_ROOT`` inside the fast
function, which doesn't exist yet on the fast path → the Termux fast path
NameError'd on --version and nobody noticed. One implementation, imported
by both the fast path and the module constants, makes that drift
structurally impossible; the parity guard test would have caught eb4040242
the day it landed.

``hermes_cli/config.py``'s ``get_container_exec_info()`` reads the same
``.container-mode`` file; keep the file-format assumptions here and there in
sync (this module deliberately only PROBES existence/typos cheaply and errs
toward the slow path, which then does the authoritative parse).
"""

from __future__ import annotations

import os
import sys

__all__ = [
    "project_root_str",
    "ensure_project_root_on_path",
    "is_termux_env",
    "is_termux_fast_version_argv",
    "is_global_fast_version_argv",
    "is_container_startup_environment",
    "active_profile_may_override_home",
    "container_mode_may_be_active",
    "read_openai_version",
    "read_install_method",
    "print_fast_version_info",
    "try_fast_version",
]


def project_root_str() -> str:
    """Repo root as a str — the single source for main.py's PROJECT_ROOT."""
    return os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))


def ensure_project_root_on_path() -> None:
    """Put the project root at sys.path[0], deduping realpath-equivalents."""
    project_root = project_root_str()
    normalized_root = os.path.normcase(os.path.realpath(project_root))
    sys.path[:] = [
        entry
        for entry in sys.path
        if not entry
        or os.path.normcase(os.path.realpath(entry)) != normalized_root
    ]
    sys.path.insert(0, project_root)


def is_termux_env() -> bool:
    """Tiny Termux check for pre-import startup shortcuts."""
    prefix = os.environ.get("PREFIX", "")
    return bool(
        os.environ.get("TERMUX_VERSION")
        or "com.termux/files/usr" in prefix
        or prefix.startswith("/data/data/com.termux/")
    )


def is_termux_fast_version_argv(argv: list[str]) -> bool:
    return argv in (["--version"], ["-V"], ["version"])


def is_global_fast_version_argv(argv: list[str]) -> bool:
    return argv in (["--version"], ["-V"])


def is_container_startup_environment() -> bool:
    """True when we're already INSIDE a container (fast path is then safe)."""
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return True
    try:
        with open("/proc/1/cgroup", encoding="utf-8") as handle:
            cgroup = handle.read()
    except OSError:
        return False
    return "docker" in cgroup or "podman" in cgroup or "/lxc/" in cgroup


def active_profile_may_override_home(hermes_root: str) -> bool:
    """Cheap probe: does an active non-default profile redirect HERMES_HOME?"""
    active_profile = os.path.join(hermes_root, "active_profile")
    try:
        if os.path.exists(active_profile):
            with open(active_profile, encoding="utf-8") as handle:
                active = handle.read().strip()
            return bool(active and active != "default")
    except (OSError, UnicodeDecodeError):
        pass
    return False


def _resolved_home() -> str:
    hermes_home = os.environ.get("HERMES_HOME", "").strip()
    if hermes_home:
        return hermes_home
    return os.path.join(os.path.expanduser("~"), ".hermes")


def container_mode_may_be_active() -> bool:
    """Conservative probe for NixOS container-mode routing.

    False positives are fine (we fall through to the slow path, whose
    ``get_container_exec_info()`` does the authoritative check and routes
    into the container). False negatives are NOT fine — they'd print the
    host's version instead of the container's. Hence: any profile
    ambiguity → assume container mode may be active.
    """
    if os.environ.get("HERMES_DEV") == "1":
        return False
    if is_container_startup_environment():
        return False

    hermes_home = os.environ.get("HERMES_HOME", "").strip()
    if hermes_home:
        if os.path.exists(os.path.join(hermes_home, ".container-mode")):
            return True
        parent_name = os.path.basename(os.path.dirname(os.path.normpath(hermes_home)))
        return (
            parent_name != "profiles"
            and active_profile_may_override_home(hermes_home)
        )

    default_home = os.path.join(os.path.expanduser("~"), ".hermes")
    if active_profile_may_override_home(default_home):
        return True
    return os.path.exists(os.path.join(default_home, ".container-mode"))


def read_openai_version() -> str | None:
    """Read OpenAI SDK version without importing ``importlib.metadata``."""
    for base in sys.path:
        if not base:
            base = os.getcwd()
        version_file = os.path.join(base, "openai", "_version.py")
        try:
            with open(version_file, encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped.startswith("__version__"):
                        continue
                    _key, _sep, value = stripped.partition("=")
                    value = value.split("#", 1)[0].strip().strip("\"'")
                    return value or None
        except OSError:
            continue
    return None


def read_install_method() -> str | None:
    """Read the installer's ``.install_method`` stamp, if present.

    Only the stamp (step 1 of ``config.detect_install_method``'s resolution
    order) — the managed/git/pip fallbacks need heavier imports and stay on
    the slow path. On the fast path home ambiguity is already excluded:
    ``container_mode_may_be_active()`` bails to the slow path whenever a
    non-default profile might redirect HERMES_HOME.
    """
    stamp = os.path.join(_resolved_home(), ".install_method")
    try:
        with open(stamp, encoding="utf-8") as handle:
            method = handle.read().strip().lower()
        return method or None
    except OSError:
        return None


def print_fast_version_info() -> None:
    from hermes_cli import __release_date__, __version__

    print(f"Hermes Agent v{__version__} ({__release_date__})")
    print(f"Install directory: {project_root_str()}")
    install_method = read_install_method()
    if install_method:
        print(f"Install method: {install_method}")

    print(f"Python: {sys.version.split()[0]}")

    openai_version = read_openai_version()
    print(f"OpenAI SDK: {openai_version}" if openai_version else "OpenAI SDK: Not installed")
    print("Run 'hermes version' for update status.")


def try_fast_version(argv: list[str] | None = None) -> bool:
    """Handle ``hermes --version`` before the heavy import wall.

    Termux keeps its historical contract (also accepts the ``version``
    subcommand + the HERMES_TERMUX_DISABLE_FAST_CLI escape hatch). Everywhere
    else: only ``--version``/``-V`` (the ``version`` subcommand stays on the
    slow path for full output incl. update check), and never when container
    mode may need to route the command into the container.
    """
    if argv is None:
        argv = sys.argv[1:]
    is_termux = is_termux_env()
    if is_termux and os.environ.get("HERMES_TERMUX_DISABLE_FAST_CLI") == "1":
        return False
    if is_termux:
        if not is_termux_fast_version_argv(argv):
            return False
    elif not is_global_fast_version_argv(argv):
        return False
    elif container_mode_may_be_active():
        return False

    print_fast_version_info()
    return True
