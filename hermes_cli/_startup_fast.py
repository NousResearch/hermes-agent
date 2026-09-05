"""Pre-import startup fast paths — THE canonical lightweight helpers.

Imported by ``hermes_cli/main.py`` BEFORE its heavy import wall (config, argparse tree, logging,
providers). Everything here must stay **stdlib-only and cheap** (os/sys file probes; no yaml, no
hermes_cli.config, no argparse). Exists so version-printing stops being reimplemented as
``*_fast()`` copies in main.py that duplicate project-root / container / profile detection.
"""

from __future__ import annotations

import os
import sys

__all__ = [
    "project_root_str", "ensure_project_root_on_path", "is_termux_env",
    "is_termux_fast_version_argv", "is_global_fast_version_argv",
    "is_container_startup_environment", "active_profile_may_override_home",
    "container_mode_may_be_active", "read_openai_version", "read_install_method",
    "print_fast_version_info", "try_fast_version",
]


def _read_text(path: str) -> str | None:
    """Read a small text file, or None when it is missing/unreadable."""
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError):
        return None


def project_root_str() -> str:
    """Repo root as a str — the single source for main.py's PROJECT_ROOT."""
    return os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))


def ensure_project_root_on_path() -> None:
    """Put the project root at sys.path[0], deduping realpath-equivalents."""
    project_root = project_root_str()
    normalized_root = os.path.normcase(os.path.realpath(project_root))
    sys.path[:] = [entry for entry in sys.path
                   if not entry or os.path.normcase(os.path.realpath(entry)) != normalized_root]
    sys.path.insert(0, project_root)


def is_termux_env() -> bool:
    """Tiny Termux check for pre-import startup shortcuts."""
    prefix = os.environ.get("PREFIX", "")
    return bool(os.environ.get("TERMUX_VERSION") or "com.termux/files/usr" in prefix
                or prefix.startswith("/data/data/com.termux/"))


def is_termux_fast_version_argv(argv: list[str]) -> bool:
    return argv in (["--version"], ["-V"])


is_global_fast_version_argv = is_termux_fast_version_argv


def is_container_startup_environment() -> bool:
    """True when we're already INSIDE a container (fast path is then safe)."""
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return True
    cgroup = _read_text("/proc/1/cgroup") or ""
    return "docker" in cgroup or "podman" in cgroup or "/lxc/" in cgroup


def active_profile_may_override_home(hermes_root: str) -> bool:
    """Cheap probe: does an active non-default profile redirect HERMES_HOME?"""
    active = (_read_text(os.path.join(hermes_root, "active_profile")) or "").strip()
    return bool(active and active != "default")


def _default_home() -> str:
    return os.path.join(os.path.expanduser("~"), ".hermes")


def _resolved_home() -> str:
    return os.environ.get("HERMES_HOME", "").strip() or _default_home()


def container_mode_may_be_active() -> bool:
    """Conservative probe for NixOS container-mode routing.

    False positives are fine (the slow path does the authoritative check). False negatives are NOT —
    they'd print the host's version instead of the container's — so any profile ambiguity means "may
    be active".
    """
    if os.environ.get("HERMES_DEV") == "1" or is_container_startup_environment():
        return False
    hermes_home = os.environ.get("HERMES_HOME", "").strip()
    if hermes_home:
        if os.path.exists(os.path.join(hermes_home, ".container-mode")):
            return True
        parent_name = os.path.basename(os.path.dirname(os.path.normpath(hermes_home)))
        return parent_name != "profiles" and active_profile_may_override_home(hermes_home)
    default_home = _default_home()
    return active_profile_may_override_home(default_home) or os.path.exists(
        os.path.join(default_home, ".container-mode"))


def read_openai_version() -> str | None:
    """Read OpenAI SDK version without importing ``importlib.metadata``."""
    for base in sys.path:
        version_file = os.path.join(base or os.getcwd(), "openai", "_version.py")
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
    """The installer's ``.install_method`` stamp, if present.

    Only the stamp (step 1 of ``config.detect_install_method``'s resolution order) — the
    managed/git/pip fallbacks need heavier imports and stay on the slow path.
    """
    method = _read_text(os.path.join(_resolved_home(), ".install_method"))
    return (method or "").strip().lower() or None


def print_fast_version_info(*, check_updates: bool = True) -> None:
    """THE canonical ``hermes --version`` output (also used by /version).

    Every lazy block degrades gracefully — a broken/heavy import can never take the basic version
    output down.
    """
    # Registry-owned banner label (includes "· upstream <sha>" for git installs); banner.py keeps
    # rich/prompt_toolkit lazy, so this import is light.
    try:
        from hermes_cli.banner import format_banner_version_label

        print(format_banner_version_label())
    except Exception:
        from hermes_cli import __release_date__, __version__

        print(f"Hermes Agent v{__version__} ({__release_date__})")
    print(f"Install directory: {project_root_str()}")
    # Authoritative resolver first (code-scoped stamp → managed → nix → git → pip; also self-heals
    # poisoned shared-home 'docker' stamps); cheap stdlib stamp probe only if it fails.
    try:
        from pathlib import Path

        from hermes_cli.config import detect_install_method

        install_method = detect_install_method(Path(project_root_str()))
    except Exception:
        install_method = read_install_method()
    if install_method:
        print(f"Install method: {install_method}")
    print(f"Python: {sys.version.split()[0]}")
    openai_version = read_openai_version()
    print(f"OpenAI SDK: {openai_version}" if openai_version else "OpenAI SDK: Not installed")
    if not check_updates:
        return
    # Synchronous update status — bounded by check_for_updates' own subprocess/network timeouts
    # and its 6-hour cache; any failure prints nothing.
    try:
        from hermes_cli.banner import UPDATE_AVAILABLE_NO_COUNT, check_for_updates
        from hermes_cli.config import recommended_update_command

        behind = check_for_updates()
        if behind == UPDATE_AVAILABLE_NO_COUNT:
            print(f"Update available — run '{recommended_update_command()}'")
        elif behind and behind > 0:
            commits_word = "commit" if behind == 1 else "commits"
            print(f"Update available: {behind} {commits_word} behind — run '{recommended_update_command()}'")
        elif behind == 0:
            print("Up to date")
    except Exception:
        pass


def try_fast_version(argv: list[str] | None = None) -> bool:
    """Handle ``hermes --version`` before the heavy import wall.

    Only ``--version``/``-V`` (``--version`` carries the full output incl. update status), and never
    when container mode may need to route the command into the container. Termux keeps the
    HERMES_TERMUX_DISABLE_FAST_CLI escape hatch.
    """
    if argv is None:
        argv = sys.argv[1:]
    is_termux = is_termux_env()
    if is_termux and os.environ.get("HERMES_TERMUX_DISABLE_FAST_CLI") == "1":
        return False
    if is_termux:
        if not is_termux_fast_version_argv(argv):
            return False
    elif not is_global_fast_version_argv(argv) or container_mode_may_be_active():
        return False
    print_fast_version_info()
    return True
