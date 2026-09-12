"""Shims to stop the old updater doing work until relaunch.

An updater already in memory imports these names after replacing its checkout.
They must not install anything, delegate to PM, or return a falsy value that
would send that old process down its pip fallback. New code must not use them.
"""

from pathlib import Path
from typing import NoReturn

from hermes_cli._old_updater import stop_for_relaunch


def _reload_hermes_constants() -> None:
    # Shim to suppress old updater work until relaunch. Never re-execute live globals.
    return None


def ensure_uv(*args, **kwargs) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch. Older releases
    # expect a tuple, newer ones a path. Exit before either can consume it.
    stop_for_relaunch()


def update_managed_uv(*args, **kwargs) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch.
    stop_for_relaunch()


def resolve_uv(*args, **kwargs) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not enable pip.
    stop_for_relaunch()


def managed_python_env(
    project_root: Path | None = None,
    *,
    install_dir: Path | None = None,
    base_env: dict[str, str] | None = None,
    **kwargs,
) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not prepare a child.
    stop_for_relaunch()


def rebuild_venv(
    uv_bin: str, venv_dir: Path, python_version: str = "3.11", **kwargs
) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not claim a rebuild.
    stop_for_relaunch()
