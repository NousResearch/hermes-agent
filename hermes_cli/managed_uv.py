"""Shims to stop the old updater doing work until relaunch.

An updater already in memory imports these names after replacing its checkout.
They must not install anything, delegate to PM, or return a falsy value that
would send that old process down its pip fallback. New code must not use them.
"""

from pathlib import Path
from typing import NoReturn

from hermes_cli._old_updater import stop_historical_updater


def _reload_hermes_constants() -> NoReturn:
    # Shim to suppress old updater work until relaunch. Callers dereference the
    # result, so None crashes. Stop without re-executing live globals.
    stop_historical_updater("hermes_cli.managed_uv._reload_hermes_constants")


def ensure_uv(*args, **kwargs) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch. Older releases
    # expect a tuple, newer ones a path. Exit before either can consume it.
    stop_historical_updater("hermes_cli.managed_uv.ensure_uv", ImportError)


def update_managed_uv(*args, **kwargs) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch.
    stop_historical_updater("hermes_cli.managed_uv.update_managed_uv", ImportError)


def resolve_uv(*args, **kwargs) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not enable pip.
    stop_historical_updater("hermes_cli.managed_uv.resolve_uv", ImportError)


def managed_python_env(
    project_root: Path | None = None,
    *,
    install_dir: Path | None = None,
    base_env: dict[str, str] | None = None,
    **kwargs,
) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not prepare a child.
    stop_historical_updater("hermes_cli.managed_uv.managed_python_env", ImportError)


def rebuild_venv(
    uv_bin: str, venv_dir: Path, python_version: str = "3.11", **kwargs
) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not claim a rebuild.
    stop_historical_updater("hermes_cli.managed_uv.rebuild_venv", ImportError)
