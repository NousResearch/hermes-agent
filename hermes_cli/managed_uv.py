"""Shims to stop the old updater doing work until relaunch.

An updater already in memory imports these names after replacing its checkout.
They must not install anything, delegate to PM, or return a falsy value that
would send that old process down its pip fallback. New code must not use them.

Since #124881 the handoff is frame-scoped (the tools.lazy_deps.install_specs
pattern, #127672): only a genuinely historical updater entrypoint -- one
without the current updater's ``_hermes_current_updater_frame`` sentinel local
-- transfers control to the takeover child. A live process on the current tree
that reaches one of these retired names is refused with a catchable
``ImportError`` (or gets an inert value) instead of exiting into an
update-rebuild loop from healthy serve/dashboard startup.
"""

from pathlib import Path
from typing import NoReturn

from hermes_cli._old_updater import in_historical_update, stop_for_relaunch


def _reload_hermes_constants():
    # Inside a real updater: never re-execute the swapped constants file in the
    # old process. A live tree's import is already current, so hand back the
    # imported module without reloading; callers only dereference attributes.
    if in_historical_update():
        stop_for_relaunch()
    import hermes_constants

    return hermes_constants


def ensure_uv(*args, **kwargs) -> NoReturn:
    # Retired. Inside a real updater, hand off before either historical return
    # contract (tuple-era / path-era) can be consumed. Elsewhere, refuse:
    # returning a falsy value would send a caller down its pip fallback.
    if in_historical_update():
        stop_for_relaunch()
    raise ImportError(
        "hermes_cli.managed_uv.ensure_uv is retired; runtime dependency "
        "resolution is unavailable."
    )


def update_managed_uv(*args, **kwargs) -> NoReturn:
    # Hand off inside a real updater; refuse live callers (see ensure_uv).
    if in_historical_update():
        stop_for_relaunch()
    raise ImportError(
        "hermes_cli.managed_uv.update_managed_uv is retired; uv updates "
        "belong to PM's updater."
    )


def resolve_uv(*args, **kwargs) -> NoReturn:
    # Hand off inside a real updater, not enable pip; refuse live callers.
    if in_historical_update():
        stop_for_relaunch()
    raise ImportError(
        "hermes_cli.managed_uv.resolve_uv is retired; runtime uv resolution "
        "is unavailable."
    )


def managed_python_env(
    project_root: Path | None = None,
    *,
    install_dir: Path | None = None,
    base_env: dict[str, str] | None = None,
    **kwargs,
) -> NoReturn:
    # Hand off inside a real updater, not prepare a child; refuse live callers.
    if in_historical_update():
        stop_for_relaunch()
    raise ImportError(
        "hermes_cli.managed_uv.managed_python_env is retired; environment "
        "preparation belongs to PM."
    )


def rebuild_venv(
    uv_bin: str, venv_dir: Path, python_version: str = "3.11", **kwargs
) -> NoReturn:
    # Hand off inside a real updater, not claim a rebuild; refuse live callers.
    if in_historical_update():
        stop_for_relaunch()
    raise ImportError(
        "hermes_cli.managed_uv.rebuild_venv is retired; venv rebuilds belong "
        "to PM's updater."
    )
