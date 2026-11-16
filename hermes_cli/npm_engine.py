"""Stop an already-running old updater before its retired npm retry path."""

from typing import NoReturn

from hermes_cli._old_updater import stop_historical_updater


def maybe_repair_npm_engine(*args, **kwargs) -> NoReturn:
    # Returning would let old callers retry or continue their stale update.
    stop_historical_updater("hermes_cli.npm_engine.maybe_repair_npm_engine")
