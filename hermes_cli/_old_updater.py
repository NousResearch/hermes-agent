"""Shims to stop the old updater doing work until relaunch."""

import sys
from typing import NoReturn


def stop_for_relaunch() -> NoReturn:
    """Do not return: old callers would fall back to pip or claim completion."""
    print(
        "You're updating from an older version of Hermes Agent. "
        "To complete this update, run `hermes` again.",
        file=sys.stderr,
    )
    raise SystemExit(0)
