"""Shims to stop the old updater doing work until relaunch."""

import sys
from typing import NoReturn


def stop_for_relaunch() -> NoReturn:
    """Do not return: old callers would fall back to pip or claim completion."""
    print(
        "This updater is running code from before the checkout changed. "
        "Stopping without installing dependencies; relaunch Hermes to continue.",
        file=sys.stderr,
    )
    raise SystemExit(0)
