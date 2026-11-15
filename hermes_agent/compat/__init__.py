"""Old module names keep importing after a module moves."""
from __future__ import annotations

import sys


def install() -> None:
    """Put the non-strict compat finder at the front of ``sys.meta_path``, once per process.

    Stdlib only and idempotent. A finder already installed (the strict one in tests) is kept.
    """
    if any(type(finder).__name__ == "Finder" and type(finder).__module__.startswith("hermes_agent.compat")
           for finder in sys.meta_path):
        return
    from hermes_agent.compat.finder import Finder
    from hermes_agent.compat.moves import MOVES

    sys.meta_path.insert(0, Finder(MOVES))
