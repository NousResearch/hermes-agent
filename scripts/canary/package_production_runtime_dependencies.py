#!/usr/bin/env python3
"""Compatibility CLI for the wheel-packaged runtime dependency boundary."""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)
from gateway import production_runtime_dependencies as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation._main())

# Preserve legacy imports while keeping exactly one implementation.  Replacing
# this module object also makes monkeypatching private compatibility helpers
# affect the implementation module, as it did before the move.
sys.modules[__name__] = _implementation
