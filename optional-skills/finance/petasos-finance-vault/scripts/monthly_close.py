#!/usr/bin/env python3
"""Petasos — monthly close: freeze month + snapshot into 4-Archives (v2).

Thin CLI wrapper over petasos.close (same usage/flags as v1.1.0):
  monthly_close.py [YYYY-MM]  (default: previous month)
Idempotent: month already in closed_months.json → skip freeze, refresh
snapshot only. Exit nonzero on drift or unmet preconditions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from petasos.knob import load_knobs        # Lane A owns knob resolution (§5)
except ImportError:                            # INTEGRATION SEAM (M3): lane A not landed
    from petasos.laneb_knobs import load_knobs

from petasos import close

if __name__ == '__main__':
    sys.exit(close.main(load=load_knobs))
