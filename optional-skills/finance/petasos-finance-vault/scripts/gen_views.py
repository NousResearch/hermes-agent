#!/usr/bin/env python3
"""Petasos gen_views — regenerate views + run ALL verification gates (v2).

Thin CLI wrapper over petasos.views (same usage/flags as v1.1.0):
  PETASOS_VAULT=/path python3 gen_views.py
  (OBSIDIAN_VAULT_PATH honored as fallback; PETASOS_LEDGER_DIR overrides the
   default '2-Areas/Finance/_Ledger' location inside the vault.)

Exit 0 only when every gate passes; the pipeline treats nonzero as fatal.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from petasos.knob import load_knobs        # Lane A owns knob resolution (§5)
except ImportError:                            # INTEGRATION SEAM (M3): lane A not landed
    from petasos.laneb_knobs import load_knobs

from petasos import views

if __name__ == '__main__':
    sys.exit(views.main(load_knobs))
