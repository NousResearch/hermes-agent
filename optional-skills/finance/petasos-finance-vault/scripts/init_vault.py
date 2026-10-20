#!/usr/bin/env python3
"""Petasos init — scaffold a new finance vault (v2).

Thin CLI wrapper over petasos.scaffold (same usage/flags as v1.1.0):
  python3 init_vault.py /path/to/vault
  PETASOS_VAULT=/path python3 init_vault.py

Idempotent, never overwrites: seeds the ledger tree, starter balances.json
(one synthetic placeholder per account kind), rules.md from the skill
template, and a vault-side banks.yaml that SHADOWS the bundled examples.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from petasos import scaffold

if __name__ == '__main__':
    sys.exit(scaffold.main())
