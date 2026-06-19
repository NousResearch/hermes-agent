#!/usr/bin/env python3
"""Run the investment-assistant symbol-store CLI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.investment_assistant.symbol_store_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
