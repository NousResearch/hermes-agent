#!/usr/bin/env python3
"""One-shot paper literature search → ranked JSON + Feishu IM."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "paper_search_feishu_deliver.py"
raise SystemExit(runpy.run_path(str(_TARGET), run_name="__main__") or 0)
