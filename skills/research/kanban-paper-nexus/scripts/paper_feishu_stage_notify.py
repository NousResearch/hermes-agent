#!/usr/bin/env python3
"""Shim: paper-nexus → shared kanban-feishu-live notify."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_LIVE = Path(__file__).resolve().parents[3] / "devops" / "kanban-feishu-live" / "scripts"
if "--board" not in sys.argv:
    sys.argv = [sys.argv[0], "--board", "paper-nexus", *sys.argv[1:]]
runpy.run_path(str(_LIVE / "kanban_feishu_stage_notify.py"), run_name="__main__")
