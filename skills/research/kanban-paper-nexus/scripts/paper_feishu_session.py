#!/usr/bin/env python3
"""Shim: re-export shared session helpers for paper-nexus."""
from __future__ import annotations

import sys
from pathlib import Path

_LIVE = Path(__file__).resolve().parents[3] / "devops" / "kanban-feishu-live" / "scripts"
if str(_LIVE) not in sys.path:
    sys.path.insert(0, str(_LIVE))

from kanban_feishu_boards import get_board_config  # noqa: E402
from kanban_feishu_session import (  # noqa: E402
    load_session,
    new_session,
    save_session,
    session_path,
)

BOARD = "paper-nexus"
_cfg = get_board_config(BOARD)
STAGES = _cfg["stages"]
