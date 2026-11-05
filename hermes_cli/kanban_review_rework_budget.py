"""Opt-in per-board review churn guard using native Kanban config.

Review rejection history comes from durable task_events, not agent memory.
No review limits apply unless the board is explicitly listed in config.yaml.
"""
from __future__ import annotations

import os


def max_review_rejections() -> int | None:
    board = os.environ.get("HERMES_KANBAN_BOARD", "").strip().lower()
    if not board:
        return None

    from hermes_cli.config import load_config_readonly

    cfg = load_config_readonly().get("kanban") or {}
    boards = cfg.get("review_rework_boards") or []
    if isinstance(boards, str):
        boards = boards.split(",")
    if not isinstance(boards, (list, tuple)):
        raise ValueError("kanban.review_rework_boards must be a list of board names")
    if board not in {str(name).strip().lower() for name in boards}:
        return None

    raw = cfg.get("max_review_rejections", 0)
    try:
        limit = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid kanban.max_review_rejections (expected 0..20)") from exc
    if not 0 <= limit <= 20:
        raise ValueError("Invalid kanban.max_review_rejections (expected 0..20)")
    return limit or None
