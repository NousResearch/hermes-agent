"""Dispatch spawn callable adaptation (legacy two-argument consumers)."""
from typing import Optional


def call_spawn_fn(spawn_fn, task, workspace: str, board: Optional[str], *, conn=None) -> Optional[int]:
    """Pass board only when supported; retain legacy spawn callable signatures."""
    import inspect
    from hermes_cli.kanban_lane_coordinator import resolve_settings, spawn_reserved, current_settings
    try:
        has_board = "board" in inspect.signature(spawn_fn).parameters
    except (TypeError, ValueError):
        has_board = False

    def invoke(task, workspace, board):
        return spawn_fn(task, workspace, board=board) if has_board else spawn_fn(task, workspace)

    settings = resolve_settings(current_settings.get())
    if not settings.enabled:
        return invoke(task, workspace, board)
    from hermes_cli.kanban_lane_coordinator import LaneDeferred
    if conn is None:
        raise LaneDeferred("claimed board connection required")
    return spawn_reserved(conn, task, workspace, board, invoke, settings)
