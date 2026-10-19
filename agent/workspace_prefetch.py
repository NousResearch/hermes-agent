"""Take the session-start workspace snapshot while the agent builds, off the first prompt's path."""

import logging
import time
from concurrent.futures import Future

from agent.coding_context import build_coding_workspace_block, resolve_runtime_mode
from agent.runtime_cwd import resolve_context_cwd
from tools.daemon_pool import DaemonThreadPoolExecutor

logger = logging.getLogger(__name__)
# Daemon workers: a slow probe never holds interpreter exit; submit() keeps the caller's profile scope.
_executor = DaemonThreadPoolExecutor(max_workers=2, thread_name_prefix="hermes-workspace-prefetch")
# Turn 1 replays the prefetch only when it was waiting on the build: Desktop creates its session on send, and a
# TUI prompt can arrive during the launch build. A session opened now and typed into later probes at turn 1, as
# before, so its snapshot is never older than the first prompt by more than this.
WAITING_TURN_WINDOW_S = 5.0


def start_workspace_snapshot_prefetch(*, platform, config=None) -> tuple[str, Future] | None:
    """Resolve posture under the bound session/profile scope before starting any git work."""
    from agent.system_prompt import _workspace_pin_key

    key = _workspace_pin_key()
    mode = resolve_runtime_mode(platform=platform, cwd=resolve_context_cwd(), config=config)
    if not mode.is_coding:
        return None
    return key, _executor.submit(build_coding_workspace_block, mode.cwd)


def attach_workspace_snapshot_prefetch(agent, prefetch: tuple[str, Future]) -> None:
    """Hand the prefetch to the built agent, stamped with the end of the build."""
    agent._workspace_snapshot_prefetch = (*prefetch, time.monotonic())


def take_prefetched_workspace_block(agent, cwd_key) -> str | None:
    """Consume once; a moved workspace, a turn that was not waiting on the build, or a failed probe falls back to
    the normal builder."""
    prefetch = vars(agent).pop("_workspace_snapshot_prefetch", None)
    if prefetch is None:
        return None
    key, future, built_at = prefetch
    if key != cwd_key or time.monotonic() - built_at > WAITING_TURN_WINDOW_S:
        future.cancel()  # a no-op once the probe has run
        return None
    # cancel() only succeeds while the probe is still queued behind other sessions' probes;
    # probing inline then is never slower than waiting for the pool.
    if future.cancel():
        return None
    try:
        return future.result()
    except Exception:  # health: allow BLE001 -- speculative probe failure must fall back to turn-1 probing
        logger.debug("Workspace snapshot prefetch failed", exc_info=True)
        return None
