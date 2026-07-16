"""One-session external ACP executor for Kanban tasks.

This is a task execution path, deliberately separate from model providers and
API modes. The dispatcher has already claimed the task; this process owns one
native ACP session and writes exactly one terminal lifecycle transition.
"""
from __future__ import annotations

import os
import shlex
import threading
from typing import Optional

from agent.copilot_acp_client import CopilotACPClient
from hermes_cli import kanban_db as kb

_DEFAULT_COMMANDS = {
    "claude-code": ("npx", ["--yes", "@agentclientprotocol/claude-agent-acp"]),
    "codex": ("codex-acp", ["--stdio"]),
}
HEARTBEAT_INTERVAL_SECONDS = 60.0


def command_for(executor: str) -> tuple[str, list[str]]:
    """Resolve the ACP adapter command for a supported external harness."""
    if executor not in _DEFAULT_COMMANDS:
        raise ValueError(f"unsupported ACP executor: {executor}")
    prefix = "HERMES_CLAUDE_CODE_ACP" if executor == "claude-code" else "HERMES_CODEX_ACP"
    command = os.getenv(prefix + "_COMMAND", "").strip() or _DEFAULT_COMMANDS[executor][0]
    raw_args = os.getenv(prefix + "_ARGS", "").strip()
    return command, shlex.split(raw_args) if raw_args else list(_DEFAULT_COMMANDS[executor][1])


def _heartbeat(task_id: str, board: Optional[str], claim_lock: Optional[str]) -> None:
    """Best-effort lease extension while an ACP adapter works on the task."""
    if not claim_lock:
        return
    try:
        with kb.connect_closing(board=board) as conn:
            kb.heartbeat_claim(conn, task_id, claimer=claim_lock)
    except Exception:
        # The adapter may still produce a useful terminal outcome; do not kill
        # it merely because one best-effort heartbeat could not be persisted.
        pass


def _heartbeat_loop(
    stop: threading.Event,
    task_id: str,
    board: Optional[str],
    claim_lock: Optional[str],
) -> None:
    while not stop.wait(HEARTBEAT_INTERVAL_SECONDS):
        _heartbeat(task_id, board, claim_lock)


def run_task(*, executor: str, task_id: str, workspace: str, board: Optional[str] = None) -> str:
    """Run one ACP session and atomically complete or block its claimed task."""
    with kb.connect_closing(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if not task:
            raise ValueError(f"unknown task {task_id}")
        if task.current_run_id is None:
            raise RuntimeError(f"task {task_id} has no active Kanban run")
        context = kb.build_worker_context(conn, task_id)
        run_id = task.current_run_id
        claim_lock = task.claim_lock

    command, args = command_for(executor)
    prompt = (
        "You are the sole native external coding-harness session for this already-scoped "
        "Kanban task. Work only in the supplied cwd; do not orchestrate child tasks. "
        "Follow project rules, verify the requested work, and return a concise factual "
        "handoff with tests run.\n\n"
        + context
    )
    stop = threading.Event()
    _heartbeat(task_id, board, claim_lock)
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(stop, task_id, board, claim_lock),
        name=f"acp-task-heartbeat-{task_id}",
        daemon=True,
    )
    heartbeat.start()
    try:
        text, _reasoning = CopilotACPClient(
            acp_command=command,
            acp_args=args,
            acp_cwd=workspace,
        )._run_prompt(prompt, timeout_seconds=900.0)
    except Exception as exc:
        with kb.connect_closing(board=board) as conn:
            kb.block_task(
                conn,
                task_id,
                reason=f"External {executor} ACP session failed: {exc}",
                kind="capability",
                expected_run_id=run_id,
            )
        raise
    finally:
        stop.set()
        heartbeat.join(timeout=1.0)

    summary = text.strip() or f"External {executor} ACP session completed."
    with kb.connect_closing(board=board) as conn:
        completed = kb.complete_task(
            conn,
            task_id,
            summary=summary,
            metadata={"executor": executor, "acp_command": command},
            expected_run_id=run_id,
        )
    if not completed:
        raise RuntimeError("task was reclaimed or reached a terminal state")
    return text


def main() -> int:
    run_task(
        executor=os.environ["HERMES_KANBAN_EXECUTOR"],
        task_id=os.environ["HERMES_KANBAN_TASK"],
        workspace=os.environ["HERMES_KANBAN_WORKSPACE"],
        board=os.getenv("HERMES_KANBAN_BOARD"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
