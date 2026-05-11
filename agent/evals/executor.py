"""Eval executor — thin wrapper that runs an AIAgent against an eval case prompt.

This module isolates the agent invocation so it can be mocked in unit tests
and swapped for different backends later (e.g., a lightweight local model).
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from acp_adapter.session import _clear_task_cwd, _register_task_cwd
from hermes_cli.config import load_config
from hermes_cli.runtime_provider import resolve_runtime_provider

logger = logging.getLogger(__name__)


class EvalTimeoutError(TimeoutError):
    """Raised when an eval agent invocation exceeds its wall-clock budget."""


@contextmanager
def _eval_timeout(seconds: float | int):
    """Best-effort wall-clock timeout for synchronous eval runs.

    SIGALRM is available only in the main thread on POSIX systems. In other
    contexts we skip enforcement rather than breaking worker-thread callers.
    """
    if not seconds or seconds <= 0:
        yield
        return
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGALRM"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise EvalTimeoutError(f"eval timed out after {seconds}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass
class AgentResult:
    """Compact payload from a single agent invocation."""
    response_text: str = ""
    iterations: int = 0
    error: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


def run_agent_for_eval(
    prompt: str,
    workdir: str,
    timeout_seconds: int = 60,
    model: Optional[str] = None,
    max_iterations: int = 30,
) -> AgentResult:
    """Run the Hermes AIAgent against *prompt* with *workdir* as CWD.

    The agent is created in quiet mode with no memory and no context files so
    eval runs stay isolated and reproducible.

    Returns an AgentResult with the response text and metadata.
    """
    prev_cwd = os.getcwd()
    try:
        os.chdir(workdir)

        from run_agent import AIAgent

        config = load_config()
        model_cfg = config.get("model")
        default_model = ""
        config_provider = None
        if isinstance(model_cfg, dict):
            default_model = str(model_cfg.get("default") or "")
            config_provider = model_cfg.get("provider")
        elif isinstance(model_cfg, str) and model_cfg.strip():
            default_model = model_cfg.strip()

        eval_session_id = f"eval-{uuid.uuid4().hex[:12]}"
        eval_task_id = f"eval-task-{uuid.uuid4().hex[:12]}"
        _register_task_cwd(eval_task_id, workdir)

        kwargs = {
            "platform": "eval",
            "quiet_mode": True,
            "skip_context_files": True,
            "skip_memory": True,
            "session_id": eval_session_id,
            "max_iterations": max_iterations,
            "model": model or default_model,
        }
        try:
            runtime = resolve_runtime_provider(requested=config_provider)
            kwargs.update(
                {
                    "provider": runtime.get("provider"),
                    "api_mode": runtime.get("api_mode"),
                    "base_url": runtime.get("base_url"),
                    "api_key": runtime.get("api_key"),
                    "command": runtime.get("command"),
                    "args": list(runtime.get("args") or []),
                }
            )
        except Exception:
            logger.debug("Eval executor falling back to default provider resolution", exc_info=True)

        agent = AIAgent(**kwargs)

        with _eval_timeout(timeout_seconds):
            result = agent.run_conversation(
                user_message=prompt,
                system_message=(
                    "You are being evaluated. Work in the current directory. "
                    "Complete the task precisely. Do not ask clarifying questions."
                ),
                task_id=eval_task_id,
            )

        response_text = ""
        if isinstance(result, dict):
            response_text = (
                result.get("final_response")
                or result.get("response")
                or result.get("content")
                or ""
            )
        elif isinstance(result, str):
            response_text = result

        return AgentResult(
            response_text=str(response_text)[:2000],
            iterations=getattr(agent, "iteration_count", 0),
            raw={
                "model": getattr(agent, "model", ""),
                "response_preview": str(response_text)[:500],
            },
        )
    except Exception as exc:
        logger.warning("Eval agent execution failed: %s", exc)
        return AgentResult(
            error=str(exc),
            raw={"exception_type": type(exc).__name__},
        )
    finally:
        try:
            if 'eval_task_id' in locals():
                _clear_task_cwd(eval_task_id)
        except Exception:
            logger.debug("Failed to clear eval task cwd", exc_info=True)
        os.chdir(prev_cwd)
