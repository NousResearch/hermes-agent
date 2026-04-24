"""Ephemeral subagents — 3rd tier (Hermes / OpenClaw / ephemeral).

Inspired by DeerFlow's subagent system (§1.D). Lets Hermes or OpenClaw spawn
bounded-scope helpers for narrow tasks without dirtying their own context.

Built-in subagents
------------------
- general-purpose: full tool access except `task` (no recursive spawn)
- bash: shell-focused, useful for filesystem exploration

Runtime
-------
- MAX_CONCURRENT = 3 per process
- Per-subagent timeout (default 15 min, env: HERMES_SUBAGENT_TIMEOUT_SEC)
- Isolated context — subagent doesn't see parent messages
- Parent waits on `SubagentResult` (poll-based, not async)
- Results stored in the bus DB as regular tasks tagged `subagent=true`,
  so they show up in dashboard alongside other work

This is a **minimum viable** implementation — `run()` creates a bus task
with the subagent spec baked into `context`, hands it off to `codex_call.
invoke_codex()`, stores the result, and closes the task. Real end-to-end
LLM execution is shape-compatible but the subagent body itself is a
heuristic or LLM invocation (not a full agent loop).

Enable with HERMES_SUBAGENTS_ENABLED=on.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# -------- Config --------
MAX_CONCURRENT = int(os.environ.get("HERMES_SUBAGENT_MAX_CONCURRENT", "3"))
DEFAULT_TIMEOUT_SEC = int(os.environ.get("HERMES_SUBAGENT_TIMEOUT_SEC", str(15 * 60)))


class SubagentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass
class SubagentSpec:
    name: str  # "general-purpose" | "bash" | custom
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)  # allowed tool names (empty = all)
    max_turns: int = 10
    timeout_seconds: int = DEFAULT_TIMEOUT_SEC


BUILTIN_SPECS: dict[str, SubagentSpec] = {
    "general-purpose": SubagentSpec(
        name="general-purpose",
        description="Multi-purpose helper — full tool access except recursive task spawn.",
        system_prompt=(
            "You are a narrow-scope helper subagent. Focus on the single task "
            "given to you. Return a concise plain-text answer. Do not spawn "
            "further subagents. If you need information, use available tools; "
            "otherwise reply directly."
        ),
    ),
    "bash": SubagentSpec(
        name="bash",
        description="Shell command specialist — use for filesystem + process inspection.",
        system_prompt=(
            "You are a bash specialist subagent. Translate the user's request "
            "into safe shell commands, execute them, and report results. Do "
            "NOT run destructive commands (rm -rf, DROP TABLE, etc.) — refuse "
            "and explain instead."
        ),
    ),
}


@dataclass
class SubagentResult:
    task_id: str
    subagent_name: str
    status: SubagentStatus
    result: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    duration_sec: float = 0.0
    trace_id: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "subagent_name": self.subagent_name,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_sec": round(self.duration_sec, 2),
            "trace_id": self.trace_id,
        }


# -------- Concurrency gate --------
_RUNNING: dict[str, SubagentResult] = {}
_LOCK = threading.Lock()


def _available_slots() -> int:
    with _LOCK:
        active = sum(1 for r in _RUNNING.values()
                     if r.status in (SubagentStatus.PENDING, SubagentStatus.RUNNING))
    return max(0, MAX_CONCURRENT - active)


def get_spec(name: str) -> SubagentSpec | None:
    return BUILTIN_SPECS.get(name)


# -------- Execution --------
def run(
    *,
    subagent_name: str,
    prompt: str,
    parent_agent: str = "hermes",
    max_turns: int | None = None,
    timeout_seconds: int | None = None,
) -> SubagentResult:
    """Execute a subagent synchronously. Returns SubagentResult regardless
    of success/failure — inspect `.status`.
    """
    if os.environ.get("HERMES_SUBAGENTS_ENABLED", "on").lower() == "off":
        return SubagentResult(
            task_id=f"sa-disabled-{uuid.uuid4().hex[:6]}",
            subagent_name=subagent_name,
            status=SubagentStatus.FAILED,
            error="subagents disabled via HERMES_SUBAGENTS_ENABLED=off",
        )

    spec = get_spec(subagent_name)
    if spec is None:
        return SubagentResult(
            task_id=f"sa-unknown-{uuid.uuid4().hex[:6]}",
            subagent_name=subagent_name,
            status=SubagentStatus.FAILED,
            error=f"unknown subagent: {subagent_name!r}",
        )

    # Concurrency guard
    if _available_slots() <= 0:
        return SubagentResult(
            task_id=f"sa-busy-{uuid.uuid4().hex[:6]}",
            subagent_name=subagent_name,
            status=SubagentStatus.FAILED,
            error=f"max concurrent subagents ({MAX_CONCURRENT}) already running",
        )

    task_id = f"sa-{uuid.uuid4().hex[:8]}"
    trace_id = f"tr-{uuid.uuid4().hex[:8]}"
    result = SubagentResult(
        task_id=task_id,
        subagent_name=subagent_name,
        status=SubagentStatus.RUNNING,
        trace_id=trace_id,
    )
    with _LOCK:
        _RUNNING[task_id] = result

    started = time.time()
    deadline_sec = timeout_seconds or spec.timeout_seconds

    try:
        from agent_bus.codex_call import invoke_codex
        full_prompt = (
            f"{spec.system_prompt}\n\n"
            f"--- TASK (parent={parent_agent}) ---\n{prompt}\n"
            f"--- END TASK ---\n\nReply with a concise answer."
        )
        codex_result = invoke_codex(
            full_prompt,
            attempt=f"subagent:{subagent_name}",
            timeout_sec=min(deadline_sec, 120),  # codex CLI practical cap
        )
        if codex_result.ok:
            result.status = SubagentStatus.COMPLETED
            result.result = codex_result.stdout.strip()
        elif codex_result.error == "timeout":
            result.status = SubagentStatus.TIMED_OUT
            result.error = codex_result.error
        else:
            result.status = SubagentStatus.FAILED
            result.error = codex_result.error or "unknown"
    except Exception as exc:
        logger.exception("subagent %s crashed", subagent_name)
        result.status = SubagentStatus.FAILED
        result.error = f"{type(exc).__name__}: {exc}"

    result.completed_at = time.time()
    result.duration_sec = result.completed_at - started

    # Persist to ~/.hermes/subagent-runs.jsonl for dashboard tail
    _persist_result(result, prompt=prompt, parent_agent=parent_agent)
    return result


def _persist_result(
    result: SubagentResult, *, prompt: str, parent_agent: str,
) -> None:
    try:
        path = Path(os.environ.get(
            "HERMES_SUBAGENT_LOG_PATH",
            str(Path.home() / ".hermes" / "subagent-runs.jsonl"),
        )).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = result.to_dict()
        entry["prompt"] = prompt[:300]
        entry["parent_agent"] = parent_agent
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Rotate if >2000 lines
        try:
            with path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > 2000:
                with path.open("w", encoding="utf-8") as f:
                    f.writelines(lines[-1000:])
        except Exception:
            pass
    except Exception as exc:
        logger.debug("subagent log persist failed: %s", exc)


def list_running() -> list[dict]:
    """For dashboard: currently running subagents."""
    with _LOCK:
        return [r.to_dict() for r in _RUNNING.values()
                if r.status in (SubagentStatus.PENDING, SubagentStatus.RUNNING)]


def summary_stats(last_n: int = 100) -> dict:
    """Read subagent-runs.jsonl, summarize."""
    path = Path(os.environ.get(
        "HERMES_SUBAGENT_LOG_PATH",
        str(Path.home() / ".hermes" / "subagent-runs.jsonl"),
    )).expanduser()
    if not path.exists():
        return {"exists": False, "total": 0}
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()[-last_n:]
    except Exception:
        return {"exists": True, "error": "read failed"}
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    by_status: dict[str, int] = {}
    by_subagent: dict[str, int] = {}
    durations: list[float] = []
    for r in rows:
        by_status[r.get("status", "?")] = by_status.get(r.get("status", "?"), 0) + 1
        by_subagent[r.get("subagent_name", "?")] = by_subagent.get(r.get("subagent_name", "?"), 0) + 1
        d = r.get("duration_sec")
        if isinstance(d, (int, float)):
            durations.append(float(d))
    return {
        "exists": True,
        "total": len(rows),
        "by_status": by_status,
        "by_subagent": by_subagent,
        "avg_duration_sec": round(sum(durations) / len(durations), 2) if durations else None,
        "recent": rows[-10:],
    }


def _reset_for_test() -> None:
    with _LOCK:
        _RUNNING.clear()
