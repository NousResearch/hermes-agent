"""Bridge DAG nodes to Hermes delegate_task (thread offload)."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional

from orchestration.honcho_context import honcho_memory_preamble, optional_honcho_env_patch
from orchestration.types import GraphTaskSpec


@contextmanager
def _temporary_environ(updates: Mapping[str, str]) -> Iterator[None]:
    previous: dict[str, Optional[str]] = {}
    try:
        for key, val in updates.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = val
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _sync_delegate(
    parent_agent: Any,
    spec: GraphTaskSpec,
    *,
    orch_id: str,
) -> dict[str, Any]:
    from tools.delegate_tool import delegate_task

    honcho_block = honcho_memory_preamble(orch_id=orch_id, role=f"worker-{spec.task_id}")
    merged_ctx = "\n\n".join(
        x for x in (spec.context, honcho_block) if x and str(x).strip()
    )

    env_updates = dict(spec.profile_env or {})
    env_updates.update(optional_honcho_env_patch())

    with _temporary_environ(env_updates):
        raw = delegate_task(
            goal=spec.goal,
            context=merged_ctx or None,
            toolsets=list(spec.toolsets) if spec.toolsets else None,
            parent_agent=parent_agent,
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "raw": raw}
    results = payload.get("results") if isinstance(payload, dict) else None
    if not results:
        return {"ok": False, "payload": payload}
    first = results[0]
    status = first.get("status")
    ok = status == "completed"
    return {
        "ok": ok,
        "status": status,
        "summary": first.get("summary"),
        "error": first.get("error"),
        "payload": payload,
    }


async def run_delegate_node(
    parent_agent: Any,
    spec: GraphTaskSpec,
    *,
    orch_id: str,
) -> dict[str, Any]:
    """Execute a single DAG node via delegate_task in a worker thread."""

    return await asyncio.to_thread(_sync_delegate, parent_agent, spec, orch_id=orch_id)
