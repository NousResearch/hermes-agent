"""Real Hermes runtime — bridges into ``run_agent.AIAgent``.

Falls back gracefully if ``run_agent`` cannot be imported (e.g. on a stripped
test profile): the runtime then refuses to run tasks but the rest of the
office still functions.

The real agent runs on a worker thread (via :func:`asyncio.to_thread`) because
``AIAgent.run_conversation`` is fully synchronous.  Hermes's existing
``tool_start_callback`` / ``tool_complete_callback`` / ``thinking_callback``
hooks are wired through to our :class:`EventBus`-shaped ``on_event``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from ..models import Activity, ActivityEvent, Employee, Task
from . import EventCallback, Runtime, TaskResult

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class HermesRuntime(Runtime):
    name = "hermes"

    def __init__(self) -> None:
        try:
            from run_agent import AIAgent  # noqa: F401
            self._available = True
        except Exception as exc:  # pragma: no cover - import-time guard
            logger.warning("HermesRuntime unavailable: %s", exc)
            self._available = False

    async def run_task(
        self,
        employee: Employee,
        task: Task,
        on_event: EventCallback,
    ) -> TaskResult:
        async def emit(kind: str, text: str, **meta) -> None:
            await on_event(
                ActivityEvent(
                    ts=_now(),
                    employee_id=employee.id,
                    department_id=employee.department_id,
                    kind=kind,                # type: ignore[arg-type]
                    text=text,
                    meta=meta,
                )
            )

        if not self._available:
            await emit(
                "error",
                "Hermes runtime not available (run_agent import failed). "
                "Run `pip install -e .[office]` and retry.",
            )
            return TaskResult(status="failed", error="run_agent import failed")

        await emit("state_change", f"{employee.name} starts working", to=Activity.WORKING)

        loop = asyncio.get_running_loop()

        # Thread-safe wrappers for Hermes's sync callbacks → our async emit.
        def _on_tool_start(tool_name: str, args: dict | None = None, *_, **__) -> None:
            asyncio.run_coroutine_threadsafe(
                emit("tool_call", f"calls {tool_name}(...)", tool=tool_name),
                loop,
            )

        def _on_tool_complete(tool_name: str, result, *_, **__) -> None:
            text = f"{tool_name} returned"
            asyncio.run_coroutine_threadsafe(
                emit("tool_result", text, tool=tool_name),
                loop,
            )

        def _on_thinking(text: str, *_, **__) -> None:
            if not text:
                return
            snippet = text if len(text) <= 280 else text[:277] + "…"
            asyncio.run_coroutine_threadsafe(
                emit("assistant", snippet),
                loop,
            )

        try:
            from run_agent import AIAgent
        except Exception as exc:  # pragma: no cover
            await emit("error", f"run_agent import failed: {exc}")
            return TaskResult(status="failed", error=str(exc))

        def _build_and_run() -> dict:
            sys_prompt = employee.system_prompt or None
            agent = AIAgent(
                base_url=employee.base_url,
                provider=employee.provider,
                model=employee.model,
                enabled_toolsets=employee.enabled_toolsets or None,
                quiet_mode=True,
                ephemeral_system_prompt=sys_prompt,
                tool_start_callback=_on_tool_start,
                tool_complete_callback=_on_tool_complete,
                thinking_callback=_on_thinking,
            )
            try:
                # Different Hermes versions expose `chat()` or `run_conversation()`;
                # fall back across both for forward-compat.
                if hasattr(agent, "chat"):
                    final = agent.chat(task.text)
                elif hasattr(agent, "run_conversation"):
                    final = agent.run_conversation(task.text)
                else:
                    raise RuntimeError("AIAgent has neither chat() nor run_conversation()")
                summary = (final or "").strip()
                return {"summary": summary[:8000], "ok": True, "error": None}
            except Exception as exc:  # noqa: BLE001
                return {"summary": "", "ok": False, "error": repr(exc)}

        result = await asyncio.to_thread(_build_and_run)

        if result["ok"]:
            await emit("assistant", result["summary"] or "(empty response)")
            await emit("state_change", f"{employee.name} returns to rest", to=Activity.RESTING)
            return TaskResult(
                status="done",
                summary=result["summary"],
                tokens_in=0,
                tokens_out=0,
            )
        await emit("error", result["error"] or "unknown error")
        await emit("state_change", f"{employee.name} returns to rest", to=Activity.RESTING)
        return TaskResult(status="failed", error=result["error"])
