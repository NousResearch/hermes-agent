"""Append-only typed step history.

``AgentMemory`` is the source of truth for what happened during a run. It is
append-only by contract — callers must not mutate past steps. This is what
makes ``replay()`` deterministic and lets ``step_callbacks`` reason about
history without needing to defensive-copy.

JSONL roundtrip is lossless across all step types, including
``governance_decisions`` and ``failure`` payloads.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

from .interfaces import GovernanceDecision
from .steps import (
    ActionStep,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    StepFailure,
    TaskStep,
    ToolCall,
    ToolOutput,
)


class AgentMemory:
    """Append-only typed step history."""

    def __init__(self) -> None:
        self._steps: list[MemoryStep] = []

    # ---- mutation -----------------------------------------------------------

    def append(self, step: MemoryStep) -> None:
        if not isinstance(step, MemoryStep):
            raise TypeError(f"AgentMemory only accepts MemoryStep instances, got {type(step).__name__}")
        self._steps.append(step)

    # ---- read access --------------------------------------------------------

    @property
    def steps(self) -> tuple[MemoryStep, ...]:
        return tuple(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self) -> Iterator[MemoryStep]:
        return iter(self._steps)

    def last(self) -> MemoryStep | None:
        return self._steps[-1] if self._steps else None

    def task_step(self) -> TaskStep | None:
        for step in self._steps:
            if isinstance(step, TaskStep):
                return step
        return None

    def action_steps(self) -> tuple[ActionStep, ...]:
        return tuple(s for s in self._steps if isinstance(s, ActionStep))

    def final_answer(self) -> Any:
        for step in reversed(self._steps):
            if isinstance(step, FinalAnswerStep):
                return step.output
        return None

    # ---- serialization ------------------------------------------------------

    def to_messages(self) -> list[dict[str, Any]]:
        """Project step history into chat-completion shape.

        Denied tool outputs are surfaced as tool messages flagged
        ``is_error=True`` so the model sees why a proposed call did not
        execute and can adapt.
        """
        messages: list[dict[str, Any]] = []
        for step in self._steps:
            if isinstance(step, TaskStep):
                messages.append({"role": "user", "content": step.task})
            elif isinstance(step, PlanningStep):
                messages.append({"role": "assistant", "content": f"<plan>\n{step.plan}\n</plan>"})
            elif isinstance(step, ActionStep):
                if step.model_output:
                    messages.append({"role": "assistant", "content": step.model_output})
                for call in step.tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [{"id": call.id, "name": call.name, "arguments": call.arguments}],
                        }
                    )
                for output in step.tool_outputs:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": output.id,
                            "name": output.name,
                            "content": _stringify(output.output),
                            "is_error": output.is_error,
                        }
                    )
            elif isinstance(step, FinalAnswerStep):
                messages.append({"role": "assistant", "content": _stringify(step.output)})
        return messages

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(_step_to_jsonable(s), ensure_ascii=False) for s in self._steps)

    @classmethod
    def from_jsonl(cls, blob: str) -> "AgentMemory":
        memory = cls()
        for line in blob.splitlines():
            line = line.strip()
            if not line:
                continue
            memory.append(_step_from_jsonable(json.loads(line)))
        return memory

    # ---- replay -------------------------------------------------------------

    def replay(self, detailed: bool = False) -> str:
        lines: list[str] = []
        for step in self._steps:
            lines.append(f"Step {step.step_number:>3} · {type(step).__name__}")
            if isinstance(step, TaskStep):
                lines.append(f"  task: {_truncate(step.task, 200 if not detailed else 10_000)}")
            elif isinstance(step, PlanningStep):
                lines.append(f"  plan: {_truncate(step.plan, 200 if not detailed else 10_000)}")
                if step.failure:
                    lines.append(f"  fail: {step.failure.kind}: {step.failure.message}")
            elif isinstance(step, ActionStep):
                if step.model_output:
                    lines.append(f"  out:  {_truncate(step.model_output, 200 if not detailed else 10_000)}")
                for decision in step.governance_decisions:
                    if decision.verdict != "allow":
                        lines.append(f"  gov:  {decision.verdict} {decision.tool_name} ({decision.reason})")
                for call in step.tool_calls:
                    lines.append(f"  call: {call.name}({_truncate(json.dumps(call.arguments), 120)})")
                for out in step.tool_outputs:
                    tag = "ERR" if out.is_error else "ok "
                    lines.append(f"  {tag}:  {_truncate(_stringify(out.output), 200 if not detailed else 10_000)}")
                if step.failure:
                    lines.append(f"  fail: {step.failure.kind}: {step.failure.message}")
            elif isinstance(step, FinalAnswerStep):
                lines.append(f"  ⇒     {_truncate(_stringify(step.output), 200 if not detailed else 10_000)}")
                lines.append(f"  reason: {step.triggered_by}")
                if step.failure:
                    lines.append(f"  fail: {step.failure.kind}: {step.failure.message}")
        return "\n".join(lines)


# ----- helpers ----------------------------------------------------------------


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _step_to_jsonable(step: MemoryStep) -> dict[str, Any]:
    payload = asdict(step)
    payload["__type__"] = type(step).__name__
    return payload


def _step_from_jsonable(payload: dict[str, Any]) -> MemoryStep:
    type_name = payload.pop("__type__")
    failure = _failure_from_payload(payload.pop("failure", None))

    if type_name == "TaskStep":
        return TaskStep(
            step_number=payload["step_number"],
            started_at=payload.get("started_at", 0.0),
            task=payload.get("task", ""),
            images=tuple(payload.get("images", [])),
        )
    if type_name == "PlanningStep":
        return PlanningStep(
            step_number=payload["step_number"],
            started_at=payload.get("started_at", 0.0),
            model_input_messages=tuple(payload.get("model_input_messages", [])),
            plan=payload.get("plan", ""),
            facts=payload.get("facts", ""),
            duration_s=payload.get("duration_s", 0.0),
            failure=failure,
        )
    if type_name == "ActionStep":
        return ActionStep(
            step_number=payload["step_number"],
            started_at=payload.get("started_at", 0.0),
            model_input_messages=tuple(payload.get("model_input_messages", [])),
            model_output=payload.get("model_output", ""),
            tool_calls=tuple(
                ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
                for c in payload.get("tool_calls", [])
            ),
            tool_outputs=tuple(
                ToolOutput(
                    id=o["id"],
                    name=o["name"],
                    output=o["output"],
                    is_error=o.get("is_error", False),
                    is_final_answer=o.get("is_final_answer", False),
                    synthesized=o.get("synthesized", False),
                )
                for o in payload.get("tool_outputs", [])
            ),
            governance_decisions=tuple(
                GovernanceDecision(
                    call_id=d["call_id"],
                    tool_name=d["tool_name"],
                    verdict=d["verdict"],
                    reason=d.get("reason", ""),
                    policy=d.get("policy", ""),
                )
                for d in payload.get("governance_decisions", [])
            ),
            failure=failure,
            duration_s=payload.get("duration_s", 0.0),
            input_tokens=payload.get("input_tokens", 0),
            output_tokens=payload.get("output_tokens", 0),
        )
    if type_name == "FinalAnswerStep":
        return FinalAnswerStep(
            step_number=payload["step_number"],
            started_at=payload.get("started_at", 0.0),
            output=payload.get("output"),
            triggered_by=payload.get("triggered_by", "final_answer_tool"),
            failure=failure,
        )
    raise ValueError(f"Unknown step type: {type_name}")


def _failure_from_payload(data: Any) -> StepFailure | None:
    if not data:
        return None
    return StepFailure(
        kind=data["kind"],
        message=data.get("message", ""),
        details=dict(data.get("details", {})),
    )
