"""Code-driven multi-agent workflow orchestration.

This module implements the real ultracode-style split:
PLAN (bounded LLM JSON), EXECUTE (Python-spawned subagents), SYNTHESIZE
(bounded LLM merge). The model never emits a large delegate_task tool call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from agent.auxiliary_client import call_llm
from tools.delegate_tool import _get_max_concurrent_children, delegate_task

_DEFAULT_PLAN_MODEL = "kr/claude-opus-4.8"
_PLAN_MAX_TOKENS = 2000
_SYNTHESIS_MAX_TOKENS = 6000


@dataclass
class WorkflowSubtask:
    goal: str
    context: str = ""
    toolsets: Optional[List[str]] = None
    role: str = "leaf"


@dataclass
class WorkflowPlan:
    mode: str = "parallel"
    subtasks: List[WorkflowSubtask] = field(default_factory=list)
    rationale: str = ""


@dataclass
class WorkflowResult:
    task: str
    plan: WorkflowPlan
    child_results: List[Dict[str, Any]]
    final_response: str
    delegated: bool
    total_duration_seconds: float = 0.0


class WorkflowOrchestrator:
    """Runs PLAN -> EXECUTE -> SYNTHESIZE using code, not model tool calls."""

    def __init__(
        self,
        agent: Any,
        *,
        plan_model: Optional[str] = None,
        call_llm_fn: Callable[..., Any] = call_llm,
        delegate_fn: Callable[..., str] = delegate_task,
        max_children_fn: Callable[[], int] = _get_max_concurrent_children,
    ) -> None:
        self.agent = agent
        self.plan_model = plan_model or _non_thinking_model(getattr(agent, "model", "")) or _DEFAULT_PLAN_MODEL
        self.call_llm_fn = call_llm_fn
        self.delegate_fn = delegate_fn
        self.max_children_fn = max_children_fn

    def plan(self, task: str) -> WorkflowPlan:
        content = self._call_planner(task)
        payload = _extract_json_object(content)
        raw_subtasks = payload.get("subtasks") or []
        subtasks: List[WorkflowSubtask] = []
        for raw in raw_subtasks[:8]:
            if not isinstance(raw, dict):
                continue
            goal = str(raw.get("goal") or "").strip()
            if not goal:
                continue
            context = str(raw.get("context") or "").strip()
            toolsets = raw.get("toolsets")
            if toolsets is not None and not isinstance(toolsets, list):
                toolsets = None
            subtasks.append(
                WorkflowSubtask(
                    goal=goal,
                    context=context,
                    toolsets=toolsets,
                    role=str(raw.get("role") or "leaf"),
                )
            )
        if not subtasks:
            subtasks = [WorkflowSubtask(goal=task, context="Single-step fallback plan.")]
        mode = str(payload.get("mode") or "parallel").lower().strip()
        if mode not in {"parallel", "sequential"}:
            mode = "parallel"
        return WorkflowPlan(mode=mode, subtasks=subtasks, rationale=str(payload.get("rationale") or ""))

    def run(self, task: str) -> WorkflowResult:
        plan = self.plan(task)
        if len(plan.subtasks) <= 1:
            return WorkflowResult(
                task=task,
                plan=plan,
                child_results=[],
                final_response="",
                delegated=False,
            )

        child_results, total_duration = self._execute_plan(plan)
        final_response = self._synthesize(task, plan, child_results)
        return WorkflowResult(
            task=task,
            plan=plan,
            child_results=child_results,
            final_response=final_response,
            delegated=True,
            total_duration_seconds=total_duration,
        )

    def _call_planner(self, task: str) -> str:
        response = self.call_llm_fn(
            provider=getattr(self.agent, "provider", None),
            model=self.plan_model,
            base_url=getattr(self.agent, "base_url", None),
            api_key=getattr(self.agent, "api_key", None),
            messages=[
                {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ],
            temperature=0.2,
            max_tokens=_PLAN_MAX_TOKENS,
            tools=None,
            timeout=120,
        )
        return response.choices[0].message.content or "{}"

    def _execute_plan(self, plan: WorkflowPlan) -> tuple[List[Dict[str, Any]], float]:
        max_children = max(1, int(self.max_children_fn()))
        all_results: List[Dict[str, Any]] = []
        total_duration = 0.0
        prior_summary = ""

        indexed_subtasks = list(enumerate(plan.subtasks))
        if plan.mode == "sequential":
            waves = [[task] for task in indexed_subtasks]
        else:
            waves = [indexed_subtasks[i : i + max_children] for i in range(0, len(indexed_subtasks), max_children)]

        for wave_index, wave in enumerate(waves, start=1):
            delegate_tasks = []
            local_to_global: Dict[int, int] = {}
            for local_index, (global_index, task) in enumerate(wave):
                local_to_global[local_index] = global_index
                context = task.context
                if prior_summary:
                    context = f"{context}\n\nPrevious workflow results:\n{prior_summary}".strip()
                delegate_tasks.append(
                    {
                        "goal": task.goal,
                        "context": context,
                        "toolsets": task.toolsets,
                        "role": task.role or "leaf",
                    }
                )
            raw = self.delegate_fn(tasks=delegate_tasks, parent_agent=self.agent)
            payload = json.loads(raw)
            if "error" in payload:
                raise RuntimeError(str(payload["error"]))
            wave_results = payload.get("results") or []
            for fallback_local_index, item in enumerate(wave_results):
                if isinstance(item, dict):
                    local_index = _safe_int(item.get("task_index"), fallback_local_index)
                    item["task_index"] = local_to_global.get(local_index, local_index)
                    item.setdefault("wave", wave_index)
                    all_results.append(item)
            total_duration += float(payload.get("total_duration_seconds") or 0.0)
            if plan.mode == "sequential":
                prior_summary = _summarize_child_results(all_results)

        return all_results, total_duration

    def _synthesize(self, task: str, plan: WorkflowPlan, child_results: Sequence[Dict[str, Any]]) -> str:
        response = self.call_llm_fn(
            provider=getattr(self.agent, "provider", None),
            model=self.plan_model,
            base_url=getattr(self.agent, "base_url", None),
            api_key=getattr(self.agent, "api_key", None),
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": task,
                            "plan": _plan_to_dict(plan),
                            "child_results": child_results,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=_SYNTHESIS_MAX_TOKENS,
            tools=None,
            timeout=180,
        )
        return (response.choices[0].message.content or "").strip()


_PLAN_SYSTEM_PROMPT = """You are a workflow planner for a coding agent.
Return ONLY a JSON object, no markdown, no prose.
Schema: {"mode":"parallel"|"sequential","rationale":str,"subtasks":[{"goal":str,"context":str,"toolsets":list[str]|null,"role":"leaf"}]}.
Use exactly ONE subtask for trivial or inherently single-step requests.
Use multiple subtasks only when independent work can benefit from subagents or when a complex task needs staged execution.
Choose "parallel" for independent subtasks. Choose "sequential" when later subtasks depend on earlier outputs.
Keep goals one sentence. Keep context under three sentences. Max 8 subtasks.
"""

_SYNTHESIS_SYSTEM_PROMPT = """You synthesize subagent results into the final user-facing answer.
Be concise, factual, and explicit about failures. Do not claim success for a subtask whose status is not completed.
If child summaries mention files, tests, commands, or errors, preserve the important details.
"""


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = (text or "").strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        stripped = match.group(0)
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("Workflow planner returned JSON that is not an object")
    return payload


def _non_thinking_model(model: str) -> str:
    value = (model or "").strip()
    for suffix in ("-thinking-agentic", "-thinking", "-xhigh", "-high", "-low"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _plan_to_dict(plan: WorkflowPlan) -> Dict[str, Any]:
    return {
        "mode": plan.mode,
        "rationale": plan.rationale,
        "subtasks": [
            {"goal": task.goal, "context": task.context, "toolsets": task.toolsets, "role": task.role}
            for task in plan.subtasks
        ],
    }


def _summarize_child_results(results: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for result in results:
        status = result.get("status", "unknown")
        summary = result.get("summary") or result.get("error") or ""
        lines.append(f"- Task {result.get('task_index')}: {status}: {summary}")
    return "\n".join(lines)
