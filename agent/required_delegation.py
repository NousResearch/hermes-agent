"""Turn-owned requirements over the existing subagent lifecycle service.

Routing policies live in plugins. This module only enforces work a policy actually
declared: launch it, let the parent continue, and join its results before completion.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import time
from typing import Any, Iterator
import uuid

from agent.message_metadata import append_message
from agent.subagent_lifecycle import (
    SubagentHandle, SubagentLaunchRequest, SubagentLifecycleError,
    SubagentLifecycleService, SubagentResult, SubagentState,
)


@dataclass
class _Requirement:
    service: SubagentLifecycleService
    request: SubagentLaunchRequest
    handle: SubagentHandle | None = None
    result: SubagentResult | None = None
    error: str | None = None

    def refresh(self) -> None:
        if self.handle is not None:
            self.result = self.service.result(self.handle)

    def receipt(self, consumed: bool) -> dict[str, Any]:
        handle, result = self.handle, self.result
        return {
            "subagent_id": handle.subagent_id if handle else None,
            "correlation_id": getattr(self.request, "correlation_id", None),
            "state": result.terminal_state.value if result else "LAUNCH_FAILED",
            "provider": handle.provider if handle else None,
            "model": handle.model if handle else None,
            "result_hash": result.result_hash if result else None,
            "result_consumed": consumed,
            "error": self.error or (result.error_message if result else None),
        }


@dataclass
class _RequiredTurn:
    requirements: list[_Requirement] = field(default_factory=list)
    accepting: bool = True
    results_appended: bool = False
    results_consumed: bool = False
    receipt_messages: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    interrupted: bool = False

    def cancel_pending(self) -> None:
        for requirement in self.requirements:
            requirement.refresh()
            if requirement.handle is not None and not requirement.result.ready:
                requirement.service.cancel(requirement.handle, reason="owning required-delegation turn ended")
                requirement.refresh()


def _scope(agent: Any) -> _RequiredTurn | None:
    scope = getattr(agent, "_required_delegation_turn", None)
    return scope if isinstance(scope, _RequiredTurn) else None


@contextmanager
def required_delegation_turn(agent: Any) -> Iterator[None]:
    previous = getattr(agent, "_required_delegation_turn", None)
    scope = _RequiredTurn()
    agent._required_delegation_turn = scope
    try:
        yield
    finally:
        try:
            scope.cancel_pending()
        finally:
            agent._required_delegation_turn = previous


def require_subagent(
    agent: Any, service: SubagentLifecycleService, request: SubagentLaunchRequest,
) -> SubagentHandle:
    scope = _scope(agent)
    if scope is None or not scope.accepting:
        raise SubagentLifecycleError("Required delegation must be declared by a pre_llm_call hook in an active turn.")
    requirement = _Requirement(service, request)
    # Register before validation/launch: ordinary hooks log exceptions and continue.
    # An accepted requirement must survive that fail-open hook boundary.
    scope.requirements.append(requirement)
    try:
        if getattr(agent, "api_mode", None) == "codex_app_server":
            raise SubagentLifecycleError("Required delegation needs the Hermes agent loop.")
        if "delegate_task" not in getattr(agent, "valid_tool_names", ()):
            raise SubagentLifecycleError("Required delegation needs the parent's delegation toolset enabled.")
        requirement.handle = service.launch(request)
        requirement.refresh()
        return requirement.handle
    except Exception as exc:
        requirement.error = str(exc)[:2000]
        raise


def seal_required_delegations(agent: Any) -> str:
    """Freeze this turn's declarations and supply context through the existing sidecar."""
    scope = _scope(agent)
    if scope is None:
        return ""
    scope.accepting = False
    if not scope.requirements:
        return ""
    receipts = [r.receipt(False) for r in scope.requirements]
    return (
        "[Runtime-required delegation]\n"
        "The host has already launched the following required work. Continue useful independent work, "
        "gather context, or steer these subagents through delegate_task. Do not launch the same work again. "
        "The runtime will collect their results before accepting your final answer.\n"
        + json.dumps(receipts, ensure_ascii=False)
    )


def defer_required_delegation_text(agent: Any) -> bool:
    """A candidate answer must not stream as completion before required results arrive."""
    scope = _scope(agent)
    return bool(scope and scope.requirements and not scope.results_appended)


def required_delegation_launch_failure(agent: Any) -> str | None:
    scope = _scope(agent)
    if scope is not None:
        scope.error = next((r.error for r in scope.requirements if r.error), None)
        if scope.error:
            return required_delegation_failure(agent)
    return None


def _wait_for_results(agent: Any, scope: _RequiredTurn) -> None:
    for requirement in scope.requirements:
        if requirement.error:
            scope.error = requirement.error
            break
        while True:
            requirement.refresh()
            if requirement.result.ready:
                break
            if getattr(agent, "_interrupt_requested", False):
                scope.interrupted = True
                scope.error = "The parent turn was interrupted before required work completed."
                break
            budget = getattr(agent, "run_budget_seconds", None)
            started = getattr(agent, "_run_budget_started_at", None)
            if budget and started and time.time() - started >= budget:
                scope.error = "The parent run budget expired before required work completed."
                break
            requirement.service.wait(requirement.handle, timeout_seconds=0.1)
        if scope.error:
            break
        if requirement.result.terminal_state is not SubagentState.SUCCEEDED:
            scope.error = requirement.result.error_message or requirement.result.terminal_state.value
            break
        if not (requirement.result.summary or "").strip():
            scope.error = "A required subagent finished without a result."
            break
    if scope.error:
        scope.cancel_pending()


def collect_required_delegations(agent: Any, messages: list[dict]) -> tuple[bool, str | None]:
    """Return (continue_with_results, failure), without asking a model to delegate/wait."""
    scope = _scope(agent)
    if scope is None or not scope.requirements:
        return False, None
    if scope.results_appended:
        return False, required_delegation_failure(agent)
    _wait_for_results(agent, scope)
    if scope.error:
        return False, required_delegation_failure(agent)

    calls, results = [], []
    for requirement in scope.requirements:
        call_id = "call_required_" + uuid.uuid4().hex[:16]
        args = {"goal": requirement.request.goal}
        if requirement.request.context:
            args["context"] = requirement.request.context
        calls.append({"id": call_id, "type": "function", "function": {
            "name": "delegate_task", "arguments": json.dumps(args, ensure_ascii=False),
        }})
        results.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps({
            **requirement.receipt(False), "summary": requirement.result.summary,
            "origin": "runtime_required_delegation",
        }, ensure_ascii=False)})
    # Record actual host-issued work as a complete assistant/tool batch. Never
    # rewrite a sent row or inject a synthetic user message into the running loop.
    append_message(messages, {
        "role": "assistant", "content": None, "tool_calls": calls,
        "display_metadata": {"origin": "runtime_required_delegation"},
    })
    for result in results:
        append_message(messages, result)
        scope.receipt_messages[result["tool_call_id"]] = result["content"]
    agent._session_messages = messages
    scope.results_appended = True
    return True, None


def observe_required_delegation_results(agent: Any, api_messages: list[dict]) -> None:
    """Count delivery only after a parent response to a request containing the receipts.

    Compaction can remove an appended result before the next request. Conversely,
    a parent may consume it in a tool-calling response and compact it later.
    """
    scope = _scope(agent)
    if scope is None or not scope.results_appended or scope.results_consumed:
        return
    sent = {m.get("tool_call_id"): m.get("content") for m in api_messages if m.get("role") == "tool"}
    scope.results_consumed = all(sent.get(key) == value for key, value in scope.receipt_messages.items())


def required_delegation_failure(agent: Any) -> str | None:
    scope = _scope(agent)
    if scope is None or not scope.requirements or scope.results_consumed:
        return None
    detail = scope.error or "The parent turn ended before it could consume the required subagent results."
    return "Required delegation is incomplete. " + detail


def required_delegation_interrupted(agent: Any) -> bool:
    scope = _scope(agent)
    return bool(scope and scope.interrupted)


def attach_required_delegation_outcome(agent: Any, result: dict[str, Any]) -> None:
    """Also covers early returns that bypass ordinary turn finalization."""
    scope = _scope(agent)
    if scope is None or not scope.requirements:
        return
    failure = required_delegation_failure(agent)
    if failure:
        scope.cancel_pending()
        result.update(completed=False, failed=True, final_response=failure, error=failure,
                      failure_reason="required_delegation_incomplete", response_previewed=False)
    if scope.interrupted:
        result["interrupted"] = True
    result["required_delegations"] = [r.receipt(scope.results_consumed) for r in scope.requirements]
