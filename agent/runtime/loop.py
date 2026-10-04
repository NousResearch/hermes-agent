"""Provider-agnostic governance-ready multi-step loop.

The loop knows nothing about Anthropic, Gemini, Bedrock, OpenAI, or any
hermes-specific concept. Everything that varies is injected:

  * model        — ``ModelProtocol``
  * tool_handler — ``ToolHandlerProtocol``
  * governance   — ``GovernanceProtocol``  (default: DenyAllGovernance)
  * transitions  — ``TransitionGuard``     (validates step ordering)
  * clock        — ``Clock``               (SystemClock or FrozenClock)
  * id_source    — ``IdSource``            (UuidIdSource or SequentialIdSource)

The loop is **fail-closed by default**:

  * Default governance is DenyAllGovernance — every tool call is denied
    until the caller opts in.
  * ``continue_on_error=False`` by default — model errors, tool errors,
    transition rejections, and limit breaches terminate the run with a
    typed ``StepFailure`` and ``completed=False``.
  * Every tool call passes through the governance gate; denied calls
    never execute and are recorded as error outputs.
  * State mutations go through the typed ``RunState`` API; raw dict
    mutation is not possible.

Every step records ``started_at`` from the injected ``Clock`` so traces
are byte-identical under replay.
"""

from __future__ import annotations

from typing import Any, Callable

from .callbacks import CallbackRegistry
from .governance import DenyAllGovernance, GovernanceGate
from .interfaces import (
    FINAL_ANSWER_TOOL,
    Clock,
    GovernanceContext,
    GovernanceDecision,
    GovernanceProtocol,
    IdSource,
    ModelOutput,
    ModelProtocol,
    SystemClock,
    ToolHandlerProtocol,
    UuidIdSource,
    failure_from_exception,
)
from .memory import AgentMemory
from .result import RunResult, Timing, TokenUsage
from .state import RunState
from .steps import (
    ActionStep,
    FailureKind,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    StepFailure,
    TaskStep,
    ToolCall,
    ToolOutput,
)
from .transitions import TransitionGuard

FinalAnswerCheck = Callable[[Any, AgentMemory], tuple[bool, str]]  # (ok, reason)


class MultiStepLoop:
    """Hardened provider-agnostic loop.

    Constructor is deliberately verbose. Every gate is explicit; no
    permissive defaults that hide policy decisions.
    """

    def __init__(
        self,
        model: ModelProtocol,
        tool_handler: ToolHandlerProtocol,
        *,
        tools: list[dict[str, Any]] | None = None,
        max_steps: int = 20,
        planning_interval: int | None = None,
        callbacks: CallbackRegistry | None = None,
        final_answer_checks: list[FinalAnswerCheck] | None = None,
        final_answer_tool_name: str = FINAL_ANSWER_TOOL,
        governance: GovernanceProtocol | None = None,
        transitions: TransitionGuard | None = None,
        clock: Clock | None = None,
        id_source: IdSource | None = None,
        continue_on_error: bool = False,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if planning_interval is not None and planning_interval < 1:
            raise ValueError("planning_interval must be >= 1 or None")

        self.model = model
        self.tool_handler = tool_handler
        self.tools = tools or []
        self.max_steps = max_steps
        self.planning_interval = planning_interval
        self.callbacks = callbacks or CallbackRegistry()
        self.final_answer_checks = list(final_answer_checks or [])
        self.final_answer_tool_name = final_answer_tool_name
        self.continue_on_error = continue_on_error

        # Fail-closed default: deny every tool call unless explicit policy.
        self._gate = GovernanceGate(governance or DenyAllGovernance())
        self.transitions = transitions or TransitionGuard()
        self._clock: Clock = clock or SystemClock()
        self._id_source: IdSource = id_source or UuidIdSource()

        self.memory = AgentMemory()
        self.state = RunState(self._clock)
        self._interrupted = False
        # Per-run caches — invalidated at the start of every run() and
        # updated as steps land via _append(). Avoids O(N²) memory scans.
        self._cached_task_text: str = ""
        self._cached_prior_tool_names: list[str] = []
        self._monotonic_step: int = 0  # last step_number actually appended

    # ---- public API ---------------------------------------------------------

    def interrupt(self) -> None:
        """Cooperative interrupt — flagged between steps."""
        self._interrupted = True

    def run(self, task: str, images: tuple[str, ...] = ()) -> RunResult:
        # MultiStepLoop instances are single-use. State is frozen at the end
        # of every run; reusing the instance would either silently rebuild
        # over a frozen state or surface a misleading StateFrozenError mid-
        # step. Refuse cleanly so the caller knows to build a new loop.
        if self.state.frozen or len(self.memory) > 0:
            raise RuntimeError(
                "MultiStepLoop is single-use; construct a new instance per run()"
            )

        started_at = self._clock.now()
        ok = self._append(TaskStep(step_number=0, started_at=started_at, task=task, images=images))
        if not ok:
            return self._fail("transition_rejected", "could not append TaskStep", started_at, TokenUsage())

        token_usage = TokenUsage()

        for step_number in range(1, self.max_steps + 1):
            if self._interrupted:
                return self._terminate(
                    output=self._last_useful_output(),
                    triggered_by="interrupted",
                    failure=StepFailure(kind="interrupted", message="run interrupted by caller"),
                    started_at=started_at,
                    token_usage=token_usage,
                    completed=False,
                )

            # Optional planning step. Planning gets its own monotonic
            # step_number so it never collides with the action step that
            # follows in the same iteration — consumers indexing by
            # step_number expect uniqueness.
            if (
                self.planning_interval
                and step_number > 1
                and (step_number - 1) % self.planning_interval == 0
            ):
                plan_number = self._monotonic_step + 1
                plan = self._planning_step(plan_number)
                appended = self._append(plan)
                if not appended:
                    return self._fail("transition_rejected", "could not append PlanningStep", started_at, token_usage)
                if plan.failure and not self._should_continue(plan.failure):
                    return self._terminate_failure(plan.failure, started_at, token_usage)

            # Action step — also takes the next monotonic slot.
            action_number = self._monotonic_step + 1
            action = self._action_step(action_number)
            token_usage = token_usage + TokenUsage(action.input_tokens, action.output_tokens)
            appended = self._append(action)
            if not appended:
                return self._fail("transition_rejected", "could not append ActionStep", started_at, token_usage)

            if action.failure and not self._should_continue(action.failure):
                return self._terminate_failure(action.failure, started_at, token_usage)

            # Allowed final-answer tool output?
            for out in action.tool_outputs:
                if out.is_final_answer and not out.is_error:
                    ok_check, reason = self._check_final_answer(out.output)
                    if ok_check:
                        return self._terminate(
                            output=out.output,
                            triggered_by="final_answer_tool",
                            failure=None,
                            started_at=started_at,
                            token_usage=token_usage,
                            completed=True,
                        )
                    self.state.append(
                        "rejected_final_answers",
                        {"output": out.output, "reason": reason},
                        reason="final_answer_check rejected tool output",
                    )

            # Plain-text response with no tool calls — implicit final answer.
            if not action.tool_calls and action.model_output:
                ok_check, reason = self._check_final_answer(action.model_output)
                if ok_check:
                    return self._terminate(
                        output=action.model_output,
                        triggered_by="empty_tool_calls",
                        failure=None,
                        started_at=started_at,
                        token_usage=token_usage,
                        completed=True,
                    )
                self.state.append(
                    "rejected_final_answers",
                    {"output": action.model_output, "reason": reason},
                    reason="final_answer_check rejected plain text",
                )

        # ``for`` exhausted without returning.
        return self._terminate(
            output=self._last_useful_output(),
            triggered_by="max_steps",
            failure=StepFailure(
                kind="limit_exceeded",
                message=f"reached max_steps={self.max_steps}",
                details={"limit": "max_steps", "max_steps": self.max_steps},
            ),
            started_at=started_at,
            token_usage=token_usage,
            completed=False,
        )

    # ---- step builders ------------------------------------------------------

    def _action_step(self, step_number: int) -> ActionStep:
        started = self._clock.now()
        messages = self.memory.to_messages()
        # Cached on _append, not rebuilt — was O(N) per step previously.
        prior_tool_names = tuple(self._cached_prior_tool_names)

        # 1. Model call.
        try:
            output: ModelOutput = self.model.generate(messages, tools=self.tools)
        except Exception as exc:
            return ActionStep(
                step_number=step_number,
                started_at=started,
                model_input_messages=tuple(messages),
                failure=failure_from_exception("model_error", exc),
                duration_s=self._clock.now() - started,
            )

        # 2. Assign ids to any tool calls missing one. The contract is
        # "every call has a non-empty id" — provider shims should supply
        # one; the fallback covers shims that pass the model id through
        # raw and need a stable substitute when it's empty/None.
        normalized_calls = tuple(
            ToolCall(
                id=(call.id or self._id_source.call_id()),
                name=call.name,
                arguments=call.arguments,
            )
            for call in output.tool_calls
        )
        # Defensive: surface a misbehaving shim early rather than letting
        # governance see an empty-id call.
        for c in normalized_calls:
            if not c.id:
                raise RuntimeError(
                    f"ToolCall produced with empty id (name={c.name!r}); "
                    "shim must populate ToolCall.id or rely on IdSource"
                )

        # 3. Governance gate — decide each call.
        context = GovernanceContext(
            step_number=step_number,
            task=self._cached_task_text,
            prior_tool_names=prior_tool_names,
            state_snapshot=self.state.snapshot(),
        )
        decisions: tuple[GovernanceDecision, ...] = ()
        allowed_calls: list[ToolCall] = []
        denied_outputs: list[ToolOutput] = []
        if normalized_calls:
            decisions = tuple(self._gate.evaluate(list(normalized_calls), context))
            for call, decision in zip(normalized_calls, decisions, strict=True):
                if decision.verdict == "allow":
                    allowed_calls.append(call)
                else:
                    denied_outputs.append(
                        ToolOutput(
                            id=call.id,
                            name=call.name,
                            output={
                                "denied": True,
                                "verdict": decision.verdict,
                                "reason": decision.reason,
                                "policy": decision.policy,
                            },
                            is_error=True,
                            is_final_answer=False,
                            synthesized=True,
                        )
                    )

        # 4. Tool execution for allowed calls only.
        tool_outputs: tuple[ToolOutput, ...] = tuple(denied_outputs)
        failure: StepFailure | None = None
        if allowed_calls:
            try:
                results = self.tool_handler.handle(allowed_calls)
            except Exception as exc:
                failure = failure_from_exception(
                    "tool_error",
                    exc,
                    extra={"allowed_call_count": len(allowed_calls)},
                )
            else:
                tool_outputs = tool_outputs + tuple(
                    ToolOutput(
                        id=r.id,
                        name=r.name,
                        output=r.output,
                        is_error=r.is_error,
                        is_final_answer=(
                            r.is_final_answer or r.name == self.final_answer_tool_name
                        ),
                        synthesized=r.synthesized,
                    )
                    for r in results
                )

        # 5. Fail-closed: if every call was denied, surface a governance_denied failure.
        if normalized_calls and not allowed_calls and failure is None:
            failure = StepFailure(
                kind="governance_denied",
                message=f"all {len(normalized_calls)} tool calls denied by governance",
                details={"policy": getattr(self._gate.policy, "policy_id", type(self._gate.policy).__name__)},
            )

        return ActionStep(
            step_number=step_number,
            started_at=started,
            model_input_messages=tuple(messages),
            model_output=output.content,
            tool_calls=normalized_calls,
            tool_outputs=tool_outputs,
            governance_decisions=decisions,
            failure=failure,
            duration_s=self._clock.now() - started,
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
        )

    def _planning_step(self, step_number: int) -> PlanningStep:
        started = self._clock.now()
        messages = self.memory.to_messages()
        messages.append({"role": "user", "content": "<plan-request>Write a brief plan for remaining work.</plan-request>"})
        try:
            output = self.model.generate(messages, tools=None)
            plan_text = output.content
            failure: StepFailure | None = None
        except Exception as exc:
            plan_text = ""
            failure = failure_from_exception("model_error", exc)
        return PlanningStep(
            step_number=step_number,
            started_at=started,
            model_input_messages=tuple(messages),
            plan=plan_text,
            duration_s=self._clock.now() - started,
            failure=failure,
        )

    # ---- helpers ------------------------------------------------------------

    def _append(self, step: MemoryStep) -> bool:
        """Validate transition, append, dispatch callbacks. Returns False if rejected."""
        previous = self.memory.last()
        ok, reason = self.transitions.check(previous, step)
        if not ok:
            # Record a rejection via state, do not append the offending step.
            self.state.append(
                "transition_rejections",
                {"step_number": step.step_number, "reason": reason},
                reason=f"transition_guard: {reason}",
            )
            return False
        self.memory.append(step)
        # Maintain caches consumed by _action_step / GovernanceContext so
        # we don't rescan memory every iteration.
        if isinstance(step, TaskStep):
            self._cached_task_text = step.task
        elif isinstance(step, ActionStep):
            self._cached_prior_tool_names.extend(c.name for c in step.tool_calls)
        self._monotonic_step = max(self._monotonic_step, step.step_number)
        self.callbacks.dispatch(step, self.state)
        if self.state.get("__interrupt__"):
            self._interrupted = True
        return True

    def _check_final_answer(self, output: Any) -> tuple[bool, str]:
        for check in self.final_answer_checks:
            ok, reason = check(output, self.memory)
            if not ok:
                return False, reason
        return True, ""

    def _should_continue(self, failure: StepFailure) -> bool:
        """Fail-closed: only continue if the caller opted in."""
        return self.continue_on_error

    def _last_useful_output(self) -> Any:
        for step in reversed(self.memory.steps):
            if isinstance(step, ActionStep) and step.failure is None and step.model_output:
                return step.model_output
        return None

    # ---- termination --------------------------------------------------------

    def _terminate(
        self,
        *,
        output: Any,
        triggered_by: str,
        failure: StepFailure | None,
        started_at: float,
        token_usage: TokenUsage,
        completed: bool,
    ) -> RunResult:
        final = FinalAnswerStep(
            step_number=self._monotonic_step + 1,
            started_at=self._clock.now(),
            output=output,
            triggered_by=triggered_by,
            failure=failure,
        )
        appended = self._append(final)
        if not appended:
            # Transition guard refused (e.g. duplicate terminal). Record a
            # state entry and proceed — the run is over either way.
            self.state.append(
                "terminal_append_rejections",
                {"triggered_by": triggered_by},
                reason="transition_guard refused FinalAnswerStep",
            )
        return self._finalize(output, started_at, token_usage, triggered_by, completed)

    def _terminate_failure(
        self,
        failure: StepFailure,
        started_at: float,
        token_usage: TokenUsage,
    ) -> RunResult:
        return self._terminate(
            output=self._last_useful_output(),
            triggered_by=failure.kind,
            failure=failure,
            started_at=started_at,
            token_usage=token_usage,
            completed=False,
        )

    def _fail(self, kind: FailureKind, message: str, started_at: float, token_usage: TokenUsage) -> RunResult:
        return self._terminate_failure(
            StepFailure(kind=kind, message=message),
            started_at,
            token_usage,
        )

    def _finalize(
        self,
        output: Any,
        started_at: float,
        token_usage: TokenUsage,
        termination: str,
        completed: bool,
    ) -> RunResult:
        ended_at = self._clock.now()
        steps_payload = tuple(_step_payload(s) for s in self.memory.steps)
        result_state = self.state.snapshot()
        self.state.freeze()
        return RunResult(
            output=output,
            steps=steps_payload,
            state=result_state,
            token_usage=token_usage,
            timing=Timing(started_at=started_at, ended_at=ended_at),
            completed=completed,
            termination_reason=termination,
        )


# ---- payload projection ------------------------------------------------------


def _step_payload(step: MemoryStep) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(step).__name__,
        "step_number": step.step_number,
        "started_at": step.started_at,
    }
    if isinstance(step, TaskStep):
        payload["task"] = step.task
    elif isinstance(step, PlanningStep):
        payload["plan"] = step.plan
        if step.failure:
            payload["failure"] = _failure_payload(step.failure)
    elif isinstance(step, ActionStep):
        payload["model_output"] = step.model_output
        payload["tool_calls"] = [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in step.tool_calls
        ]
        payload["tool_outputs"] = [
            {
                "id": o.id,
                "name": o.name,
                "output": o.output,
                "is_error": o.is_error,
                "is_final_answer": o.is_final_answer,
                "synthesized": o.synthesized,
            }
            for o in step.tool_outputs
        ]
        payload["governance_decisions"] = [
            {
                "call_id": d.call_id,
                "tool_name": d.tool_name,
                "verdict": d.verdict,
                "reason": d.reason,
                "policy": d.policy,
            }
            for d in step.governance_decisions
        ]
        payload["input_tokens"] = step.input_tokens
        payload["output_tokens"] = step.output_tokens
        payload["duration_s"] = step.duration_s
        if step.failure:
            payload["failure"] = _failure_payload(step.failure)
    elif isinstance(step, FinalAnswerStep):
        payload["output"] = step.output
        payload["triggered_by"] = step.triggered_by
        if step.failure:
            payload["failure"] = _failure_payload(step.failure)
    return payload


def _failure_payload(failure: StepFailure) -> dict[str, Any]:
    return {"kind": failure.kind, "message": failure.message, "details": dict(failure.details)}
