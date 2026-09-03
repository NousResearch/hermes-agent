"""Session-scoped state machine for bounded automatic review checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from functools import partial
import json
from typing import Any, Callable

from agent.review_runner import ReviewRequest, ReviewResult


@dataclass(frozen=True)
class ReviewCheckpointConfig:
    """Behavior that the session owner explicitly enables."""

    enabled: bool = False
    max_revisions: int = 2
    failure_policy: str = "continue"


@dataclass(frozen=True)
class ReviewCheckpointDecision:
    """A core-loop action derived from one review result."""

    action: str
    result: ReviewResult | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ReviewCheckpointRoute:
    """Exact auxiliary route selected by the user for this session."""

    provider: str
    model: str
    require_distinct_from_main: bool = True
    timeout_seconds: float = 60.0


@dataclass
class _InFlight:
    done: threading.Event


class ReviewCheckpointController:
    """Own idempotency, cancellation, and verdict transitions for one session."""

    def __init__(
        self,
        *,
        session_id: str,
        config: ReviewCheckpointConfig,
        run_review: Callable[[ReviewRequest], ReviewResult],
        emit: Callable[[dict], None] | None = None,
    ) -> None:
        if not session_id.strip():
            raise ValueError("session_id is required")
        if config.max_revisions < 0:
            raise ValueError("max_revisions must be non-negative")
        if config.failure_policy not in {"continue", "block"}:
            raise ValueError("failure_policy must be 'continue' or 'block'")
        if not callable(run_review):
            raise ValueError("run_review must be callable")

        self.session_id = session_id
        self.config = config
        self._run_review = run_review
        self._emit = emit
        self._lock = threading.RLock()
        self._completed: dict[str, ReviewCheckpointDecision] = {}
        self._inflight: dict[str, _InFlight] = {}
        self._cancelled: set[str] = set()

    def cancel(self, checkpoint_id: str) -> None:
        """Cancel a checkpoint; an already returned late result stays discarded."""

        if not checkpoint_id:
            return
        with self._lock:
            if checkpoint_id in self._completed:
                return
            self._cancelled.add(checkpoint_id)
            if checkpoint_id not in self._inflight:
                self._completed[checkpoint_id] = self._cancelled_decision(
                    checkpoint_id
                )

    def evaluate(self, request: ReviewRequest) -> ReviewCheckpointDecision:
        """Evaluate once, sharing one billed call across duplicate checkpoint IDs."""

        if not self.config.enabled:
            return ReviewCheckpointDecision(action="continue")
        if request.session_id != self.session_id:
            return ReviewCheckpointDecision(
                action="block",
                reason="session_mismatch",
            )

        with self._lock:
            cached = self._completed.get(request.checkpoint_id)
            if cached is not None:
                return cached
            if request.checkpoint_id in self._cancelled:
                decision = self._cancelled_decision(request.checkpoint_id)
                self._completed[request.checkpoint_id] = decision
                return decision
            pending = self._inflight.get(request.checkpoint_id)
            if pending is None:
                pending = _InFlight(done=threading.Event())
                self._inflight[request.checkpoint_id] = pending
                owner = True
            else:
                owner = False

        if not owner:
            pending.done.wait()
            with self._lock:
                return self._completed[request.checkpoint_id]

        try:
            try:
                result = self._run_review(request)
            except Exception:
                result = ReviewResult(
                    checkpoint_id=request.checkpoint_id,
                    status="unavailable",
                    summary="Review runner failed before producing a safe result.",
                    unavailable_reason="runner_error",
                )

            if not isinstance(result, ReviewResult):
                result = ReviewResult(
                    checkpoint_id=request.checkpoint_id,
                    status="unavailable",
                    summary="Review runner returned an invalid result.",
                    unavailable_reason="invalid_review_result",
                )
            elif result.checkpoint_id != request.checkpoint_id:
                result = ReviewResult(
                    checkpoint_id=request.checkpoint_id,
                    status="unavailable",
                    summary="Review result did not match the active checkpoint.",
                    unavailable_reason="checkpoint_mismatch",
                )

            with self._lock:
                cancelled = request.checkpoint_id in self._cancelled
            decision = (
                self._cancelled_decision(request.checkpoint_id)
                if cancelled
                else self._decision_for(request, result)
            )
        finally:
            with self._lock:
                # Keep a safe terminal object even if an unexpected mapping bug
                # occurs, so duplicate waiters never hang or trigger a second call.
                if "decision" not in locals():
                    decision = ReviewCheckpointDecision(
                        action=self.config.failure_policy,
                        result=ReviewResult(
                            checkpoint_id=request.checkpoint_id,
                            status="unavailable",
                            unavailable_reason="controller_error",
                        ),
                        reason="controller_error",
                    )
                self._completed[request.checkpoint_id] = decision
                current = self._inflight.pop(request.checkpoint_id, None)
                if current is not None:
                    current.done.set()

        self._emit_decision(request, decision)
        return decision

    @staticmethod
    def _cancelled_decision(checkpoint_id: str) -> ReviewCheckpointDecision:
        result = ReviewResult(
            checkpoint_id=checkpoint_id,
            status="cancelled",
            unavailable_reason="cancelled",
        )
        return ReviewCheckpointDecision(
            action="cancelled",
            result=result,
            reason="cancelled",
        )

    def _decision_for(
        self,
        request: ReviewRequest,
        result: ReviewResult,
    ) -> ReviewCheckpointDecision:
        if result.status != "completed":
            return ReviewCheckpointDecision(
                action=self.config.failure_policy,
                result=result,
                reason=result.unavailable_reason or result.status,
            )
        if result.verdict == "PASS":
            return ReviewCheckpointDecision(action="continue", result=result)
        if result.verdict == "REVISE":
            if request.attempt < self.config.max_revisions:
                return ReviewCheckpointDecision(action="revise", result=result)
            return ReviewCheckpointDecision(
                action="ask_user",
                result=result,
                reason="revision_limit_reached",
            )
        if result.verdict == "ASK_USER":
            return ReviewCheckpointDecision(action="ask_user", result=result)
        if result.verdict == "BLOCK":
            return ReviewCheckpointDecision(action="block", result=result)
        return ReviewCheckpointDecision(
            action=self.config.failure_policy,
            result=result,
            reason="invalid_verdict",
        )

    def _emit_decision(
        self,
        request: ReviewRequest,
        decision: ReviewCheckpointDecision,
    ) -> None:
        if self._emit is None:
            return
        result = decision.result
        event = {
            "checkpoint_id": request.checkpoint_id,
            "session_id": self.session_id,
            "phase": request.phase,
            "attempt": request.attempt,
            "action": decision.action,
            "status": result.status if result is not None else "skipped",
            "verdict": result.verdict if result is not None else None,
            "reason": decision.reason,
            "actual_route": (
                dict(result.actual_route)
                if result is not None and result.actual_route is not None
                else None
            ),
            "failure_policy": self.config.failure_policy,
        }
        try:
            self._emit(event)
        except Exception:
            # Observability must not change checkpoint semantics.
            pass


class ReviewCheckpointRuntime:
    """Build bounded requests and delegate state transitions to the controller."""

    def __init__(
        self,
        *,
        session_id: str,
        route: ReviewCheckpointRoute,
        controller: ReviewCheckpointController,
    ) -> None:
        if not route.provider.strip() or not route.model.strip():
            raise ValueError("an exact review provider and model are required")
        if route.timeout_seconds <= 0:
            raise ValueError("review timeout must be positive")
        if controller.session_id != session_id:
            raise ValueError("runtime and controller sessions must match")
        self.session_id = session_id
        self.route = route
        self.controller = controller

    @property
    def enabled(self) -> bool:
        return self.controller.config.enabled

    def evaluate(
        self,
        *,
        checkpoint_id: str,
        phase: str,
        attempt: int,
        objective: str,
        constraints: tuple[str, ...],
        candidate: dict[str, Any],
        main_provider: str,
        main_model: str,
    ) -> ReviewCheckpointDecision:
        request = ReviewRequest(
            checkpoint_id=checkpoint_id,
            session_id=self.session_id,
            phase=phase,
            attempt=attempt,
            objective=objective,
            constraints=constraints,
            candidate=candidate,
            provider=self.route.provider,
            model=self.route.model,
            main_provider=main_provider,
            main_model=main_model,
            require_distinct_from_main=self.route.require_distinct_from_main,
            timeout_seconds=self.route.timeout_seconds,
        )
        return self.controller.evaluate(request)

    def cancel(self, checkpoint_id: str) -> None:
        self.controller.cancel(checkpoint_id)


def create_review_checkpoint_runtime(
    *,
    session_id: str,
    provider: str,
    model: str,
    enabled: bool = True,
    max_revisions: int = 2,
    failure_policy: str = "continue",
    require_distinct_from_main: bool = True,
    timeout_seconds: float = 60.0,
    emit: Callable[[dict], None] | None = None,
    run_review_fn: Callable[[ReviewRequest], ReviewResult] | None = None,
) -> ReviewCheckpointRuntime:
    """Create the trusted per-session runtime used by core checkpoint seams."""

    if run_review_fn is None:
        from agent.review_backend import (
            complete_subscription_review,
            resolve_subscription_review_route,
        )
        from agent.review_runner import run_review

        run_review_fn = partial(
            run_review,
            resolve_route=resolve_subscription_review_route,
            complete=complete_subscription_review,
        )
    config = ReviewCheckpointConfig(
        enabled=enabled,
        max_revisions=max_revisions,
        failure_policy=failure_policy,
    )
    controller = ReviewCheckpointController(
        session_id=session_id,
        config=config,
        run_review=run_review_fn,
        emit=emit,
    )
    return ReviewCheckpointRuntime(
        session_id=session_id,
        route=ReviewCheckpointRoute(
            provider=provider,
            model=model,
            require_distinct_from_main=require_distinct_from_main,
            timeout_seconds=timeout_seconds,
        ),
        controller=controller,
    )


def _objective_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()[:8_000] or "Review the current turn"
    return "Review the current turn"


def _argument_keys(raw: Any) -> list[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "{}")
        except (TypeError, ValueError):
            return []
    if isinstance(raw, dict):
        return sorted(str(key) for key in raw)[:100]
    return []


def review_tool_checkpoint(
    agent: Any,
    *,
    turn_id: str,
    attempt: int,
    user_message: Any,
    assistant_content: str,
    tool_calls: list[Any],
) -> ReviewCheckpointDecision:
    """Review an action bundle immediately before its handlers may execute."""

    runtime = getattr(agent, "review_checkpoint_runtime", None)
    if not isinstance(runtime, ReviewCheckpointRuntime) or not runtime.enabled:
        return ReviewCheckpointDecision(action="continue")

    from agent.tool_result_classification import tool_may_have_side_effect

    actions = []
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        name = str(getattr(function, "name", "") or "")
        actions.append({
            "tool": name,
            "effect": "state_change" if tool_may_have_side_effect(name) else "read",
            "argument_keys": _argument_keys(getattr(function, "arguments", None)),
            "redacted_arguments": {},
        })
    return runtime.evaluate(
        checkpoint_id=f"{turn_id}:plan:{attempt}",
        phase="plan",
        attempt=attempt,
        objective=_objective_text(user_message),
        constraints=("Do not execute tools during review.",),
        candidate={
            "summary": str(assistant_content or "")[:16_000],
            "actions": actions,
        },
        main_provider=str(getattr(agent, "provider", "") or ""),
        main_model=str(getattr(agent, "model", "") or ""),
    )


def review_final_checkpoint(
    agent: Any,
    *,
    turn_id: str,
    attempt: int,
    user_message: Any,
    final_response: str,
    evidence: list[str] | None = None,
) -> ReviewCheckpointDecision:
    """Review a candidate answer before any final-response surface receives it."""

    runtime = getattr(agent, "review_checkpoint_runtime", None)
    if not isinstance(runtime, ReviewCheckpointRuntime) or not runtime.enabled:
        return ReviewCheckpointDecision(action="continue")
    return runtime.evaluate(
        checkpoint_id=f"{turn_id}:final:{attempt}",
        phase="final",
        attempt=attempt,
        objective=_objective_text(user_message),
        constraints=("Review only the candidate answer and bounded evidence.",),
        candidate={
            "summary": str(final_response or "")[:32_000],
            "evidence": [str(item)[:2_000] for item in (evidence or [])[:20]],
        },
        main_provider=str(getattr(agent, "provider", "") or ""),
        main_model=str(getattr(agent, "model", "") or ""),
    )


def configure_review_output_hold(agent: Any) -> bool:
    """Enable candidate buffering only while this session's runtime is enabled."""

    runtime = getattr(agent, "review_checkpoint_runtime", None)
    enabled = isinstance(runtime, ReviewCheckpointRuntime) and runtime.enabled
    agent._review_hold_output = enabled
    agent._review_held_stream_chunks = []
    agent._review_held_stream_chars = 0
    agent._review_held_stream_overflow = False
    agent._review_release_interim_once = False
    return enabled


def discard_review_output(agent: Any) -> None:
    """Discard a candidate that did not pass its checkpoint."""

    agent._review_held_stream_chunks = []
    agent._review_held_stream_chars = 0
    agent._review_held_stream_overflow = False
    agent._review_release_interim_once = False


def release_review_output(agent: Any) -> bool:
    """Release already-scrubbed candidate deltas after a checkpoint passes."""

    if not getattr(agent, "_review_hold_output", False):
        return True
    if getattr(agent, "_review_held_stream_overflow", False):
        discard_review_output(agent)
        return False

    held = list(getattr(agent, "_review_held_stream_chunks", []) or [])
    callbacks = [
        callback
        for callback in (
            getattr(agent, "stream_delta_callback", None),
            getattr(agent, "_stream_callback", None),
        )
        if callback is not None
    ]
    for chunk in held:
        delivered = False
        for callback in callbacks:
            try:
                callback(chunk)
                delivered = True
            except Exception:
                pass
        try:
            from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

            enqueue_plugin_stream_hook(
                "on_stream_delta",
                **agent._stream_hook_base_payload(),
                delta=chunk,
                kind="text",
            )
        except Exception:
            pass
        if delivered:
            try:
                agent._record_streamed_assistant_text(chunk)
            except Exception:
                pass

    agent._review_held_stream_chunks = []
    agent._review_held_stream_chars = 0
    agent._review_release_interim_once = True
    return True
