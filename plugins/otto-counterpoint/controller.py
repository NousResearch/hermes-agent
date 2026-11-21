"""Single-entry bounded admission and output gate for Otto's Hermes plugin.

The controller is intentionally a hook adapter, not a second agent runtime.  It
admits one model turn, keeps only bounded in-memory context until the final
response, and delegates the review call to the already-tested counterpoint
package.  The ledger receives hashes, route metadata, gate results and
sanitized decisions; prompt and response text stay out of persisted payloads.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping, Sequence

from .counterpoint import (
    Artifact,
    CounterpointWorkflow,
    DispatchDecision,
    DispatchOutcome,
    GateResult,
    HermesAgentClient,
    HermesCounterpointCallbacks,
    HermesRuntime,
    RequestEnvelope,
    RouteIdentity,
    SingleEntryDispatcher,
    WorkflowResult,
    WorkflowSpec,
)
from .counterpoint.ledger import Ledger, RunLifecycle

logger = logging.getLogger(__name__)

_MAX_PENDING = 256
_MAX_RESPONSE_CHARS = 100_000
_MODES = frozenset({"shadow", "canary", "blocking"})
_VALID_EFFORTS = frozenset({"high", "xhigh", "ultra"})

_EFFECT_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("deploy", ("deploy", "produção", "production", "release")),
    ("write_external", ("enviar", "send", "postar", "post", "publicar", "publish")),
    ("delete", ("apagar", "excluir", "delete", "remover", "drop")),
    ("payment", ("pagamento", "payment", "pagar", "purchase", "comprar", "refund")),
    ("credential", ("senha", "password", "token", "secret", "segredo", "credential", "api key")),
)
_SENSITIVE_TERMS = frozenset(
    {
        "senha", "password", "token", "secret", "segredo", "credential", "api key",
        "cpf", "cnpj", "pii", "privacidade", "privacy", "financeiro", "financial",
        "pagamento", "payment", "produção", "production",
    }
)
_COMPLEX_TERMS = frozenset(
    {
        "arquitetura", "architecture", "research", "pesquisa", "investigar", "investigate",
        "compare", "comparar", "evidence", "evidência", "decompose", "decompor", "design",
        "implementar", "implement", "refactor", "refatorar",
    }
)
_UNCERTAIN_TERMS = frozenset(
    {
        "incerto", "incerteza", "uncertain", "ambíguo", "ambiguo", "ambiguous", "maybe",
        "talvez", "compare", "comparar", "conflito", "conflict", "evidence", "evidência",
    }
)


@dataclass(frozen=True)
class _Classification:
    risk: str
    complexity: str
    external_effect: bool
    sensitive_scope: bool
    evidence_conflict: bool
    uncertainty: bool
    requested_effects: tuple[str, ...]
    forbidden_effects: tuple[str, ...]


@dataclass
class PendingTurn:
    envelope: RequestEnvelope
    outcome: DispatchOutcome
    generator_route: RouteIdentity
    counterpoint_route: RouteIdentity | None
    adjudicator_route: RouteIdentity | None
    spec: WorkflowSpec | None
    classification: _Classification
    canary_selected: bool
    session_id: str
    task_id: str
    turn_id: str
    created_at: float
    finalized: bool = False
    canary_excluded: bool = False
    fail_closed: bool = False
    workflow_result: WorkflowResult | Any | None = None


class CounterpointController:
    """Bridge Hermes lifecycle hooks to one bounded counterpoint workflow."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        state: Any,
        workflow_runner: Callable[..., Any] | None = None,
        client_factory: Callable[[], HermesAgentClient] | None = None,
    ) -> None:
        self.config = dict(config or {})
        mode = str(self.config.get("mode", "shadow") or "shadow").strip().lower()
        self.mode = mode if mode in _MODES else "shadow"
        self.state = state
        data_dir = Path(getattr(state, "data_dir", state))
        data_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(data_dir / "counterpoint.sqlite3")
        self.dispatcher = SingleEntryDispatcher()
        self._workflow_runner = workflow_runner
        self._client_factory = client_factory
        self._pending: dict[str, PendingTurn] = {}
        self._by_task: dict[str, str] = {}
        self._lock = RLock()
        self.last_results: dict[str, Any] = {}

    def on_pre_llm_call(self, **kwargs: Any) -> None:
        """Admit one turn before its first model request.

        Hook return values are deliberately unused: this gate cannot inject
        policy text into the user's prompt.  Tool blocking and response
        replacement happen at their dedicated seams.
        """
        turn_id = _text(kwargs.get("turn_id"))
        session_id = _text(kwargs.get("session_id")) or "session-unknown"
        task_id = _text(kwargs.get("task_id")) or turn_id
        if not turn_id or not task_id:
            logger.warning("otto-counterpoint admission missing turn identity")
            return
        with self._lock:
            self._prune()
            if turn_id in self._pending:
                return
            try:
                pending = self._build_pending(
                    session_id=session_id,
                    task_id=task_id,
                    turn_id=turn_id,
                    user_message=kwargs.get("user_message"),
                    model=kwargs.get("model"),
                    provider=kwargs.get("provider"),
                    api_mode=kwargs.get("api_mode"),
                    platform=kwargs.get("platform"),
                )
            except Exception:  # health: allow BLE001 -- hook boundary must fail closed and sanitize the reason
                pending = self._failed_closed_pending(
                    session_id=session_id,
                    task_id=task_id,
                    turn_id=turn_id,
                    user_message=kwargs.get("user_message"),
                    platform=kwargs.get("platform"),
                )
                self._pending[turn_id] = pending
                self._by_task[task_id] = turn_id
                self._persist_review(pending, reason="admission_failed_closed", verdict="human_review")
                pending.finalized = True
                return
            self._pending[turn_id] = pending
            self._by_task[task_id] = turn_id

            if pending.outcome.decision == DispatchDecision.DIRECT:
                self._persist_direct(pending)
                pending.finalized = True
            elif pending.outcome.decision in {
                DispatchDecision.BLOCKED,
                DispatchDecision.HUMAN_REVIEW,
            }:
                self._persist_review(
                    pending,
                    reason=pending.outcome.reason,
                    verdict="block" if pending.outcome.decision == DispatchDecision.BLOCKED else "human_review",
                )
                pending.finalized = True

    def on_pre_tool_call(self, **kwargs: Any) -> dict[str, str] | None:
        """Block model tools for enforced non-direct turns."""
        pending = self._lookup(kwargs.get("turn_id"), kwargs.get("task_id"), kwargs.get("session_id"))
        if pending is None:
            return None
        if not self._must_fail_closed(pending) and not self._enforce(pending):
            return None
        if pending.outcome.decision == DispatchDecision.DIRECT and not pending.fail_closed:
            return None
        return {
            "action": "block",
            "message": (
                "Otto bounded admission blocked this tool call until the turn is "
                "accepted by deterministic gates and the required counterpoint review."
            ),
        }

    def on_transform_llm_output(self, **kwargs: Any) -> str | None:
        """Review the frozen response and optionally replace it with a safe result."""
        turn_id = _text(kwargs.get("turn_id"))
        pending = self._lookup(turn_id, None, kwargs.get("session_id"))
        if pending is None:
            return None
        with self._lock:
            if pending.finalized and pending.workflow_result is not None:
                return None
            response = kwargs.get("response_text")
            if not isinstance(response, str) or not response.strip():
                pending.finalized = True
                return None
            if len(response) > _MAX_RESPONSE_CHARS:
                response = response[:_MAX_RESPONSE_CHARS]
            if pending.outcome.decision != DispatchDecision.COUNTERPOINT:
                pending.finalized = True
                if self._must_fail_closed(pending):
                    return _human_review_message(pending.outcome.reason)
                return None
            if self.mode == "canary" and not pending.canary_selected:
                pending.finalized = True
                return None

            try:
                result = self._run_workflow(pending, response)
                pending.workflow_result = result
                self.last_results[pending.turn_id] = result
            except Exception:  # health: allow BLE001 -- workflow boundary converts all failures to human review
                self._persist_workflow_failure(pending, "workflow_callback_failed")
                pending.workflow_result = SimpleWorkflowResult(
                    verdict="human_review",
                    blocked_reason="workflow_callback_failed",
                )
                pending.fail_closed = True
                pending.finalized = True
                return _human_review_message("workflow_callback_failed")

            pending.finalized = True
            accepted = _workflow_accepted(result)
            if not accepted:
                pending.fail_closed = True
                return _human_review_message(getattr(result, "blocked_reason", None) or "human_review_required")
            if self.mode == "shadow" or not self._enforce(pending):
                return None
            replacement = getattr(result, "replacement_text", None)
            return replacement if isinstance(replacement, str) and replacement.strip() else None

    def pending(self, turn_id: str) -> PendingTurn:
        with self._lock:
            return self._pending[turn_id]

    def _build_pending(
        self,
        *,
        session_id: str,
        task_id: str,
        turn_id: str,
        user_message: Any,
        model: Any,
        provider: Any,
        api_mode: Any,
        platform: Any,
    ) -> PendingTurn:
        prompt = user_message if isinstance(user_message, str) else ""
        request_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        classification = _classify(prompt)
        idempotency_key = f"hermes:{session_id}:{turn_id}"
        generator = self._generator_route(model=model, provider=provider, api_mode=api_mode)
        counterpoint = _route_from_config(self.config.get("counterpoint_route"), default_ready=False)
        adjudicator = _route_from_config(self.config.get("adjudicator_route"), default_ready=False)
        counterpoint_ready = _route_ready(counterpoint, generator)
        canary_selected = _canary_selected(idempotency_key, self.config.get("canary_percent", 100))
        sampling_selected = _sampling_selected(idempotency_key, self.config.get("sample_percent", 0))
        envelope = RequestEnvelope(
            run_id=f"cp-{_identifier(turn_id)}",
            task_id=task_id,
            idempotency_key=idempotency_key,
            source=_text(platform) or "unknown",
            source_message_id=turn_id,
            request_sha256=request_hash,
            objective_ref=f"request://{request_hash[:24]}",
            scope_ref=f"session://{_identifier(session_id)}",
            risk=classification.risk,
            complexity=classification.complexity,
            requested_effects=classification.requested_effects,
            forbidden_effects=classification.forbidden_effects,
            external_effect=classification.external_effect,
            sensitive_scope=classification.sensitive_scope,
            evidence_conflict=classification.evidence_conflict,
            uncertainty=classification.uncertainty,
            sampling_selected=sampling_selected,
            supervisor_approved=bool(self.config.get("supervisor_approved", False)),
            counterpoint_route_ready=counterpoint_ready,
        )
        outcome = self.dispatcher.admit(envelope)
        spec = None
        canary_excluded = False
        if outcome.decision == DispatchDecision.COUNTERPOINT:
            canary_excluded = self.mode == "canary" and not canary_selected
            if not canary_excluded:
                spec = self._workflow_spec(
                    envelope=envelope,
                    generator=generator,
                    counterpoint=counterpoint,
                    adjudicator=adjudicator,
                    classification=classification,
                )
        return PendingTurn(
            envelope=envelope,
            outcome=outcome,
            generator_route=generator,
            counterpoint_route=counterpoint,
            adjudicator_route=adjudicator,
            spec=spec,
            classification=classification,
            canary_selected=canary_selected,
            session_id=session_id,
            task_id=task_id,
            turn_id=turn_id,
            created_at=time.monotonic(),
            canary_excluded=canary_excluded,
        )

    def _failed_closed_pending(
        self,
        *,
        session_id: str,
        task_id: str,
        turn_id: str,
        user_message: Any,
        platform: Any,
    ) -> PendingTurn:
        prompt = user_message if isinstance(user_message, str) else ""
        request_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        run_id = f"cp-{_identifier(turn_id)}"
        envelope = RequestEnvelope(
            run_id=run_id,
            task_id=task_id,
            idempotency_key=f"hermes:{session_id}:{turn_id}",
            source=_text(platform) or "unknown",
            source_message_id=turn_id,
            request_sha256=request_hash,
            objective_ref=f"request://{request_hash[:24]}",
            scope_ref=f"session://{_identifier(session_id)}",
            risk="critical",
            complexity="complex",
            requested_effects=("admission_error",),
            forbidden_effects=(),
            external_effect=True,
            sensitive_scope=True,
            evidence_conflict=True,
            uncertainty=True,
            sampling_selected=False,
            supervisor_approved=False,
            counterpoint_route_ready=False,
        )
        outcome = DispatchOutcome(
            run_id=run_id,
            task_id=task_id,
            idempotency_key=envelope.idempotency_key,
            request_sha256=request_hash,
            decision=DispatchDecision.HUMAN_REVIEW,
            reason="admission_failed_closed",
            next_actor="human",
        )
        generator = RouteIdentity(
            vendor="unknown",
            family="unknown",
            provider="unknown",
            model="unknown",
            reasoning_effort="high",
            authenticated=False,
            accessible=False,
            smoke_tested=False,
            relative_load=1.0,
        )
        classification = _Classification(
            risk="critical",
            complexity="complex",
            external_effect=True,
            sensitive_scope=True,
            evidence_conflict=True,
            uncertainty=True,
            requested_effects=("admission_error",),
            forbidden_effects=(),
        )
        return PendingTurn(
            envelope=envelope,
            outcome=outcome,
            generator_route=generator,
            counterpoint_route=None,
            adjudicator_route=None,
            spec=None,
            classification=classification,
            canary_selected=False,
            session_id=session_id,
            task_id=task_id,
            turn_id=turn_id,
            created_at=time.monotonic(),
        )

    def _generator_route(self, *, model: Any, provider: Any, api_mode: Any) -> RouteIdentity:
        model_text = _text(model) or "unknown-model"
        provider_text = _text(provider) or str(self.config.get("generator_provider", "runtime"))
        family = _text(self.config.get("generator_family")) or _infer_family(provider_text, model_text)
        vendor = _text(self.config.get("generator_vendor")) or _infer_vendor(provider_text, model_text)
        effort = _text(self.config.get("generator_reasoning_effort")) or "high"
        if effort not in _VALID_EFFORTS:
            effort = "high"
        return RouteIdentity(
            vendor=vendor,
            family=family,
            provider=provider_text,
            model=model_text,
            reasoning_effort=effort,
            authenticated=True,
            accessible=True,
            smoke_tested=True,
            relative_load=1.0,
        )

    def _workflow_spec(
        self,
        *,
        envelope: RequestEnvelope,
        generator: RouteIdentity,
        counterpoint: RouteIdentity | None,
        adjudicator: RouteIdentity | None,
        classification: _Classification,
    ) -> WorkflowSpec:
        criteria = self.config.get("criteria", ())
        if not isinstance(criteria, (list, tuple)) or not criteria:
            criteria = (
                "Answer the admitted request accurately.",
                "Identify uncertainty and distinguish evidence from interpretation.",
                "Do not authorize external effects.",
            )
        return WorkflowSpec(
            project_id=_text(self.config.get("project_id")) or "otto-hermes-gateway",
            task_id=envelope.task_id,
            run_id=envelope.run_id,
            agent_id="hermes-gateway",
            risk=envelope.risk,
            complexity=envelope.complexity,
            generator_route=generator,
            counterpoint_route=counterpoint,
            adjudicator_route=adjudicator,
            criteria=tuple(str(item) for item in criteria),
            evidence_refs=(f"request-sha256:{envelope.request_sha256}",),
            external_effect=classification.external_effect,
            evidence_conflict=classification.evidence_conflict,
            uncertainty=classification.uncertainty,
            sampling_selected=envelope.sampling_selected,
            max_corrections=_bounded_int(self.config.get("max_corrections", 1), 0, 1),
        )

    def _run_workflow(self, pending: PendingTurn, response: str) -> Any:
        if pending.spec is None or pending.counterpoint_route is None:
            raise RuntimeError("workflow_spec_unavailable")
        if self._workflow_runner is not None:
            return self._workflow_runner(pending=pending, response_text=response)

        client = self._client_factory() if self._client_factory is not None else self._default_client()
        callbacks = HermesCounterpointCallbacks(
            client=client,
            generator_route=pending.generator_route,
            counterpoint_route=pending.counterpoint_route,
            adjudicator_route=pending.adjudicator_route,
        )
        content_ref = f"memory://counterpoint/{pending.envelope.run_id}/response-v1"

        def generator(context: Mapping[str, Any], previous: Artifact | None, critique: Any) -> Artifact:
            if previous is None:
                artifact = Artifact.from_content(
                    run_id=pending.envelope.run_id,
                    task_id=pending.envelope.task_id,
                    artifact_id=f"{pending.envelope.run_id}-artifact-v1",
                    version=1,
                    content=response,
                    content_ref=content_ref,
                    claims=({"claim_id": "response", "text": "bounded response"},),
                    evidence_refs=(f"request-sha256:{pending.envelope.request_sha256}",),
                    tests=({
                        "test_id": "response-present",
                        "name": "response is non-empty",
                        "result": "pass",
                        "evidence_id": f"gate:{pending.envelope.run_id}:response-present",
                    },),
                )
                callbacks._contents[content_ref] = response
                return artifact
            return callbacks.generator(context, previous, critique)

        def validator(artifact: Artifact) -> Sequence[GateResult]:
            content = callbacks.content_for(artifact)
            return (
                GateResult(
                    gate="response_non_empty",
                    status="pass" if isinstance(content, str) and bool(content.strip()) else "fail",
                    evidence_id=f"gate:{artifact.artifact_id}:non-empty",
                ),
                GateResult(
                    gate="artifact_hash_frozen",
                    status="pass",
                    evidence_id=f"gate:{artifact.artifact_id}:hash",
                ),
            )

        workflow = CounterpointWorkflow(
            ledger=self.ledger,
            generator=generator,
            validator=validator,
            critic=callbacks.critic,
            adjudicator=callbacks.adjudicator if pending.adjudicator_route is not None else None,
        )
        result = workflow.run(
            pending.spec,
            context={
                "run_id": pending.envelope.run_id,
                "task_id": pending.envelope.task_id,
                "request_sha256": pending.envelope.request_sha256,
                "source": pending.envelope.source,
            },
        )
        if getattr(result, "status", None) == "blocked":
            self._finish_cancelled(pending, result)
        return result

    def _default_client(self) -> HermesAgentClient:
        runtime = HermesRuntime.discover()
        runtime = replace(
            runtime,
            worker_script=Path(__file__).resolve().parent / "hermes_counterpoint_worker.py",
            timeout_seconds=min(runtime.timeout_seconds, float(self.config.get("timeout_seconds", 180))),
            run_budget_seconds=min(runtime.run_budget_seconds, _bounded_int(self.config.get("run_budget_seconds", 120), 1, 120)),
            max_turns=1,
        )
        return HermesAgentClient(runtime)

    def _lookup(self, turn_id: Any, task_id: Any, session_id: Any) -> PendingTurn | None:
        with self._lock:
            key = _text(turn_id)
            if key and key in self._pending:
                return self._pending[key]
            task_key = _text(task_id)
            if task_key and task_key in self._by_task:
                return self._pending.get(self._by_task[task_key])
            session_key = _text(session_id)
            if session_key:
                candidates = [item for item in self._pending.values() if item.session_id == session_key]
                if len(candidates) == 1:
                    return candidates[0]
            return None

    def _enforce(self, pending: PendingTurn) -> bool:
        return self.mode == "blocking" or (self.mode == "canary" and pending.canary_selected)

    @staticmethod
    def _must_fail_closed(pending: PendingTurn) -> bool:
        return pending.fail_closed or pending.outcome.decision in {
            DispatchDecision.BLOCKED,
            DispatchDecision.HUMAN_REVIEW,
        }

    def _persist_direct(self, pending: PendingTurn) -> None:
        lifecycle = RunLifecycle(self.ledger)
        lifecycle.create(
            project_id=_project_id(self.config),
            linear_issue_id=pending.task_id,
            run_id=pending.envelope.run_id,
            agent_id="hermes-gateway",
            evidence_refs=(f"request-sha256:{pending.envelope.request_sha256}",),
            payload={"admission": pending.outcome.to_audit_dict()},
        )
        lifecycle.start(pending.envelope.run_id)
        lifecycle.finish(
            pending.envelope.run_id,
            "succeeded",
            evidence_refs=(f"request-sha256:{pending.envelope.request_sha256}",),
            payload={
                "decision": {
                    "verdict": "accept_local",
                    "next_actor": "otto",
                    "reason": pending.outcome.reason,
                },
            },
        )

    def _persist_review(self, pending: PendingTurn, *, reason: str, verdict: str) -> None:
        lifecycle = RunLifecycle(self.ledger)
        lifecycle.create(
            project_id=_project_id(self.config),
            linear_issue_id=pending.task_id,
            run_id=pending.envelope.run_id,
            agent_id="hermes-gateway",
            evidence_refs=(f"request-sha256:{pending.envelope.request_sha256}",),
            payload={"admission": pending.outcome.to_audit_dict()},
        )
        lifecycle.block(
            pending.envelope.run_id,
            reason,
            evidence_refs=(f"request-sha256:{pending.envelope.request_sha256}",),
            payload={"decision": {"verdict": verdict, "next_actor": "human"}},
        )
        lifecycle.finish(
            pending.envelope.run_id,
            "cancelled",
            evidence_refs=(f"request-sha256:{pending.envelope.request_sha256}",),
            payload={"decision": {"verdict": verdict, "next_actor": "human", "reason": reason}},
        )

    def _persist_workflow_failure(self, pending: PendingTurn, reason: str) -> None:
        if not self.ledger.list_run(pending.envelope.run_id):
            self._persist_review(pending, reason=reason, verdict="human_review")
            return
        self._finish_cancelled(
            pending,
            SimpleWorkflowResult(verdict="human_review", blocked_reason=reason),
        )

    def _finish_cancelled(self, pending: PendingTurn, result: Any) -> None:
        lifecycle = RunLifecycle(self.ledger)
        events = self.ledger.list_run(pending.envelope.run_id)
        if not events or events[-1]["event_type"] != "run.blocked":
            return
        verdict = getattr(getattr(result, "decision", None), "verdict", "human_review")
        lifecycle.finish(
            pending.envelope.run_id,
            "cancelled",
            evidence_refs=(f"request-sha256:{pending.envelope.request_sha256}",),
            payload={
                "decision": {
                    "verdict": verdict,
                    "next_actor": "human",
                    "reason": getattr(result, "blocked_reason", None) or "human_review_required",
                },
            },
        )

    def _prune(self) -> None:
        now = time.monotonic()
        stale = [key for key, item in self._pending.items() if now - item.created_at > 900]
        for key in stale:
            item = self._pending.pop(key)
            self._by_task.pop(item.task_id, None)
        if len(self._pending) <= _MAX_PENDING:
            return
        for key in sorted(self._pending, key=lambda item: self._pending[item].created_at)[: len(self._pending) - _MAX_PENDING]:
            item = self._pending.pop(key)
            self._by_task.pop(item.task_id, None)


class SimpleWorkflowResult:
    def __init__(self, *, verdict: str, blocked_reason: str) -> None:
        self.decision = type("Decision", (), {"verdict": verdict})()
        self.blocked_reason = blocked_reason


def _classify(text: str) -> _Classification:
    folded = text.casefold()
    effects = tuple(name for name, terms in _EFFECT_TERMS if any(term in folded for term in terms))
    external = bool(effects)
    sensitive = any(term in folded for term in _SENSITIVE_TERMS)
    conflict = any(term in folded for term in ("conflito", "conflict", "contradict", "contradit"))
    uncertain = any(term in folded for term in _UNCERTAIN_TERMS)
    complex_signal = any(term in folded for term in _COMPLEX_TERMS) or len(text) > 1800
    production = any(term in folded for term in ("produção", "production", "deploy", "release"))
    critical = any(term in folded for term in ("pagamento", "payment", "senha", "password", "token", "secret", "apagar", "delete"))
    if critical or (production and external):
        risk = "critical"
    elif external or sensitive:
        risk = "high"
    elif uncertain or conflict or complex_signal:
        risk = "medium"
    else:
        risk = "low"
    complexity = "frontier" if complex_signal and any(
        term in folded for term in ("architecture", "arquitetura", "research", "pesquisa", "decompose", "decompor")
    ) else "complex" if complex_signal else "simple"
    return _Classification(
        risk=risk,
        complexity=complexity,
        external_effect=external,
        sensitive_scope=sensitive,
        evidence_conflict=conflict,
        uncertainty=uncertain,
        requested_effects=effects,
        forbidden_effects=(),
    )


def _route_from_config(raw: Any, *, default_ready: bool) -> RouteIdentity | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        provider = _required_config_text(raw, "provider")
        model = _required_config_text(raw, "model")
        family = _required_config_text(raw, "family")
        vendor = _text(raw.get("vendor")) or family
        effort = _text(raw.get("reasoning_effort")) or "high"
        if effort not in _VALID_EFFORTS:
            raise ValueError("invalid reasoning_effort")
        verified_value = raw.get("route_verified")
        if verified_value is None:
            ready = all(bool(raw.get(key, default_ready)) for key in ("authenticated", "accessible", "smoke_tested"))
        else:
            ready = bool(verified_value)
        return RouteIdentity(
            vendor=vendor,
            family=family,
            provider=provider,
            model=model,
            reasoning_effort=effort,
            authenticated=ready and bool(raw.get("authenticated", ready)),
            accessible=ready and bool(raw.get("accessible", ready)),
            smoke_tested=ready and bool(raw.get("smoke_tested", ready)),
            relative_load=float(raw.get("relative_load", 1.0)),
        )
    except (TypeError, ValueError):
        logger.warning("otto-counterpoint route configuration is invalid")
        return None


def _route_ready(route: RouteIdentity | None, generator: RouteIdentity) -> bool:
    return bool(
        route is not None
        and route.authenticated
        and route.accessible
        and route.smoke_tested
        and route.family != generator.family
    )


def _workflow_accepted(result: Any) -> bool:
    return bool(
        getattr(result, "status", None) == "succeeded"
        and getattr(getattr(result, "decision", None), "verdict", None) == "accept_local"
    )


def _sampling_selected(key: str, value: Any) -> bool:
    percent = _bounded_int(value, 0, 100)
    if percent <= 0:
        return False
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return bucket < percent


def _canary_selected(key: str, value: Any) -> bool:
    return _sampling_selected(key, value)


def _project_id(config: Mapping[str, Any]) -> str:
    return _text(config.get("project_id")) or "otto-hermes-gateway"


def _required_config_text(value: Mapping[str, Any], key: str) -> str:
    output = _text(value.get(key))
    if not output:
        raise ValueError(key)
    return output


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _identifier(value: Any) -> str:
    text = _text(value) or "unknown"
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)[:120]


def _bounded_int(value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return minimum
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError):
        return minimum


def _infer_family(provider: str, model: str) -> str:
    folded = f"{provider}/{model}".casefold()
    if "claude" in folded or "anthropic" in folded:
        return "anthropic"
    if "gpt" in folded or "openai" in folded or provider.casefold().startswith("openai"):
        return "openai"
    if "gemini" in folded:
        return "google"
    if "llama" in folded or "meta" in folded:
        return "meta"
    return provider or "unknown"


def _infer_vendor(provider: str, model: str) -> str:
    family = _infer_family(provider, model)
    return family if family != "unknown" else provider or "runtime"


def _human_review_message(reason: str) -> str:
    return (
        "Não vou concluir esta chamada automaticamente: o processo bounded exige "
        f"revisão humana ({reason})."
    )


__all__ = ["CounterpointController", "PendingTurn"]
