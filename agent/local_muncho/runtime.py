"""Local Muncho runtime coordinator and hook helpers."""

from __future__ import annotations

import contextvars
import json
import os
import re
import uuid
from typing import Any, Mapping, Sequence

from agent.local_muncho.brain import CanonicalBrain
from agent.local_muncho.evidence import validate_final_output as _validate_final_output
from agent.local_muncho.policy import classify_tool_action
from agent.local_muncho.types import (
    ApprovalRecord,
    AuditEvent,
    CodexTaskCreate,
    CodexTaskPatch,
    EvidenceValidationResult,
    GuardDecision,
    HeartbeatPayload,
    HeartbeatResult,
    KnowledgeContext,
    LeaseAssertion,
    LockRequest,
    RuntimeContext,
    RuntimeEvent,
    ToolEvidence,
    WorkerContract,
    utc_ts,
)


_current_context: contextvars.ContextVar[RuntimeContext | None] = contextvars.ContextVar(
    "local_muncho_runtime_context",
    default=None,
)
_current_brain: contextvars.ContextVar[CanonicalBrain | None] = contextvars.ContextVar(
    "local_muncho_canonical_brain",
    default=None,
)

_SENSITIVE_KEY_RE = re.compile(r"(token|secret|password|api[_-]?key|dsn|credential)", re.IGNORECASE)
_SENSITIVE_VALUE_RE = re.compile(
    r"(\bBearer\s+[A-Za-z0-9._~+/=-]{12,}|\bsk-[A-Za-z0-9]{16,})",
    re.IGNORECASE,
)


def set_current_runtime_context(context: RuntimeContext | None):
    return _current_context.set(context)


def reset_current_runtime_context(token) -> None:
    _current_context.reset(token)


def set_current_canonical_brain(brain: CanonicalBrain | None):
    return _current_brain.set(brain)


def reset_current_canonical_brain(token) -> None:
    _current_brain.reset(token)


def _runtime_cfg(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    block = config.get("muncho_runtime")
    return block if isinstance(block, Mapping) else config


def _load_runtime_config() -> Mapping[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        return load_config_readonly().get("muncho_runtime", {}) or {}
    except Exception:
        return {}


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return _SENSITIVE_VALUE_RE.sub("***", value)
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    return value


def redact_mapping(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if _SENSITIVE_KEY_RE.search(str(key)):
            redacted[str(key)] = "***"
        else:
            redacted[str(key)] = _redact_value(value)
    return redacted


def context_from_env() -> RuntimeContext:
    return RuntimeContext(
        lane=os.getenv("HERMES_MUNCHO_RUNTIME_LANE") or None,
        session_id=os.getenv("HERMES_SESSION_ID", ""),
        platform=os.getenv("HERMES_SESSION_PLATFORM", ""),
        user_id=os.getenv("HERMES_SESSION_USER_ID", ""),
        chat_id=os.getenv("HERMES_SESSION_CHAT_ID", ""),
        thread_id=os.getenv("HERMES_SESSION_THREAD_ID", ""),
        message_id=os.getenv("HERMES_SESSION_MESSAGE_ID", ""),
        profile=os.getenv("HERMES_PROFILE", ""),
    )


def context_from_agent(agent: Any) -> RuntimeContext:
    explicit = getattr(agent, "_local_muncho_context", None)
    if isinstance(explicit, RuntimeContext):
        return explicit
    env_context = context_from_env()
    return RuntimeContext(
        lane=getattr(agent, "local_muncho_lane", None) or env_context.lane,
        session_id=getattr(agent, "session_id", "") or env_context.session_id,
        platform=getattr(agent, "platform", "") or env_context.platform,
        user_id=getattr(agent, "user_id", "") or env_context.user_id,
        chat_id=getattr(agent, "chat_id", "") or env_context.chat_id,
        thread_id=str(getattr(agent, "thread_id", "") or env_context.thread_id),
        message_id=env_context.message_id,
        profile=env_context.profile,
    )


def evidence_from_messages(messages: Sequence[Mapping[str, Any]] | None) -> list[ToolEvidence]:
    call_names: dict[str, str] = {}
    evidence: list[ToolEvidence] = []
    for message in messages or []:
        if message.get("role") == "assistant":
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, Mapping):
                    continue
                call_id = str(tool_call.get("id") or "")
                function = tool_call.get("function") or {}
                name = function.get("name") if isinstance(function, Mapping) else None
                if call_id and name:
                    call_names[call_id] = str(name)
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        parsed: Any = content
        success = False
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = content
        if isinstance(parsed, Mapping):
            success = bool(parsed.get("success") or parsed.get("ok") or parsed.get("approved"))
            if "error" in parsed:
                success = False
        call_id = str(message.get("tool_call_id") or "")
        evidence.append(
            ToolEvidence(
                tool_name=str(message.get("name") or call_names.get(call_id) or "tool"),
                tool_call_id=call_id,
                result=parsed,
                success=success,
                durable_ref=str(parsed.get("id") or parsed.get("message_id") or "")
                if isinstance(parsed, Mapping)
                else "",
            )
        )
    return evidence


class LocalMunchoRuntime:
    def __init__(
        self,
        config: Mapping[str, Any] | None,
        context: RuntimeContext,
        brain: CanonicalBrain | None = None,
    ) -> None:
        self.config = dict(_runtime_cfg(config))
        self.context = context
        self.brain = brain

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        context: RuntimeContext,
        brain: CanonicalBrain | None = None,
    ) -> "LocalMunchoRuntime":
        cfg = _runtime_cfg(config)
        if brain is None:
            brain_cfg = cfg.get("brain", {}) if isinstance(cfg.get("brain"), Mapping) else {}
            if cfg.get("allow_stub") and brain_cfg.get("mode") == "stub":
                from agent.local_muncho.testing import InMemoryCanonicalBrain

                brain = InMemoryCanonicalBrain()
        return cls(cfg, context, brain)

    def enabled(self) -> bool:
        if not bool(self.config.get("enabled", False)):
            return False
        lane = str(self.config.get("lane") or "internal-support")
        return self.context.lane == lane

    def _allow_or_fail_open(self, reason: str) -> GuardDecision:
        if bool(self.config.get("fail_open", False)):
            return GuardDecision.allow(f"fail_open:{reason}")
        return GuardDecision.block(reason, code="local_muncho_block")

    def _lease_failure(self, reason: str) -> LeaseAssertion:
        return LeaseAssertion(False, reason=reason)

    def load_context(self, scope: str = "internal-support") -> KnowledgeContext:
        if not self.enabled():
            return KnowledgeContext(scope=scope, text="", version="")
        if self.brain is None:
            return KnowledgeContext(scope=scope, text="", version="")
        max_chars = int(self.config.get("knowledge_max_chars") or 24000)
        return self.brain.load_knowledge_context(scope, max_chars=max_chars)

    def emit_heartbeat(
        self,
        *,
        status: str,
        event_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> HeartbeatResult:
        if not self.enabled():
            return HeartbeatResult(True, reason="runtime disabled")
        if self.brain is None:
            return HeartbeatResult(False, reason="canonical brain unavailable")
        payload = HeartbeatPayload(
            runtime_id=str(self.config.get("runtime_id") or "local-muncho"),
            runtime_kind=str(self.config.get("runtime_kind") or "local-primary"),
            status=status,
            event_type=event_type,
            metadata=redact_mapping(metadata),
        )
        try:
            lease = self.brain.refresh_active_lease(
                payload,
                ttl_seconds=int(self.config.get("lease_ttl_seconds") or 90),
            )
            self.brain.write_runtime_event(
                RuntimeEvent(
                    event_type=event_type,
                    status=status,
                    context=self.context,
                    metadata=payload.metadata,
                )
            )
            return HeartbeatResult(True, reason="heartbeat emitted", lease=lease)
        except Exception as exc:
            return HeartbeatResult(False, reason=f"heartbeat failed: {exc}")

    def assert_active_lease(
        self,
        *,
        action: str,
        lock_key: str | None = None,
        case_id: str | None = None,
        approval_class: str | None = None,
    ) -> LeaseAssertion:
        if not self.enabled():
            return LeaseAssertion(True, reason="runtime disabled")
        if self.brain is None:
            return self._lease_failure("canonical brain unavailable")
        try:
            lease = self.brain.read_active_lease()
        except Exception as exc:
            return self._lease_failure(f"lease read failed: {exc}")
        if lease is None:
            return self._lease_failure("active lease missing")
        expected_owner = str(self.config.get("runtime_id") or "local-muncho")
        if lease.lease_owner != expected_owner:
            return self._lease_failure("active lease owner mismatch")
        active_runtime = str(lease.active_runtime or "")
        allowed_runtime = {
            "local",
            "local-primary",
            str(self.config.get("runtime_kind") or "local-primary"),
        }
        if active_runtime not in allowed_runtime:
            return self._lease_failure("active lease runtime mismatch")
        if lease.is_expired():
            return self._lease_failure("active lease expired")
        flags = {str(flag).lower() for flag in (lease.flags or ())}
        if "force-cloud" in flags:
            return self._lease_failure("force-cloud flag active")
        if "pause-all" in flags:
            return self._lease_failure("pause-all flag active")
        if approval_class and lease.approval_classes:
            if approval_class not in set(lease.approval_classes):
                return self._lease_failure("approval class not covered by active lease")
        if lock_key:
            try:
                handle = self.brain.acquire_lock(
                    lock_key,
                    {"token": str(uuid.uuid4()), "action": action, "case_id": case_id or ""},
                    ttl_seconds=int(self.config.get("lease_ttl_seconds") or 90),
                )
            except Exception as exc:
                return self._lease_failure(f"lock acquire failed: {exc}")
            if not handle.acquired:
                return self._lease_failure("scoped lock unavailable")
        return LeaseAssertion(True, reason="active lease ok", lease=lease)

    def guard_tool_action(
        self,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> GuardDecision:
        if not self.enabled():
            return GuardDecision.allow("runtime disabled")
        policy = classify_tool_action(tool_name, args)
        if not policy.requires_lease:
            return GuardDecision.allow("read-only action")
        assertion = self.assert_active_lease(
            action=policy.action,
            approval_class=policy.approval_class,
        )
        if not assertion.allowed:
            return self._allow_or_fail_open(assertion.reason)
        return GuardDecision.allow(assertion.reason)

    def validate_final_output(
        self,
        text: str,
        *,
        evidence: Sequence[ToolEvidence],
    ) -> EvidenceValidationResult:
        if not self.enabled():
            return EvidenceValidationResult(True, reason="runtime disabled")
        assertion = self.assert_active_lease(action="final_output")
        if not assertion.allowed:
            replacement = _blocked_text(assertion.reason)
            return EvidenceValidationResult(
                False,
                reason=assertion.reason,
                evidence_gaps=(assertion.reason,),
                replacement_text=replacement,
            )
        result = _validate_final_output(text, context=self.context, evidence=evidence)
        if not result.allowed:
            self._audit("final_output_validation", "blocked", result.reason)
            return result
        final_assertion = self.assert_active_lease(action="final_output_recheck")
        if not final_assertion.allowed:
            replacement = _blocked_text(final_assertion.reason)
            return EvidenceValidationResult(
                False,
                reason=final_assertion.reason,
                evidence_gaps=(final_assertion.reason,),
                replacement_text=replacement,
            )
        return result

    def guard_worker_spawn(
        self,
        contract: WorkerContract,
        *,
        source: str,
    ) -> GuardDecision:
        if not self.enabled():
            return GuardDecision.allow("runtime disabled")
        assertion = self.assert_active_lease(
            action=f"worker_spawn:{source}",
            approval_class="worker_spawn",
        )
        if not assertion.allowed:
            return self._allow_or_fail_open(assertion.reason)
        if self.brain is not None:
            try:
                self.brain.write_runtime_event(
                    RuntimeEvent(
                        event_type="worker_spawn_guard",
                        status="allowed",
                        context=self.context,
                        metadata={
                            "source": source,
                            "task_count": contract.task_count,
                        },
                    )
                )
            except Exception:
                pass
        return GuardDecision.allow(assertion.reason)

    def guard_approval_mutation(
        self,
        *,
        action: str,
        session_key: str,
        choice: str = "",
    ) -> GuardDecision:
        if not self.enabled():
            return GuardDecision.allow("runtime disabled")
        assertion = self.assert_active_lease(action=f"approval:{action}", approval_class="approval")
        if not assertion.allowed:
            return self._allow_or_fail_open(assertion.reason)
        if self.brain is not None:
            try:
                self.brain.record_approval(
                    ApprovalRecord(
                        approval_id=f"{action}:{session_key}:{int(utc_ts())}",
                        session_key=session_key,
                        choice=choice,
                        context=self.context,
                        metadata={"action": action},
                    )
                )
            except Exception:
                pass
        return GuardDecision.allow(assertion.reason)

    def create_codex_task(self, title: str, metadata: Mapping[str, Any] | None = None) -> str | None:
        if not self.enabled() or self.brain is None:
            return None
        try:
            return self.brain.create_codex_task(
                CodexTaskCreate(title=title, context=self.context, metadata=redact_mapping(metadata))
            )
        except Exception:
            return None

    def update_codex_task(self, task_id: str | None, patch: CodexTaskPatch) -> None:
        if not task_id or not self.enabled() or self.brain is None:
            return
        try:
            self.brain.update_codex_task(task_id, patch)
        except Exception:
            pass

    def _audit(self, action: str, outcome: str, reason: str, metadata: Mapping[str, Any] | None = None) -> None:
        if self.brain is None:
            return
        try:
            self.brain.write_audit_log(
                AuditEvent(
                    action=action,
                    outcome=outcome,
                    context=self.context,
                    reason=reason,
                    metadata=redact_mapping(metadata),
                )
            )
        except Exception:
            pass


def guard_internal_error_decision(
    runtime: LocalMunchoRuntime,
    exc: BaseException,
    *,
    guard_name: str,
) -> GuardDecision | None:
    """Fail closed for guard wrapper errors only when this runtime is active."""

    try:
        enabled = runtime.enabled()
    except Exception:
        enabled = False
    if not enabled:
        return None

    detail = str(exc) or exc.__class__.__name__
    reason = f"Local Muncho runtime {guard_name} guard internal error: {detail}"
    return GuardDecision.block(
        reason,
        code="local_muncho_guard_error",
        replacement_text=_blocked_text(reason),
    )


def guard_internal_error_decision_for_agent(
    agent: Any,
    exc: BaseException,
    *,
    guard_name: str,
) -> GuardDecision | None:
    return guard_internal_error_decision(
        get_runtime_for_agent(agent),
        exc,
        guard_name=guard_name,
    )


def guard_internal_error_decision_for_current_context(
    exc: BaseException,
    *,
    guard_name: str,
) -> GuardDecision | None:
    return guard_internal_error_decision(
        get_current_runtime(),
        exc,
        guard_name=guard_name,
    )


def _blocked_text(reason: str) -> str:
    return "\n".join(
        [
            "VERDICT: BLOCKED",
            "TL;DR: Local Muncho runtime blocked the response before delivery.",
            "CATEGORY: runtime_guard",
            "EVIDENCE_CHECKED: active-lease",
            f"EVIDENCE_GAP: {reason}",
            "STATUS: blocked",
            "NEXT_ACTION: restore local lease or hand off to cloud runtime",
            "APPROVAL_NEEDED: yes",
            "RISK: unsafe-runtime-transition",
        ]
    )


def get_current_runtime(
    *,
    context: RuntimeContext | None = None,
    brain: CanonicalBrain | None = None,
    config: Mapping[str, Any] | None = None,
) -> LocalMunchoRuntime:
    return LocalMunchoRuntime.from_config(
        config if config is not None else _load_runtime_config(),
        context or _current_context.get() or context_from_env(),
        brain if brain is not None else _current_brain.get(),
    )


def get_runtime_for_agent(agent: Any) -> LocalMunchoRuntime:
    brain = getattr(agent, "_local_muncho_brain", None) or _current_brain.get()
    config = getattr(agent, "_local_muncho_config", None)
    return LocalMunchoRuntime.from_config(
        config if isinstance(config, Mapping) else _load_runtime_config(),
        context_from_agent(agent),
        brain,
    )


def guard_tool_action_for_agent(
    agent: Any,
    tool_name: str,
    args: Mapping[str, Any],
) -> GuardDecision:
    return get_runtime_for_agent(agent).guard_tool_action(tool_name, args)


def guard_tool_action_for_current_context(
    tool_name: str,
    args: Mapping[str, Any],
) -> GuardDecision:
    return get_current_runtime().guard_tool_action(tool_name, args)


def validate_final_response_for_agent(
    agent: Any,
    text: str,
    messages: Sequence[Mapping[str, Any]] | None,
) -> str:
    runtime = get_runtime_for_agent(agent)
    try:
        result = runtime.validate_final_output(text, evidence=evidence_from_messages(messages))
    except Exception as exc:
        decision = guard_internal_error_decision(
            runtime,
            exc,
            guard_name="final_output",
        )
        if decision is not None and not decision.allowed:
            return decision.replacement_text or decision.message or text
        return text
    return result.replacement_text or text


def guard_worker_spawn_for_agent(
    agent: Any,
    contract: WorkerContract,
    *,
    source: str,
) -> GuardDecision:
    return get_runtime_for_agent(agent).guard_worker_spawn(contract, source=source)


def guard_approval_mutation_for_current_context(
    *,
    action: str,
    session_key: str,
    choice: str = "",
) -> GuardDecision:
    return get_current_runtime().guard_approval_mutation(
        action=action,
        session_key=session_key,
        choice=choice,
    )


def tool_block_result(decision: GuardDecision) -> str:
    return json.dumps(
        {
            "error": decision.message or decision.reason or "Local Muncho runtime blocked this action",
            "status": "blocked",
            "blocked_by": "local_muncho_runtime",
            "code": decision.code or "local_muncho_block",
        },
        ensure_ascii=False,
    )
