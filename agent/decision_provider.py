"""Provider-neutral, shadow-only decision seam for Hermes.

The runtime owns deterministic safety facts, privacy screening, logging, and replay
metadata.  A Jev adapter is just a provider implementing :class:`DecisionProvider`;
no Jev package is required, and no decision here can bypass Hermes approvals or
tool guardrails while the layer is in shadow mode.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

CONSUMERS = (
    "pre_action", "completion", "loop", "context_sieve",
    "failure_classification", "retry", "specialist", "notification",
)
DEFAULT_THRESHOLDS = {
    "pre_action": 0.95,
    "completion": 0.80,
    "loop": 0.85,
    "context_sieve": 0.90,
    "failure_classification": 0.80,
    "retry": 0.85,
    "specialist": 0.80,
    "notification": 0.80,
}
DEFAULT_POLICIES = {
    # fail-closed means a provider failure never supplies an allow decision.  The
    # existing deterministic approval/guardrail path still owns execution.
    "pre_action": {"timeout_seconds": 1.5, "fail_mode": "closed"},
    "completion": {"timeout_seconds": 2.0, "fail_mode": "open"},
    "loop": {"timeout_seconds": 1.0, "fail_mode": "open"},
    "context_sieve": {"timeout_seconds": 2.0, "fail_mode": "open"},
    "failure_classification": {"timeout_seconds": 2.0, "fail_mode": "open"},
    "retry": {"timeout_seconds": 2.0, "fail_mode": "open"},
    "specialist": {"timeout_seconds": 2.0, "fail_mode": "open"},
    "notification": {"timeout_seconds": 2.0, "fail_mode": "open"},
}

_DONE_RE = re.compile(r"\b(done|complete|completed|finished|fixed|resolved)\b", re.I)
_CONSTRAINT_RE = re.compile(
    r"\b(?:must|must not|never|do not|don't|only|without|preserve|keep|ask before|requires? approval)\b",
    re.I,
)
_BOUNDARY_RE = re.compile(
    r"\b(?:approval|approve|destructive|publish(?:ing)?|financial|payment|permission|secret|credential|exfiltrat)\w*\b",
    re.I,
)
_SENSITIVE_PATH_RE = re.compile(
    r"(?:^|[\\/])(?:\.env(?:\.[^\\/]*)?|id_rsa|id_ed25519|credentials(?:\.[^\\/]*)?|secret(?:s)?(?:\.[^\\/]*)?)$",
    re.I,
)
_SENSITIVE_TEXT_RE = re.compile(
    r"(?:-----BEGIN [^-]+ PRIVATE KEY-----|\b(?:password|passwd|token|secret|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|client[_ -]?secret|authorization|cookie)\s*[:=])",
    re.I,
)
_PRIVATE_CODE_RE = re.compile(
    r"```[\s\S]+```|(?:^|\n)\s*(?:def|class|import|from|const|let|var|function|SELECT|#!/)\b",
    re.I,
)
_EXFIL_TOOL_RE = re.compile(r"(?:send|email|message|upload|publish|share|post|notify|browser_type)", re.I)
_EXFIL_COMMAND_RE = re.compile(r"\b(?:curl|wget|scp|rsync|git\s+push|ftp|nc|netcat)\b", re.I)
_PRESERVE_RE = re.compile(
    r"\b(?:error|exception|traceback|failed?|failure|warning|approval|approved|denied|permission|timeout|not found|repro|flaky|non[- ]?reproducible)\b",
    re.I,
)
_ERROR_RE = re.compile(r"\b(?:error|exception|traceback|failed?|failure|exit\s*code\s*[1-9]|fatal)\b", re.I)
_APPROVAL_RE = re.compile(r"\b(?:approval|approved|denied|permission|authorize|authorise|pending)\b", re.I)
_NON_REPRO_RE = re.compile(r"\b(?:non[- ]?reproducible|not reproducible|flaky|race condition|intermittent|could not reproduce)\b", re.I)
_PATH_RE = re.compile(r"(?<![\w-])(?:~|/|\.{0,2}/)?(?:[\w.-]+[\\/])*[\w.-]+(?:\.[\w.-]+)?")
_PRIVATE_ARG_KEYS = frozenset({
    "command", "cmd", "content", "patch", "body", "message", "text", "headers", "input", "data",
})
_PRIVATE_CONTENT_TOOLS = frozenset({"read_file", "search_files", "write_file", "patch", "execute_code"})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8", "replace")).hexdigest()


def _clip(value: Any, limit: int = 4000) -> str:
    text = str(value or "").replace("\x00", " ")
    if len(text) <= limit:
        return text
    half = max(32, (limit - 24) // 2)
    return f"{text[:half]} …[truncated]… {text[-half:]}"


def _safe_error_code(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).lower()
    return text if re.fullmatch(r"[a-z0-9_.-]{1,40}", text) else "provider_error"


def _message_text(message: Any) -> str:
    if not isinstance(message, Mapping):
        return _clip(message, 2000)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text") or part.get("content") or "")
            for part in content
            if isinstance(part, Mapping)
        )
    return str(content or "")


def _bounded(value: Any, *, depth: int = 0) -> Any:
    """Bound provider input without changing the local evidence being hashed."""
    if depth > 4:
        return "[depth limited]"
    if isinstance(value, Mapping):
        return {str(k): _bounded(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bounded(v, depth=depth + 1) for v in value[:64]]
    if isinstance(value, str):
        return _clip(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _clip(value)


def _argument_projection(args: Mapping[str, Any]) -> dict[str, Any]:
    """Expose shapes and hashes, not private code or command text, to a provider."""
    projected: dict[str, Any] = {}
    for key, value in args.items():
        name = str(key)
        if name in _PRIVATE_ARG_KEYS:
            projected[name] = {"type": type(value).__name__, "chars": len(str(value)), "sha256": _digest(value)}
        elif name in {"path", "workdir", "mode", "offset", "limit", "name", "url"}:
            projected[name] = _clip(value, 512)
        elif isinstance(value, (bool, int, float)) or value is None:
            projected[name] = value
        else:
            projected[name] = {"type": type(value).__name__, "sha256": _digest(value)}
    return projected


@dataclass(frozen=True)
class HardConstraint:
    constraint_id: str
    kind: str
    source: str
    text_digest: str


class ConstraintLedger:
    """Deterministic, non-verbatim ledger of user constraints and safety boundaries."""

    def __init__(self, constraints: Sequence[HardConstraint] = (), boundaries: Sequence[str] = ()):
        self.constraints = tuple(constraints)
        self.safety_boundaries = tuple(sorted(set(boundaries)))

    @classmethod
    def from_messages(cls, messages: Sequence[Any] | None) -> "ConstraintLedger":
        constraints: list[HardConstraint] = []
        boundaries: set[str] = set()
        for index, message in enumerate(messages or []):
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            text = _message_text(message)
            if _BOUNDARY_RE.search(text):
                for boundary in ("approval", "destructive", "publishing", "financial", "permission", "secret"):
                    if re.search(rf"\b{boundary.rstrip('ing')}\w*\b", text, re.I):
                        boundaries.add(boundary)
            for sentence in re.split(r"(?<=[.!?\n])\s+", text):
                normalized = " ".join(sentence.split())
                if not normalized or not _CONSTRAINT_RE.search(normalized):
                    continue
                kind = "safety" if _BOUNDARY_RE.search(normalized) else "user"
                digest = _digest(normalized)
                constraints.append(HardConstraint(
                    constraint_id=f"c{index}_{digest[:12]}",
                    kind=kind,
                    source=f"user:{index}",
                    text_digest=digest,
                ))
        return cls(constraints, tuple(boundaries))

    def to_metadata(self) -> dict[str, Any]:
        return {
            "count": len(self.constraints),
            "constraints": [
                {
                    "constraint_id": item.constraint_id,
                    "kind": item.kind,
                    "source": item.source,
                    "text_digest": item.text_digest,
                }
                for item in self.constraints
            ],
            "safety_boundaries": list(self.safety_boundaries),
        }


@dataclass(frozen=True)
class DecisionRequest:
    consumer: str
    request_id: str
    payload: Mapping[str, Any]
    constraints: Mapping[str, Any]


@dataclass(frozen=True)
class DecisionResult:
    decision: str = "unknown"
    confidence: float | None = None
    provider: str = "none"
    rationale: str = ""
    error: str | None = None
    timed_out: bool = False
    model: str | None = None
    service_version: str | None = None

    @classmethod
    def from_value(cls, value: Any, *, provider: str) -> "DecisionResult":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return cls(provider=provider, error="invalid_result")
        raw_confidence = value.get("confidence")
        try:
            confidence = max(0.0, min(1.0, float(raw_confidence))) if raw_confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        return cls(
            decision=str(value.get("decision", value.get("choice", value.get("action", value.get("label", "unknown")))) or "unknown").lower(),
            confidence=confidence,
            provider=provider,
            rationale=_clip(value.get("rationale", ""), 240),
            error=_safe_error_code(value.get("error")),
            model=_clip(value.get("model", ""), 160) or None,
            service_version=_clip(value.get("service_version", ""), 160) or None,
        )


class DecisionProvider(Protocol):
    """Minimal adapter seam; Jev and future providers implement this contract."""

    name: str

    def decide(self, request: DecisionRequest, *, timeout_seconds: float) -> DecisionResult | Mapping[str, Any] | None:
        ...


class NullDecisionProvider:
    name = "none"

    def decide(self, request: DecisionRequest, *, timeout_seconds: float):
        return DecisionResult(provider=self.name)


class SubprocessDecisionProvider:
    """Optional JSON adapter for a local command such as ``node hermes_jev_adapter.cjs``."""

    name = "hermes-bridge-jev"

    def __init__(
        self,
        command: Sequence[str] | str,
        *,
        name: str = "hermes-bridge-jev",
        provider_config: Mapping[str, Any] | None = None,
    ):
        if isinstance(command, str):
            command = shlex.split(command)
        self.command = tuple(str(part) for part in command if str(part))
        if not self.command:
            raise ValueError("decision adapter command is empty")
        self.name = str(name or self.name)
        self.provider_config = dict(provider_config or {})

    def decide(self, request: DecisionRequest, *, timeout_seconds: float):
        from agent.secret_scope import get_secret
        from tools.environments.local import build_subprocess_env

        env = build_subprocess_env()
        if api_key := get_secret("OPENROUTER_API_KEY"):
            env["OPENROUTER_API_KEY"] = api_key
        completed = subprocess.run(
            self.command,
            input=_canonical({
                "consumer": request.consumer,
                "request_id": request.request_id,
                "payload": request.payload,
                "constraints": request.constraints,
                "provider_config": self.provider_config,
            }),
            capture_output=True,
            text=True,
            timeout=max(0.05, float(timeout_seconds)),
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            return DecisionResult(provider=self.name, error=f"exit_{completed.returncode}")
        try:
            return json.loads(completed.stdout or "{}")
        except (TypeError, ValueError):
            return DecisionResult(provider=self.name, error="invalid_json")


@dataclass(frozen=True)
class DecisionPolicy:
    threshold: float
    timeout_seconds: float
    fail_mode: str


@dataclass(frozen=True)
class PrivacyVerdict:
    allowed: bool
    safe_value: Any
    reason: str = ""


@dataclass(frozen=True)
class DecisionObservation:
    consumer: str
    request_id: str
    result: DecisionResult
    policy: DecisionPolicy
    privacy_allowed: bool
    skipped_reason: str = ""
    threshold_met: bool = False
    shadow: bool = True

    @property
    def candidate_action(self) -> str:
        return self.result.decision if self.threshold_met else "unknown"

    @property
    def fail_closed(self) -> bool:
        return self.policy.fail_mode == "closed"


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        try:
            from agent.redact import redact_sensitive_text

            return redact_sensitive_text(value, force=True)
        except Exception:
            return value
    return value


def _contains_sensitive_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_sensitive_path(k) or _contains_sensitive_path(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_sensitive_path(item) for item in value)
    if not isinstance(value, str):
        return False
    for token in re.findall(r"[^\s\"'<>]+", value):
        if _SENSITIVE_PATH_RE.search(token):
            return True
    return False


def privacy_gate(value: Any) -> PrivacyVerdict:
    """Fail closed before provider egress; no raw sensitive value reaches an adapter."""
    scrubbed = _redact(value)
    raw_text = _canonical(value)
    scrubbed_text = _canonical(scrubbed)
    if _contains_sensitive_path(value):
        return PrivacyVerdict(False, {}, "sensitive_path")
    if raw_text != scrubbed_text or _SENSITIVE_TEXT_RE.search(raw_text):
        return PrivacyVerdict(False, {}, "secret_or_sensitive_text")
    if _PRIVATE_CODE_RE.search(raw_text):
        return PrivacyVerdict(False, {}, "private_code")
    return PrivacyVerdict(True, _bounded(scrubbed))


def _parse_policy(consumer: str, cfg: Mapping[str, Any], thresholds: Mapping[str, Any]) -> DecisionPolicy:
    defaults = DEFAULT_POLICIES.get(consumer, DEFAULT_POLICIES["completion"])
    raw = cfg.get(consumer, {}) if isinstance(cfg, Mapping) else {}
    raw = raw if isinstance(raw, Mapping) else {}
    try:
        threshold = float(thresholds.get(consumer, DEFAULT_THRESHOLDS.get(consumer, 0.8)))
    except (TypeError, ValueError):
        threshold = DEFAULT_THRESHOLDS.get(consumer, 0.8)
    try:
        timeout = float(raw.get("timeout_seconds", defaults["timeout_seconds"]))
    except (TypeError, ValueError):
        timeout = float(defaults["timeout_seconds"])
    fail_mode = str(raw.get("fail_mode", defaults["fail_mode"])).lower()
    return DecisionPolicy(
        threshold=max(0.0, min(1.0, threshold)),
        timeout_seconds=max(0.05, min(30.0, timeout)),
        fail_mode=fail_mode if fail_mode in {"open", "closed"} else str(defaults["fail_mode"]),
    )


def _call_with_timeout(provider: DecisionProvider, request: DecisionRequest, timeout_seconds: float) -> DecisionResult:
    result: dict[str, Any] = {}

    def run() -> None:
        try:
            value = provider.decide(request, timeout_seconds=timeout_seconds)
            result["value"] = DecisionResult.from_value(value, provider=getattr(provider, "name", "provider"))
        except subprocess.TimeoutExpired:
            result["value"] = DecisionResult(provider=getattr(provider, "name", "provider"), error="timeout", timed_out=True)
        except Exception as exc:  # provider failures never cross the Hermes safety boundary
            logger.debug("decision provider failed", exc_info=True)
            result["value"] = DecisionResult(
                provider=getattr(provider, "name", "provider"),
                error=type(exc).__name__,
            )

    context = contextvars.copy_context()
    worker = threading.Thread(
        target=lambda: context.run(run), name="hermes-decision-provider", daemon=True
    )
    worker.start()
    worker.join(max(0.05, timeout_seconds))
    if worker.is_alive():
        return DecisionResult(provider=getattr(provider, "name", "provider"), error="timeout", timed_out=True)
    return result.get("value") or DecisionResult(provider=getattr(provider, "name", "provider"), error="no_result")


class DecisionRuntime:
    """Per-agent decision runtime. All consumers are shadow-only until explicitly evolved."""

    def __init__(self, config: Mapping[str, Any] | None = None, *, provider: DecisionProvider | None = None):
        cfg = config if isinstance(config, Mapping) else {}
        # Enforcement is intentionally not implemented in this first local slice;
        # deterministic Hermes policy remains the sole owner until a later review.
        self.mode = "shadow"
        self.requested_mode = str(cfg.get("mode", "shadow") or "shadow").lower()
        self.shadow = True
        self.provider_name = str(cfg.get("provider", "none") or "none")
        self.model = str(cfg.get("model", "") or "")
        self.service_version = str(cfg.get("service_version", "") or "")
        raw_thresholds = cfg.get("thresholds")
        thresholds: Mapping[str, Any] = raw_thresholds if isinstance(raw_thresholds, Mapping) else {}
        raw_policies = cfg.get("policies")
        policies: Mapping[str, Any] = raw_policies if isinstance(raw_policies, Mapping) else {}
        self.thresholds = {**DEFAULT_THRESHOLDS, **thresholds}
        self.policies = {
            consumer: _parse_policy(consumer, policies, self.thresholds)
            for consumer in CONSUMERS
        }
        raw_sieve_cfg = cfg.get("context_sieve")
        sieve_cfg: Mapping[str, Any] = raw_sieve_cfg if isinstance(raw_sieve_cfg, Mapping) else {}
        self.sieve_enabled = bool(sieve_cfg.get("enabled", True))
        try:
            self.sieve_min_chars = max(1, int(sieve_cfg.get("min_chars", 8000)))
        except (TypeError, ValueError):
            self.sieve_min_chars = 8000
        self.provider = provider or self._provider_from_config(cfg)
        logging_cfg = cfg.get("logging", True)
        if isinstance(logging_cfg, Mapping):
            self.log_enabled = bool(logging_cfg.get("enabled", True))
            configured_path = logging_cfg.get("path", "")
        else:
            self.log_enabled = bool(logging_cfg)
            configured_path = cfg.get("log_path", "")
        self.log_path = Path(configured_path) if configured_path else get_hermes_home() / "decision_logs" / "replay.jsonl"
        self._log_lock = threading.Lock()
        self._loop_history: deque[tuple[str, str, bool]] = deque(maxlen=32)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None, *, provider: DecisionProvider | None = None) -> "DecisionRuntime":
        return cls(config, provider=provider)

    @staticmethod
    def _provider_from_config(config: Mapping[str, Any]) -> DecisionProvider:
        provider_name = str(config.get("provider", "none") or "none").lower()
        if provider_name in {"none", "null", "disabled", "off"}:
            return NullDecisionProvider()
        command = config.get("adapter_command") or config.get("command")
        if not command:
            adapter_path = str(config.get("adapter_path", "") or "").strip()
            if adapter_path:
                path = Path(adapter_path).expanduser()
                if path.is_file():
                    command = ("node", str(path))
        if command:
            try:
                provider_config = {
                    key: config[key]
                    for key in ("model", "service_version", "endpoint", "adapter_timeout_ms", "max_retries")
                    if config.get(key) not in (None, "")
                }
                if "adapter_timeout_ms" in provider_config:
                    provider_config["timeout_ms"] = provider_config.pop("adapter_timeout_ms")
                return SubprocessDecisionProvider(
                    command,
                    name=str(config.get("provider") or "hermes-bridge-jev"),
                    provider_config=provider_config,
                )
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid decision provider command")
        return NullDecisionProvider()

    def _log(self, record: Mapping[str, Any]) -> None:
        if not self.log_enabled:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            line = _canonical(record) + "\n"
            with self._log_lock:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                try:
                    os.chmod(self.log_path, 0o600)
                except OSError:
                    pass
        except OSError:
            logger.debug("decision replay log unavailable", exc_info=True)

    def evaluate(
        self,
        consumer: str,
        payload: Mapping[str, Any],
        *,
        messages: Sequence[Any] | None = None,
        privacy_input: Any = None,
        constraints: ConstraintLedger | None = None,
        private_content: bool = False,
    ) -> DecisionObservation:
        consumer = consumer if consumer in CONSUMERS else "completion"
        policy = self.policies[consumer]
        ledger = constraints or ConstraintLedger.from_messages(messages)
        raw_input = payload if privacy_input is None else {"payload": payload, "source": privacy_input}
        privacy = (
            PrivacyVerdict(False, {}, "private_content")
            if private_content else privacy_gate(raw_input)
        )
        request_id = _digest({"consumer": consumer, "payload": payload, "constraints": ledger.to_metadata()})[:24]
        request = DecisionRequest(
            consumer=consumer,
            request_id=request_id,
            payload=privacy.safe_value if privacy.allowed else {},
            constraints=ledger.to_metadata(),
        )
        result = DecisionResult(provider=getattr(self.provider, "name", "none"))
        skipped_reason = ""
        if not privacy.allowed:
            skipped_reason = privacy.reason
        elif isinstance(self.provider, NullDecisionProvider):
            skipped_reason = "no_provider"
        else:
            result = _call_with_timeout(self.provider, request, policy.timeout_seconds)
            if result.error:
                skipped_reason = result.error
        threshold_met = result.confidence is not None and result.confidence >= policy.threshold
        self._log({
            "record_type": "decision_shadow",
            "schema_version": 1,
            "timestamp": time.time(),
            "consumer": consumer,
            "request_id": request_id,
            "input_sha256": _digest(raw_input),
            "payload_keys": sorted(str(key) for key in payload),
            "constraints": ledger.to_metadata(),
            "privacy_allowed": privacy.allowed,
            "privacy_reason": privacy.reason,
            "provider": result.provider,
            "provider_model": result.model or self.model or None,
            "provider_service_version": result.service_version or self.service_version or None,
            "decision": result.decision,
            "confidence": result.confidence,
            "threshold": policy.threshold,
            "threshold_met": threshold_met,
            "fail_mode": policy.fail_mode,
            "timed_out": result.timed_out,
            "error": _safe_error_code(result.error),
            "shadow": self.shadow,
            "candidate_action": result.decision if threshold_met else "unknown",
            # Labels can be added offline to this same JSONL file for calibration.
            "outcome": None,
        })
        return DecisionObservation(
            consumer=consumer,
            request_id=request_id,
            result=result,
            policy=policy,
            privacy_allowed=privacy.allowed,
            skipped_reason=skipped_reason,
            threshold_met=threshold_met,
            shadow=self.shadow,
        )

    def observe_loop(
        self,
        agent: Any,
        tool_name: str,
        args: Mapping[str, Any],
        result: Any,
        *,
        is_error: bool,
        messages: Sequence[Any] | None = None,
        private_content: bool = False,
    ) -> DecisionObservation:
        signature = _digest({"tool": tool_name, "args": args})
        result_hash = _digest(result)
        repeat_count = sum(1 for item in self._loop_history if item[:2] == (signature, result_hash)) + 1
        try:
            from agent.tool_guardrails import IDEMPOTENT_TOOL_NAMES

            is_progress = not is_error and tool_name not in IDEMPOTENT_TOOL_NAMES
        except Exception:
            is_progress = not is_error
        self._loop_history.append((signature, result_hash, is_progress))
        existing = getattr(getattr(agent, "_tool_guardrails", None), "halt_decision", None)
        payload = {
            "tool": tool_name,
            "args_sha256": signature,
            "result_sha256": result_hash,
            "is_error": bool(is_error),
            "repeat_count": repeat_count,
            "no_progress_candidate": repeat_count >= 2 and not is_progress,
            "existing_guardrail": existing.to_metadata() if existing is not None else None,
            "active_incident": bool(getattr(agent, "_incident_state", None)),
        }
        return self.evaluate(
            "loop", payload, messages=messages, privacy_input=result, private_content=private_content,
        )

    def observe_failure(
        self,
        agent: Any,
        error: Any,
        *,
        tool_name: str = "",
        context: Mapping[str, Any] | None = None,
        messages: Sequence[Any] | None = None,
        private_content: bool = False,
    ) -> DecisionObservation:
        error_text = _result_text(error)
        payload = {
            "error_text": _clip(error_text, 4000),
            "error_sha256": _digest(error_text),
            "error_chars": len(error_text),
            "tool": tool_name,
            "context": _bounded(context or {}),
            "deterministic_failure_owner": "hermes_tool_execution",
        }
        return self.evaluate(
            "failure_classification",
            payload,
            messages=messages,
            privacy_input=error_text,
            private_content=private_content,
        )

    def observe_retry(
        self,
        agent: Any,
        failure: str,
        *,
        attempts: int = 0,
        context: Mapping[str, Any] | None = None,
        messages: Sequence[Any] | None = None,
    ) -> DecisionObservation:
        payload = {
            "failure": str(failure or "unknown")[:80],
            "attempts": max(0, int(attempts)),
            "context": _bounded(context or {}),
            "deterministic_retry_owner": "hermes_retry_and_tool_guardrails",
        }
        return self.evaluate("retry", payload, messages=messages, privacy_input=payload)

    def observe_specialist(
        self,
        agent: Any,
        objective: Any,
        *,
        context: Mapping[str, Any] | None = None,
        prior_failures: Sequence[Any] = (),
        messages: Sequence[Any] | None = None,
        private_content: bool = False,
    ) -> DecisionObservation:
        objective_text = _message_text(objective)
        payload = {
            "objective": _clip(objective_text, 4000),
            "objective_sha256": _digest(objective_text),
            "objective_chars": len(objective_text),
            "context": _bounded(context or {}),
            "prior_failures": _bounded(list(prior_failures)[:16]),
            "deterministic_route_owner": "hermes_profile_and_escalation_policy",
        }
        return self.evaluate(
            "specialist",
            payload,
            messages=messages,
            privacy_input=objective_text,
            private_content=private_content,
        )

    def observe_notification(
        self,
        agent: Any,
        current_state: Mapping[str, Any],
        *,
        previous_baseline: Mapping[str, Any] | None = None,
        messages: Sequence[Any] | None = None,
    ) -> DecisionObservation:
        payload = {
            "current_state": _bounded(current_state),
            "previous_baseline": _bounded(previous_baseline or {}),
            "deterministic_notification_owner": "hermes_surface_notification_policy",
        }
        return self.evaluate("notification", payload, messages=messages, privacy_input=payload)


def _runtime_for(agent: Any) -> DecisionRuntime:
    runtime = getattr(agent, "_decision_runtime", None)
    if isinstance(runtime, DecisionRuntime):
        return runtime
    config: Mapping[str, Any] = {}
    try:
        from hermes_cli.config import load_config_readonly

        loaded = load_config_readonly()
        config = loaded.get("decision_provider", {}) if isinstance(loaded, Mapping) else {}
    except Exception:
        pass
    runtime = DecisionRuntime.from_mapping(config)
    try:
        agent._decision_runtime = runtime
    except Exception:
        pass
    return runtime


def _workspace_root(task_id: str) -> Path | None:
    try:
        from tools.file_tools_paths import _resolve_base_dir

        return Path(_resolve_base_dir(task_id))
    except Exception:
        try:
            from agent.runtime_cwd import resolve_agent_cwd

            return Path(resolve_agent_cwd()).resolve()
        except Exception:
            return None


def _pre_action_payload(agent: Any, tool_name: str, args: Mapping[str, Any], task_id: str, existing_block: str | None) -> dict[str, Any]:
    command = str(args.get("command") or args.get("cmd") or "")
    destructive = False
    targets: list[str] = []
    try:
        from agent.tool_dispatch_helpers import _extract_file_mutation_targets, _is_destructive_command

        destructive = tool_name in {"write_file", "patch"} or _is_destructive_command(command)
        targets = _extract_file_mutation_targets(tool_name, dict(args))
    except Exception:
        destructive = tool_name in {"write_file", "patch"}
    root = _workspace_root(task_id)
    resolved: list[str] = []
    outside = False
    try:
        from tools.file_tools_paths import _resolve_path_for_task

        for target in targets[:32]:
            path = Path(_resolve_path_for_task(target, task_id))
            resolved.append(str(path))
            if root is not None and not path.is_relative_to(root):
                outside = True
    except Exception:
        outside = bool(targets and root is None)
    workdir = str(args.get("workdir") or "")
    if workdir and root is not None:
        try:
            outside = outside or not Path(workdir).expanduser().resolve().is_relative_to(root)
        except Exception:
            outside = True
    exfiltration = bool(_EXFIL_TOOL_RE.search(tool_name) or _EXFIL_COMMAND_RE.search(command))
    return {
        "tool": tool_name,
        "args": _argument_projection(args),
        "destructive_candidate": destructive,
        "scope": "outside_workspace" if outside else ("inside_workspace" if resolved else "unscoped"),
        "workspace_root": str(root) if root else "unknown",
        "target_paths": resolved,
        "exfiltration_candidate": exfiltration,
        "existing_scope_block": bool(existing_block),
        "existing_hard_boundary_owner": "hermes_approval_guardrails",
    }


def record_pre_action(agent: Any, tool_name: str, args: Mapping[str, Any], task_id: str, *, existing_block: str | None = None) -> DecisionObservation:
    payload = _pre_action_payload(agent, tool_name, args, task_id, existing_block)
    return _runtime_for(agent).evaluate(
        "pre_action",
        payload,
        messages=getattr(agent, "_session_messages", None),
        # The projection contains hashes/types only for private arguments, so Jev can
        # observe the boundary without receiving file/code/command payloads.
        privacy_input=payload,
    )


def record_failure(
    agent: Any,
    error: Any,
    *,
    tool_name: str = "",
    context: Mapping[str, Any] | None = None,
    private_content: bool = False,
) -> DecisionObservation:
    return _runtime_for(agent).observe_failure(
        agent,
        error,
        tool_name=tool_name,
        context=context,
        messages=getattr(agent, "_session_messages", None),
        private_content=private_content,
    )


def record_retry(
    agent: Any,
    failure: str,
    *,
    attempts: int = 0,
    context: Mapping[str, Any] | None = None,
) -> DecisionObservation:
    return _runtime_for(agent).observe_retry(
        agent,
        failure,
        attempts=attempts,
        context=context,
        messages=getattr(agent, "_session_messages", None),
    )


def record_specialist(
    agent: Any,
    objective: Any,
    *,
    context: Mapping[str, Any] | None = None,
    prior_failures: Sequence[Any] = (),
    private_content: bool = False,
) -> DecisionObservation:
    return _runtime_for(agent).observe_specialist(
        agent,
        objective,
        context=context,
        prior_failures=prior_failures,
        messages=getattr(agent, "_session_messages", None),
        private_content=private_content,
    )


def _result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        if isinstance(value.get("text_summary"), str):
            return value["text_summary"]
        parts = value.get("content")
        if isinstance(parts, list):
            return "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, Mapping))
    return str(value or "")


def _paths_from_text(text: str) -> list[str]:
    seen: list[str] = []
    for match in _PATH_RE.findall(text):
        candidate = match.strip(".,:;()[]{}")
        if len(candidate) > 2 and candidate not in seen and ("/" in candidate or "." in Path(candidate).name):
            seen.append(candidate)
    return seen[:32]


def build_context_sieve_payload(agent: Any, tool_name: str, args: Mapping[str, Any], result: Any, *, is_error: bool) -> dict[str, Any] | None:
    text = _result_text(result)
    runtime = _runtime_for(agent)
    if not runtime.sieve_enabled or len(text) < runtime.sieve_min_chars:
        return None
    lines = text.splitlines() or [text]
    keep_indexes = {0, 1, 2, max(0, len(lines) - 3), max(0, len(lines) - 2), len(lines) - 1}
    keep_indexes.update(index for index, line in enumerate(lines) if _PRESERVE_RE.search(line))
    keep_indexes = {index for index in keep_indexes if 0 <= index < len(lines)}
    ranges: list[list[int]] = []
    for index in sorted(keep_indexes):
        if ranges and index <= ranges[-1][1] + 1:
            ranges[-1][1] = index
        else:
            ranges.append([index, index])
    exact_error = "\n".join(lines[index] for index in sorted(keep_indexes) if _ERROR_RE.search(lines[index]))
    paths = list(dict.fromkeys(
        [str(path) for path in getattr(agent, "_turn_file_mutation_paths", set())]
        + [str(args.get(key)) for key in ("path", "file", "file_path") if args.get(key)]
        + _paths_from_text(text)
    ))[:32]
    sample = "\n".join(lines[index] for index in sorted(keep_indexes))
    return {
        "tool": tool_name,
        "output_chars": len(text),
        "output_sha256": _digest(text),
        # Keep retention evidence local; private code/file output never leaves Hermes.
        "sample_sha256": _digest(sample),
        "sample_chars": len(sample),
        "candidate_line_ranges": ranges,
        "preserve_exact_error": bool(exact_error or is_error),
        "exact_error_sha256": _digest(exact_error or text) if (exact_error or is_error) else None,
        "active_file_paths": paths,
        "unresolved_failure": bool(is_error or getattr(agent, "_turn_failed_file_mutations", None)),
        "approval_evidence": bool(_APPROVAL_RE.search(text)),
        "non_reproducible_evidence": bool(_NON_REPRO_RE.search(text)),
        # The existing compressor/incident checkpoint remains the mutation owner.
        "compaction_owner": "incident_context_hygiene+context_compressor",
    }


def record_tool_result(agent: Any, tool_name: str, args: Mapping[str, Any], result: Any, *, is_error: bool, observed: bool = True) -> None:
    if not observed:
        return
    runtime = _runtime_for(agent)
    runtime.observe_loop(
        agent,
        tool_name,
        args,
        result,
        is_error=is_error,
        messages=getattr(agent, "_session_messages", None),
        private_content=tool_name in _PRIVATE_CONTENT_TOOLS,
    )
    if is_error:
        failure = runtime.observe_failure(
            agent,
            result,
            tool_name=tool_name,
            context={"args_sha256": _digest(args)},
            messages=getattr(agent, "_session_messages", None),
            private_content=tool_name in _PRIVATE_CONTENT_TOOLS,
        )
        runtime.observe_retry(
            agent,
            failure.result.decision,
            attempts=int(getattr(agent, "_incident_failed_repair_attempts", 0) or 0),
            context={
                "tool": tool_name,
                "failure_observation": failure.result.decision,
                "loop_candidate": True,
            },
            messages=getattr(agent, "_session_messages", None),
        )
    sieve_payload = build_context_sieve_payload(agent, tool_name, args, result, is_error=is_error)
    if sieve_payload is not None:
        runtime.evaluate(
            "context_sieve",
            sieve_payload,
            messages=getattr(agent, "_session_messages", None),
            privacy_input=_result_text(result),
            private_content=tool_name in _PRIVATE_CONTENT_TOOLS,
        )


def record_completion(agent: Any, final_response: Any, *, changed_paths: Sequence[str] = ()) -> DecisionObservation:
    verification: dict[str, Any] = {}
    try:
        from agent.verification_evidence import verification_status

        verification = verification_status(
            session_id=getattr(agent, "session_id", None),
            cwd=getattr(agent, "runtime_cwd", None),
        )
    except Exception:
        verification = {"status": "unavailable"}
    response = str(final_response or "")
    claims_done = bool(_DONE_RE.search(response))
    verification_status_value = str(verification.get("status") or "unavailable").lower()
    artifact_exists = getattr(agent, "_decision_artifact_exists", claims_done)
    if not isinstance(artifact_exists, bool):
        artifact_exists = claims_done
    tests_required = getattr(agent, "_decision_tests_required", verification.get("tests_required", False))
    tests_passed = getattr(agent, "_decision_tests_passed", verification.get("tests_passed"))
    if tests_passed is None:
        tests_passed = verification_status_value == "passed"
    payload = {
        "claims_done": claims_done,
        "response_sha256": _digest(response),
        "response_chars": len(response),
        "changed_paths": sorted(str(path) for path in changed_paths)[:32],
        "failed_file_mutations": sorted(str(path) for path in (getattr(agent, "_turn_failed_file_mutations", {}) or {}))[:32],
        "verification": _bounded(verification),
        "artifact_exists": artifact_exists,
        "verification_recorded": verification_status_value not in {"unavailable", "disabled", "not_applicable", "unverified"},
        "tests_required": bool(tests_required),
        "tests_passed": bool(tests_passed),
        "active_incident": bool(getattr(agent, "_incident_state", None)),
        "existing_stop_verification_enabled": bool(getattr(agent, "_verification_stop_nudges", 0) or False),
        "deterministic_completion_owner": "hermes_stop_gates+verification_evidence",
    }
    return _runtime_for(agent).evaluate(
        "completion",
        payload,
        messages=getattr(agent, "_session_messages", None),
    )


def record_notification(
    agent: Any,
    final_response: Any,
    *,
    changed_paths: Sequence[str] = (),
    interrupted: bool = False,
    failed: bool = False,
) -> DecisionObservation:
    response = str(final_response or "")
    state = {
        "claims_done": bool(_DONE_RE.search(response)),
        "response_sha256": _digest(response),
        "response_chars": len(response),
        "changed_paths": sorted(str(path) for path in changed_paths)[:32],
        "interrupted": bool(interrupted),
        "failed": bool(failed),
        "deterministic_notification_owner": "hermes_surface_notification_policy",
    }
    return _runtime_for(agent).observe_notification(
        agent,
        state,
        previous_baseline=getattr(agent, "_decision_notification_baseline", None),
        messages=getattr(agent, "_session_messages", None),
    )


def replay_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if isinstance(item, dict) and item.get("record_type") == "decision_shadow":
                    records.append(item)
    except OSError:
        pass
    return records


def calibrate_thresholds(records: Sequence[Mapping[str, Any]], *, candidates: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95), min_samples: int = 1) -> dict[str, dict[str, Any]]:
    """Small jevcal-style offline scaffold: choose a threshold per consumer from labels."""
    grouped: dict[str, list[tuple[float, bool]]] = {}
    for record in records:
        consumer = str(record.get("consumer") or "")
        confidence = record.get("confidence")
        outcome = record.get("outcome")
        if consumer not in CONSUMERS or confidence is None or outcome is None:
            continue
        try:
            grouped.setdefault(consumer, []).append((float(confidence), bool(outcome)))
        except (TypeError, ValueError):
            continue
    calibrated: dict[str, dict[str, Any]] = {}
    for consumer, samples in grouped.items():
        if len(samples) < max(1, int(min_samples)):
            continue
        best = max(
            ((sum((confidence >= threshold) == label for confidence, label in samples) / len(samples), threshold) for threshold in candidates),
            key=lambda item: (item[0], -item[1]),
        )
        calibrated[consumer] = {"threshold": best[1], "accuracy": best[0], "samples": len(samples)}
    return calibrated


__all__ = [
    "CONSUMERS", "DEFAULT_POLICIES", "DEFAULT_THRESHOLDS", "ConstraintLedger", "DecisionObservation",
    "DecisionPolicy", "DecisionProvider", "DecisionRequest", "DecisionResult", "DecisionRuntime",
    "HardConstraint", "NullDecisionProvider", "SubprocessDecisionProvider", "build_context_sieve_payload",
    "calibrate_thresholds", "privacy_gate", "record_completion", "record_failure", "record_notification",
    "record_pre_action", "record_retry", "record_specialist", "record_tool_result", "replay_records",
]
