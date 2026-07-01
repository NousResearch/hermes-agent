"""AI-assisted loop guard for gateway message dispatch and delivery.

The guard is deliberately outside the agent conversation.  It decides whether a
message should enter the agent loop or leave the gateway at all, using a small
set of deterministic safety signals plus an optional auxiliary LLM judge for
semantic agent-agent recursion.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Optional

from gateway.loop_state import LoopStateStore, default_loop_state_path

logger = logging.getLogger(__name__)

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_BLOCKING_ACTIONS = {"suppress", "quarantine_pair", "ask_human"}
_RUNTIME_INTENTS = {"status", "progress", "shutdown", "restart", "error", "notice"}

_META_STATUS_RE = re.compile(
    r"("
    r"gateway\s+(?:restarted|restart(?:ing)?|online|shutdown|stopp(?:ed|ing))"
    r"|hermes\s+(?:gateway|update)"
    r"|previous\s+(?:agent\s+)?run\s+was\s+interrupted"
    r"|auto-?resume"
    r"|processing\s+(?:started|complete|cancelled|failed)"
    r"|handler\s+returned\s+empty"
    r"|auxiliary\s+.+\s+failed"
    r"|runtime\s+notice"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_RESTART_REQUEST_RE = re.compile(
    r"("
    r"(?:^|\b)/(?:restart|update)\b"
    r"|(?:reinicia|reiniciar|rearranca|restart)\s+(?:tu\s+|el\s+|the\s+)?(?:gateway|hermes)"
    r"|systemctl\s+(?:--user\s+)?restart\s+.+hermes"
    r"|hermes\s+gateway\s+restart"
    r")",
    re.IGNORECASE,
)

_ACK_ONLY_RE = re.compile(
    r"^(?:[\s\W]*(?:ok|vale|recibido|entendido|done|ack|acknowledged|noted|sin accion|no action|no actuo|hilo cerrado|closed)[\s\W]*){1,8}$",
    re.IGNORECASE,
)


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return default
    return bool(value)


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _split_identities(raw: Any) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        items = raw.split(",")
    elif isinstance(raw, Iterable):
        items = list(raw)
    else:
        return set()
    return {str(item).strip().lower() for item in items if str(item).strip()}


def _risk_at_least(risk: str, threshold: str) -> bool:
    return _RISK_ORDER.get(risk, 0) >= _RISK_ORDER.get(threshold, 0)


def _agent_slug(identity: str) -> str:
    local = (identity or "hermes").split("@", 1)[0].strip().lower()
    return re.sub(r"[^a-z0-9_.-]+", "-", local) or "hermes"


@dataclass
class LoopGuardConfig:
    enabled: bool = True
    mode: str = "protect"  # observe|protect|strict
    ai_enabled: bool = False
    ai_timeout_seconds: float = 12.0
    quarantine_ttl_seconds: int = 7200
    agent_identities: set[str] = field(default_factory=set)
    local_identity: str = ""
    agent_id: str = ""
    state_path: str = ""
    window_seconds: int = 1800
    max_agent_pair_events: int = 6
    max_hop_count: int = 4
    duplicate_window_seconds: int = 1800

    @classmethod
    def from_mapping(
        cls,
        mapping: Dict[str, Any] | None,
        *,
        local_identity: str = "",
    ) -> "LoopGuardConfig":
        mapping = mapping or {}
        env_identities = _split_identities(os.getenv("HERMES_LOOP_GUARD_AGENT_IDENTITIES"))
        cfg_identities = _split_identities(
            mapping.get("agent_identities")
            or mapping.get("agent_email_addresses")
            or mapping.get("agent_addresses")
        )
        identities = env_identities | cfg_identities
        if local_identity:
            identities.add(local_identity.strip().lower())

        state_path = (
            os.getenv("HERMES_LOOP_GUARD_STATE_PATH")
            or str(mapping.get("state_path") or "")
            or str(default_loop_state_path())
        )
        mode = (os.getenv("HERMES_LOOP_GUARD_MODE") or str(mapping.get("mode") or "protect")).strip().lower()
        if mode not in {"observe", "protect", "strict", "off"}:
            mode = "protect"

        return cls(
            enabled=_bool(os.getenv("HERMES_LOOP_GUARD_ENABLED"), _bool(mapping.get("enabled"), True)) and mode != "off",
            mode=mode,
            ai_enabled=_bool(os.getenv("HERMES_LOOP_GUARD_AI_ENABLED"), _bool(mapping.get("ai_enabled"), False)),
            ai_timeout_seconds=_float(mapping.get("ai_timeout_seconds"), 12.0),
            quarantine_ttl_seconds=_int(mapping.get("quarantine_ttl_seconds"), 7200),
            agent_identities=identities,
            local_identity=(local_identity or str(mapping.get("local_identity") or "")).strip().lower(),
            agent_id=str(mapping.get("agent_id") or "").strip() or _agent_slug(local_identity),
            state_path=state_path,
            window_seconds=_int(mapping.get("window_seconds"), 1800),
            max_agent_pair_events=_int(mapping.get("max_agent_pair_events"), 6),
            max_hop_count=_int(mapping.get("max_hop_count"), 4),
            duplicate_window_seconds=_int(mapping.get("duplicate_window_seconds"), 1800),
        )


@dataclass
class LoopContext:
    platform: str
    direction: str  # inbound|outbound
    local_identity: str
    remote_identity: str
    text: str
    subject: str = ""
    message_id: str = ""
    in_reply_to: str = ""
    headers: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pair_key(self) -> str:
        return LoopStateStore.pair_key(self.platform, self.local_identity, self.remote_identity)


@dataclass
class LoopDecision:
    risk: str = "low"
    category: str = "normal"
    should_dispatch_to_agent: bool = True
    should_send_reply: bool = True
    recommended_action: str = "allow"
    confidence: float = 1.0
    reason: str = ""
    owner_summary: str = ""
    source: str = "deterministic"

    @classmethod
    def allow(cls, *, reason: str = "allowed", source: str = "deterministic") -> "LoopDecision":
        return cls(risk="low", category="normal", reason=reason, source=source)

    @classmethod
    def from_mapping(cls, data: Dict[str, Any], *, source: str) -> "LoopDecision":
        risk = str(data.get("risk") or "low").lower()
        if risk not in _RISK_ORDER:
            risk = "low"
        action = str(data.get("recommended_action") or "allow").lower()
        category = str(data.get("category") or "unclear").lower()
        should_dispatch = data.get("should_dispatch_to_agent")
        should_send = data.get("should_send_reply")
        if should_dispatch is None:
            should_dispatch = action not in _BLOCKING_ACTIONS
        if should_send is None:
            should_send = action not in _BLOCKING_ACTIONS
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return cls(
            risk=risk,
            category=category,
            should_dispatch_to_agent=bool(should_dispatch),
            should_send_reply=bool(should_send),
            recommended_action=action,
            confidence=max(0.0, min(1.0, confidence)),
            reason=str(data.get("reason") or ""),
            owner_summary=str(data.get("owner_summary") or ""),
            source=source,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "risk": self.risk,
            "category": self.category,
            "should_dispatch_to_agent": self.should_dispatch_to_agent,
            "should_send_reply": self.should_send_reply,
            "recommended_action": self.recommended_action,
            "confidence": self.confidence,
            "reason": self.reason,
            "owner_summary": self.owner_summary,
            "source": self.source,
        }


class AgentLoopGuard:
    """Gateway-level loop guard with optional AI semantic judge."""

    def __init__(self, config: LoopGuardConfig, *, state: LoopStateStore | None = None):
        self.config = config
        self.state = state or LoopStateStore(config.state_path)

    @classmethod
    def from_platform_config(cls, platform_config: Any, *, local_identity: str = "") -> "AgentLoopGuard":
        extra = getattr(platform_config, "extra", {}) or {}
        mapping = extra.get("loop_guard", extra.get("agent_loop_guard", {}))
        cfg = LoopGuardConfig.from_mapping(mapping, local_identity=local_identity)
        return cls(cfg)

    def is_agent_identity(self, value: str) -> bool:
        ident = (value or "").strip().lower()
        return bool(ident and ident in self.config.agent_identities)

    def context_is_agent_to_agent(self, ctx: LoopContext) -> bool:
        headers = {str(k).lower(): str(v) for k, v in (ctx.headers or {}).items()}
        remote_has_hermes_header = bool(headers.get("x-hermes-origin-agent"))
        local_is_agent = self.is_agent_identity(ctx.local_identity) or bool(self.config.local_identity)
        remote_is_agent = self.is_agent_identity(ctx.remote_identity) or remote_has_hermes_header
        return local_is_agent and remote_is_agent

    async def evaluate(self, ctx: LoopContext, *, stage: str) -> LoopDecision:
        """Evaluate a message at ``pre_dispatch`` or ``pre_send`` stage."""

        if not self.config.enabled:
            return LoopDecision.allow(reason="loop guard disabled")

        deterministic = self._deterministic_decision(ctx)
        decision = deterministic
        if self.config.ai_enabled and self._should_call_ai(ctx, deterministic):
            ai_decision = await self._call_ai_judge(ctx, deterministic)
            if ai_decision is not None:
                decision = self._combine_decisions(deterministic, ai_decision)

        decision = self._apply_mode(decision)
        self._record(ctx, decision, stage=stage)
        return decision

    def _deterministic_decision(self, ctx: LoopContext) -> LoopDecision:
        pair_key = ctx.pair_key
        quarantine = self.state.get_quarantine(pair_key)
        if quarantine is not None:
            return LoopDecision(
                risk="critical",
                category=str(quarantine.get("category") or "agent_agent_loop"),
                should_dispatch_to_agent=False,
                should_send_reply=False,
                recommended_action="suppress",
                confidence=1.0,
                reason=f"pair is quarantined until {quarantine.get('expires_at')}",
            )

        is_agent_pair = self.context_is_agent_to_agent(ctx)
        headers = {str(k).lower(): str(v) for k, v in (ctx.headers or {}).items()}
        intent = (headers.get("x-hermes-intent") or str(ctx.metadata.get("hermes_intent") or "")).strip().lower()
        hop_count = self._hop_count(ctx, headers)
        text = ctx.text or ""
        subject = ctx.subject or ""
        recent = self.state.recent_events(pair_key, window_seconds=self.config.window_seconds)
        duplicate_count = self.state.duplicate_count(
            pair_key,
            text,
            window_seconds=self.config.duplicate_window_seconds,
        )

        if is_agent_pair and self._is_restart_loop(text, intent):
            return LoopDecision(
                risk="critical",
                category="restart_loop",
                should_dispatch_to_agent=False,
                should_send_reply=False,
                recommended_action="quarantine_pair",
                confidence=1.0,
                reason="agent-to-agent message attempts to restart/update the gateway or reports restart recursion",
                owner_summary="Suppressed an agent-to-agent gateway restart loop.",
            )

        if is_agent_pair and hop_count >= self.config.max_hop_count:
            return LoopDecision(
                risk="high",
                category="agent_agent_loop",
                should_dispatch_to_agent=False,
                should_send_reply=False,
                recommended_action="quarantine_pair",
                confidence=0.95,
                reason=f"Hermes hop count {hop_count} exceeded limit {self.config.max_hop_count}",
            )

        if is_agent_pair and self._is_meta_status(text, subject, intent):
            return LoopDecision(
                risk="high",
                category="meta_status_loop",
                should_dispatch_to_agent=False,
                should_send_reply=False,
                recommended_action="suppress",
                confidence=0.9,
                reason="agent-to-agent runtime/status notice should not enter or leave the agent conversation",
            )

        if is_agent_pair and len(recent) >= self.config.max_agent_pair_events:
            return LoopDecision(
                risk="high",
                category="agent_agent_loop",
                should_dispatch_to_agent=False,
                should_send_reply=False,
                recommended_action="quarantine_pair",
                confidence=0.9,
                reason=f"{len(recent)} recent agent-agent events in the loop window",
            )

        if is_agent_pair and duplicate_count >= 1:
            return LoopDecision(
                risk="high",
                category="duplicate_ack_loop" if self._is_low_information_ack(text) else "agent_agent_loop",
                should_dispatch_to_agent=False,
                should_send_reply=False,
                recommended_action="suppress",
                confidence=0.85,
                reason="duplicate low-novelty agent-agent message in recent window",
            )

        if is_agent_pair and self._is_low_information_ack(text):
            return LoopDecision(
                risk="medium",
                category="duplicate_ack_loop",
                should_dispatch_to_agent=True,
                should_send_reply=True,
                recommended_action="allow_once",
                confidence=0.75,
                reason="first low-information agent acknowledgement; allow once and watch for repetition",
            )

        return LoopDecision.allow(reason="no loop signal")

    def _hop_count(self, ctx: LoopContext, headers: Dict[str, str]) -> int:
        raw = ctx.metadata.get("hermes_hop_count") or headers.get("x-hermes-hop-count") or 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def _is_restart_loop(self, text: str, intent: str) -> bool:
        return intent == "restart" or bool(_RESTART_REQUEST_RE.search(text or ""))

    def _is_meta_status(self, text: str, subject: str, intent: str) -> bool:
        if intent in _RUNTIME_INTENTS:
            return True
        combined = f"{subject}\n{text}"
        return bool(_META_STATUS_RE.search(combined))

    def _is_low_information_ack(self, text: str) -> bool:
        normalized = " ".join((text or "").strip().lower().split())
        if not normalized:
            return True
        if len(normalized) <= 120 and _ACK_ONLY_RE.match(normalized):
            return True
        if len(normalized) <= 220 and any(
            phrase in normalized
            for phrase in (
                "no realizo ninguna accion",
                "no realizo ninguna acción",
                "no additional action",
                "nothing else to do",
                "hilo cerrado",
                "thread closed",
            )
        ):
            return True
        return False

    def _should_call_ai(self, ctx: LoopContext, deterministic: LoopDecision) -> bool:
        if not self.context_is_agent_to_agent(ctx):
            return False
        if deterministic.risk in {"high", "critical"}:
            return False  # deterministic guard already has enough confidence
        return True

    async def _call_ai_judge(self, ctx: LoopContext, deterministic: LoopDecision) -> Optional[LoopDecision]:
        payload = {
            "stage_context": {
                "platform": ctx.platform,
                "direction": ctx.direction,
                "local_identity": ctx.local_identity,
                "remote_identity": ctx.remote_identity,
                "subject": ctx.subject[:240],
                "text_excerpt": (ctx.text or "")[:1800],
                "headers": {
                    key: value
                    for key, value in (ctx.headers or {}).items()
                    if str(key).lower().startswith("x-hermes-")
                },
            },
            "deterministic_decision": deterministic.as_dict(),
            "allowed_actions": ["allow", "allow_once", "suppress", "quarantine_pair", "notify_owner", "ask_human"],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Hermes gateway's internal loop-safety judge. "
                    "Classify whether an agent-to-agent message advances a human goal or is recursively "
                    "processing runtime/progress/restart/status/meta chatter. Return STRICT JSON only with: "
                    "risk low|medium|high|critical, category normal|agent_agent_loop|meta_status_loop|"
                    "restart_loop|duplicate_ack_loop|tool_failure_loop|unclear, should_dispatch_to_agent, "
                    "should_send_reply, recommended_action, confidence, reason, owner_summary. "
                    "Do not address the remote agent; this decision is internal."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        def _call() -> Any:
            from agent.auxiliary_client import call_llm

            return call_llm(
                task="loop_guard",
                messages=messages,
                temperature=0,
                max_tokens=500,
                timeout=self.config.ai_timeout_seconds,
            )

        try:
            response = await asyncio.to_thread(_call)
            content = response.choices[0].message.content
            data = self._parse_json_object(str(content or ""))
            if not isinstance(data, dict):
                return None
            return LoopDecision.from_mapping(data, source="ai")
        except Exception as exc:
            logger.warning("Loop guard AI judge failed; using deterministic decision: %s", exc)
            return None

    def _parse_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

    def _combine_decisions(self, deterministic: LoopDecision, ai_decision: LoopDecision) -> LoopDecision:
        if _RISK_ORDER.get(ai_decision.risk, 0) > _RISK_ORDER.get(deterministic.risk, 0):
            return ai_decision
        if _RISK_ORDER.get(ai_decision.risk, 0) == _RISK_ORDER.get(deterministic.risk, 0) and ai_decision.confidence >= deterministic.confidence:
            return ai_decision
        return deterministic

    def _apply_mode(self, decision: LoopDecision) -> LoopDecision:
        mode = self.config.mode
        if mode == "observe":
            return replace(
                decision,
                should_dispatch_to_agent=True,
                should_send_reply=True,
                recommended_action="allow",
                reason=f"observe mode: would have {decision.recommended_action} ({decision.reason})",
            )

        threshold = "medium" if mode == "strict" else "high"
        should_block = _risk_at_least(decision.risk, threshold) or decision.recommended_action in {"quarantine_pair", "ask_human"}
        if should_block:
            return replace(
                decision,
                should_dispatch_to_agent=False,
                should_send_reply=False,
                recommended_action="quarantine_pair" if decision.recommended_action == "quarantine_pair" else "suppress",
            )
        return decision

    def _record(self, ctx: LoopContext, decision: LoopDecision, *, stage: str) -> None:
        try:
            self.state.add_event(
                ctx.pair_key,
                direction=ctx.direction,
                text=ctx.text,
                subject=ctx.subject,
                risk=decision.risk,
                category=decision.category,
                action=decision.recommended_action,
                stage=stage,
            )
            if decision.recommended_action == "quarantine_pair" and self.config.mode != "observe":
                self.state.set_quarantine(
                    ctx.pair_key,
                    ttl_seconds=self.config.quarantine_ttl_seconds,
                    reason=decision.reason,
                    category=decision.category,
                )
        except Exception:
            logger.debug("Loop guard state update failed", exc_info=True)

    def outbound_headers(self, *, to_addr: str, metadata: Dict[str, Any] | None = None, inbound_hop_count: int = 0) -> Dict[str, str]:
        metadata = metadata or {}
        intent = str(metadata.get("hermes_intent") or "assistant_reply")
        reply_policy = str(metadata.get("hermes_reply_policy") or "human-or-allow-once")
        try:
            if "hermes_hop_count" in metadata:
                hop_count = int(metadata.get("hermes_hop_count") or 0)
            else:
                hop_count = int(inbound_hop_count or 0) + 1
        except (TypeError, ValueError):
            hop_count = 1
        return {
            "X-Hermes-Origin-Agent": str(metadata.get("hermes_origin_agent") or self.config.agent_id or _agent_slug(self.config.local_identity)),
            "X-Hermes-Origin-Address": self.config.local_identity,
            "X-Hermes-Intent": intent,
            "X-Hermes-Hop-Count": str(max(1, hop_count)),
            "X-Hermes-Reply-Policy": reply_policy,
        }
