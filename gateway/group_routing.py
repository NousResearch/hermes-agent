"""Gateway-level WhatsApp group routing policy.

This module is deliberately deterministic: it runs before any LLM reply
planning and decides whether an incoming WhatsApp message may enter the
public-response path. Skipped messages are still logged and ingested into the
session transcript/task-open-loop state so Hermes stays in the loop without
speaking in groups by default.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

RoutingAction = Literal[
    "no_reply_silent_ingest",
    "no_reply_task_update",
    "private_alert_to_jacob",
    "authorized_private_follow_up",
    "public_group_reply",
    "ask_jacob_privately",
    "ignore_non_actionable_noise",
]

AddresseeClassification = Literal[
    "addressed_to_hermes",
    "addressed_to_jacob",
    "addressed_to_specific_other_people",
    "addressed_to_group",
    "ambient_context",
    "ambiguous",
]

_GATEWAY_ACTION = Literal["skip", "allow", "rewrite"]

_DIRECT_HERMES_RE = re.compile(
    r"(?ix)"
    r"(?:^|[\s>¿¡,.;:!\-])"
    r"(?:@?(?:jack|hermes)|/(?:jack|hermes))"
    r"(?:\s*[:,;\-]|\s+|$)"
)

_JACOB_RE = re.compile(r"(?i)(?:^|[\s@,.;:!\-])jacob(?:\s*[:,;\-]|\s+|$)")

_ACTIONABLE_RE = re.compile(
    r"(?ix)\b("
    r"can\s+you|could\s+you|please|pls|confirm|follow\s+up|handle\s+this|"
    r"support|let\s+me\s+know|available|pending|block(?:ed|er)?|stuck|"
    r"need(?:s|ed)?|schedule|deliver|send|pay|payment|meeting|call|"
    r"puedes|podr[ií]as|me\s+ayudas|ap[oó]yalo|apoyarlo|ay[uú]dalo|"
    r"confirma(?:s|r)?|confirmaci[oó]n|disponibilidad|queda\s+pendiente|"
    r"av[ií]same|dale\s+seguimiento|seguimiento|necesito\s+que|por\s+favor|"
    r"pendiente|bloquead[oa]|atorad[oa]|pago|entrega|reuni[oó]n|llamada"
    r")\b"
)

_BLOCKER_RE = re.compile(
    r"(?ix)\b(block(?:ed|er)?|stuck|urgent|asap|cannot|can't|failed|failure|"
    r"bloquead[oa]|atorad[oa]|urgente|no\s+puedo|fall[oó]|error|problema)\b"
)

_DEADLINE_RE = re.compile(
    r"(?ix)\b("
    r"today|tomorrow|tonight|this\s+(?:morning|afternoon|evening|week)|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"hoy|ma[nñ]ana|esta\s+(?:ma[nñ]ana|tarde|noche|semana)|"
    r"lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo|"
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm)|\d{1,2}/\d{1,2}(?:/\d{2,4})?"
    r")\b"
)

_ACK_OR_PARROT_RE = re.compile(
    r"(?ix)^\s*(noted|understood|ok(?:ay)?|on\s+it|got\s+it|sounds\s+good|"
    r"anotado|entendido|de\s+acuerdo|va|sale|listo)\s*[.!]*\s*$"
)

_ALLOWED_ACTION_VERBS_RE = re.compile(
    r"(?ix)\b("
    r"send|ask|remind|summari[sz]e|coordinate|follow\s+up|reply|tell|"
    r"message|dm|contact|avisa|pregunta|recuerda|resume|coordina|"
    r"dale\s+seguimiento|manda|env[ií]a|contesta|escribe"
    r")\b"
)

_NAME_TAG_RE = re.compile(r"@([\w.+\-]+)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_platform_name(platform: Any) -> str:
    return str(getattr(platform, "value", platform) or "").lower()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _normalize_jid(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    # Baileys can expose linked-device JIDs as ``user:device@server`` while
    # native WhatsApp mentions use the bare ``user@server``.  Mention matching
    # must compare those canonical bare JIDs; replacing ':' with '@' produces
    # ``user@device@server`` and makes real @Jack mentions look like mentions
    # of another participant.
    if ":" in text and "@" in text:
        local, server = text.split("@", 1)
        text = f"{local.split(':', 1)[0]}@{server}"
    return text


def _lower_set(values: Iterable[Any]) -> set[str]:
    return {str(v).strip().lower() for v in values if str(v).strip()}


def _get_extra(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    return getattr(config, "extra", {}) if config is not None else {}


def _deep_get(mapping: Any, *keys: str, default: Any = None) -> Any:
    cur = mapping
    for key in keys:
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return default
        if cur is None:
            return default
    return cur


@dataclass
class GroupRoutingConfig:
    enabled: bool = True
    group_mode: str = "mention_only"
    allowed_public_reply: bool = False
    allowed_dm_followup: bool = False
    noise_tolerance: str = "low"
    default_followup_channel: str = "private_jacob"
    hermes_aliases: list[str] = field(default_factory=lambda: ["jack", "hermes"])
    jacob_aliases: list[str] = field(default_factory=lambda: ["jacob"])

    @classmethod
    def from_gateway_config(cls, config: Any) -> "GroupRoutingConfig":
        raw = {}
        # GatewayConfig keeps top-level user config in ``raw`` in newer builds,
        # but tests and some paths pass dicts. Accept both shapes and platform
        # extra overrides for compatibility.
        if isinstance(config, dict):
            raw = _deep_get(config, "gateway", "whatsapp_group_routing", default={}) or {}
        else:
            raw = getattr(config, "whatsapp_group_routing", None) or {}
            raw = raw or _deep_get(getattr(config, "raw", {}), "gateway", "whatsapp_group_routing", default={}) or {}
        platform_extra = {}
        try:
            from gateway.config import Platform
            platforms = getattr(config, "platforms", None)
            platform_cfg = platforms.get(Platform.WHATSAPP) if platforms is not None else None
            platform_extra = _get_extra(platform_cfg)
        except Exception:
            platform_extra = {}
        platform_policy = platform_extra.get("group_routing") if isinstance(platform_extra, dict) else {}
        if isinstance(platform_policy, dict):
            raw = {**raw, **platform_policy}
        aliases = raw.get("aliases", {}) if isinstance(raw.get("aliases"), dict) else {}
        hermes_aliases = aliases.get("hermes") or aliases.get("jack") or raw.get("hermes_aliases")
        jacob_aliases = aliases.get("jacob") or raw.get("jacob_aliases")
        return cls(
            enabled=_coerce_bool(raw.get("enabled", True)),
            group_mode=str(raw.get("group_mode", "mention_only") or "mention_only"),
            allowed_public_reply=_coerce_bool(raw.get("allowed_public_reply", False)),
            allowed_dm_followup=_coerce_bool(raw.get("allowed_dm_followup", False)),
            noise_tolerance=str(raw.get("noise_tolerance", "low") or "low"),
            default_followup_channel=str(raw.get("default_followup_channel", "private_jacob") or "private_jacob"),
            hermes_aliases=[str(a).lower() for a in _as_list(hermes_aliases or ["jack", "hermes"])],
            jacob_aliases=[str(a).lower() for a in _as_list(jacob_aliases or ["jacob"])],
        )


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ExtractedTask:
    task_title: str
    requested_action: str
    owner: Optional[str]
    requester: Optional[str]
    affected_people: list[str]
    due_date: Optional[str]
    pending_confirmations: list[str]
    blocker_indicators: list[str]
    source_platform: str
    source_group: Optional[str]
    source_message_id: Optional[str]
    public_reply_authorization: bool
    follow_up_mode: str
    next_follow_up_time: Optional[str]
    confidence: float
    missing_information: list[str] = field(default_factory=list)


@dataclass
class WhatsAppRoutingDecision:
    platform: str
    chat_type: str
    chat_id: Optional[str]
    message_id: Optional[str]
    sender_id: Optional[str]
    addressee_classification: AddresseeClassification
    selected_action: RoutingAction
    public_reply_allowed: bool
    should_call_llm_reply_generator: bool
    should_call_send_message: bool
    should_send_reaction: bool
    should_ingest_context: bool
    should_extract_task: bool
    should_alert_jacob_privately: bool
    reason: str
    gateway_action: _GATEWAY_ACTION
    group_mode: str = "mention_only"
    context_updated: bool = False
    task_update_created: bool = False
    extracted_task: Optional[ExtractedTask] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("gateway_action", None)
        return data

    def as_hook_result(self) -> dict[str, Any]:
        result = {"action": self.gateway_action, "reason": self.reason, "routing_decision": self.to_dict()}
        if self.gateway_action == "skip":
            result["skip"] = True
        return result


class WhatsAppGroupRoutingPolicy:
    def __init__(self, config: GroupRoutingConfig | None = None):
        self.config = config or GroupRoutingConfig()

    @classmethod
    def from_gateway_config(cls, config: Any) -> "WhatsAppGroupRoutingPolicy":
        return cls(GroupRoutingConfig.from_gateway_config(config))

    def evaluate(self, event: Any) -> WhatsAppRoutingDecision:
        source = getattr(event, "source", None)
        raw = getattr(event, "raw_message", None) or {}
        platform = _as_platform_name(getattr(source, "platform", ""))
        chat_type = str(getattr(source, "chat_type", "") or ("group" if raw.get("isGroup") else "dm"))
        chat_id = getattr(source, "chat_id", None) or raw.get("chatId")
        message_id = getattr(event, "message_id", None) or raw.get("messageId")
        sender_id = getattr(source, "user_id", None) or raw.get("senderId") or raw.get("from")
        text = str(getattr(event, "text", "") or raw.get("body") or "")

        if platform != "whatsapp" or chat_type != "group" or not self.config.enabled:
            return WhatsAppRoutingDecision(
                platform=platform,
                chat_type=chat_type,
                chat_id=chat_id,
                message_id=message_id,
                sender_id=sender_id,
                addressee_classification="addressed_to_hermes" if chat_type == "dm" else "ambient_context",
                selected_action="public_group_reply" if chat_type == "group" else "public_group_reply",
                public_reply_allowed=True,
                should_call_llm_reply_generator=True,
                should_call_send_message=True,
                should_send_reaction=True,
                should_ingest_context=False,
                should_extract_task=False,
                should_alert_jacob_privately=False,
                reason="not_whatsapp_group_or_policy_disabled",
                gateway_action="allow",
                group_mode=self.config.group_mode,
            )

        classification, reason = self.classify_addressee(text, raw, sender_id)
        actionable = self.is_actionable(text)
        urgent_blocker = self.is_urgent_blocker(text)

        if classification == "addressed_to_hermes":
            public_allowed = self.config.allowed_public_reply or self.config.group_mode in {"mention_only", "authorized_coordination", "active_ops"}
            return WhatsAppRoutingDecision(
                platform="whatsapp",
                chat_type="group",
                chat_id=chat_id,
                message_id=message_id,
                sender_id=sender_id,
                addressee_classification=classification,
                selected_action="public_group_reply",
                public_reply_allowed=public_allowed,
                should_call_llm_reply_generator=public_allowed,
                should_call_send_message=public_allowed,
                should_send_reaction=public_allowed,
                should_ingest_context=True,
                should_extract_task=actionable,
                should_alert_jacob_privately=False,
                reason=reason,
                gateway_action="allow" if public_allowed else "skip",
                group_mode=self.config.group_mode,
            )

        if urgent_blocker:
            return WhatsAppRoutingDecision(
                platform="whatsapp",
                chat_type="group",
                chat_id=chat_id,
                message_id=message_id,
                sender_id=sender_id,
                addressee_classification=classification,
                selected_action="private_alert_to_jacob",
                public_reply_allowed=False,
                should_call_llm_reply_generator=False,
                should_call_send_message=False,
                should_send_reaction=False,
                should_ingest_context=True,
                should_extract_task=True,
                should_alert_jacob_privately=True,
                reason="private_alert_preferred",
                gateway_action="skip",
                group_mode=self.config.group_mode,
            )

        selected: RoutingAction
        if actionable:
            selected = "no_reply_task_update"
        elif classification == "ambient_context":
            selected = "ignore_non_actionable_noise"
        else:
            selected = "no_reply_silent_ingest"

        skip_reason = {
            "addressed_to_specific_other_people": "addressed_to_specific_other_people",
            "addressed_to_group": "ambient_group_context",
            "addressed_to_jacob": "group_message_not_addressed_to_hermes",
            "ambient_context": "ambient_group_context",
            "ambiguous": "group_message_not_addressed_to_hermes",
        }.get(classification, "group_message_not_addressed_to_hermes")

        return WhatsAppRoutingDecision(
            platform="whatsapp",
            chat_type="group",
            chat_id=chat_id,
            message_id=message_id,
            sender_id=sender_id,
            addressee_classification=classification,
            selected_action=selected,
            public_reply_allowed=False,
            should_call_llm_reply_generator=False,
            should_call_send_message=False,
            should_send_reaction=False,
            should_ingest_context=True,
            should_extract_task=actionable,
            should_alert_jacob_privately=False,
            reason=skip_reason,
            gateway_action="skip",
            group_mode=self.config.group_mode,
        )

    def classify_addressee(self, text: str, raw: dict[str, Any], sender_id: Any = None) -> tuple[AddresseeClassification, str]:
        mentioned_ids = {_normalize_jid(v) for v in raw.get("mentionedIds") or raw.get("mentionedJid") or [] if _normalize_jid(v)}
        bot_ids = {_normalize_jid(v) for v in raw.get("botIds") or [] if _normalize_jid(v)}
        quoted_participant = _normalize_jid(raw.get("quotedParticipant"))
        raw_body = str(raw.get("body") or "")
        normalized_text = " ".join((text or "").split())
        raw_normalized_text = " ".join(raw_body.split())
        combined_text = " ".join(part for part in [normalized_text, raw_normalized_text] if part)
        lower_text = normalized_text.lower()
        lower_combined_text = combined_text.lower()

        adapter_trigger_reason = raw.get("_hermes_direct_group_trigger_reason")
        if adapter_trigger_reason:
            return "addressed_to_hermes", str(adapter_trigger_reason)

        if bot_ids and mentioned_ids:
            if mentioned_ids & bot_ids:
                return "addressed_to_hermes", "mentioned_hermes_metadata"
            return "addressed_to_specific_other_people", "addressed_to_specific_other_people"

        # WhatsApp/Baileys may expose a display/native mention as raw body text
        # (for example ``@Jack Assistant`` or ``@277365414441152``) and the
        # adapter may already have stripped it out of event.text before this
        # hard gate runs. Check the raw body as well, otherwise a real @Jack
        # group mention can be silently misclassified as a mention of another
        # participant.
        if bot_ids and raw_normalized_text:
            raw_lower = raw_normalized_text.lower()
            for bot_id in bot_ids:
                bare_id = bot_id.split("@", 1)[0].lower()
                if bare_id and (f"@{bare_id}" in raw_lower or raw_lower.startswith(bare_id)):
                    return "addressed_to_hermes", "mentioned_hermes_text"

        if quoted_participant and bot_ids and quoted_participant in bot_ids:
            return "addressed_to_hermes", "reply_to_hermes_message"

        if self._direct_hermes_text_address(combined_text):
            return "addressed_to_hermes", "direct_hermes_text_address"

        if self._direct_jacob_text_address(combined_text):
            return "addressed_to_jacob", "addressed_to_jacob"

        text_tags = _NAME_TAG_RE.findall(combined_text)
        if text_tags:
            hermes_aliases = _lower_set(self.config.hermes_aliases)
            if any(tag.lower().strip("@") in hermes_aliases for tag in text_tags):
                return "addressed_to_hermes", "direct_hermes_text_address"
            return "addressed_to_specific_other_people", "addressed_to_specific_other_people"

        if _ACTIONABLE_RE.search(lower_combined_text):
            # With no metadata/name target, phrases like "can you" in a group are
            # ambiguous. Low-noise default is still no public reply.
            return "ambiguous", "group_message_not_addressed_to_hermes"

        if lower_combined_text and any(token in lower_combined_text for token in ("everyone", "equipo", "team", "todos", "grupo")):
            return "addressed_to_group", "ambient_group_context"
        return "ambient_context", "ambient_group_context"

    def _direct_hermes_text_address(self, text: str) -> bool:
        if not text:
            return False
        for alias in self.config.hermes_aliases:
            alias_re = re.compile(
                rf"(?i)(?:^\s*(?:/?@?{re.escape(alias)})\b\s*[:,;\-]?|@{re.escape(alias)}\b)"
            )
            if alias_re.search(text):
                return True
        return bool(_DIRECT_HERMES_RE.match(text))

    def _direct_jacob_text_address(self, text: str) -> bool:
        if not text:
            return False
        for alias in self.config.jacob_aliases:
            alias_re = re.compile(rf"(?i)(?:^|[\s@,.;:!\-]){re.escape(alias)}(?:\s*[:,;\-]|\s+|$)")
            if alias_re.search(text):
                return True
        return bool(_JACOB_RE.search(text))

    @staticmethod
    def is_actionable(text: str) -> bool:
        return bool(_ACTIONABLE_RE.search(text or ""))

    @staticmethod
    def is_urgent_blocker(text: str) -> bool:
        return bool(_BLOCKER_RE.search(text or ""))

    def extract_task(self, event: Any, decision: WhatsAppRoutingDecision) -> Optional[ExtractedTask]:
        if not decision.should_extract_task:
            return None
        source = getattr(event, "source", None)
        raw = getattr(event, "raw_message", None) or {}
        text = str(getattr(event, "text", "") or raw.get("body") or "").strip()
        mentioned = [str(v) for v in (raw.get("mentionedIds") or raw.get("mentionedJid") or [])]
        owner = mentioned[0] if mentioned else None
        missing: list[str] = []
        if owner is None and decision.addressee_classification in {"ambiguous", "addressed_to_group"}:
            missing.append("owner")
        deadline_match = _DEADLINE_RE.search(text)
        due_date = deadline_match.group(0) if deadline_match else None
        blocker_indicators = [m.group(0) for m in _BLOCKER_RE.finditer(text)]
        pending = []
        if re.search(r"(?i)confirm|confirma|confirmaci[oó]n|me\s+confirmas|por\s+favor\s+confirma", text):
            pending.append("confirmation")
        action = _summarize_action(text)
        task = ExtractedTask(
            task_title=_title_from_text(text),
            requested_action=action,
            owner=owner,
            requester=getattr(source, "user_id", None) or raw.get("senderId"),
            affected_people=mentioned,
            due_date=due_date,
            pending_confirmations=pending,
            blocker_indicators=blocker_indicators,
            source_platform=decision.platform,
            source_group=getattr(source, "chat_id", None) or raw.get("chatId"),
            source_message_id=decision.message_id,
            public_reply_authorization=decision.public_reply_allowed,
            follow_up_mode=self.config.default_followup_channel,
            next_follow_up_time=due_date,
            confidence=0.75 if owner else 0.55,
            missing_information=missing,
        )
        decision.extracted_task = task
        return task

    @staticmethod
    def should_suppress_public_reply(draft: str, source_text: str = "", *, owns_next_action: bool = False) -> bool:
        text = (draft or "").strip()
        if not text:
            return True
        if _ACK_OR_PARROT_RE.match(text):
            return True
        if not owns_next_action and source_text:
            source_words = {w.lower() for w in re.findall(r"\w{4,}", source_text)}
            draft_words = {w.lower() for w in re.findall(r"\w{4,}", text)}
            if draft_words and len(draft_words & source_words) / max(len(draft_words), 1) > 0.72:
                return True
        return False


def _title_from_text(text: str) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= 90:
        return clean or "Operational follow-up"
    return clean[:87].rstrip() + "..."


def _summarize_action(text: str) -> str:
    clean = " ".join((text or "").split())
    return clean or "Review message and update open loop"


class SilentGroupStateStore:
    """Persists skipped context and extracted open loops locally."""

    def __init__(self, session_store: Any = None):
        self.session_store = session_store
        base = get_hermes_home()
        self.log_path = base / "logs" / "whatsapp_group_routing.jsonl"
        self.state_path = base / "gateway" / "whatsapp_group_open_loops.jsonl"

    def apply(self, event: Any, decision: WhatsAppRoutingDecision, policy: WhatsAppGroupRoutingPolicy) -> WhatsAppRoutingDecision:
        if decision.should_extract_task:
            task = policy.extract_task(event, decision)
            decision.task_update_created = task is not None
            if task is not None:
                self._append_jsonl(self.state_path, {"timestamp": _utc_now(), "task": asdict(task), "decision": decision.to_dict()})
        if decision.should_ingest_context:
            decision.context_updated = self._append_observed_transcript(event, decision)
        self.log_decision(event, decision)
        return decision

    def _append_observed_transcript(self, event: Any, decision: WhatsAppRoutingDecision) -> bool:
        if not self.session_store:
            return False
        try:
            source = getattr(event, "source", None)
            if source is None:
                return False
            # Use a group-scoped source so later addressed messages in the same
            # group can retrieve skipped context regardless of sender.
            try:
                import dataclasses
                shared_source = dataclasses.replace(source, user_id=None, user_name=None, user_id_alt=None)
            except Exception:
                shared_source = source
            session_entry = self.session_store.get_or_create_session(shared_source)
            sender = getattr(source, "user_name", None) or getattr(source, "user_id", None) or "unknown"
            content = {
                "observed_whatsapp_group_context": True,
                "sender": sender,
                "text": getattr(event, "text", "") or "",
                "decision": decision.to_dict(),
            }
            entry = {
                "role": "user",
                "content": json.dumps(content, ensure_ascii=False, sort_keys=True),
                "timestamp": _utc_now(),
                "observed": True,
                "routing_decision": decision.to_dict(),
            }
            if decision.message_id:
                entry["message_id"] = str(decision.message_id)
            self.session_store.append_to_transcript(session_entry.session_id, entry)
            return True
        except Exception as exc:
            logger.warning("WhatsApp group routing: failed to ingest skipped context: %s", exc)
            return False

    def log_decision(self, event: Any, decision: WhatsAppRoutingDecision) -> None:
        record = {
            "timestamp": _utc_now(),
            **decision.to_dict(),
            "context_updated": decision.context_updated,
            "task_update_created": decision.task_update_created,
            "skip_reason": decision.reason,
        }
        self._append_jsonl(self.log_path, record)
        logger.info("whatsapp_group_routing_decision %s", json.dumps(record, ensure_ascii=False, sort_keys=True))

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception as exc:
            logger.warning("WhatsApp group routing: failed to append %s: %s", path, exc)


def route_whatsapp_group_event(event: Any, *, gateway_config: Any = None, session_store: Any = None) -> WhatsAppRoutingDecision:
    policy = WhatsAppGroupRoutingPolicy.from_gateway_config(gateway_config)
    decision = policy.evaluate(event)
    if decision.gateway_action == "skip" or decision.should_ingest_context or decision.should_extract_task:
        decision = SilentGroupStateStore(session_store=session_store).apply(event, decision, policy)
    return decision


__all__ = [
    "ExtractedTask",
    "GroupRoutingConfig",
    "SilentGroupStateStore",
    "WhatsAppGroupRoutingPolicy",
    "WhatsAppRoutingDecision",
    "route_whatsapp_group_event",
]
