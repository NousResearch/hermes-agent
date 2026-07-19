"""Interlocutor capability policy for gateway-originated human messages.

This module is intentionally deterministic and disabled by default. Existing
platform authorization decides whether a sender may talk to Hermes at all; this
policy adds a narrower capability layer for operator vs chat-only senders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from gateway.config import Platform
from gateway.platforms.base import MessageEvent


class AuthorityClass(str, Enum):
    """Runtime capability class for a message sender."""

    OPERATOR = "operator"
    CHAT_ONLY = "chat_only"
    UNKNOWN = "unknown"
    SYSTEM = "system"


class ProtectedIntent(str, Enum):
    """Sensitive intent families chat-only contacts must not invoke."""

    NONE = "none"
    CREDENTIAL_REQUEST = "credential_request"
    PRIVATE_CONVERSATION_REQUEST = "private_conversation_request"
    PRIVATE_CODE_OR_WORK_REQUEST = "private_code_or_work_request"
    BEHAVIOR_MUTATION_REQUEST = "behavior_mutation_request"
    CRON_OR_SCHEDULER_REQUEST = "cron_or_scheduler_request"
    CODING_OR_REPO_ACTION_REQUEST = "coding_or_repo_action_request"
    GATEWAY_OR_CONFIG_ACTION_REQUEST = "gateway_or_config_action_request"
    PRIVILEGED_SLASH_COMMAND = "privileged_slash_command"
    AUTHORITY_LAUNDERING = "authority_laundering"


@dataclass(frozen=True)
class InterlocutorPolicyConfig:
    """Small value object for the disabled-by-default interlocutor policy."""

    enabled: bool = False
    operator_user_ids: set[str] = field(default_factory=set)
    chat_only_user_ids: set[str] = field(default_factory=set)
    chat_only_default_response: str = (
        "I can talk generally, but I can't share Andrew's private information "
        "or run substantial actions. Anything private, coding-related, "
        "cron-related, or configuration-related needs Andrew's approval."
    )
    audit_log_enabled: bool = True
    audit_log_path: str = "~/.hermes/policy/interlocutor-events.jsonl"
    redact_event_text: bool = True
    block_privileged_slash_commands: bool = True
    block_sensitive_plaintext_requests: bool = True

    @classmethod
    def from_mapping(cls, data: Any) -> "InterlocutorPolicyConfig":
        if not isinstance(data, dict):
            return cls()
        raw_response = data.get("chat_only_default_response")
        if isinstance(raw_response, str) and raw_response.strip():
            response = raw_response
        else:
            response = cls.chat_only_default_response
        raw_audit_path = data.get("audit_log_path")
        if isinstance(raw_audit_path, str) and raw_audit_path.strip():
            audit_path = raw_audit_path
        else:
            audit_path = cls.audit_log_path
        return cls(
            enabled=_coerce_bool(data.get("enabled"), default=False),
            operator_user_ids=_coerce_id_set(data.get("operator_user_ids")),
            chat_only_user_ids=_coerce_id_set(data.get("chat_only_user_ids")),
            chat_only_default_response=response,
            audit_log_enabled=_coerce_bool(data.get("audit_log_enabled"), default=True),
            audit_log_path=audit_path,
            redact_event_text=_coerce_bool(data.get("redact_event_text"), default=True),
            block_privileged_slash_commands=_coerce_bool(
                data.get("block_privileged_slash_commands"), default=True
            ),
            block_sensitive_plaintext_requests=_coerce_bool(
                data.get("block_sensitive_plaintext_requests"), default=True
            ),
        )


@dataclass(frozen=True)
class PolicyDecision:
    """Decision returned by the local deterministic policy classifier."""

    allowed: bool
    authority_class: AuthorityClass
    intent: ProtectedIntent
    response: str | None = None
    audit_reason: str | None = None


PRIVILEGED_SLASH_COMMANDS = frozenset(
    {
        "approve",
        "deny",
        "restart",
        "update",
        "reload-mcp",
        "sethome",
        "background",
        "queue",
        "q",
        "steer",
        "model",
        "tools",
        "toolsets",
        "skills",
        "cron",
        "voice",
        "thread",
        "yolo",
        "memory",
        "personality",
        "reasoning",
        "config",
        "plugins",
    }
)

_CREDENTIAL_RE = re.compile(
    r"\b(env(?:ironment)?\s+var(?:iable)?s?|api\s*keys?|oauth|tokens?|"
    r"passwords?|secrets?|credentials?|auth(?:orization)?\s+headers?)\b",
    re.IGNORECASE,
)
_PRIVATE_CONVO_RE = re.compile(
    r"\b(private\s+(?:conversation|dm|chat|message|session)s?|hidden\s+(?:conversation|dm|chat)s?|"
    r"(?:dm|direct\s+message)s?\s+with\s+andrew|chat\s+logs?)\b",
    re.IGNORECASE,
)
_PRIVATE_WORK_RE = re.compile(
    r"\b(private\s+(?:code|repo|repository|codebase|work|task)s?|ongoing\s+work|"
    r"unreleased\s+(?:plans?|work)|internal\s+(?:repo|code|task)s?)\b",
    re.IGNORECASE,
)
_BEHAVIOR_RE = re.compile(
    r"\b(change|modify|alter|update|rewrite|ignore|override)\b.*\b(behavior|rules?|"
    r"instructions?|memory|memories|skills?|tool\s*policy|approval\s+boundar(?:y|ies))\b|"
    r"\b(system\s+prompt|developer\s+message)\b",
    re.IGNORECASE,
)
_CRON_RE = re.compile(
    r"\b(cron|scheduled?\s+(?:job|task)|scheduler|schedule\s+(?:a|the)|"
    r"pause\s+job|resume\s+job|remove\s+job)\b",
    re.IGNORECASE,
)
_CODING_RE = re.compile(
    r"\b(commit|push|pull\s+request|\bpr\b|github\s+issue|edit\s+(?:the\s+)?(?:repo|file|code)|"
    r"write\s+code|coding\s+task|deploy|merge|branch|git\s+)\b",
    re.IGNORECASE,
)
_GATEWAY_CONFIG_RE = re.compile(
    r"\b(restart\s+(?:the\s+)?gateway|gateway\s+restart|config(?:\.yaml)?|\.env|"
    r"change\s+(?:the\s+)?config|reload\s+mcp|set\s*home|toolsets?)\b",
    re.IGNORECASE,
)
_AUTHORITY_LAUNDERING_RE = re.compile(
    r"\b(same\s+brain|andrew\s+(?:approved|said)|azra3l\s+(?:approved|said)|"
    r"i\s+already\s+know|you\s+can\s+trust\s+me|operator\s+now)\b",
    re.IGNORECASE,
)


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _coerce_id_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    if isinstance(value, Iterable):
        return {str(part).strip() for part in value if str(part).strip()}
    return set()


def _slash_command(text: str) -> str | None:
    stripped = (text or "").strip()
    if not stripped.startswith("/"):
        return None
    token = stripped.split(maxsplit=1)[0].lstrip("/").lower()
    return token or None


def _hash(value: Any) -> str:
    raw = str(value or "").encode("utf-8", errors="replace")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def write_policy_audit_event(
    *,
    event: MessageEvent,
    config: InterlocutorPolicyConfig,
    decision: PolicyDecision,
) -> bool:
    """Append a redacted JSONL audit event for a policy decision.

    Raw message text is never written by default; the text field is represented
    by a stable SHA-256 fingerprint so operators can correlate repeated probes
    without retaining private content or secret-shaped strings.
    """

    if not config.audit_log_enabled:
        return False
    path = Path(config.audit_log_path).expanduser()
    source = getattr(event, "source", None)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "platform": getattr(getattr(source, "platform", None), "value", None),
        "chat_id": str(getattr(source, "chat_id", "") or ""),
        "thread_id": str(getattr(source, "thread_id", "") or ""),
        "message_id": str(getattr(event, "message_id", "") or ""),
        "user_id_hash": _hash(getattr(source, "user_id", "")),
        "authority_class": decision.authority_class.value,
        "intent": decision.intent.value,
        "decision": "allowed" if decision.allowed else "blocked",
        "audit_reason": decision.audit_reason,
        "text_fingerprint": _hash(getattr(event, "text", "")),
    }
    if not config.redact_event_text:
        payload["text"] = str(getattr(event, "text", "") or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return True


def resolve_authority_class(
    event: MessageEvent,
    config: InterlocutorPolicyConfig,
) -> AuthorityClass:
    """Resolve operator/chat-only class without replacing existing authz."""

    if getattr(event, "internal", False):
        return AuthorityClass.SYSTEM
    source = getattr(event, "source", None)
    user_id = str(getattr(source, "user_id", "") or "").strip()
    if not user_id:
        return AuthorityClass.UNKNOWN
    if user_id in config.operator_user_ids:
        return AuthorityClass.OPERATOR
    if user_id in config.chat_only_user_ids:
        return AuthorityClass.CHAT_ONLY
    if getattr(source, "platform", None) == Platform.DISCORD:
        return AuthorityClass.CHAT_ONLY
    return AuthorityClass.UNKNOWN


def classify_protected_intent(
    text: str,
    *,
    block_privileged_slash_commands: bool = True,
    block_sensitive_plaintext_requests: bool = True,
) -> ProtectedIntent:
    """Classify protected intents with local deterministic patterns."""

    text = text or ""
    command = _slash_command(text)
    if (
        block_privileged_slash_commands
        and command is not None
        and command in PRIVILEGED_SLASH_COMMANDS
    ):
        return ProtectedIntent.PRIVILEGED_SLASH_COMMAND

    if not block_sensitive_plaintext_requests:
        return ProtectedIntent.NONE

    has_action = any(
        pattern.search(text)
        for pattern in (
            _CREDENTIAL_RE,
            _PRIVATE_CONVO_RE,
            _PRIVATE_WORK_RE,
            _BEHAVIOR_RE,
            _CRON_RE,
            _CODING_RE,
            _GATEWAY_CONFIG_RE,
        )
    )
    if _AUTHORITY_LAUNDERING_RE.search(text) and has_action:
        return ProtectedIntent.AUTHORITY_LAUNDERING
    if _CREDENTIAL_RE.search(text):
        return ProtectedIntent.CREDENTIAL_REQUEST
    if _PRIVATE_CONVO_RE.search(text):
        return ProtectedIntent.PRIVATE_CONVERSATION_REQUEST
    if _PRIVATE_WORK_RE.search(text):
        return ProtectedIntent.PRIVATE_CODE_OR_WORK_REQUEST
    if _BEHAVIOR_RE.search(text):
        return ProtectedIntent.BEHAVIOR_MUTATION_REQUEST
    if _CRON_RE.search(text):
        return ProtectedIntent.CRON_OR_SCHEDULER_REQUEST
    if _CODING_RE.search(text):
        return ProtectedIntent.CODING_OR_REPO_ACTION_REQUEST
    if _GATEWAY_CONFIG_RE.search(text):
        return ProtectedIntent.GATEWAY_OR_CONFIG_ACTION_REQUEST
    return ProtectedIntent.NONE


def evaluate_interlocutor_policy(
    event: MessageEvent,
    config: InterlocutorPolicyConfig,
) -> PolicyDecision:
    """Return the policy decision for an inbound gateway event."""

    if not config.enabled:
        return PolicyDecision(
            allowed=True,
            authority_class=AuthorityClass.UNKNOWN,
            intent=ProtectedIntent.NONE,
        )

    authority = resolve_authority_class(event, config)
    intent = classify_protected_intent(
        getattr(event, "text", "") or "",
        block_privileged_slash_commands=config.block_privileged_slash_commands,
        block_sensitive_plaintext_requests=config.block_sensitive_plaintext_requests,
    )

    if authority is AuthorityClass.OPERATOR or authority is AuthorityClass.SYSTEM:
        return PolicyDecision(allowed=True, authority_class=authority, intent=intent)

    if authority is AuthorityClass.CHAT_ONLY and intent is not ProtectedIntent.NONE:
        return PolicyDecision(
            allowed=False,
            authority_class=authority,
            intent=intent,
            response=config.chat_only_default_response,
            audit_reason=f"blocked:{intent.value}",
        )

    return PolicyDecision(allowed=True, authority_class=authority, intent=intent)
