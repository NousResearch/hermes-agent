"""Runtime transport profiles for provider/model execution.

Hermes historically routed requests with a provider-wide ``api_mode`` flag.
This module adds an explicit transport profile so future providers can resolve
execution protocol from ``provider + model`` while preserving the legacy
``api_mode`` contract for existing callers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
OPENAI_RESPONSES = "openai_responses"
ANTHROPIC_MESSAGES = "anthropic_messages"
GOOGLE_GENERATE_CONTENT = "google_generate_content"

LEGACY_API_MODES = {
    OPENAI_CHAT_COMPLETIONS: "chat_completions",
    OPENAI_RESPONSES: "codex_responses",
    ANTHROPIC_MESSAGES: "chat_completions",
    GOOGLE_GENERATE_CONTENT: "chat_completions",
}

_PROVIDER_DEFAULT_TRANSPORTS = {
    "openrouter": OPENAI_CHAT_COMPLETIONS,
    "custom": OPENAI_CHAT_COMPLETIONS,
    "nous": OPENAI_CHAT_COMPLETIONS,
    "nous-api": OPENAI_CHAT_COMPLETIONS,
    "openai-codex": OPENAI_RESPONSES,
    "zai": OPENAI_CHAT_COMPLETIONS,
    "kimi-coding": OPENAI_CHAT_COMPLETIONS,
    "minimax": OPENAI_CHAT_COMPLETIONS,
    "minimax-cn": OPENAI_CHAT_COMPLETIONS,
}


@dataclass(frozen=True)
class TransportProfile:
    provider: str
    transport: str
    api_mode: str
    base_url: str = ""
    model: str = ""
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["metadata"] is None:
            data.pop("metadata")
        return data


def legacy_api_mode_for_transport(transport: Optional[str]) -> str:
    return LEGACY_API_MODES.get((transport or "").strip(), "chat_completions")


def default_transport_for_provider(provider: Optional[str]) -> str:
    normalized = (provider or "openrouter").strip().lower()
    return _PROVIDER_DEFAULT_TRANSPORTS.get(normalized, OPENAI_CHAT_COMPLETIONS)


def build_transport_profile(
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    transport: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> TransportProfile:
    normalized_provider = (provider or "openrouter").strip().lower()
    resolved_transport = (transport or default_transport_for_provider(normalized_provider)).strip().lower()
    return TransportProfile(
        provider=normalized_provider,
        transport=resolved_transport,
        api_mode=legacy_api_mode_for_transport(resolved_transport),
        base_url=(base_url or "").rstrip("/"),
        model=(model or "").strip(),
        metadata=dict(metadata) if metadata else None,
    )


def is_responses_transport(value: TransportProfile | str | None) -> bool:
    if isinstance(value, TransportProfile):
        return value.transport == OPENAI_RESPONSES
    return (value or "").strip().lower() == OPENAI_RESPONSES
