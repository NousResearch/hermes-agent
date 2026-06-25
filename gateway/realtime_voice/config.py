"""Configuration primitives for experimental realtime voice sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

DEFAULT_REALTIME_TOOLS: tuple[str, ...] = (
    "ask_agent",
    "start_agent_task",
    "get_agent_task_status",
    "summarize_agent_task",
    "ask_context",
)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        return default
    return bool(value)


def _coerce_int(value: Any, default: int, *, minimum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    return result


def _clean_nonempty_string(value: Any, default: str) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return default


@dataclass(frozen=True)
class RealtimeVoiceConfig:
    """Profile-level config for the experimental gateway realtime voice layer.

    This is intentionally provider-neutral. Discord voice can opt into the
    realtime command surface with ``discord.voice_backend: realtime`` while the
    realtime provider, model, voice, duration caps, and tiny tool bridge live in
    the top-level ``realtime_voice`` block.
    """

    provider: str = "xai"
    model: str = "grok-voice-latest"
    voice: str = "ara"
    max_session_minutes: int = 20
    max_background_tasks: int = 3
    transcript_to_text_channel: bool = True
    allow_tools: tuple[str, ...] = DEFAULT_REALTIME_TOOLS
    providers: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RealtimeVoiceConfig":
        if not isinstance(data, Mapping):
            return cls()

        raw_tools = data.get("allow_tools", DEFAULT_REALTIME_TOOLS)
        if isinstance(raw_tools, (list, tuple)):
            allow_tools = tuple(
                item.strip()
                for item in raw_tools
                if isinstance(item, str) and item.strip()
            )
        elif raw_tools is None:
            allow_tools = DEFAULT_REALTIME_TOOLS
        else:
            allow_tools = DEFAULT_REALTIME_TOOLS

        raw_providers = data.get("providers", {})
        providers: dict[str, Any] = {}
        if isinstance(raw_providers, Mapping):
            providers = {
                str(name).strip().lower(): value
                for name, value in raw_providers.items()
                if str(name).strip()
            }

        return cls(
            provider=_clean_nonempty_string(data.get("provider"), "xai").lower(),
            model=_clean_nonempty_string(data.get("model"), "grok-voice-latest"),
            voice=_clean_nonempty_string(data.get("voice"), "ara"),
            max_session_minutes=_coerce_int(
                data.get("max_session_minutes"),
                20,
                minimum=1,
            ),
            max_background_tasks=_coerce_int(
                data.get("max_background_tasks"),
                3,
                minimum=0,
            ),
            transcript_to_text_channel=_coerce_bool(
                data.get("transcript_to_text_channel"),
                True,
            ),
            allow_tools=allow_tools,
            providers=providers,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "voice": self.voice,
            "max_session_minutes": self.max_session_minutes,
            "max_background_tasks": self.max_background_tasks,
            "transcript_to_text_channel": self.transcript_to_text_channel,
            "allow_tools": list(self.allow_tools),
            "providers": dict(self.providers),
        }
