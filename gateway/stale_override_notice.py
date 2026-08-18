"""Idle notices for stale session-scoped model/reasoning overrides.

This module contains the policy-only pieces.  GatewayRunner owns the actual
hold/resume flow because it already owns session overrides and platform
adapters; keeping matching and comparison here makes the contract independently
testable and avoids adding more policy to gateway/run.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from hermes_constants import VALID_REASONING_EFFORTS


NOTICE_MODES = {"off", "info_only", "confirm"}
MODEL_POLICIES = {"off", "non_default"}
REASONING_POLICIES = {"off", "above_default", "non_default"}

# Ordered from disabled through the canonical efforts accepted by Hermes.
_REASONING_RANK = {
    "none": -1,
    **{effort: rank for rank, effort in enumerate(VALID_REASONING_EFFORTS)},
}


def _choice(value: Any, allowed: set[str], default: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    return normalized if normalized in allowed else default


def _positive_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _channels(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("home",)
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return ("home",)
    return tuple(str(item).strip().lower() for item in value if str(item).strip())


@dataclass
class StaleOverrideNoticeConfig:
    """Configuration for first-message-after-idle override notices."""

    mode: str = "off"
    idle_minutes: float = 60.0
    model: str = "non_default"
    reasoning: str = "above_default"
    channels: tuple[str, ...] = ("home",)

    @classmethod
    def from_dict(cls, data: Any) -> "StaleOverrideNoticeConfig":
        raw = data if isinstance(data, Mapping) else {}
        return cls(
            mode=_choice(raw.get("mode"), NOTICE_MODES, "off"),
            idle_minutes=_positive_float(raw.get("idle_minutes"), 60.0),
            model=_choice(raw.get("model"), MODEL_POLICIES, "non_default"),
            reasoning=_choice(
                raw.get("reasoning"), REASONING_POLICIES, "above_default"
            ),
            channels=_channels(raw.get("channels")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "idle_minutes": self.idle_minutes,
            "model": self.model,
            "reasoning": self.reasoning,
            "channels": list(self.channels),
        }


def _platform_value(source: Any) -> str:
    platform = getattr(source, "platform", None)
    return str(getattr(platform, "value", platform) or "").strip().lower()


def source_matches_channels(
    source: Any,
    selectors: Iterable[str],
    *,
    home_channel: Any = None,
) -> bool:
    """Return whether *source* is in the configured channel scope.

    An empty selector list means all channels. Supported selectors are ``*``,
    ``home``, ``platform:*``, ``platform:chat_id``, and
    ``platform:chat_id:thread_id``. A chat selector without a thread matches the
    whole chat, including its threads; a thread selector is exact.
    """

    normalized = tuple(
        str(item).strip().lower() for item in selectors if str(item).strip()
    )
    if not normalized:
        return True

    platform = _platform_value(source)
    chat_id = str(getattr(source, "chat_id", None) or "").strip().lower()
    thread_id = str(getattr(source, "thread_id", None) or "").strip().lower()
    chat_selector = f"{platform}:{chat_id}"
    thread_selector = f"{chat_selector}:{thread_id}" if thread_id else ""

    for selector in normalized:
        if selector == "*" or selector == f"{platform}:*":
            return True
        if selector == chat_selector or (
            thread_selector and selector == thread_selector
        ):
            return True
        if selector == "home" and home_channel is not None:
            home_platform = _platform_value(home_channel)
            home_chat = (
                str(getattr(home_channel, "chat_id", None) or "").strip().lower()
            )
            home_thread = (
                str(getattr(home_channel, "thread_id", None) or "").strip().lower()
            )
            if platform == home_platform and chat_id == home_chat:
                if not home_thread or thread_id == home_thread:
                    return True
    return False


def route_label(model: Any, provider: Any) -> str:
    model_text = str(model or "unknown")
    provider_text = str(provider or "").strip()
    return f"{provider_text}/{model_text}" if provider_text else model_text


def routes_differ(
    current_model: Any,
    current_provider: Any,
    default_model: Any,
    default_provider: Any,
) -> bool:
    current = (
        str(current_provider or "").strip().lower(),
        str(current_model or "").strip().lower(),
    )
    default = (
        str(default_provider or "").strip().lower(),
        str(default_model or "").strip().lower(),
    )
    return current != default


def reasoning_effort(config: Optional[Mapping[str, Any]]) -> str:
    """Normalize a Hermes reasoning config to an ordered effort label."""

    if config is not None and config.get("enabled") is False:
        return "none"
    effort = str((config or {}).get("effort") or "medium").strip().lower()
    return effort if effort in _REASONING_RANK else "medium"


def reasoning_matches_policy(
    policy: str,
    override: Optional[Mapping[str, Any]],
    default: Optional[Mapping[str, Any]],
) -> bool:
    if policy == "off" or override is None:
        return False
    current_effort = reasoning_effort(override)
    default_effort = reasoning_effort(default)
    if policy == "above_default":
        return _REASONING_RANK[current_effort] > _REASONING_RANK[default_effort]
    return current_effort != default_effort


@dataclass(frozen=True)
class OverrideNoticeDecision:
    model_stale: bool = False
    reasoning_stale: bool = False
    current_route: str = ""
    default_route: str = ""
    current_reasoning: str = ""
    default_reasoning: str = ""

    @property
    def triggered(self) -> bool:
        return self.model_stale or self.reasoning_stale

    def choices(self) -> list[dict[str, Any]]:
        choices = [
            {
                "value": "continue",
                "label": "✅ Continue with current settings",
                # The label already carries the affirmative marker. Keeping
                # is_current false avoids a duplicate generic checkmark in
                # Telegram while Discord still shows the emoji.
                "is_current": False,
            }
        ]
        if self.model_stale:
            choices.append({
                "value": "default_model",
                "label": "↩️ Restore default model",
                "is_current": False,
            })
        if self.reasoning_stale:
            choices.append({
                "value": "default_reasoning",
                "label": "↩️ Restore default reasoning",
                "is_current": False,
            })
        if self.model_stale and self.reasoning_stale:
            choices.append({
                "value": "defaults",
                "label": "↩️ Restore all defaults",
                "is_current": False,
            })
        return choices

    def message(self, idle_minutes: float, *, held: bool) -> str:
        lines = [
            "Session override still active",
            f"This is the first message after about {max(1, round(idle_minutes))} minutes idle.",
        ]
        if self.model_stale:
            lines.append(
                f"Model: `{self.current_route}` (default: `{self.default_route}`)"
            )
        if self.reasoning_stale:
            lines.append(
                f"Reasoning: `{self.current_reasoning}` "
                f"(default: `{self.default_reasoning}`)"
            )
        if held:
            lines.append(
                "Choose how to send the pending message. It will not be sent if this prompt expires."
            )
        else:
            lines.append(
                "The message is continuing without interruption (info-only mode)."
            )
        return "\n".join(lines)
