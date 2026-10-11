"""Validate native token measurements before provider SDK coercion."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


class NativeUsageError(ValueError):
    code = "invalid_usage"

    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(
            f"Invalid native usage at {field}: expected a non-negative integer"
        )


def _field(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)


def _count(value: Any, path: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise NativeUsageError(path)
    return value


def _details(usage: Any, key: str) -> Any:
    value = _field(usage, key)
    if (
        value is not None
        and not isinstance(value, Mapping)
        and not hasattr(value, "__dict__")
    ):
        raise NativeUsageError(f"usage.{key}")
    return value


@dataclass(frozen=True)
class NativeUsageCounts:
    input: int | None
    output: int | None
    total: int | None
    cached: int = 0
    cache_write: int = 0
    reasoning: int = 0

    @classmethod
    def read(
        cls,
        usage: Any,
        *,
        kind: Literal["gemini", "chat", "codex", "anthropic", "bedrock"],
    ) -> "NativeUsageCounts":
        if not isinstance(usage, Mapping) and not hasattr(usage, "__dict__"):
            raise NativeUsageError("usage")
        fields = {
            "gemini": ("promptTokenCount", "candidatesTokenCount", "totalTokenCount"),
            "chat": ("prompt_tokens", "completion_tokens", "total_tokens"),
            "codex": ("input_tokens", "output_tokens", "total_tokens"),
            "anthropic": ("input_tokens", "output_tokens", None),
            "bedrock": ("inputTokens", "outputTokens", "totalTokens"),
        }[kind]
        input_tokens, output_tokens, total_tokens = (
            _count(_field(usage, key), f"usage.{key}") if key is not None else None
            for key in fields
        )
        cached = cache_write = reasoning = None
        if kind == "gemini":
            cached = _count(
                _field(usage, "cachedContentTokenCount"),
                "usage.cachedContentTokenCount",
            )
            reasoning = _count(
                _field(usage, "thoughtsTokenCount"), "usage.thoughtsTokenCount"
            )
        elif kind == "bedrock":
            cached = _count(
                _field(usage, "cacheReadInputTokens"), "usage.cacheReadInputTokens"
            )
            cache_write = _count(
                _field(usage, "cacheWriteInputTokens"), "usage.cacheWriteInputTokens"
            )
        elif kind == "anthropic":
            cached = _count(
                _field(usage, "cache_read_input_tokens"),
                "usage.cache_read_input_tokens",
            )
            cache_write = _count(
                _field(usage, "cache_creation_input_tokens"),
                "usage.cache_creation_input_tokens",
            )
            details = _details(usage, "cache_creation")
            for key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
                _count(_field(details, key), f"usage.cache_creation.{key}")
        else:
            input_key = (
                "prompt_tokens_details" if kind == "chat" else "input_tokens_details"
            )
            output_key = (
                "completion_tokens_details"
                if kind == "chat"
                else "output_tokens_details"
            )
            details = _details(usage, input_key)
            cached = _count(
                _field(details, "cached_tokens"), f"usage.{input_key}.cached_tokens"
            )
            cache_write = _count(
                _field(details, "cache_write_tokens"),
                f"usage.{input_key}.cache_write_tokens",
            )
            creation = _count(
                _field(details, "cache_creation_tokens"),
                f"usage.{input_key}.cache_creation_tokens",
            )
            if cache_write is None:
                cache_write = creation
            details = _details(usage, output_key)
            reasoning = _count(
                _field(details, "reasoning_tokens"),
                f"usage.{output_key}.reasoning_tokens",
            )
            if kind == "chat":
                for key in (
                    "audio_tokens",
                    "accepted_prediction_tokens",
                    "rejected_prediction_tokens",
                ):
                    _count(_field(details, key), f"usage.{output_key}.{key}")
                _count(
                    _field(_details(usage, input_key), "audio_tokens"),
                    f"usage.{input_key}.audio_tokens",
                )
        return cls(
            input_tokens,
            output_tokens,
            total_tokens,
            cached or 0,
            cache_write or 0,
            reasoning or 0,
        )


def validate_openai_usage(
    data: object, *, kind: Literal["chat", "codex"] = "chat"
) -> None:
    if not isinstance(data, dict):
        return
    if data.get("object") == "response" or str(data.get("type", "")).startswith(
        "response."
    ):
        kind = "codex"
    completion = data.get("data")
    if (
        not data.get("choices")
        and data.get("success") is not False
        and isinstance(completion, dict)
        and completion.get("choices")
    ):
        validate_openai_usage(completion, kind=kind)
    if isinstance(data.get("response"), dict):
        validate_openai_usage(data["response"], kind="codex")
    usage = data.get("usage")
    if usage is None:
        return
    counts = NativeUsageCounts.read(usage, kind=kind)
    if kind != "chat":
        return
    if counts.input is None or counts.output is None:
        data["usage"] = None
        return
    if counts.total is None:
        usage["total_tokens"] = counts.input + counts.output


def validate_anthropic_usage(data: object) -> None:
    if not isinstance(data, dict):
        return
    if isinstance(data.get("message"), dict):
        validate_anthropic_usage(data["message"])
    if (usage := data.get("usage")) is not None:
        NativeUsageCounts.read(usage, kind="anthropic")
