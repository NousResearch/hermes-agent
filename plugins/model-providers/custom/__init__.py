"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances and OpenAI-compatible reasoning endpoints (GLM-5.2 on
Volcengine ARK, vLLM, llama.cpp). Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled → top-level reasoning_effort="none"
    (Ollama /v1/chat/completions ignores think=False — ollama#14820)
    + extra_body.think = False for /api/chat and proxies
  - reasoning_config enabled + effort → top-level reasoning_effort
    (the native OpenAI-compatible format GLM/ARK expect; unset omits it
    so the endpoint's server default applies)
  - tool-loop payloads with no plain user query get a synthetic user
    turn (Ollama qwen3.8 renderer 500s otherwise — ollama#17778)
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Ollama's qwen3.8 renderer validateMessages() requires at least one
# user-role message that is not a <tool_response> wrapper. Hermes tool
# loops and compaction can legally send [system, assistant(tool_calls),
# tool] — inject a continuation user turn for the wire only.
_OLLAMA_USER_QUERY_PLACEHOLDER = (
    "Continue with the current task using the latest tool results."
)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _has_plain_user_query(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        text = _message_text(msg.get("content")).strip()
        if text and "<tool_response>" not in text:
            return True
    return False


def _looks_like_tool_loop(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool" or msg.get("tool_calls"):
            return True
        if msg.get("role") == "user" and "<tool_response>" in _message_text(
            msg.get("content")
        ):
            return True
    return False


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false and num_ctx support."""

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep a plain user query on the wire for local Ollama tool loops.

        Does not mutate the caller's list. The placeholder is request-only.
        """
        if not messages or _has_plain_user_query(messages):
            return messages
        if not _looks_like_tool_loop(messages):
            return messages

        prepared = list(messages)
        insert_at = 0
        while insert_at < len(prepared):
            msg = prepared[insert_at]
            if not isinstance(msg, dict) or msg.get("role") != "system":
                break
            insert_at += 1
        prepared.insert(
            insert_at,
            {"role": "user", "content": _OLLAMA_USER_QUERY_PLACEHOLDER},
        )
        return prepared

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # Ollama context window
        if ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_num_ctx
            extra_body["options"] = options

        # Reasoning / thinking control for custom OpenAI-compatible endpoints
        # (GLM-5.2 on Volcengine ARK, vLLM, Ollama, llama.cpp, …).
        #
        #   - disabled  → extra_body.think = False (Ollama's thinking-off flag)
        #   - enabled + effort set → TOP-LEVEL reasoning_effort string, the
        #     format GLM-5.2/ARK and other OpenAI-compatible reasoning APIs
        #     expect (GLM documents "high" and "max"; "max" is its default).
        #   - enabled + no effort  → omit both, so the endpoint applies its own
        #     server-side default (do NOT force a level the user didn't pick).
        #
        # We deliberately do NOT emit ``think=True`` on enable: it is an
        # Ollama-only flag and thinking is already server-default-on for these
        # backends, so forcing it risks a 400 on GLM/vLLM endpoints that don't
        # recognize it. Mirrors the DeepSeek/Zai profile precedent.
        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)
            if _effort == "none" or _enabled is False:
                # Ollama's /v1/chat/completions silently ignores
                # extra_body.think (only /api/chat honours it — ollama#14820)
                # but respects the top-level reasoning_effort field, so both
                # are needed to actually stop a thinking-capable model from
                # reasoning (#25758). Endpoints that recognize neither simply
                # ignore them.
                top_level["reasoning_effort"] = "none"
                extra_body["think"] = False
            elif _effort:
                # Clamp the internal ladder onto the endpoint's wire
                # vocabulary (shared policy in agent.reasoning_effort).
                # qwen3.8 is a narrower local-Ollama set; other custom
                # OpenAI-compat endpoints keep the wide vocabulary.
                # Forwarding "ultra" verbatim is a guaranteed 400 (#89503).
                from agent.reasoning_effort import (
                    clamp_effort,
                    custom_endpoint_efforts,
                )

                top_level["reasoning_effort"] = clamp_effort(
                    _effort, custom_endpoint_efforts(ctx.get("model"))
                )

        return extra_body, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Custom/Ollama: base_url is user-configured; fetch if set."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


custom = CustomProfile(
    name="custom",
    aliases=(
        "ollama",
        "local",
        "vllm",
        "llamacpp",
        "llama.cpp",
        "llama-cpp",
    ),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
    # Without this, no max_tokens is sent and Ollama falls back to its internal
    # num_predict=128, truncating responses after a few tokens (#39281). This is
    # only a floor used when the user hasn't set model.max_tokens — they can
    # override per-model — so we set it generously rather than lowballing it.
    default_max_tokens=65536,
)

register_provider(custom)
