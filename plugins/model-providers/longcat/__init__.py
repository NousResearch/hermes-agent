"""LongCat's OpenAI-compatible API with binary thinking and reasoning replay."""

from providers import register_provider
from providers.base import ProviderProfile


class LongCatProfile(ProviderProfile):
    def build_api_kwargs_extras(self, *, reasoning_config=None, **context):
        config = reasoning_config if isinstance(reasoning_config, dict) else {}
        disabled = config.get("enabled") is False or config.get("effort") == "none"
        return {"thinking": {"type": "disabled" if disabled else "enabled"}}, {}

    def prepare_messages(self, messages):
        # Tool-call-only assistant turns may have null content in durable history;
        # LongCat requires a string on the wire. Leave the cached history untouched.
        return [
            {**message, "content": ""}
            if message.get("role") == "assistant" and message.get("content") is None
            else message
            for message in messages
        ]


register_provider(LongCatProfile(
    name="longcat", display_name="LongCat", description="LongCat (direct API)",
    signup_url="https://longcat.chat/platform/api_keys", env_vars=("LONGCAT_API_KEY",),
    base_url="https://api.longcat.chat/openai/v1", fallback_models=("LongCat-2.0",),
    requires_reasoning_echo=True,
))
