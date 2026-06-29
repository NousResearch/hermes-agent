"""AnyRouter provider profile.

AnyRouter (https://anyrouter.dev) is a unified model gateway with an
OpenAI-style ``/v1`` surface and ``vendor/model`` routing slugs. This profile
wires the request-shaping that AnyRouter's API actually documents and accepts:

  - ``extra_body.session_id`` — sticky-session id grouping related requests in
    the Request Logs dashboard (body wins over the ``x-session-id`` header).
  - ``extra_body.provider`` — request-level routing preferences (only / ignore
    / order / sort / allow_fallbacks / max_price …).
  - ``extra_body.reasoning`` — reasoning effort passthrough; AnyRouter
    translates it to each upstream's native thinking control.
  - App-attribution headers (``HTTP-Referer`` / ``X-AnyRouter-Title`` /
    ``X-AnyRouter-Source`` / ``X-AnyRouter-Categories``) so Hermes traffic is
    credited to the agent in AnyRouter's public app rankings.

See: https://anyrouter.dev/docs/api-reference/chat-completions and
https://anyrouter.dev/docs/features/app-attribution
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class AnyRouterProfile(ProviderProfile):
    """AnyRouter gateway — session, routing-preference, reasoning passthrough."""

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        prefs = context.get("provider_preferences")
        if prefs:
            body["provider"] = prefs
        return body

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Forward the reasoning config as ``extra_body.reasoning``.

        AnyRouter accepts the OpenAI-style ``reasoning`` object natively
        (``effort`` / ``enabled`` / ``exclude`` / ``max_tokens``) and owns the
        translation to each upstream's thinking control, so the profile simply
        passes the caller's intent through.
        """
        extra_body: dict[str, Any] = {}
        if supports_reasoning:
            if reasoning_config is not None:
                extra_body["reasoning"] = dict(reasoning_config)
            else:
                extra_body["reasoning"] = {"enabled": True, "effort": "medium"}
        return extra_body, {}


anyrouter = AnyRouterProfile(
    name="anyrouter",
    env_vars=("ANYROUTER_API_KEY", "ANYROUTER_BASE_URL"),
    display_name="AnyRouter",
    description="AnyRouter — OpenRouter-compatible unified model gateway",
    signup_url="https://anyrouter.dev",
    base_url="https://anyrouter.dev/api/v1",
    models_url="https://anyrouter.dev/api/v1/models",
    # App attribution — credits Hermes traffic in AnyRouter's app rankings.
    default_headers={
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-AnyRouter-Title": "Hermes Agent",
        "X-AnyRouter-Source": "cli-agent",
        "X-AnyRouter-Categories": "cli-agent",
    },
    fallback_models=(
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
        "deepseek/deepseek-chat",
        "google/gemini-3-flash-preview",
        "qwen/qwen3-plus",
    ),
)

register_provider(anyrouter)
