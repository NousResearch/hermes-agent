"""Google Gemini (AI Studio) provider profile.

Reports api_mode="chat_completions" but supplies its own client (``client.GeminiClient``): each call
goes to the Interactions API (``interactions.py``) or to ``generateContent`` (GeminiNativeClient), per
model. This profile also carries auth/endpoint metadata and the thinking_config translation hook.
"""

import logging
from typing import Any

from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


_GEMINI_FREE_TIER_NOTICE = (
    "", "❌ This Google API key is on the free tier (<= 250 requests/day for gemini-2.5-flash).",
    "   Hermes typically makes 3-10 API calls per user turn (tool iterations + auxiliary tasks),",
    "   so the free tier is exhausted after a handful of messages and cannot sustain",
    "   an agent session.", "",
    "   To use Gemini with Hermes, enable billing on your Google Cloud project and regenerate",
    "   the key in a billing-enabled project: https://aistudio.google.com/apikey", "",
    "   Alternatives with workable free usage: DeepSeek, OpenRouter (free models), Groq, Nous.", "",
    "Not saving Gemini as the default provider.")


class GeminiProfile(ProviderProfile):
    """Gemini — translate reasoning_config to thinking_config in extra_body."""

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        """Native: ``thinking_config``; OpenAI-compat /openai subpath:
        ``extra_body.google.thinking_config`` (snake_case)."""
        from agent.gemini_model_family import build_gemini_thinking_config, snake_case_gemini_thinking_config
        from agent.gemini_urls import is_gemini_openai_compat_base_url, is_native_gemini_base_url
        from agent.reasoning_effort import gemini_top_thinking_level

        raw = build_gemini_thinking_config(context.get("model") or "", context.get("reasoning_config"))
        if not raw:
            return {}
        raw = gemini_top_thinking_level(context.get("model"), context.get("reasoning_config"), raw)
        base_url = context.get("base_url") or self.base_url
        if self.name == "gemini" and is_gemini_openai_compat_base_url(base_url):
            thinking_config = snake_case_gemini_thinking_config(raw)
            return {"extra_body": {"google": {"thinking_config": thinking_config}}} if thinking_config else {}
        from .long_decode import long_decode_max_output_tokens

        # Long decode exists only on native generateContent (not /openai, not proxies).
        budget = long_decode_max_output_tokens(context.get("model") or "", context.get("reasoning_config"))
        return {"thinking_config": raw, **({"max_output_tokens": budget} if budget and is_native_gemini_base_url(base_url) else {})}

    def create_client(self, **client_kwargs: Any) -> Any:
        """The plugin's client (Interactions or generateContent per model) on a native Gemini base; None on
        the OpenAI-compatible ``/openai`` surface and proxies, which keep the standard OpenAI client."""
        from agent.gemini_urls import is_native_gemini_base_url

        base_url = str(client_kwargs.get("base_url") or self.base_url or "")
        if not is_native_gemini_base_url(base_url):
            return None
        allowed = {"api_key", "base_url", "default_headers", "timeout", "http_client"}
        kwargs = {k: v for k, v in client_kwargs.items() if k in allowed}
        if client_kwargs.get("auxiliary_chain"):
            from .native_adapter import GeminiNativeClient

            return GeminiNativeClient(**kwargs)
        from .client import GeminiClient

        if "http_client" not in kwargs and (http_client := _keepalive_http_client(base_url)) is not None:
            kwargs["http_client"] = http_client
        return GeminiClient(**kwargs)

    def implicit_max_output_tokens(self, *, model: str | None = None, base_url: str = "") -> int | None:
        from agent.gemini_model_family import GEMINI_DEFAULT_MAX_OUTPUT_TOKENS

        return GEMINI_DEFAULT_MAX_OUTPUT_TOKENS

    def validate_setup_api_key(self, api_key: str, *, base_url: str = "") -> bool:
        """Gemini free-tier gate: refuse free-tier keys during setup; best-effort on errors."""
        from .native_adapter import probe_gemini_tier

        print("  Checking Gemini API tier...")
        tier = probe_gemini_tier(api_key, base_url or self.base_url)
        if tier == "free":
            for line in _GEMINI_FREE_TIER_NOTICE:
                print(line)
            return False
        print("  Tier check: paid ✓" if tier == "paid" else "  Tier check: could not verify (proceeding anyway).")
        print()
        return True


def _keepalive_http_client(base_url: str) -> Any:
    """The keepalive client (and the provider's ``ssl_ca_cert`` / ``ssl_verify``) the core gives a native
    Gemini client; the profile hook receives the client kwargs after the core has consumed the TLS keys."""
    try:
        from agent.process_bootstrap import build_keepalive_http_client
        from agent.ssl_verify import resolve_httpx_verify
        from hermes_cli.config_providers import get_custom_provider_tls_settings

        tls = get_custom_provider_tls_settings(base_url)
        verify = resolve_httpx_verify(ca_bundle=tls.get("ssl_ca_cert"), ssl_verify=tls.get("ssl_verify"), base_url=base_url)
        return build_keepalive_http_client(base_url, verify=verify)
    except Exception:  # GeminiNativeClient then builds its own httpx client
        logger.warning("Gemini: keepalive client unavailable; using a default client", exc_info=True)
        return None


gemini = GeminiProfile(
    name="gemini", aliases=("google", "google-gemini", "google-ai-studio"), api_mode="chat_completions",
    display_name="Google AI Studio", description="Google AI Studio (Native Gemini API)",
    env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta", auth_type="api_key",
    supports_vision=True,
    default_aux_model="gemini-3.6-flash",
    forwards_max_tokens=True,
    hostnames=("generativelanguage.googleapis.com",),
    models_dev_id="google",
    replays_tool_call_signatures=True,
    static_models=(
        # Gemini 4 after it: [0] is the silent default when no model is configured, and a flagship never is.
        "gemini-3.8-flash", "gemini-4-neon", "gemini-4-hydrogen", "gemini-4-carbon", "gemini-4-argon",
        "gemini-3.7-flash", "gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-3.6-flash",
        "gemini-3.1-flash-lite-preview",
    ),
    picker_after="huggingface",
    picker_group=("google", "Google Gemini", "Google AI Studio (API key)"),
    catalog_aliases=True,
    base_url_env_var="GEMINI_BASE_URL",
    registry_after="lmstudio",
    strict_client=True,
)
