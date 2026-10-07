"""Google Gemini (AI Studio) provider profile.

Reports api_mode="chat_completions" but runs on GeminiNativeClient; this
profile carries auth/endpoint metadata and the thinking_config translation hook.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class GeminiProfile(ProviderProfile):
    """Gemini — translate reasoning_config to thinking_config in extra_body."""

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        """Native: ``thinking_config``; OpenAI-compat /openai subpath:
        ``extra_body.google.thinking_config`` (snake_case)."""
        from agent.transports.chat_completions import (
            _build_gemini_thinking_config,
            _is_gemini_openai_compat_base_url,
            _snake_case_gemini_thinking_config,
        )

        raw = _build_gemini_thinking_config(context.get("model") or "", context.get("reasoning_config"))
        if not raw:
            return {}
        if self.name == "gemini" and _is_gemini_openai_compat_base_url(context.get("base_url") or self.base_url):
            thinking_config = _snake_case_gemini_thinking_config(raw)
            return {"extra_body": {"google": {"thinking_config": thinking_config}}} if thinking_config else {}
        return {"thinking_config": raw}

    def create_client(self, **client_kwargs: Any) -> Any | None:
        """Native REST client when ``base_url`` is the native surface, else None (the compat
        surface and other hosts take the standard client). ``httpx_verify`` comes from the main
        agent path only: it gets a keepalive transport carrying that TLS decision, while auxiliary
        callers omit it and the client builds its own transport."""
        from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

        base_url = str(client_kwargs.get("base_url", "") or "")
        if not is_native_gemini_base_url(base_url):
            return None
        safe_kwargs = {
            k: v for k, v in client_kwargs.items()
            if k in {"api_key", "base_url", "default_headers", "timeout", "http_client"}
        }
        if "http_client" not in safe_kwargs and "httpx_verify" in client_kwargs:
            from agent.process_bootstrap import build_keepalive_http_client

            keepalive_http = build_keepalive_http_client(base_url, verify=client_kwargs["httpx_verify"])
            if keepalive_http is not None:
                safe_kwargs["http_client"] = keepalive_http
        return GeminiNativeClient(**safe_kwargs)


gemini = GeminiProfile(
    name="gemini", aliases=("google", "google-gemini", "google-ai-studio"), api_mode="chat_completions",
    env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta", auth_type="api_key",
    default_aux_model="gemini-3.6-flash", strict_client=True,
)

register_provider(gemini)
