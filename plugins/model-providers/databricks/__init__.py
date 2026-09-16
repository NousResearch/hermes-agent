"""Databricks Unity Gateway provider metadata."""

from urllib.parse import urlsplit, urlunsplit

from providers import register_provider
from providers.base import ProviderProfile


_GPT_5_6_RESPONSES_SERVICES = frozenset({
    "gpt-5-6-sol",
    "gpt-5-6-terra",
    "gpt-5-6-luna",
})


def _service_name(model: str | None) -> str:
    return str(model or "").strip().lower().rsplit(".", 1)[-1]


def _service_family(model: str | None) -> str:
    service = _service_name(model)
    if service.startswith(("claude-", "databricks-claude-")):
        return "anthropic"
    if service.startswith(("gemini-", "databricks-gemini-")):
        return "gemini"
    if service in _GPT_5_6_RESPONSES_SERVICES:
        return "openai_responses"
    return "chat_completions"


def _gateway_base_url(configured_base_url: str, path: str) -> str:
    parsed = urlsplit(str(configured_base_url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return configured_base_url
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


class DatabricksProfile(ProviderProfile):
    """Unity Gateway model-specific transport selection."""

    def resolve_api_mode(self, model: str | None, configured_mode: str) -> str:
        family = _service_family(model)
        if family == "openai_responses":
            return "codex_responses"
        if family == "anthropic":
            return "anthropic_messages"
        return "chat_completions"

    def resolve_base_url(self, model: str | None, configured_base_url: str) -> str:
        path = {
            "anthropic": "/ai-gateway/anthropic",
            "gemini": "/ai-gateway/gemini/v1beta",
        }.get(_service_family(model), "/ai-gateway/mlflow/v1")
        return _gateway_base_url(configured_base_url, path)

    def create_client(self, **client_kwargs):
        base_url = str(client_kwargs.get("base_url") or "").rstrip("/")
        token_provider = client_kwargs.get("api_key")
        if not base_url.endswith("/ai-gateway/gemini/v1beta") or not callable(token_provider):
            return None
        from agent.gemini_native_adapter import GeminiNativeClient
        return GeminiNativeClient(
            api_key="databricks",
            base_url=base_url,
            bearer_token_provider=token_provider,
            **{
                key: value for key, value in client_kwargs.items()
                if key in {"default_headers", "timeout", "http_client"}
            },
        )


databricks = DatabricksProfile(
    name="databricks",
    display_name="Databricks Unity Gateway",
    env_vars=(),
    base_url="",
    auth_type="oauth_external",
    supports_health_check=False,
    supports_stream_options=False,
    responses_event_stale_timeout_seconds=120.0,
)

register_provider(databricks)
