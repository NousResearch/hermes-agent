"""Perplexity Agent API provider profile (Responses API at api.perplexity.ai/v1).

``PERPLEXITY_API_KEY`` is also the Search API web backend key, so ``ambient_credentials`` keeps a
web-search-only key from selecting Perplexity for inference, auxiliary calls, or the model picker.
Model ids are vendor-prefixed (``anthropic/claude-opus-5-5``), hence ``aggregator``; the picker reads
the live ``GET /v1/models``.
"""

from hermes_cli.version_info import hermes_user_agent
from providers import register_provider
from providers.base import LazyHeaders, ProviderProfile


class PerplexityProfile(ProviderProfile):
    def api_mode_for_model(self, model: str, *, api_key: str | None = None) -> str | None:
        """Responses for every model: ``/v1/chat/completions`` is a 404 on the Agent API."""
        return "codex_responses"


perplexity = PerplexityProfile(
    name="perplexity", aliases=("pplx",), catalog_aliases=True, aggregator=True,
    display_name="Perplexity", description="Perplexity (Agent API)",
    signup_url="https://www.perplexity.ai/account/api", api_mode="codex_responses",
    env_vars=("PERPLEXITY_API_KEY",), base_url="https://api.perplexity.ai/v1", auth_type="api_key",
    ambient_credentials=True, overlay_aliases=("pplx",),
    provider_overlay={"transport": "codex_responses", "base_url_override": "https://api.perplexity.ai/v1",
                      "is_aggregator": True},
    default_headers=LazyHeaders({
        "User-Agent": lambda: hermes_user_agent("Hermes-Agent"), "X-Pplx-Integration": "hermes-agent",
    }),
    fallback_models=("anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5", "openai/gpt-5.5"),
    default_aux_model="anthropic/claude-haiku-5-5",
)

register_provider(perplexity)
