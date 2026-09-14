"""DeepInfra provider profile (chat surface; image-gen/TTS/STT are wired via
their own plugin subsystems)."""

from providers import register_provider
from providers.base import ProviderProfile


class _DeepInfraProfile(ProviderProfile):
    """DeepInfra profile with live vision-default discovery, so shared vision
    resolution in ``agent/auxiliary_client.py`` stays provider-agnostic."""

    def default_vision_model(self):  # type: ignore[override]
        """First vision-capable *chat* model from the live catalog, or None. Key-gated so a box
        without DEEPINFRA_API_KEY never pays the round-trip; requires the ``chat`` surface tag so
        an image-gen model carrying a ``vision`` tag can't be picked as a chat vision backend."""
        from agent.secret_scope import get_secret

        if not (get_secret("DEEPINFRA_API_KEY") or "").strip():
            return None
        try:
            from hermes_cli.models import _fetch_deepinfra_models_by_tag
            items = _fetch_deepinfra_models_by_tag("chat")
        except Exception:
            return None
        for item in items or []:
            metadata = item.get("metadata") or {}
            tags = metadata.get("tags") if isinstance(metadata, dict) else None
            if isinstance(tags, list) and "vision" in tags and item.get("id"):
                return item["id"]
        return None


deepinfra = _DeepInfraProfile(
    name="deepinfra", aliases=("deep-infra", "deepinfra-ai"), display_name="DeepInfra",
    description="DeepInfra — 100+ open models, pay-per-use", signup_url="https://deepinfra.com/dash/api_keys",
    env_vars=("DEEPINFRA_API_KEY", "DEEPINFRA_BASE_URL"), base_url="https://api.deepinfra.com/v1/openai",
    auth_type="api_key",
    default_max_tokens=None,  # DeepInfra applies its documented per-model limit
    # The only hardcoded DeepInfra model: aux resolution is synchronous, so it
    # can't wait on a catalog round-trip. Everything else is discovered live.
    default_aux_model="deepseek-ai/DeepSeek-V4-Flash",
    # Empty on purpose: the live catalog is the source of truth; an empty picker
    # beats silently routing to a retired model.
    fallback_models=(),
)

register_provider(deepinfra)
