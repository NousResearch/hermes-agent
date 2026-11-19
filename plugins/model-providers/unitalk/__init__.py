"""Unitalk's OpenAI-compatible model gateway for Hermes."""

from providers import register_provider
from providers.base import ProviderProfile


class UnitalkProfile(ProviderProfile):
    def prepare_pool_entry(self, entry, api_key):
        # Environment keys are alternatives, not rotation candidates. Returning
        # no pool key sends resolution through Hermes' ordered env-key ladder.
        source = getattr(entry, "source", "")
        if any(source == f"env:{name}" or source.startswith(f"env:{name}_")
               for name in self.env_vars):
            return ""
        return api_key


register_provider(UnitalkProfile(
    name="unitalk",
    display_name="Unitalk",
    description="Unitalk model gateway",
    env_vars=("UNITALK_API_KEY", "OPENAI_API_KEY"),
    base_url="https://llmgateway.unitalk.ai/v1",
    auth_type="api_key",
    api_mode="chat_completions",
))
