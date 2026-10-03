"""xAI (Grok) provider profile."""

from hermes_cli.version_info import get_version_info
from providers import register_provider
from providers.base import ProviderProfile
from copy import copy
from .reasoning import cached_efforts, cached_model_ids, catalog_knows_model, legacy_efforts, prepare_catalog


class XaiProfile(ProviderProfile):
    _catalog_key = None

    def prepare_reasoning_catalog(self, *, api_key, base_url):
        bound = copy(self)
        bound._catalog_key = prepare_catalog(base_url, api_key)
        return bound

    def supported_reasoning_efforts(self, model):
        from agent.models_dev_reasoning import get_reasoning_effort_override
        declared = cached_efforts(self._catalog_key, model)
        override = get_reasoning_effort_override("xai", model, catalog_hit=catalog_knows_model(self._catalog_key, model))
        if override is not None:
            return override
        return declared if declared is not None else legacy_efforts(model)

    def fetch_models(self, *, api_key=None, base_url=None, timeout=5):
        bound = self.prepare_reasoning_catalog(api_key=api_key, base_url=base_url or self.base_url)
        return cached_model_ids(bound._catalog_key)


xai = XaiProfile(
    name="xai", aliases=("grok", "x-ai", "x.ai"), api_mode="codex_responses", env_vars=("XAI_API_KEY",),
    base_url="https://api.x.ai/v1", auth_type="api_key",
    default_headers={"User-Agent": f"Hermes-Agent/{get_version_info().base_version}"},
)

register_provider(xai)
