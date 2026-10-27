"""Provider detection used by the turn-retry credential-refresh gates."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from providers.base import ProviderProfile


class _RefreshingProfile(ProviderProfile):
    def refresh_runtime_credentials(self, *, reason: str):
        return None


def test_refresh_provider_check_accepts_alias_spellings():
    """`/model` and profile configs can leave an alias (any case) as the
    provider spelling; the recovery gates must not silently skip them."""
    import providers
    from agent.conversation_loop import _credential_refresh_profile

    providers.list_providers()  # discover first so the patched registry restores to the real one
    acme = _RefreshingProfile(name="acme-refresh", aliases=("acme-cloud-refresh", "acme"),
                              base_url="https://api.acme-refresh.test", hostnames=("acme-refresh.test",))

    def _owner(provider, base_url=""):
        profile = _credential_refresh_profile(SimpleNamespace(provider=provider, base_url=base_url))
        return profile.name if profile else None

    with patch.dict(providers._REGISTRY, {acme.name: acme}), \
            patch.dict(providers._ALIASES, {alias: acme.name for alias in acme.aliases}), \
            patch.object(providers, "_PROVIDER_LIST_CACHE", None):
        assert _owner("acme-refresh") == "acme-refresh"
        assert _owner("acme-cloud-refresh") == "acme-refresh"
        assert _owner("ACME-Cloud-Refresh") == "acme-refresh"
        assert _owner("acme") == "acme-refresh"
        # URL fallback: unnormalized provider but the profile's base URL.
        assert _owner("custom", "https://api.acme-refresh.test") == "acme-refresh"
        assert _owner("openrouter", "https://openrouter.ai/api/v1") is None
