"""The direct-Moonshot pickers must not offer ``kimi-k2.5``: platform.kimi.ai/docs/models lists it
under "Deprecated Models", discontinued on 2026-08-31 and no longer maintained or supported. Other
hosts keep their own catalogs (DashScope, OpenCode Zen/Go, Novita still serve the id)."""

from hermes_cli.models import _PROVIDER_MODELS

# api.moonshot.ai / api.moonshot.cn — the three lists served straight by Moonshot.
_DIRECT_MOONSHOT_PROVIDERS = ("kimi-coding", "kimi-coding-cn", "moonshot")


def test_retired_kimi_k2_5_is_absent_from_direct_moonshot_pickers():
    for provider in _DIRECT_MOONSHOT_PROVIDERS:
        assert "kimi-k2.5" not in _PROVIDER_MODELS[provider], provider


def test_direct_moonshot_pickers_still_offer_kimi_k3():
    """kimi-k3 is still a current model on the vendor's list; dropping kimi-k2.5 must not take it along."""
    for provider in _DIRECT_MOONSHOT_PROVIDERS:
        assert "kimi-k3" in _PROVIDER_MODELS[provider], provider


def test_other_hosts_keep_their_own_kimi_k2_5_entries():
    """Only the direct-Moonshot lists are retired-by-vendor; third-party hosts are unaffected."""
    for provider in ("alibaba", "opencode-zen", "opencode-go"):
        assert "kimi-k2.5" in _PROVIDER_MODELS[provider], provider
