"""Unit tests for the Cloud Temple LLMaaS provider profile."""

from __future__ import annotations


def test_cloud_temple_profile_registered():
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("cloud-temple")
    assert profile is not None, "cloud-temple provider profile must be registered"
    assert profile.display_name == "Cloud Temple"
    assert profile.base_url == "https://api.ai.cloud-temple.com/v1"
    assert profile.env_vars == ("CLOUD_TEMPLE_API_KEY", "CLOUD_TEMPLE_BASE_URL")
    assert profile.default_aux_model == "qwen3.6:35b"
    assert profile.fallback_models == ("qwen3.6:35b", "gemma4:31b")


def test_cloud_temple_aliases_resolve_to_profile():
    import model_tools  # noqa: F401
    import providers

    canonical = providers.get_provider_profile("cloud-temple")
    assert providers.get_provider_profile("cloud_temple") is canonical
    assert providers.get_provider_profile("cloudtemple") is canonical


def test_cloud_temple_model_picker_uses_profile_fallback_without_key(monkeypatch):
    import model_tools  # noqa: F401
    from hermes_cli.models import provider_model_ids

    monkeypatch.delenv("CLOUD_TEMPLE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUD_TEMPLE_BASE_URL", raising=False)

    assert provider_model_ids("cloud-temple") == ["qwen3.6:35b", "gemma4:31b"]
