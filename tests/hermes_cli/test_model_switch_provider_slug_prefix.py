"""Regression tests: typed ``/model <provider>/<model>`` must switch provider.

Before the PATH A0 fix in ``switch_model``, a typed ``provider/model`` string
was treated as a model name on the CURRENT provider. The leading provider
slug was stripped by per-provider normalization and the bare model was
searched in the wrong catalog — an error when it was missing there, or a
silent no-op stay on the current provider when a same-named model existed on
both (e.g. ``kimi-k2.5`` on opencode-zen and opencode-go).

The fix routes ``/model opencode-zen/gpt-5.5``-style input through the same
explicit-provider path the picker uses, so the typed form and the picker
behave identically. These tests pin the resolution contract:

* A resolvable provider-slug prefix switches providers.
* Aggregator-vendor slugs native to the current catalog (e.g.
  ``openai/gpt-5.5`` on OpenRouter) are NOT hijacked as provider hops.
* A prefix that resolves to the current provider (or its group) keeps the
  normal same-provider path.
"""

import pytest

import hermes_cli.model_switch as ms
from hermes_cli.model_switch import switch_model

_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


@pytest.fixture(autouse=True)
def _fast_path(monkeypatch):
    """Keep switch_model off the network and credential machinery.

    The resolution steps under test are: provider-slug detection, PATH A
    provider resolution, and the final target_provider/new_model. Live
    catalog fetches, credential lookup, and metadata calls are stubbed.
    """
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "test-key",
            "base_url": "https://example.test/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *a, **k: dict(_MOCK_VALIDATION),
    )
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None
    )


@pytest.fixture
def fake_catalogs(monkeypatch):
    """Pin each provider's model catalog for the native-slug guard."""

    def _catalog(provider: str, **kwargs):
        return {
            "openrouter": ["openai/gpt-5.5", "anthropic/claude-sonnet-4.6"],
            "opencode-go": ["kimi-k2.5", "deepseek-v4-flash"],
            "opencode-zen": ["kimi-k2.5", "gpt-5.5"],
        }.get(provider, [])

    monkeypatch.setattr(ms, "list_provider_models", _catalog)


def test_go_to_zen_typed_provider_slug_switches_provider(fake_catalogs):
    """/model opencode-zen/gpt-5.5 while on opencode-go moves to Zen."""
    result = switch_model(
        raw_input="opencode-zen/gpt-5.5",
        current_provider="opencode-go",
        current_model="deepseek-v4-flash",
        user_providers={},
        custom_providers=[],
    )
    assert result.success
    # opencode-zen resolves to the "opencode" group; the runtime endpoint is
    # Zen (https://opencode.ai/zen/v1), and the model is the bare Zen model.
    assert result.target_provider == "opencode"
    assert result.provider_changed is True
    assert result.new_model == "gpt-5.5"


def test_zen_to_go_typed_provider_slug_switches_provider(fake_catalogs):
    """/model opencode-go/kimi-k3 while on opencode-zen moves to Go."""
    result = switch_model(
        raw_input="opencode-go/kimi-k3",
        current_provider="opencode-zen",
        current_model="gpt-5.5",
        user_providers={},
        custom_providers=[],
    )
    assert result.success
    assert result.target_provider == "opencode-go"
    assert result.provider_changed is True
    assert result.new_model == "kimi-k3"


def test_typed_slug_model_present_on_both_catalogs_still_switches(fake_catalogs):
    """kimi-k2.5 exists on both providers; the typed slug must still switch.

    This is the silent no-op regression: before the fix the bare model was
    found in the current catalog and the switch "succeeded" without ever
    leaving the current provider.
    """
    result = switch_model(
        raw_input="opencode-zen/kimi-k2.5",
        current_provider="opencode-go",
        current_model="deepseek-v4-flash",
        user_providers={},
        custom_providers=[],
    )
    assert result.success
    assert result.target_provider == "opencode"
    assert result.provider_changed is True
    assert result.new_model == "kimi-k2.5"


def test_openrouter_native_vendor_slug_is_not_hijacked(fake_catalogs):
    """openai/gpt-5.5 is a native OpenRouter model, not a provider hop."""
    result = switch_model(
        raw_input="openai/gpt-5.5",
        current_provider="openrouter",
        current_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        user_providers={},
        custom_providers=[],
    )
    assert result.success
    assert result.target_provider == "openrouter"
    assert result.provider_changed is False
    assert result.new_model == "openai/gpt-5.5"


def test_openrouter_anthropic_vendor_slug_is_not_hijacked(fake_catalogs):
    """anthropic/... on OpenRouter is a vendor slug (anthropic is also a
    real Hermes provider id — the native-catalog guard must win)."""
    result = switch_model(
        raw_input="anthropic/claude-sonnet-4.6",
        current_provider="openrouter",
        current_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        user_providers={},
        custom_providers=[],
    )
    assert result.success
    assert result.target_provider == "openrouter"
    assert result.provider_changed is False
    assert result.new_model == "anthropic/claude-sonnet-4.6"


def test_same_provider_prefix_keeps_normal_path(fake_catalogs):
    """/model opencode-go/kimi-k3 while already on opencode-go stays."""
    result = switch_model(
        raw_input="opencode-go/kimi-k3",
        current_provider="opencode-go",
        current_model="deepseek-v4-flash",
        user_providers={},
        custom_providers=[],
    )
    assert result.success
    assert result.target_provider == "opencode-go"
    assert result.provider_changed is False
    assert result.new_model == "kimi-k3"


def test_group_id_prefix_keeps_normal_path(fake_catalogs):
    """opencode/<model> while on the opencode group (Zen) is same-provider."""
    result = switch_model(
        raw_input="opencode/kimi-k2.5",
        current_provider="opencode",
        current_model="gpt-5.5",
        user_providers={},
        custom_providers=[],
    )
    assert result.success
    assert result.target_provider == "opencode"
    assert result.provider_changed is False
    assert result.new_model == "kimi-k2.5"


def test_unresolvable_prefix_falls_through_to_normal_path(fake_catalogs):
    """A prefix that is not a known provider keeps the old model-name path."""
    result = switch_model(
        raw_input="notaprovider/foo",
        current_provider="opencode-go",
        current_model="deepseek-v4-flash",
        user_providers={},
        custom_providers=[],
    )
    # Falls through: the provider never changes, the string is handled as a
    # model name on the current provider (and ultimately rejected there).
    assert result.target_provider == "opencode-go"
    assert result.provider_changed is False


def test_bare_provider_slug_still_switches_to_default_model(fake_catalogs):
    """/model opencode-go (no model part) remains a provider-only switch."""
    result = switch_model(
        raw_input="opencode-go",
        current_provider="opencode-zen",
        current_model="gpt-5.5",
        user_providers={},
        custom_providers=[],
    )
    assert result.success
    assert result.target_provider == "opencode-go"
    assert result.provider_changed is True


def test_empty_catalog_routing_aggregator_slug_preserved(monkeypatch):
    """On a routing aggregator, an empty models.dev catalog must not turn a
    native vendor slug into a provider hop (hermes-sweeper #75777 review)."""
    monkeypatch.setattr(ms, "list_provider_models", lambda *a, **k: [])
    result = switch_model(
        raw_input="anthropic/claude-sonnet-4.6",
        current_provider="openrouter",
        current_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        user_providers={},
        custom_providers=[],
    )
    assert result.success
    assert result.target_provider == "openrouter"
    assert result.provider_changed is False
    assert result.new_model == "anthropic/claude-sonnet-4.6"
    assert result.typed_provider_hop is False


def test_empty_catalog_flat_reseller_still_hops(monkeypatch):
    """Flat-namespace resellers (opencode-go/zen) never consult the catalog:
    a typed provider/model string still switches even when models.dev is
    empty."""
    monkeypatch.setattr(ms, "list_provider_models", lambda *a, **k: [])
    result = switch_model(
        raw_input="opencode-zen/gpt-5.5",
        current_provider="opencode-go",
        current_model="deepseek-v4-flash",
        user_providers={},
        custom_providers=[],
    )
    assert result.success
    assert result.target_provider == "opencode"
    assert result.provider_changed is True
    assert result.new_model == "gpt-5.5"
    assert result.typed_provider_hop is True


def test_typed_provider_hop_signal_set_for_typed_hop(fake_catalogs):
    """result.typed_provider_hop must be True for a typed provider hop."""
    result = switch_model(
        raw_input="opencode-zen/gpt-5.5",
        current_provider="opencode-go",
        current_model="deepseek-v4-flash",
        user_providers={},
        custom_providers=[],
    )
    assert result.success
    assert result.provider_changed is True
    assert result.typed_provider_hop is True


def test_typed_provider_hop_signal_not_set_for_native_slug(fake_catalogs):
    """Native aggregator slugs (openai/gpt-5.5 on OpenRouter) are NOT hops."""
    result = switch_model(
        raw_input="openai/gpt-5.5",
        current_provider="openrouter",
        current_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        user_providers={},
        custom_providers=[],
    )
    assert result.success
    assert result.provider_changed is False
    assert result.typed_provider_hop is False


def test_typed_provider_hop_signal_not_set_for_plain_model(fake_catalogs):
    """Plain model names never set the typed-hop signal."""
    result = switch_model(
        raw_input="deepseek-v4-flash",
        current_provider="opencode-go",
        current_model="kimi-k2.5",
        user_providers={},
        custom_providers=[],
    )
    assert result.success
    assert result.typed_provider_hop is False


def test_resolve_persist_behavior_typed_hop_is_session_only():
    """A typed provider hop follows the --provider session-only rule unless
    --global forces persistence (resolve_persist_behavior rule 4)."""
    assert ms.resolve_persist_behavior(False, False, typed_provider_hop=True) is False
    assert ms.resolve_persist_behavior(False, False, typed_provider_hop=False) is False
    assert ms.resolve_persist_behavior(True, False, typed_provider_hop=True) is True
    assert ms.resolve_persist_behavior(False, True, typed_provider_hop=True) is False
    assert ms.resolve_persist_behavior(False, False, is_once=True, typed_provider_hop=True) is False
