"""Regression: the blackbox recorder must fold a leaked composite
``<provider>/<model>`` model id (recorded with an EMPTY provider) back into the
split ``(provider, model)`` pair every correctly-recorded turn uses — so it stops
producing a phantom "Spend by model" row on tokens.ace.

Ace's 2026-07-05 report: a single $0 / 0-token desktop turn recorded as
``claude-api-proxy-f6/claude-haiku-4-5`` with no provider, instead of
``model=claude-haiku-4-5`` + ``provider=claude-api-proxy-f6`` like every other
Haiku turn — so it never rolled up with the bare model on the chart.

The repair lives at the RECORDING boundary (``plugins/blackbox``), not in the
live model-resolution path: a slash-bearing model on a custom/aggregator endpoint
carries prompt-caching/routing meaning, so rewriting it during inference is
unsafe; rewriting only the telemetry row is not.

NOTE the function contract: ``_recover_provider_from_model(model, provider)`` takes
``(model, provider)`` and returns ``(provider, model)`` — the split pair in the
order the record/pricing code consumes it.
"""
from __future__ import annotations

from unittest.mock import patch

import plugins.blackbox as bb


# A registry/aggregator view that does NOT depend on live fleet plugins, so the
# test asserts the SPLIT LOGIC, not plugin discovery under a hermetic HERMES_HOME.
_FAKE_REGISTRY = {
    "claude-api-proxy-f6": object(),
    "claude-bridge-f2": object(),
    "claude-app": object(),
    "anthropic": object(),
    "nous": object(),
}
_FAKE_AGGREGATORS = frozenset({"openrouter", "nous", "kilocode"})


def _recover(model, provider):
    """Call the recorder helper with a stubbed provider registry/aggregator set.
    Returns ``(provider, model)`` (the helper's contract)."""
    with (
        patch("hermes_cli.auth.PROVIDER_REGISTRY", _FAKE_REGISTRY),
        patch("hermes_cli.model_normalize._AGGREGATOR_PROVIDERS", _FAKE_AGGREGATORS),
    ):
        return bb._recover_provider_from_model(model, provider)


def test_composite_empty_provider_is_split():
    """THE bug: composite lane model + empty provider → split ``(provider, model)``."""
    assert _recover("claude-api-proxy-f6/claude-haiku-4-5", "") == (
        "claude-api-proxy-f6",
        "claude-haiku-4-5",
    )


def test_composite_empty_provider_none_is_split():
    """None provider is treated the same as empty."""
    assert _recover("claude-bridge-f2/claude-opus-4-8", None) == (
        "claude-bridge-f2",
        "claude-opus-4-8",
    )


def test_correctly_split_turn_is_untouched():
    """A turn that already has a provider is NEVER rewritten — the guard is the
    empty provider column, so the overwhelming majority of turns pass through
    unchanged (helper echoes back its ``(model, provider)`` inputs as
    ``(provider, model)``)."""
    # bare model + real provider → unchanged
    assert _recover("claude-haiku-4-5", "claude-api-proxy-f6") == (
        "claude-api-proxy-f6",
        "claude-haiku-4-5",
    )
    # Even a slash-bearing model is left alone when the provider is already set
    # (this is how legit aggregator turns record: openrouter + anthropic/… ).
    assert _recover("anthropic/claude-haiku-4.5", "openrouter") == (
        "openrouter",
        "anthropic/claude-haiku-4.5",
    )


def test_unknown_prefix_empty_provider_is_NOT_split():
    """An empty-provider slug whose prefix is NOT a registered provider is left
    verbatim — a genuine vendor slug (``meta-llama/llama-3.1``) must not be torn
    apart or have a provider fabricated. Provider stays empty, model unchanged."""
    assert _recover("meta-llama/llama-3.1", "") == ("", "meta-llama/llama-3.1")


def test_aggregator_prefix_empty_provider_is_NOT_split():
    """``nous`` is registered but is an AGGREGATOR — an empty-provider
    ``nous/<model>`` keeps its vendor prefix (aggregator routes carry it
    legitimately)."""
    assert _recover("nous/hermes-4", "") == ("", "nous/hermes-4")


def test_bare_model_empty_provider_is_untouched():
    """No slash → nothing to recover; the pair is returned unchanged."""
    assert _recover("claude-haiku-4-5", "") == ("", "claude-haiku-4-5")


def test_recovery_survives_registry_import_failure():
    """If the registry can't be imported, the recorder must not crash — it
    returns the pair unchanged (telemetry must never break a turn)."""
    with patch.dict("sys.modules", {"hermes_cli.auth": None}):
        assert bb._recover_provider_from_model(
            "claude-api-proxy-f6/claude-haiku-4-5", ""
        ) == ("", "claude-api-proxy-f6/claude-haiku-4-5")


def test_build_record_writes_the_split_pair():
    """E2E through ``_build_record``: a leaked composite turn is stored with the
    split provider/model, so ``by_model`` rolls it up with the bare form."""
    with (
        patch("hermes_cli.auth.PROVIDER_REGISTRY", _FAKE_REGISTRY),
        patch("hermes_cli.model_normalize._AGGREGATOR_PROVIDERS", _FAKE_AGGREGATORS),
    ):
        rec = bb._build_record(
            session_id="s1",
            interrupted=False,
            model="claude-api-proxy-f6/claude-haiku-4-5",
            platform="desktop",
            provider="",
            user_message="hi",
            final_response="hey",
            turn_usage={"api_calls": 1, "input_tokens": 0, "output_tokens": 0,
                        "calls": []},
            cfg={"store_text": False},
            kwargs={},
        )
    assert rec is not None
    assert rec.model == "claude-haiku-4-5"
    assert rec.provider == "claude-api-proxy-f6"


def test_build_record_names_virtual_moa_preset_explicitly():
    """Spend-by-model must not show the ambiguous bare model name ``default``."""
    rec = bb._build_record(
        session_id="moa-session",
        interrupted=False,
        model="default",
        platform="cli",
        provider="moa",
        user_message="ping",
        final_response="pong",
        turn_usage={"api_calls": 1, "input_tokens": 0, "output_tokens": 0,
                    "calls": []},
        cfg={"store_text": False},
        kwargs={},
    )
    assert rec is not None
    assert rec.model == "moa/default"
    assert rec.provider == "moa"


if __name__ == "__main__":  # pragma: no cover
    import pytest

    pytest.main([__file__, "-v"])
