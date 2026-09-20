"""Rows the Nous gateway serves on the user's ChatGPT subscription: CLI picker chrome and the
silent default for a free-tier account."""

import hermes_cli.models as models_mod
from hermes_cli import models_pricing
from hermes_cli.auth_model_picker import _ModelPickerRows

_PAID = {"prompt": "0.000003", "completion": "0.000015"}
_SHARED = {**_PAID, "billing_mode": "openai_token_sharing"}
_FREE = {"prompt": "0", "completion": "0"}


def test_picker_row_names_the_subscription_instead_of_a_price():
    rows = _ModelPickerRows(["openai/gpt", "anthropic/claude"], {"openai/gpt": _SHARED, "anthropic/claude": _PAID},
                            current_model="", sale_chrome=True)
    assert "ChatGPT subscription" in rows.label("openai/gpt")
    assert "$" not in rows.label("openai/gpt")
    assert "ChatGPT subscription" not in rows.label("anthropic/claude")
    assert "$3.00" in rows.label("anthropic/claude")


def _patch_default_inputs(monkeypatch, model_ids, pricing):
    monkeypatch.setattr(models_mod, "get_curated_nous_model_ids", lambda: list(model_ids))
    monkeypatch.setattr(models_mod, "check_nous_free_tier", lambda **kw: True)
    monkeypatch.setattr(models_mod, "union_with_portal_free_recommendations", lambda ids, pr, url="", **kw: (ids, pr))
    monkeypatch.setattr(models_mod, "get_preferred_silent_default_model", lambda provider="openrouter": "not/listed")
    monkeypatch.setattr(models_pricing, "get_pricing_for_provider", lambda slug, **kw: pricing)
    monkeypatch.setattr(models_pricing, "nous_policy_allowed_ids", lambda **kw: None)


def test_free_tier_silent_default_stays_a_free_model(monkeypatch):
    """A subscription-billed row listed first must not become the default: it spends the user's
    ChatGPT plan limits, which they opt into from the picker."""
    _patch_default_inputs(monkeypatch, ["openai/gpt", "free/model"], {"openai/gpt": _SHARED, "free/model": _FREE})
    assert models_mod.recommended_nous_default_model()["model"] == "free/model"


def test_free_tier_silent_default_falls_back_to_a_subscription_billed_model(monkeypatch):
    _patch_default_inputs(monkeypatch, ["openai/gpt", "anthropic/claude"], {"openai/gpt": _SHARED, "anthropic/claude": _PAID})
    assert models_mod.recommended_nous_default_model()["model"] == "openai/gpt"
