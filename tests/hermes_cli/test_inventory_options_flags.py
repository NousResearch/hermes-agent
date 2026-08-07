"""Tests for ``build_model_options_payload``'s ``include_pricing`` flag.

The desktop Model settings page never renders pricing, yet the shared
model-options payload historically hardcoded ``pricing=True`` — triggering
sequential per-provider network fetches (8s timeout each) that dominated the
page's cold-load latency. ``include_pricing`` lets such surfaces opt out while
keeping the default (pricing ON) for the picker/onboarding consumers that DO
render pricing. Capabilities stay on regardless (cheap CPU lookups).
"""

import hermes_cli.inventory as inv
from hermes_cli.inventory import ConfigContext


def _ctx() -> ConfigContext:
    return ConfigContext(
        current_provider="openrouter",
        current_model="a/model",
        current_base_url="",
        user_providers={},
        custom_providers=[],
    )


def _stub_rows(monkeypatch):
    """Replace the provider discovery + enrichment passes with recorders.

    ``list_authenticated_providers`` is stubbed so no credential probing or
    network happens; ``_apply_pricing`` / ``_apply_capabilities`` are replaced
    with recorders so the test asserts WHICH enrichment ran, not its output.
    """
    rows = [{"slug": "openrouter", "name": "OpenRouter", "models": ["a/model"]}]
    monkeypatch.setattr(inv, "_moa_provider_row", lambda *a, **k: None)
    calls = {"pricing": 0, "capabilities": 0}

    def fake_list(*args, **kwargs):
        # Return fresh copies so in-place enrichment can't leak across tests.
        return [dict(r) for r in rows]

    def fake_pricing(r, **kwargs):
        calls["pricing"] += 1
        for row in r:
            row["pricing"] = {"a/model": {"input": "$3", "output": "$15"}}

    def fake_capabilities(r):
        calls["capabilities"] += 1

    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers", fake_list
    )
    monkeypatch.setattr(inv, "_apply_pricing", fake_pricing)
    monkeypatch.setattr(inv, "_apply_capabilities", fake_capabilities)
    return calls


def test_include_pricing_default_runs_pricing(monkeypatch):
    """Default (no flag) keeps pricing ON — backward compatible for pickers."""
    calls = _stub_rows(monkeypatch)

    inv.build_model_options_payload(_ctx())

    assert calls["pricing"] == 1
    assert calls["capabilities"] == 1


def test_include_pricing_false_skips_pricing_keeps_capabilities(monkeypatch):
    """Opting out skips pricing but keeps the (cheap) capabilities pass."""
    calls = _stub_rows(monkeypatch)

    inv.build_model_options_payload(_ctx(), include_pricing=False)

    assert calls["pricing"] == 0
    assert calls["capabilities"] == 1


def test_include_pricing_false_omits_pricing_keys(monkeypatch):
    """With pricing off, rows carry no pricing/free_tier keys; default adds them."""
    _stub_rows(monkeypatch)

    off = inv.build_model_options_payload(_ctx(), include_pricing=False)
    on = inv.build_model_options_payload(_ctx())

    assert "pricing" not in off["providers"][0]
    assert "pricing" in on["providers"][0]
