"""The OpenRouter free-model picker row and its flow.

The row is a *picker action* over the live ``/v1/models`` catalog, not a provider: it must
persist its choice under the existing ``openrouter`` provider so ``--provider``/``/model``
resolution keeps working. These tests pin that contract, the two-part filter (zero-priced AND
tool-capable), and the cache/stale-list behaviour.
"""

from __future__ import annotations

import pytest

from hermes_cli import models_openrouter_free as free_mod

# Two real-shaped catalog items: one genuinely free + tool-capable, one free but NOT
# tool-capable, one priced. The filter must keep only the first.
LIVE_ITEMS = [
    {"id": "vendor/free-tool:free", "pricing": {"prompt": "0", "completion": "0"},
     "supported_parameters": ["tools", "temperature"]},
    {"id": "vendor/free-no-tools:free", "pricing": {"prompt": "0", "completion": "0"},
     "supported_parameters": ["temperature"]},
    {"id": "vendor/paid:paid", "pricing": {"prompt": "0.0000012", "completion": "0"},
     "supported_parameters": ["tools"]},
    {"id": "vendor/free-missing-params:free", "pricing": {"prompt": "0", "completion": "0"}},
]


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    """Every test starts with a cold process cache."""
    monkeypatch.setattr(free_mod, "_free_cache", None)
    monkeypatch.setattr(free_mod, "_free_cache_time", 0.0)


@pytest.fixture
def stub_live(monkeypatch):
    """Stub the catalog fetch at the models.py seam the module reads it from."""
    from hermes_cli import models as models_mod
    calls = []

    def _fake_fetch(url, timeout, opener):
        calls.append(url)
        by_id = {item["id"]: item for item in LIVE_ITEMS}
        return list(LIVE_ITEMS), by_id

    monkeypatch.setattr(models_mod, "_fetch_live_catalog_index", _fake_fetch)
    return calls


def test_filter_keeps_only_free_and_tool_capable(stub_live):
    ids = free_mod.openrouter_free_model_ids(force_refresh=True)
    assert "vendor/free-tool:free" in ids
    # Tool-capability is not optional: Hermes is tool-calling-first.
    assert "vendor/free-no-tools:free" not in ids
    assert "vendor/paid:paid" not in ids
    # An absent supported_parameters field is permissive (some gateways omit it).
    assert "vendor/free-missing-params:free" in ids


def test_cache_serves_second_open_without_a_second_fetch(stub_live):
    first = free_mod.openrouter_free_model_ids()
    assert len(stub_live) == 1
    second = free_mod.openrouter_free_model_ids()
    assert first == second
    assert len(stub_live) == 1, "a picker re-open inside the TTL must not re-download the catalog"


def test_force_refresh_bypasses_the_cache(stub_live):
    free_mod.openrouter_free_model_ids()
    free_mod.openrouter_free_model_ids(force_refresh=True)
    assert len(stub_live) == 2


def test_unreachable_catalog_keeps_serving_the_last_good_list(stub_live, monkeypatch):
    from hermes_cli import models as models_mod
    good = free_mod.openrouter_free_model_ids()
    # A network blip on the next forced refresh must not empty the picker.
    monkeypatch.setattr(models_mod, "_fetch_live_catalog_index", lambda *a, **k: None)
    assert free_mod.openrouter_free_model_ids(force_refresh=True) == good


def test_picker_row_is_present_and_is_a_flat_action():
    from hermes_cli.main_provider_setup import _build_provider_picker_rows
    rows, _ = _build_provider_picker_rows({}, "openrouter", {"openrouter": "OpenRouter"}, {})
    keys = [key for key, _label, _members in rows]
    assert "openrouter-free" in keys
    row = next(r for r in rows if r[0] == "openrouter-free")
    assert row[2] == [], "it is a leaf action, not a provider group"


def test_flow_persists_under_the_openrouter_provider(monkeypatch, stub_live):
    """The contract that makes this cheap: no fake provider slug reaches the resolver."""
    # The flow imports ``_prompt_model_selection`` from hermes_cli.auth, so that is the
    # binding to patch — auth re-exports the auth_model_picker function under its own name.
    import hermes_cli.auth as auth_mod
    import hermes_cli.main_provider_setup as setup_mod
    import hermes_cli.model_setup_flows_common as common_mod

    seen = {}

    monkeypatch.setattr(
        setup_mod, "_prompt_api_key", lambda pconfig, existing, **kw: (existing or "sk-or-v1-test", False)
    )
    monkeypatch.setattr(
        auth_mod, "_prompt_model_selection",
        lambda models, **kw: (seen.update(models=list(models), provider=kw.get("confirm_provider")),
                              models[0])[1],
    )
    monkeypatch.setattr(
        common_mod, "_finish_model",
        lambda sel, provider, done, **kw: seen.update(selected=sel, persisted_provider=provider, persist_kw=kw),
    )

    free_mod._model_flow_openrouter_free({}, "some/previous-model")

    assert "vendor/free-tool:free" in seen["models"]
    assert seen["provider"] == "openrouter"
    # NOT "openrouter-free": a fake provider would have to resolve through
    # resolve_provider_full and every alias table.
    assert seen["persisted_provider"] == "openrouter"
    assert seen["persist_kw"]["api_mode"] == "chat_completions"
