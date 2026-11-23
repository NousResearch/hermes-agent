"""Regression test for the probe-budget decoupling on #103843.

``for_picker`` historically picked the custom-endpoint discovery timeout
(1.5s fast / 5s full). When the CLI main picker started forwarding
``for_picker=True`` for exhausted-pool visibility, a slow-but-working
current custom endpoint lost its models: an endpoint answering between
the two budgets fell out of ``/model``. ``fast_custom_probe`` now owns
the budget so visibility no longer shortens discovery.
"""

import pytest

from hermes_cli.model_switch import list_authenticated_providers


@pytest.fixture(autouse=True)
def _no_builtin_catalog_fetches(monkeypatch):
    """Keep the row builder independent of provider credentials and network."""
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda *_a, **_kw: [])
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", lambda *_a, **_kw: [])
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *_a, **_kw: None)
    monkeypatch.setattr("hermes_cli.models_local.fetch_ollama_local_models", lambda *_a, **_kw: None)


def _slow_endpoint_rows(monkeypatch, **extra):
    """Rows for a current custom endpoint answering after 1.5s but within 5s."""

    def _fake_live(_api_key, _api_url, _native_provider, _preserve, headers=None,
                   timeout=5.0, api_mode=None, **_kw):
        return ["slow-discovered-model"] if timeout >= 5.0 else None

    monkeypatch.setattr("hermes_cli.model_switch_providers._fetch_picker_live_models", _fake_live)
    return list_authenticated_providers(
        current_provider="custom", current_base_url="http://127.0.0.1:9999/v1",
        current_model="kept-model", probe_custom_providers=False,
        probe_current_custom_provider=True, for_picker=True, **extra)


def _custom_row(rows):
    return next(r for r in rows if r["slug"] == "custom")


def test_full_budget_keeps_slow_endpoint_models(monkeypatch):
    """for_picker visibility with fast_custom_probe=False retains the historical 5s budget."""
    row = _custom_row(_slow_endpoint_rows(monkeypatch, fast_custom_probe=False))

    assert "slow-discovered-model" in row["models"]


def test_default_callers_keep_the_fast_budget_coupling(monkeypatch):
    """fast_custom_probe=None still defers to for_picker: the legacy fast-picker behavior."""
    row = _custom_row(_slow_endpoint_rows(monkeypatch))

    assert "slow-discovered-model" not in row["models"]
    assert row["models"] == ["kept-model"]


def test_explicit_fast_probe_true_still_cuts_the_budget(monkeypatch):
    row = _custom_row(_slow_endpoint_rows(monkeypatch, fast_custom_probe=True))

    assert "slow-discovered-model" not in row["models"]
    assert row["models"] == ["kept-model"]
