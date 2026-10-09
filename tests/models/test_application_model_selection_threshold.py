"""The shared selection guard honours active profile policy on all surfaces."""
from __future__ import annotations

from application_model_selection_guards import SelectionContext, combined_selection_warning


def test_configured_context_threshold_including_disable(monkeypatch):
    import application_model_selection_guards as guards
    monkeypatch.setattr(guards, "_cost_warning", lambda *args: None)
    monkeypatch.setattr(guards, "_data_warning", lambda *args: None)

    config = {"model": {"switch_context_confirm_tokens": 0}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    ctx = SelectionContext(context_tokens=200_000, current_model="old-model")
    kwargs = {"provider": "openrouter", "selection_context": ctx}
    assert combined_selection_warning("new-model", **kwargs) is None

    config["model"]["switch_context_confirm_tokens"] = 150_000
    warning = combined_selection_warning("new-model", **kwargs)
    assert warning is not None
    assert warning.kind == "context_cache"
    assert "150,000" in warning.message

    config["model"]["switch_context_confirm_tokens"] = 250_000
    assert combined_selection_warning("new-model", **kwargs) is None
