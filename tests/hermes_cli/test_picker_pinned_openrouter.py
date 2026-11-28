"""A ``providers.openrouter`` allowlist (``discover_models: false``) survives the picker.

``list_picker_providers`` swaps the OpenRouter row's models for the live/curated
catalog. For a user-defined row whose config pins ``models`` with discovery off,
that swap puts back every model the operator removed.
"""

import hermes_cli.model_switch_providers as msp
from hermes_cli import model_switch


def _setup(monkeypatch, cfg_openrouter):
    row = {"slug": "openrouter", "name": "OpenRouter", "is_current": False, "is_user_defined": True,
           "models": ["anthropic/claude-sonnet-5-5"], "total_models": 1, "source": "user-config"}
    lister = lambda **_kw: [dict(row)]
    monkeypatch.setattr(model_switch, "list_authenticated_providers", lister)
    monkeypatch.setattr(msp, "list_authenticated_providers", lister)
    curated = lambda *a, **kw: [("openai/gpt-5.4", ""), ("x-ai/grok-5", ""), ("anthropic/claude-sonnet-5-5", "")]
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models", curated)
    if hasattr(msp, "fetch_openrouter_models"):
        monkeypatch.setattr(msp, "fetch_openrouter_models", curated)
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *_a, **_kw: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"providers": {"openrouter": cfg_openrouter}})
    monkeypatch.setattr(msp, "_with_local_runtime_row", lambda providers, *a, **kw: providers)


def _openrouter_models():
    rows = msp.list_picker_providers(current_provider="openrouter")
    return next(r for r in rows if r["slug"] == "openrouter")["models"]


def test_pinned_allowlist_is_kept(monkeypatch):
    _setup(monkeypatch, {"discover_models": False, "models": ["anthropic/claude-sonnet-5-5"]})
    assert _openrouter_models() == ["anthropic/claude-sonnet-5-5"]


def test_discovery_on_still_uses_the_catalog(monkeypatch):
    _setup(monkeypatch, {"models": ["anthropic/claude-sonnet-5-5"]})
    assert "x-ai/grok-5" in _openrouter_models()


def test_builtin_openrouter_row_honors_pinned_allowlist(monkeypatch):
    """An authenticated built-in row is not marked is_user_defined by the lister."""
    _setup(monkeypatch, {"discover_models": False, "models": ["anthropic/claude-sonnet-5-5"]})
    # Sections 1/2 emit this row when OPENROUTER_API_KEY is present. Section 3
    # skips the already-seen slug, so the guard must not require is_user_defined.
    monkeypatch.setattr(model_switch, "list_authenticated_providers", lambda **_kw: [{
        "slug": "openrouter", "name": "OpenRouter", "is_current": False,
        "is_user_defined": False, "source": "built-in",
        "models": ["anthropic/claude-sonnet-5-5", "openai/gpt-5.4"],
        "total_models": 2,
    }])
    assert _openrouter_models() == ["anthropic/claude-sonnet-5-5"]
