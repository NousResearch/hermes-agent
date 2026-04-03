"""Tests for _model_flow_openrouter credential reuse prompt.

Mirrors the pattern from test_codex_models.py — when an API key already exists,
the flow should prompt the user whether to keep, replace, set fallback, or cancel, matching
the Anthropic and OpenAI Codex flows.
"""
import builtins
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_openrouter_prompts_to_use_or_replace_key(monkeypatch, capfd):
    """When an existing key is configured, show mask + menu."""
    from hermes_cli.main import _model_flow_openrouter

    # choice "2" (replace) → new key → then model selection (empty = skip)
    choices = iter(["2", "sk-or-v1-newfromprompt", ""])

    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda key: "sk-or-v1-abc123def456" if key == "OPENROUTER_API_KEY" else (
            "sk-or-v1-fallback001" if key == "OPENROUTER_FALLBACK_API_KEY" else ""
        ),
    )
    saved = {}
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr("hermes_cli.auth._save_model_choice", lambda m: None)
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.models.model_ids",
        lambda: ["anthropic/claude-sonnet-4-20250514", "openai/gpt-4.1"],
    )
    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: None)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(choices))

    _model_flow_openrouter({}, current_model="anthropic/claude-sonnet-4-20250514")

    out, _ = capfd.readouterr()
    assert "Use existing credentials" in out
    assert "Enter new API key" in out
    assert "Set fallback API key" in out
    assert "sk-or-v1-abc123def456" not in out
    assert "sk-or-v1-abc" in out  # first 12 chars visible


def test_openrouter_uses_existing_key(monkeypatch):
    """When key exists and user chooses '1', proceed to model selection."""
    from hermes_cli.main import _model_flow_openrouter

    choices = iter(["1"])
    captured = {}

    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda key: "sk-or-v1-existing-key" if key == "OPENROUTER_API_KEY" else "",
    )
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda k, v: captured.__setitem__(k, v))
    monkeypatch.setattr("hermes_cli.auth._save_model_choice", lambda m: captured.__setitem__("model", m))
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.models.model_ids",
        lambda: ["anthropic/claude-sonnet-4-20250514"],
    )
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(choices))
    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", lambda *a, **kw: "anthropic/claude-sonnet-4-20250514")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: None)

    _model_flow_openrouter({}, current_model="test")

    assert "OPENROUTER_API_KEY" not in captured
    assert captured.get("model") == "anthropic/claude-sonnet-4-20250514"


def test_openrouter_cancel_preserves_existing(monkeypatch, capfd):
    """When key exists and user chooses '4', return without changes."""
    from hermes_cli.main import _model_flow_openrouter

    choices = iter(["4"])
    save_called = [False]

    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda key: "sk-or-v1-existing-key" if key == "OPENROUTER_API_KEY" else "",
    )
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda k, v: save_called.__setitem__(0, True))
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(choices))

    _model_flow_openrouter({}, current_model="test")

    out, _ = capfd.readouterr()
    assert "No change." in out
    assert not save_called[0]


def test_openrouter_no_key_prompts_for_new(monkeypatch, capfd):
    """When no key exists, prompt for a new key (original behavior)."""
    from hermes_cli.main import _model_flow_openrouter

    inputs = iter(["sk-or-v1-newkey"])
    saved = {}

    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda key: "",
    )
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
    monkeypatch.setattr("hermes_cli.auth._save_model_choice", lambda m: None)
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.models.model_ids",
        lambda: ["anthropic/claude-sonnet-4-20250514"],
    )
    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", lambda *a, **kw: "anthropic/claude-sonnet-4-20250514")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: None)

    _model_flow_openrouter({}, current_model="test")

    out, _ = capfd.readouterr()
    assert "No OpenRouter API key configured" in out
    assert "Use existing credentials" not in out
    assert "Enter new API key" not in out
    assert "Fallback" not in out
    assert saved.get("OPENROUTER_API_KEY") == "sk-or-v1-newkey"


def test_openrouter_replace_key_saves_new(monkeypatch, capfd):
    """When user chooses '2' and enters a new key, it should be saved."""
    from hermes_cli.main import _model_flow_openrouter

    inputs = iter(["2", "sk-or-v1-rotatedxyz", ""])
    saved = {}

    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda key: "sk-or-v1-oldkey123" if key == "OPENROUTER_API_KEY" else "",
    )
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
    monkeypatch.setattr("hermes_cli.auth._save_model_choice", lambda m: None)
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.models.model_ids",
        lambda: ["anthropic/claude-sonnet-4-20250514"],
    )
    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: None)

    _model_flow_openrouter({}, current_model="test")

    out, _ = capfd.readouterr()
    assert "API key updated" in out
    assert saved.get("OPENROUTER_API_KEY") == "sk-or-v1-rotatedxyz"


def test_openrouter_set_fallback_key(monkeypatch, capfd):
    """When user chooses '3' and enters a fallback key, it should be saved."""
    from hermes_cli.main import _model_flow_openrouter

    inputs = iter(["3", "sk-or-v1-fallbackkey99", ""])
    saved = {}

    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda key: "sk-or-v1-primary123" if key == "OPENROUTER_API_KEY" else "",
    )
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
    monkeypatch.setattr("hermes_cli.auth._save_model_choice", lambda m: None)
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.models.model_ids",
        lambda: ["anthropic/claude-sonnet-4-20250514"],
    )
    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: None)

    _model_flow_openrouter({}, current_model="test")

    out, _ = capfd.readouterr()
    assert "Fallback API key saved" in out
    assert saved.get("OPENROUTER_FALLBACK_API_KEY") == "sk-or-v1-fallbackkey99"
    assert "OPENROUTER_API_KEY" not in saved


def test_openrouter_shows_existing_fallback_key(monkeypatch, capfd):
    """When a fallback key already exists, it should be displayed in the menu."""
    from hermes_cli.main import _model_flow_openrouter

    choices = iter(["1"])
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda key: (
            "sk-or-v1-primarykey01" if key == "OPENROUTER_API_KEY"
            else "sk-or-v1-fallbackkey2" if key == "OPENROUTER_FALLBACK_API_KEY"
            else ""
        ),
    )
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda k, v: None)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(choices))
    monkeypatch.setattr("hermes_cli.auth._save_model_choice", lambda m: None)
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.models.model_ids",
        lambda: ["anthropic/claude-sonnet-4-20250514"],
    )
    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: None)

    _model_flow_openrouter({}, current_model="test")

    out, _ = capfd.readouterr()
    assert "Fallback key: sk-or-v1-fal" in out  # first 12 chars visible
