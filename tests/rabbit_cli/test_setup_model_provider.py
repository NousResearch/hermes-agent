"""Regression tests for interactive setup provider/model persistence.

Since setup_model_provider delegates to select_provider_and_model()
from rabbit_cli.main, these tests mock the delegation point and verify
that the setup wizard correctly syncs config from disk after the call.
"""

from __future__ import annotations

from rabbit_cli.config import load_config, save_config
from rabbit_cli.setup import setup_model_provider


def _clear_provider_env(monkeypatch):
    for key in (
        "RABBIT_INFERENCE_PROVIDER",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GLM_API_KEY",
        "KIMI_API_KEY",
        "MINIMAX_API_KEY",
        "MINIMAX_CN_API_KEY",
        "ANTHROPIC_TOKEN",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def _write_aux_config(task="compression", provider="gemini", model_name="gemini-2.5-flash"):
    """Simulate the aux picker writing a task override to disk."""
    cfg = load_config()
    aux = cfg.setdefault("auxiliary", {})
    entry = aux.setdefault(task, {})
    entry["provider"] = provider
    entry["model"] = model_name
    save_config(cfg)


def test_setup_model_provider_preserves_auxiliary_choices_written_by_picker(tmp_path, monkeypatch):
    """Aux choices made inside rabbit setup must survive the wizard's final save."""
    monkeypatch.setenv("RABBIT_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()
    assert config["auxiliary"]["compression"]["provider"] == "auto"

    def fake_select():
        _write_aux_config("compression", "gemini", "gemini-2.5-flash")

    monkeypatch.setattr("rabbit_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config, quick=True)
    save_config(config)  # mirrors run_setup_wizard(section="model") final save

    reloaded = load_config()
    compression = reloaded["auxiliary"]["compression"]
    assert compression["provider"] == "gemini"
    assert compression["model"] == "gemini-2.5-flash"
