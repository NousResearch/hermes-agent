"""Regression: CLI /model persist must clear stale model.base_url (#48305 sibling).

The TUI _persist_model_switch (server.py) correctly clears model.base_url
when switching to a native provider that doesn't use one.  The CLI's two
/model persist paths (text-command and picker-callback) only wrote
model.default and model.provider — leaving a stale base_url that routed
the new model at the old custom host.
"""

import types

import yaml
import pytest


def _write_config(tmp_path, content: str):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(content)
    return cfg


class TestCliModelSwitchBaseUrlPersist:
    def test_cli_model_switch_clears_stale_base_url(self, tmp_path, monkeypatch):
        """Switching from custom endpoint → native provider via CLI /model
        must clear the stale base_url, not leave it routing at the old host."""
        import cli as cli_mod

        cfg_path = _write_config(
            tmp_path,
            "model:\n"
            "  default: local-model\n"
            "  provider: custom:mylocal\n"
            "  base_url: http://localhost:1234/v1\n",
        )
        monkeypatch.setattr(cli_mod, "_hermes_home", tmp_path)

        from cli import save_config_value

        save_config_value("model.default", "claude-haiku")
        save_config_value("model.provider", "anthropic")
        save_config_value("model.base_url", None)

        saved = yaml.safe_load(cfg_path.read_text())
        assert saved["model"]["default"] == "claude-haiku"
        assert saved["model"]["provider"] == "anthropic"
        assert not saved["model"].get("base_url"), (
            f"Stale base_url not cleared: {saved['model'].get('base_url')}"
        )

    def test_cli_model_switch_preserves_new_base_url(self, tmp_path, monkeypatch):
        """Switching to a custom provider must persist its base_url."""
        import cli as cli_mod

        cfg_path = _write_config(
            tmp_path,
            "model:\n"
            "  default: gpt-4\n"
            "  provider: openai\n",
        )
        monkeypatch.setattr(cli_mod, "_hermes_home", tmp_path)

        from cli import save_config_value

        save_config_value("model.default", "local-llama")
        save_config_value("model.provider", "custom:ollama")
        save_config_value("model.base_url", "http://localhost:11434/v1")

        saved = yaml.safe_load(cfg_path.read_text())
        assert saved["model"]["default"] == "local-llama"
        assert saved["model"]["provider"] == "custom:ollama"
        assert saved["model"]["base_url"] == "http://localhost:11434/v1"
