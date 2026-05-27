"""Tests that provider selection via `hermes model` always persists correctly.

Regression tests for the bug where _save_model_choice could save config.model
as a plain string, causing subsequent provider writes (which check
isinstance(model, dict)) to silently fail — leaving the provider unset and
falling back to auto-detection.
"""

import os
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a minimal string-format config."""
    home = tmp_path / "hermes"
    home.mkdir()
    config_yaml = home / "config.yaml"
    # Start with model as a plain string — the format that triggered the bug
    config_yaml.write_text("model: some-old-model\n")
    env_file = home / ".env"
    env_file.write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Clear env vars that could interfere
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
    monkeypatch.delenv("STEPFUN_BASE_URL", raising=False)
    return home


class TestSaveModelChoiceAlwaysDict:
    def test_string_model_becomes_dict(self, config_home):
        """When config.model is a plain string, _save_model_choice must
        convert it to a dict so provider can be set afterwards."""
        from hermes_cli.auth import _save_model_choice

        _save_model_choice("kimi-k2.5")

        import yaml
        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict), (
            f"Expected model to be a dict after save, got {type(model)}: {model}"
        )
        assert model["default"] == "kimi-k2.5"

    def test_dict_model_stays_dict(self, config_home):
        """When config.model is already a dict, _save_model_choice preserves it."""
        import yaml
        (config_home / "config.yaml").write_text(
            "model:\n  default: old-model\n  provider: openrouter\n"
        )
        from hermes_cli.auth import _save_model_choice

        _save_model_choice("new-model")

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict)
        assert model["default"] == "new-model"
        assert model["provider"] == "openrouter"  # preserved


class TestProviderPersistsAfterModelSave:
    def test_update_config_for_provider_uses_atomic_yaml_write(self, config_home):
        """Provider switches should delegate config writes to atomic_yaml_write."""
        from hermes_cli.auth import _update_config_for_provider

        config_path = config_home / "config.yaml"
        original_text = config_path.read_text(encoding="utf-8")

        def _boom(path, data, **kwargs):
            assert path == config_path
            assert data["model"]["provider"] == "nous"
            assert data["model"]["base_url"] == "https://inference.example.com/v1"
            assert data["model"]["default"] == "some-old-model"
            assert kwargs["sort_keys"] is False
            raise OSError("simulated atomic write failure")

        with patch("hermes_cli.auth.atomic_yaml_write", side_effect=_boom) as mock_write:
            with pytest.raises(OSError, match="simulated atomic write failure"):
                _update_config_for_provider(
                    "nous",
                    "https://inference.example.com/v1/",
                    default_model="llama-3.3",
                )

        assert mock_write.call_count == 1
        assert config_path.read_text(encoding="utf-8") == original_text

    def test_api_key_provider_saved_when_model_was_string(self, config_home, monkeypatch):
        """_model_flow_api_key_provider must persist the provider even when
        config.model started as a plain string."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get("kimi-coding")
        if not pconfig:
            pytest.skip("kimi-coding not in PROVIDER_REGISTRY")

        # Simulate: user has a Kimi API key, model was a string
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-test-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        # Mock the model selection prompt to return "kimi-k2.5"
        # Also mock input() for the base URL prompt and builtins.input
        with patch("hermes_cli.auth._prompt_model_selection", return_value="kimi-k2.5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "kimi-coding", "old-model")

        import yaml
        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict), f"model should be dict, got {type(model)}"
        assert model.get("provider") == "kimi-coding", (
            f"provider should be 'kimi-coding', got {model.get('provider')}"
        )
        assert model.get("default") == "kimi-k2.5"

    def test_copilot_provider_saved_when_selected(self, config_home):
        """_model_flow_copilot should persist provider/base_url/model together."""
        from hermes_cli.main import _model_flow_copilot
        from hermes_cli.config import load_config

        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "copilot",
                "api_key": "gh-cli-token",
                "base_url": "https://api.githubcopilot.com",
                "source": "gh auth token",
            },
        ), patch(
            "hermes_cli.models.fetch_github_model_catalog",
            return_value=[
                {
                    "id": "gpt-4.1",
                    "capabilities": {"type": "chat", "supports": {}},
                    "supported_endpoints": ["/chat/completions"],
                },
                {
                    "id": "gpt-5.4",
                    "capabilities": {"type": "chat", "supports": {"reasoning_effort": ["low", "medium", "high"]}},
                    "supported_endpoints": ["/responses"],
                },
            ],
        ), patch(
            "hermes_cli.auth._prompt_model_selection",
            return_value="gpt-5.4",
        ), patch(
            "hermes_cli.main._prompt_reasoning_effort_selection",
            return_value="high",
        ), patch(
            "hermes_cli.auth.deactivate_provider",
        ):
            _model_flow_copilot(load_config(), "old-model")

        import yaml

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict), f"model should be dict, got {type(model)}"
        assert model.get("provider") == "copilot"
        assert model.get("base_url") == "https://api.githubcopilot.com"
        assert model.get("default") == "gpt-5.4"
        assert model.get("api_mode") == "codex_responses"
        assert config["agent"]["reasoning_effort"] == "high"

    def test_named_custom_provider_preserves_explicit_api_mode(self, config_home):
        """Named custom providers should re-activate with their saved api_mode."""
        import yaml

        from hermes_cli.main import _model_flow_named_custom

        provider_info = {
            "name": "Packy",
            "base_url": "https://packy.example.com/v1",
            "api_key": "sk-test",
            "model": "gpt-5.4",
            "api_mode": "codex_responses",
        }

        # Patch fetch_api_models so the named custom flow returns one model;
        # patch simple_term_menu to force the input() fallback; patch input to
        # auto-select the first model from the fallback prompt.
        from unittest.mock import MagicMock
        fake_menu_module = MagicMock()
        fake_menu_module.TerminalMenu.side_effect = OSError("no tty in test")
        with patch("hermes_cli.auth._save_model_choice"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("hermes_cli.models.fetch_api_models", return_value=["gpt-5.4"]), \
             patch.dict("sys.modules", {"simple_term_menu": fake_menu_module}), \
             patch("builtins.input", return_value="1"):
            _model_flow_named_custom({}, provider_info)

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict)
        assert model.get("provider") == "custom"
        assert model.get("base_url") == "https://packy.example.com/v1"
        assert model.get("api_mode") == "codex_responses"

    def test_copilot_acp_provider_saved_when_selected(self, config_home):
        """_model_flow_copilot_acp should persist provider/base_url/model together."""
        from hermes_cli.main import _model_flow_copilot_acp
        from hermes_cli.config import load_config

        with patch(
            "hermes_cli.auth.get_external_process_provider_status",
            return_value={
                "resolved_command": "/usr/local/bin/copilot",
                "command": "copilot",
                "base_url": "acp://copilot",
            },
        ), patch(
            "hermes_cli.auth.resolve_external_process_provider_credentials",
            return_value={
                "provider": "copilot-acp",
                "api_key": "copilot-acp",
                "base_url": "acp://copilot",
                "command": "/usr/local/bin/copilot",
                "args": ["--acp", "--stdio"],
                "source": "process",
            },
        ), patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "copilot",
                "api_key": "gh-cli-token",
                "base_url": "https://api.githubcopilot.com",
                "source": "gh auth token",
            },
        ), patch(
            "hermes_cli.models.fetch_github_model_catalog",
            return_value=[
                {
                    "id": "gpt-4.1",
                    "capabilities": {"type": "chat", "supports": {}},
                    "supported_endpoints": ["/chat/completions"],
                },
                {
                    "id": "gpt-5.4",
                    "capabilities": {"type": "chat", "supports": {"reasoning_effort": ["low", "medium", "high"]}},
                    "supported_endpoints": ["/responses"],
                },
            ],
        ), patch(
            "hermes_cli.auth._prompt_model_selection",
            return_value="gpt-5.4",
        ), patch(
            "hermes_cli.auth.deactivate_provider",
        ):
            _model_flow_copilot_acp(load_config(), "old-model")

        import yaml

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict), f"model should be dict, got {type(model)}"
        assert model.get("provider") == "copilot-acp"
        assert model.get("base_url") == "acp://copilot"
        assert model.get("default") == "gpt-5.4"
        assert model.get("api_mode") == "chat_completions"

    def test_opencode_go_models_are_selectable_and_persist_normalized(self, config_home, monkeypatch):
        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key")

        with patch("hermes_cli.models.fetch_api_models", return_value=["opencode-go/kimi-k2.5", "opencode-go/minimax-m2.7"]), \
             patch("hermes_cli.auth._prompt_model_selection", return_value="kimi-k2.5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "opencode-go", "opencode-go/kimi-k2.5")

        import yaml
        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict)
        assert model.get("provider") == "opencode-go"
        assert model.get("default") == "kimi-k2.5"
        assert model.get("api_mode") == "chat_completions"

    def test_opencode_go_same_provider_switch_recomputes_api_mode(self, config_home, monkeypatch):
        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key")
        (config_home / "config.yaml").write_text(
            "model:\n"
            "  default: kimi-k2.5\n"
            "  provider: opencode-go\n"
            "  base_url: https://opencode.ai/zen/go/v1\n"
            "  api_mode: chat_completions\n"
        )

        with patch("hermes_cli.models.fetch_api_models", return_value=["opencode-go/kimi-k2.5", "opencode-go/minimax-m2.5"]), \
             patch("hermes_cli.auth._prompt_model_selection", return_value="minimax-m2.5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "opencode-go", "kimi-k2.5")

        import yaml
        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict)
        assert model.get("provider") == "opencode-go"
        assert model.get("default") == "minimax-m2.5"
        assert model.get("api_mode") == "anthropic_messages"



class TestBaseUrlValidation:
    """Reject non-URL values in the base URL prompt (e.g. shell commands)."""

    def test_invalid_base_url_rejected(self, config_home, monkeypatch, capsys):
        """Typing a non-URL string should not be saved as the base URL."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get("zai")
        if not pconfig:
            pytest.skip("zai not in PROVIDER_REGISTRY")

        monkeypatch.setenv("GLM_API_KEY", "test-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config, get_env_value

        # User types a shell command instead of a URL at the base URL prompt
        with patch("hermes_cli.auth._prompt_model_selection", return_value="glm-5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value="nano ~/.hermes/.env"):
            _model_flow_api_key_provider(load_config(), "zai", "old-model")

        # The garbage value should NOT have been saved
        saved = get_env_value("GLM_BASE_URL") or ""
        assert not saved or saved.startswith(("http://", "https://")), \
            f"Non-URL value was saved as GLM_BASE_URL: {saved}"
        captured = capsys.readouterr()
        assert "Invalid URL" in captured.out

    def test_valid_base_url_accepted(self, config_home, monkeypatch):
        """A proper URL should be saved normally."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get("zai")
        if not pconfig:
            pytest.skip("zai not in PROVIDER_REGISTRY")

        monkeypatch.setenv("GLM_API_KEY", "test-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config, get_env_value

        with patch("hermes_cli.auth._prompt_model_selection", return_value="glm-5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value="https://custom.z.ai/api/paas/v4"):
            _model_flow_api_key_provider(load_config(), "zai", "old-model")

        saved = get_env_value("GLM_BASE_URL") or ""
        assert saved == "https://custom.z.ai/api/paas/v4"

    def test_empty_base_url_keeps_default(self, config_home, monkeypatch):
        """Pressing Enter (empty) should not change the base URL."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get("zai")
        if not pconfig:
            pytest.skip("zai not in PROVIDER_REGISTRY")

        monkeypatch.setenv("GLM_API_KEY", "test-key")
        monkeypatch.delenv("GLM_BASE_URL", raising=False)

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config, get_env_value

        with patch("hermes_cli.auth._prompt_model_selection", return_value="glm-5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "zai", "old-model")

        saved = get_env_value("GLM_BASE_URL") or ""
        assert saved == "", "Empty input should not save a base URL"

