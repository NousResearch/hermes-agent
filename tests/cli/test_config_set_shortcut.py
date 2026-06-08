"""Tests for set_config_value shortcut key aliasing (issue #41943).

`hermes config set provider X` must write to `model.provider`, not a dead
top-level `provider:` key.  Same for `base_url` → `model.base_url`.
"""
import textwrap

import pytest


@pytest.fixture
def config_env(tmp_path, monkeypatch):
    """Set up an isolated config.yaml + .env for set_config_value tests."""
    cfg_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    cfg_path.write_text(textwrap.dedent("""\
        model:
          provider: alibaba
          default: deepseek-v4-pro
    """))
    env_path.write_text("")

    import hermes_cli.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "get_config_path", lambda: cfg_path)
    monkeypatch.setattr(cfg_mod, "get_env_path", lambda: env_path)
    monkeypatch.setattr(cfg_mod, "is_managed", lambda: False)
    monkeypatch.setattr(cfg_mod, "ensure_hermes_home", lambda: None)

    yield cfg_path, env_path


def _load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class TestProviderShortcut:
    """`hermes config set provider X` → model.provider."""

    def test_provider_shortcut_writes_model_provider(self, config_env):
        from hermes_cli.config import set_config_value
        cfg_path, _ = config_env

        set_config_value("provider", "opencode-go")

        data = _load_yaml(cfg_path)
        assert data["model"]["provider"] == "opencode-go"
        # Dead top-level key must NOT exist
        assert "provider" not in data, (
            "Shortcut 'provider' wrote to dead top-level key instead of model.provider"
        )

    def test_base_url_shortcut_writes_model_base_url(self, config_env):
        from hermes_cli.config import set_config_value
        cfg_path, _ = config_env

        set_config_value("base_url", "http://localhost:8080/v1")

        data = _load_yaml(cfg_path)
        assert data["model"]["base_url"] == "http://localhost:8080/v1"
        assert "base_url" not in data, (
            "Shortcut 'base_url' wrote to dead top-level key instead of model.base_url"
        )

    def test_dotted_key_not_affected_by_shortcut(self, config_env):
        """Explicit `model.provider` still works (no double-wrapping)."""
        from hermes_cli.config import set_config_value
        cfg_path, _ = config_env

        set_config_value("model.provider", "anthropic")

        data = _load_yaml(cfg_path)
        assert data["model"]["provider"] == "anthropic"

    def test_non_shortcut_key_unchanged(self, config_env):
        """Keys that have no shortcut pass through unchanged."""
        from hermes_cli.config import set_config_value
        cfg_path, _ = config_env

        set_config_value("terminal.backend", "docker")

        data = _load_yaml(cfg_path)
        assert data["terminal"]["backend"] == "docker"

    def test_provider_shortcut_preserves_other_model_keys(self, config_env):
        """Setting provider via shortcut must not overwrite model.default."""
        from hermes_cli.config import set_config_value
        cfg_path, _ = config_env

        set_config_value("provider", "opencode-go")

        data = _load_yaml(cfg_path)
        assert data["model"]["provider"] == "opencode-go"
        # Original model.default should survive
        assert data["model"]["default"] == "deepseek-v4-pro"
