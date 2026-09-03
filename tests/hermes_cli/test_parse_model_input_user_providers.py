"""parse_model_input resolves user-declared config providers (providers:/custom_providers:)."""

import pytest

from hermes_cli.models import parse_model_input


def _write_provider_config(tmp_path, providers_yaml=""):
    (tmp_path / "config.yaml").write_text(providers_yaml)


def test_bare_user_declared_provider_prefix_is_split(tmp_path, monkeypatch):
    """``myrouter:gpt`` with myrouter declared under providers: splits."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_provider_config(
        tmp_path,
        "model:\n  default: glm\n  provider: myrouter\n"
        "providers:\n  myrouter:\n    base_url: http://127.0.0.1:20128/v1\n"
        "    model: glm\n    api_key: sk-test\n",
    )
    provider, model = parse_model_input("myrouter:gpt", "myrouter")
    assert provider == "myrouter"
    assert model == "gpt"


def test_named_custom_providers_entry_is_split(tmp_path, monkeypatch):
    """``custom:myrouter:model`` resolves when myrouter is a custom_providers entry."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_provider_config(
        tmp_path,
        "custom_providers:\n  - name: myrouter\n"
        "    base_url: http://127.0.0.1:20128/v1\n"
        "    api_key: sk-test\n",
    )
    provider, model = parse_model_input("custom:myrouter:ocg/ox-alpha", "custom")
    assert provider == "custom:myrouter"
    assert model == "ocg/ox-alpha"


def test_undeclared_prefix_stays_whole(tmp_path, monkeypatch):
    """A colon-bearing model with an undeclared left side is not split."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_provider_config(tmp_path, "model:\n  default: gpt\n")
    provider, model = parse_model_input(
        "anthropic/claude-3.5-sonnet:beta", "openrouter"
    )
    assert provider == "openrouter"
    assert model == "anthropic/claude-3.5-sonnet:beta"


def test_builtin_provider_prefix_still_wins(tmp_path, monkeypatch):
    """Builtin aliases keep working even when user config declares providers."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_provider_config(
        tmp_path,
        "providers:\n  myrouter:\n    base_url: http://127.0.0.1:20128/v1\n",
    )
    provider, model = parse_model_input("zai:glm-5.2", "myrouter")
    assert provider == "zai"
    assert model == "glm-5.2"
