"""Tests for provider presets feature (#1891)."""
import os
import pytest
from unittest.mock import patch, MagicMock
from hermes_cli.runtime_provider import (
    _expand_env_vars,
    get_provider_preset,
    list_provider_presets,
    get_default_preset_name,
)


SAMPLE_CONFIG = {
    "providers": {
        "local": {
            "type": "openai-compatible",
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "base_url": "http://localhost:8000/v1",
            "api_key": "dummy",
        },
        "openrouter": {
            "type": "openrouter",
            "model": "anthropic/claude-3.5-sonnet",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "${OPENROUTER_API_KEY}",
        },
        "anthropic": {
            "type": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "api_key": "${ANTHROPIC_API_KEY}",
        },
    },
    "default_provider": "local",
}


# ---------------------------------------------------------------------------
# _expand_env_vars
# ---------------------------------------------------------------------------

def test_expand_env_vars_substitutes_known_var(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret123")
    assert _expand_env_vars("${MY_KEY}") == "secret123"


def test_expand_env_vars_keeps_unknown_var():
    result = _expand_env_vars("${UNDEFINED_VAR_XYZ}")
    assert result == "${UNDEFINED_VAR_XYZ}"


def test_expand_env_vars_no_vars():
    assert _expand_env_vars("plain string") == "plain string"


def test_expand_env_vars_multiple(monkeypatch):
    monkeypatch.setenv("A", "hello")
    monkeypatch.setenv("B", "world")
    assert _expand_env_vars("${A}-${B}") == "hello-world"


# ---------------------------------------------------------------------------
# get_provider_preset
# ---------------------------------------------------------------------------

def test_get_provider_preset_returns_known():
    with patch("hermes_cli.runtime_provider.load_config", return_value=SAMPLE_CONFIG):
        preset = get_provider_preset("local")
    assert preset is not None
    assert preset["type"] == "openai-compatible"
    assert preset["model"] == "meta-llama/Llama-3.3-70B-Instruct"


def test_get_provider_preset_returns_none_for_unknown():
    with patch("hermes_cli.runtime_provider.load_config", return_value=SAMPLE_CONFIG):
        assert get_provider_preset("nonexistent") is None


def test_get_provider_preset_expands_env_vars(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-abc")
    with patch("hermes_cli.runtime_provider.load_config", return_value=SAMPLE_CONFIG):
        preset = get_provider_preset("openrouter")
    assert preset["api_key"] == "or-key-abc"


def test_get_provider_preset_no_base_url_for_anthropic():
    with patch("hermes_cli.runtime_provider.load_config", return_value=SAMPLE_CONFIG):
        preset = get_provider_preset("anthropic")
    assert preset is not None
    assert "base_url" not in preset


def test_get_provider_preset_empty_providers():
    with patch("hermes_cli.runtime_provider.load_config", return_value={}):
        assert get_provider_preset("local") is None


# ---------------------------------------------------------------------------
# list_provider_presets
# ---------------------------------------------------------------------------

def test_list_provider_presets_returns_all():
    with patch("hermes_cli.runtime_provider.load_config", return_value=SAMPLE_CONFIG):
        presets = list_provider_presets()
    assert set(presets.keys()) == {"local", "openrouter", "anthropic"}


def test_list_provider_presets_empty():
    with patch("hermes_cli.runtime_provider.load_config", return_value={}):
        assert list_provider_presets() == {}


def test_list_provider_presets_invalid_type():
    config = {"providers": "not-a-dict"}
    with patch("hermes_cli.runtime_provider.load_config", return_value=config):
        assert list_provider_presets() == {}


# ---------------------------------------------------------------------------
# get_default_preset_name
# ---------------------------------------------------------------------------

def test_get_default_preset_name():
    with patch("hermes_cli.runtime_provider.load_config", return_value=SAMPLE_CONFIG):
        assert get_default_preset_name() == "local"


def test_get_default_preset_name_empty():
    with patch("hermes_cli.runtime_provider.load_config", return_value={}):
        assert get_default_preset_name() is None


def test_get_default_preset_name_whitespace():
    config = {"default_provider": "  "}
    with patch("hermes_cli.runtime_provider.load_config", return_value=config):
        assert get_default_preset_name() is None


# ---------------------------------------------------------------------------
# cmd_provider CLI
# ---------------------------------------------------------------------------

def test_cmd_provider_list_no_presets(capsys):
    with patch("hermes_cli.runtime_provider.load_config", return_value={}):
        from hermes_cli.provider_cmd import cmd_provider
        args = MagicMock()
        args.provider_action = "list"
        cmd_provider(args)
    out = capsys.readouterr().out
    assert "No provider presets configured" in out


def test_cmd_provider_list_with_presets(capsys):
    with patch("hermes_cli.runtime_provider.load_config", return_value=SAMPLE_CONFIG):
        from hermes_cli.provider_cmd import cmd_provider
        args = MagicMock()
        args.provider_action = "list"
        cmd_provider(args)
    out = capsys.readouterr().out
    assert "local" in out
    assert "openrouter" in out
    assert "[default]" in out


def test_cmd_provider_show(capsys):
    with patch("hermes_cli.runtime_provider.load_config", return_value=SAMPLE_CONFIG):
        from hermes_cli.provider_cmd import cmd_provider
        args = MagicMock()
        args.provider_action = "show"
        args.name = "local"
        cmd_provider(args)
    out = capsys.readouterr().out
    assert "local" in out
    assert "openai-compatible" in out


def test_cmd_provider_show_unknown(capsys):
    import sys
    with patch("hermes_cli.runtime_provider.load_config", return_value=SAMPLE_CONFIG):
        from hermes_cli.provider_cmd import cmd_provider
        args = MagicMock()
        args.provider_action = "show"
        args.name = "nonexistent"
        with pytest.raises(SystemExit):
            cmd_provider(args)
