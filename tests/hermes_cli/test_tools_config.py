"""Tests for hermes_cli.tools_config platform tool persistence."""

from unittest.mock import patch

from hermes_cli.tools_config import (
    _get_platform_tools,
    _platform_toolset_summary,
    _save_platform_tools,
    _toolset_has_keys,
)


def test_get_platform_tools_uses_default_when_platform_not_configured():
    config = {}

    enabled = _get_platform_tools(config, "cli")

    assert enabled


def test_get_platform_tools_preserves_explicit_empty_selection():
    config = {"platform_toolsets": {"cli": []}}

    enabled = _get_platform_tools(config, "cli")

    assert enabled == set()


def test_platform_toolset_summary_uses_explicit_platform_list():
    config = {}

    summary = _platform_toolset_summary(config, platforms=["cli"])

    assert set(summary.keys()) == {"cli"}
    assert summary["cli"] == _get_platform_tools(config, "cli")


def test_toolset_has_keys_for_vision_accepts_codex_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(
        '{"active_provider":"openai-codex","providers":{"openai-codex":{"tokens":{"access_token": "codex-...oken","refresh_token": "codex-...oken"}}}}'
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AUXILIARY_VISION_PROVIDER", raising=False)
    monkeypatch.delenv("CONTEXT_VISION_PROVIDER", raising=False)

    assert _toolset_has_keys("vision") is True


def test_save_platform_tools_preserves_dynamic_mcp_toolsets():
    """Dynamic MCP toolsets should survive platform tool reconfiguration."""
    config = {
        "platform_toolsets": {
            "cli": ["web", "terminal", "mcp-time", "mcp-github", "mcp-custom-server"]
        }
    }

    new_selection = {"web", "browser"}

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", new_selection)

    saved_toolsets = config["platform_toolsets"]["cli"]

    assert "mcp-time" in saved_toolsets
    assert "mcp-github" in saved_toolsets
    assert "mcp-custom-server" in saved_toolsets
    assert "web" in saved_toolsets
    assert "browser" in saved_toolsets
    assert "terminal" not in saved_toolsets


def test_save_platform_tools_does_not_preserve_platform_presets():
    config = {
        "platform_toolsets": {
            "cli": ["hermes-cli", "mcp-time"]
        }
    }

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", {"web"})

    saved_toolsets = config["platform_toolsets"]["cli"]

    assert "web" in saved_toolsets
    assert "mcp-time" in saved_toolsets
    assert "hermes-cli" not in saved_toolsets


def test_save_platform_tools_handles_empty_existing_config():
    config = {}

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "telegram", {"web", "terminal"})

    saved_toolsets = config["platform_toolsets"]["telegram"]

    assert "web" in saved_toolsets
    assert "terminal" in saved_toolsets


def test_save_platform_tools_handles_invalid_existing_config():
    config = {
        "platform_toolsets": {
            "cli": "invalid-string-value"
        }
    }

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", {"web"})

    saved_toolsets = config["platform_toolsets"]["cli"]

    assert saved_toolsets == ["web"]
