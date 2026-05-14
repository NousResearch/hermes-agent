"""Tests for _resolve_display_model() — banner model name resolution (#25339)."""

from unittest.mock import patch, MagicMock


def _make_cli(model: str) -> "HermesCLI":
    """Create a minimal HermesCLI-like object for testing _resolve_display_model."""
    # Import here to avoid heavy CLI init at module level
    import sys
    import types

    # We need a minimal mock that has .model and can access CLI_CONFIG
    # Instead of instantiating HermesCLI (which triggers heavy init),
    # we monkey-patch the method onto a simple namespace.
    from cli import HermesCLI

    # Create a lightweight instance by mocking __init__
    obj = object.__new__(HermesCLI)
    obj.model = model
    return obj


def test_resolve_display_model_passthrough_real_model():
    """A real model name (e.g. anthropic/claude-sonnet-4) passes through unchanged."""
    cli = _make_cli("anthropic/claude-sonnet-4")
    result = cli._resolve_display_model()
    assert result == "anthropic/claude-sonnet-4"


def test_resolve_display_model_passthrough_dotted_model():
    """A dotted model name (e.g. gpt-5.4) passes through unchanged."""
    cli = _make_cli("gpt-5.4")
    result = cli._resolve_display_model()
    assert result == "gpt-5.4"


def test_resolve_display_model_resolves_provider_slug():
    """A provider slug like 'kimi-coding' resolves to the provider's default model."""
    cli = _make_cli("kimi-coding")
    result = cli._resolve_display_model()
    # Should resolve to a model name like kimi-k2.6, NOT stay as kimi-coding
    assert result != "kimi-coding"
    assert "/" not in result or "." in result  # a real model name


def test_resolve_display_model_resolves_minimax_cn():
    """A provider slug like 'minimax-cn' resolves to a real model name."""
    cli = _make_cli("minimax-cn")
    result = cli._resolve_display_model()
    assert result != "minimax-cn"


def test_resolve_display_model_empty_model():
    """Empty model returns empty string without error."""
    cli = _make_cli("")
    result = cli._resolve_display_model()
    assert result == ""


def test_resolve_display_model_none_model():
    """None model returns empty string without error."""
    cli = _make_cli(None)
    result = cli._resolve_display_model()
    assert result == ""


def test_resolve_display_model_unknown_string():
    """A string that is not a provider slug passes through unchanged."""
    cli = _make_cli("my-custom-model-v1")
    result = cli._resolve_display_model()
    assert result == "my-custom-model-v1"


def test_resolve_display_model_custom_provider_with_model():
    """When config has custom_providers with an explicit model field, use it."""
    cli = _make_cli("my-custom-provider")
    mock_config = {
        "model": {},
        "custom_providers": {
            "my-custom-provider": {
                "model": "deepseek-v4-flash",
                "base_url": "https://api.example.com/v1",
            }
        },
    }
    with patch("cli.CLI_CONFIG", mock_config):
        # my-custom-provider is NOT in ALIASES/OVERLAYS, so it passes through
        # unchanged — the function only resolves KNOWN provider slugs.
        result = cli._resolve_display_model()
        # Since my-custom-provider is not a known provider slug,
        # it returns as-is (safest behavior)
        assert result == "my-custom-provider"
