"""Tests for tips i18n integration."""
import pytest
from unittest.mock import patch, MagicMock


def test_tips_module_imports():
    """Test that tips module can be imported."""
    from hermes_cli.tips import get_random_tip, TIPS
    assert callable(get_random_tip)
    assert isinstance(TIPS, list)
    assert len(TIPS) > 0


def test_get_random_tip_returns_string():
    """Test that get_random_tip returns a non-empty string."""
    from hermes_cli.tips import get_random_tip
    tip = get_random_tip()
    assert isinstance(tip, str)
    assert len(tip) > 0


def test_tips_uses_i18n_when_available():
    """Test that tips prefer i18n translations when available."""
    from hermes_cli import tips

    # Mock i18n tips available
    mock_tips = ["i18n提示1", "i18n提示2", "i18n提示3"]
    with patch('hermes_cli.strings.get_tips', return_value=mock_tips):
        tip = tips.get_random_tip()
        assert tip in mock_tips


def test_tips_fallback_to_hardcoded_when_i18n_empty():
    """Test that tips fall back to hardcoded list when i18n returns empty."""
    from hermes_cli import tips

    # Mock i18n returns empty
    with patch('hermes_cli.strings.get_tips', return_value=[]):
        tip = tips.get_random_tip()
        assert tip in tips.TIPS


def test_tips_fallback_to_hardcoded_when_i18n_raises():
    """Test that tips fall back to hardcoded list when i18n raises exception."""
    from hermes_cli import tips

    # Mock i18n raises exception
    with patch('hermes_cli.strings.get_tips', side_effect=Exception("i18n error")):
        tip = tips.get_random_tip()
        assert tip in tips.TIPS


def test_tips_hardcoded_list_preserved():
    """Test that hardcoded TIPS list is preserved for fallback."""
    from hermes_cli.tips import TIPS
    # Verify TIPS has substantial content (200+ tips)
    assert len(TIPS) > 200
    # Verify some known tips exist
    assert any("/background" in tip for tip in TIPS)
    assert any("/branch" in tip for tip in TIPS)
