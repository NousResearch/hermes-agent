"""Tests for skin-aware spinner helpers in agent/display.py."""

from unittest.mock import patch as mock_patch, MagicMock

import pytest

from agent.display import (
    get_skin_waiting_faces,
    get_skin_thinking_faces,
    get_skin_thinking_verbs,
    KawaiiSpinner,
)


class TestGetSkinWaitingFaces:
    """get_skin_waiting_faces() resolves from skin, falls back to hardcoded."""

    def test_returns_skin_faces_when_configured(self):
        skin = MagicMock()
        skin.get_spinner_list.return_value = ["(⚔)", "(⛨)", "(▲)"]
        with mock_patch("agent.display._get_skin", return_value=skin):
            result = get_skin_waiting_faces()
            assert result == ["(⚔)", "(⛨)", "(▲)"]
            skin.get_spinner_list.assert_called_once_with("waiting_faces")

    def test_returns_hardcoded_when_skin_returns_empty(self):
        skin = MagicMock()
        skin.get_spinner_list.return_value = []
        with mock_patch("agent.display._get_skin", return_value=skin):
            result = get_skin_waiting_faces()
            assert result == KawaiiSpinner.KAWAII_WAITING

    def test_returns_hardcoded_when_no_skin(self):
        with mock_patch("agent.display._get_skin", return_value=None):
            result = get_skin_waiting_faces()
            assert result == KawaiiSpinner.KAWAII_WAITING

    def test_returns_hardcoded_when_skin_is_none(self):
        with mock_patch("agent.display._get_skin", return_value=None):
            assert get_skin_waiting_faces() == KawaiiSpinner.KAWAII_WAITING


class TestGetSkinThinkingFaces:
    """get_skin_thinking_faces() resolves from skin, falls back to hardcoded."""

    def test_returns_skin_faces_when_configured(self):
        skin = MagicMock()
        skin.get_spinner_list.return_value = ["(Ψ)", "(∿)", "(≈)"]
        with mock_patch("agent.display._get_skin", return_value=skin):
            result = get_skin_thinking_faces()
            assert result == ["(Ψ)", "(∿)", "(≈)"]
            skin.get_spinner_list.assert_called_once_with("thinking_faces")

    def test_returns_hardcoded_when_skin_returns_empty(self):
        skin = MagicMock()
        skin.get_spinner_list.return_value = []
        with mock_patch("agent.display._get_skin", return_value=skin):
            result = get_skin_thinking_faces()
            assert result == KawaiiSpinner.KAWAII_THINKING

    def test_returns_hardcoded_when_no_skin(self):
        with mock_patch("agent.display._get_skin", return_value=None):
            result = get_skin_thinking_faces()
            assert result == KawaiiSpinner.KAWAII_THINKING


class TestGetSkinThinkingVerbs:
    """get_skin_thinking_verbs() resolves from skin, falls back to hardcoded."""

    def test_returns_skin_verbs_when_configured(self):
        skin = MagicMock()
        skin.get_spinner_list.return_value = ["forging", "marching", "hammering plans"]
        with mock_patch("agent.display._get_skin", return_value=skin):
            result = get_skin_thinking_verbs()
            assert result == ["forging", "marching", "hammering plans"]
            skin.get_spinner_list.assert_called_once_with("thinking_verbs")

    def test_returns_hardcoded_when_skin_returns_empty(self):
        skin = MagicMock()
        skin.get_spinner_list.return_value = []
        with mock_patch("agent.display._get_skin", return_value=skin):
            result = get_skin_thinking_verbs()
            assert result == KawaiiSpinner.THINKING_VERBS

    def test_returns_hardcoded_when_no_skin(self):
        with mock_patch("agent.display._get_skin", return_value=None):
            result = get_skin_thinking_verbs()
            assert result == KawaiiSpinner.THINKING_VERBS


class TestSpinnerHelpersAresSkinIntegration:
    """Integration test: ares skin spinner values are returned correctly."""

    def test_ares_skin_waiting_faces(self):
        from hermes_cli.skin_engine import set_active_skin
        set_active_skin("ares")
        try:
            result = get_skin_waiting_faces()
            assert result == ["(⚔)", "(⛨)", "(▲)", "(<>)", "(/)"]
        finally:
            set_active_skin("default")

    def test_ares_skin_thinking_faces(self):
        from hermes_cli.skin_engine import set_active_skin
        set_active_skin("ares")
        try:
            result = get_skin_thinking_faces()
            assert result == ["(⚔)", "(⛨)", "(▲)", "(⌁)", "(<>)"]
        finally:
            set_active_skin("default")

    def test_ares_skin_thinking_verbs(self):
        from hermes_cli.skin_engine import set_active_skin
        set_active_skin("ares")
        try:
            result = get_skin_thinking_verbs()
            assert "forging" in result
            assert "marching" in result
            assert "plotting impact" in result
        finally:
            set_active_skin("default")

    def test_default_skin_falls_back_to_hardcoded(self):
        from hermes_cli.skin_engine import set_active_skin
        set_active_skin("default")
        try:
            # Default skin has empty spinner lists — should fall back to hardcoded
            assert get_skin_waiting_faces() == KawaiiSpinner.KAWAII_WAITING
            assert get_skin_thinking_faces() == KawaiiSpinner.KAWAII_THINKING
            assert get_skin_thinking_verbs() == KawaiiSpinner.THINKING_VERBS
        finally:
            set_active_skin("default")

    def test_poseidon_skin_thinking_verbs(self):
        from hermes_cli.skin_engine import set_active_skin
        set_active_skin("poseidon")
        try:
            verbs = get_skin_thinking_verbs()
            assert "charting currents" in verbs
            assert "sounding the depth" in verbs
        finally:
            set_active_skin("default")
