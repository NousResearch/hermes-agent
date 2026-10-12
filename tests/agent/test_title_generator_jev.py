import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from agent.title_generator import _jev_extract_title, generate_title, _MAX_TITLE_WORDS

class TestJevExtractTitle:
    """Unit tests for _jev_extract_title() — the `jev extract` subprocess wrapper itself."""

    def test_empty_snippet_short_circuits(self):
        with patch("agent.title_generator.subprocess.run") as mock_run:
            assert _jev_extract_title("   ") is None
        mock_run.assert_not_called()

    def test_confident_match_returns_normalized_value(self):
        proc = MagicMock(returncode=0, stdout=json.dumps({
            "fields": {"title": {"value": "fix login button", "normalized": "Fix login button", "action": "ok"}},
        }))
        with patch("agent.title_generator.subprocess.run", return_value=proc):
            assert _jev_extract_title("please fix the login button") == "Fix login button"

    def test_low_confidence_action_returns_none(self):
        proc = MagicMock(returncode=0, stdout=json.dumps({
            "fields": {"title": {"value": "fix login button", "action": "review"}},
        }))
        with patch("agent.title_generator.subprocess.run", return_value=proc):
            assert _jev_extract_title("please fix the login button") is None

    def test_nonzero_exit_returns_none(self):
        proc = MagicMock(returncode=1, stdout="")
        with patch("agent.title_generator.subprocess.run", return_value=proc):
            assert _jev_extract_title("please fix the login button") is None

    def test_missing_binary_returns_none(self):
        with patch("agent.title_generator.subprocess.run", side_effect=FileNotFoundError("jev")):
            assert _jev_extract_title("please fix the login button") is None

    def test_timeout_returns_none(self):
        with patch("agent.title_generator.subprocess.run", side_effect=subprocess.TimeoutExpired("jev", 10)):
            assert _jev_extract_title("please fix the login button") is None

    def test_malformed_json_returns_none(self):
        proc = MagicMock(returncode=0, stdout="not json")
        with patch("agent.title_generator.subprocess.run", return_value=proc):
            assert _jev_extract_title("please fix the login button") is None


@pytest.mark.parametrize("candidate", [None, " ".join(["word"] * (_MAX_TITLE_WORDS + 1)), "Fix login button on mobile"])
def test_invalid_jev_title_preserves_llm_fallback(candidate):
    response = MagicMock()
    response.choices[0].message.content = '{"title":"Investigate title resolver bug"}'
    with patch("agent.title_generator._jev_extract_title", return_value=candidate), \
         patch("agent.title_generator._title_language", return_value=""), \
         patch("agent.title_generator.call_llm", return_value=response) as llm:
        assert generate_title("Investigate title generation failures") == "Investigate title resolver bug"
    llm.assert_called_once()


def test_confident_jev_title_skips_llm():
    with patch("agent.title_generator._jev_extract_title", return_value="Repair session naming"), \
         patch("agent.title_generator._title_language", return_value=""), \
         patch("agent.title_generator.call_llm") as llm:
        assert generate_title("Please repair session naming") == "Repair session naming"
    llm.assert_not_called()


def test_pinned_language_uses_translation_capable_llm():
    response = MagicMock()
    response.choices[0].message.content = '{"title":"Reparar nombres de sesiones"}'
    with patch("agent.title_generator._jev_extract_title") as jev, \
         patch("agent.title_generator._title_language", return_value="Spanish"), \
         patch("agent.title_generator.call_llm", return_value=response):
        assert generate_title("Please repair session naming") == "Reparar nombres de sesiones"
    jev.assert_not_called()


@pytest.fixture(autouse=True)
def _enable_jev_for_tests(monkeypatch):
    monkeypatch.setattr("agent.title_generator.is_jev_enabled", lambda task: True)
