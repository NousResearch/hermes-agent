"""Tests for curator CodeAct trajectory mining integration.

Covers:
  - find_codeact_promotion_candidates_from_sessions() wiring
  - CLI promote subcommand (list/approve/reject/sessions)
"""

import json
import textwrap
from unittest.mock import patch, MagicMock

import pytest

from agent.codeact_promotion import PromotionCandidate, flag_candidate


# ---------------------------------------------------------------------------
# Curator trajectory mining
# ---------------------------------------------------------------------------


class TestCuratorTrajectoryMining:
    """Test find_codeact_promotion_candidates_from_sessions with mocked SessionDB."""

    def _make_messages_with_helpers(self, fn_name="my_helper", count=4):
        """Create message dicts with repeated run_code tool calls."""
        code = textwrap.dedent(f"""\
            def {fn_name}():
                \"\"\"A helper function.\"\"\"
                return 42
        """)
        return [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_code",
                            "arguments": json.dumps({"code": code}),
                        }
                    }
                ],
            }
            for _ in range(count)
        ]

    def test_returns_empty_on_import_error(self):
        """Should gracefully return [] when dependencies are missing."""
        from agent.curator import find_codeact_promotion_candidates_from_sessions

        with patch.dict("sys.modules", {"agent.codeact_promotion": None}):
            result = find_codeact_promotion_candidates_from_sessions()
        assert result == []

    def test_returns_candidates_above_threshold(self):
        """Functions appearing >= min_occurrences across sessions are candidates."""
        from agent import curator

        messages = self._make_messages_with_helpers("fetch_data", count=5)
        mock_db = MagicMock()
        mock_db.list_sessions.return_value = ["session_1"]
        mock_db.get_messages.return_value = messages

        with (
            patch("hermes_state.SessionDB", return_value=mock_db),
            patch("agent.curator.logger"),
        ):
            candidates = curator.find_codeact_promotion_candidates_from_sessions(
                min_occurrences=3
            )

        assert len(candidates) == 1
        assert candidates[0]["fn_name"] == "fetch_data"
        assert candidates[0]["occurrence_count"] == 5

    def test_below_threshold_filtered(self):
        from agent import curator

        messages = self._make_messages_with_helpers("rare_fn", count=2)
        mock_db = MagicMock()
        mock_db.list_sessions.return_value = ["session_1"]
        mock_db.get_messages.return_value = messages

        with (
            patch("hermes_state.SessionDB", return_value=mock_db),
            patch("agent.curator.logger"),
        ):
            candidates = curator.find_codeact_promotion_candidates_from_sessions(
                min_occurrences=3
            )

        assert len(candidates) == 0

    def test_session_list_failure_returns_empty(self):
        from agent import curator

        mock_db = MagicMock()
        mock_db.list_sessions.side_effect = RuntimeError("db error")

        with (
            patch("hermes_state.SessionDB", return_value=mock_db),
            patch("agent.curator.logger"),
        ):
            candidates = curator.find_codeact_promotion_candidates_from_sessions()

        assert candidates == []

    def test_max_sessions_respected(self):
        from agent import curator

        messages = self._make_messages_with_helpers(count=4)
        mock_db = MagicMock()
        mock_db.list_sessions.return_value = ["s1", "s2", "s3", "s4", "s5"]
        mock_db.get_messages.return_value = messages

        with (
            patch("hermes_state.SessionDB", return_value=mock_db),
            patch("agent.curator.logger"),
        ):
            curator.find_codeact_promotion_candidates_from_sessions(max_sessions=3)

        # Should only have listed 3 sessions
        mock_db.list_sessions.assert_called_once_with(limit=3)


# ---------------------------------------------------------------------------
# CLI promote subcommand
# ---------------------------------------------------------------------------


class TestCLIPromoteCommand:
    """Test the hermes curator promote CLI subcommand."""

    def test_list_empty(self, tmp_path):
        from hermes_cli.curator import cli_main

        with patch("agent.codeact_promotion.get_hermes_home", return_value=tmp_path):
            result = cli_main(["promote", "list"])
        assert result == 0

    def test_list_with_candidates(self, tmp_path, sample_candidate):
        from hermes_cli.curator import cli_main

        with patch("agent.codeact_promotion.get_hermes_home", return_value=tmp_path):
            flag_candidate(sample_candidate)
            result = cli_main(["promote", "list"])
        assert result == 0

    def test_approve_nonexistent_name(self, tmp_path):
        from hermes_cli.curator import cli_main

        with patch("agent.codeact_promotion.get_hermes_home", return_value=tmp_path):
            result = cli_main(["promote", "approve", "--name", "nonexistent"])
        assert result == 1

    def test_approve_without_name_fails(self, tmp_path):
        from hermes_cli.curator import cli_main

        with patch("agent.codeact_promotion.get_hermes_home", return_value=tmp_path):
            result = cli_main(["promote", "approve"])
        assert result == 1

    def test_approve_writes_skill(self, tmp_path, sample_candidate):
        from hermes_cli.curator import cli_main

        with patch("agent.codeact_promotion.get_hermes_home", return_value=tmp_path):
            flag_candidate(sample_candidate)
            result = cli_main(["promote", "approve", "--name", "fetch_weather"])
        assert result == 0
        skill_md = tmp_path / "skills" / "promoted" / "fetch_weather" / "SKILL.md"
        assert skill_md.exists()
        assert "fetch_weather" in skill_md.read_text()

    def test_reject_removes_from_pending(self, tmp_path, sample_candidate):
        from hermes_cli.curator import cli_main

        with patch("agent.codeact_promotion.get_hermes_home", return_value=tmp_path):
            flag_candidate(sample_candidate)
            result = cli_main(["promote", "reject", "--name", "fetch_weather"])
        assert result == 0
        from agent.codeact_promotion import load_pending

        assert load_pending() == []

    def test_sessions_empty(self, tmp_path):
        from hermes_cli.curator import cli_main

        mock_db = MagicMock()
        mock_db.list_sessions.return_value = []

        with (
            patch("hermes_state.SessionDB", return_value=mock_db),
            patch("agent.curator.logger"),
        ):
            result = cli_main(["promote", "sessions"])
        assert result == 0

    def test_sessions_with_candidates(self, tmp_path):
        from hermes_cli.curator import cli_main

        code = 'def cool_func():\n    """Does cool stuff."""\n    pass\n'
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_code",
                            "arguments": json.dumps({"code": code}),
                        }
                    }
                ],
            }
            for _ in range(5)
        ]
        mock_db = MagicMock()
        mock_db.list_sessions.return_value = ["sess_1"]
        mock_db.get_messages.return_value = messages

        with (
            patch("hermes_state.SessionDB", return_value=mock_db),
            patch("agent.curator.logger"),
        ):
            result = cli_main(["promote", "sessions"])
        assert result == 0


# Fixtures needed by TestCLIPromoteCommand


@pytest.fixture
def sample_candidate():
    return PromotionCandidate(
        fn_name="fetch_weather",
        description="Fetch current weather for a city.",
        source_code='def fetch_weather(city: str) -> str:\n    """Fetch weather."""\n    return "sunny"',
        domain="research",
        tags=["weather", "api"],
        session_id="abc123",
        occurrence_count=5,
        seen_in_sessions=["abc123", "def456"],
    )
