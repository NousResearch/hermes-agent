"""Tests for the configurable skill review quality gate in AIAgent."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _fake_routed_client():
    """Return a minimal fake provider client to satisfy AIAgent.__init__."""
    client = MagicMock()
    client.api_key = "sk-fake-test-key"
    client.base_url = "https://api.openai.com/v1/"
    client._default_headers = {}
    return client


def _make_agent(extra_prompt: str = ""):
    """Construct a minimal AIAgent with mocked provider and config."""
    import run_agent as ra

    cfg = {"skills": {"extra_review_prompt": extra_prompt}} if extra_prompt else {}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_fake_routed_client(), "openai"),
        ),
    ):
        return ra.AIAgent(
            model="claude-haiku-4-5-20251001",
            quiet_mode=True,
            skip_context_files=True,
            max_iterations=1,
        )


class TestSkillReviewExtra:
    def test_defaults_to_empty_string_when_not_configured(self):
        agent = _make_agent()
        assert agent._skill_review_extra == ""

    def test_loaded_from_config(self):
        agent = _make_agent("verify: is this generalizable?")
        assert agent._skill_review_extra == "verify: is this generalizable?"

    def test_extra_appended_to_skill_only_review(self):
        """When review_skills=True and review_memory=False, extra is appended."""
        import run_agent as ra

        agent = _make_agent("my-quality-gate")
        captured: list[str] = []

        fake_review = MagicMock()
        fake_review.run_conversation = lambda user_message, **kw: captured.append(user_message)

        with patch(
            "threading.Thread",
            side_effect=lambda *a, target, **kw: type("T", (), {"start": lambda s: target()})(),
        ):
            with patch.object(ra, "AIAgent", return_value=fake_review):
                agent._spawn_background_review(
                    messages_snapshot=[],
                    review_memory=False,
                    review_skills=True,
                )

        assert len(captured) == 1
        assert "my-quality-gate" in captured[0]

    def test_extra_NOT_appended_to_memory_only_review(self):
        """When review_skills=False, extra prompt must not appear."""
        import run_agent as ra

        agent = _make_agent("my-quality-gate")
        captured: list[str] = []

        fake_review = MagicMock()
        fake_review.run_conversation = lambda user_message, **kw: captured.append(user_message)

        with patch(
            "threading.Thread",
            side_effect=lambda *a, target, **kw: type("T", (), {"start": lambda s: target()})(),
        ):
            with patch.object(ra, "AIAgent", return_value=fake_review):
                agent._spawn_background_review(
                    messages_snapshot=[],
                    review_memory=True,
                    review_skills=False,
                )

        assert len(captured) == 1
        assert "my-quality-gate" not in captured[0]

    def test_no_extra_when_config_is_empty(self):
        """When extra_review_prompt is empty, prompt is unchanged."""
        import run_agent as ra

        agent = _make_agent()
        captured: list[str] = []

        fake_review = MagicMock()
        fake_review.run_conversation = lambda user_message, **kw: captured.append(user_message)

        with patch(
            "threading.Thread",
            side_effect=lambda *a, target, **kw: type("T", (), {"start": lambda s: target()})(),
        ):
            with patch.object(ra, "AIAgent", return_value=fake_review):
                agent._spawn_background_review(
                    messages_snapshot=[],
                    review_memory=False,
                    review_skills=True,
                )

        assert len(captured) == 1
        assert captured[0] == agent._SKILL_REVIEW_PROMPT
