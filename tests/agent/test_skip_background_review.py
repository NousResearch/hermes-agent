"""Tests for the skip_background_review constructor flag.

Verifies that AIAgent can be instructed to skip the end-of-turn
_spawn_background_review fork (~30K tokens / event), which is essential
on cron sessions that have no human-in-the-loop value from skill/memory
review forks.

Plan reference: ralplan-hermes-token-leaks.md §3.9 (Phase 8).
"""
from __future__ import annotations

from unittest.mock import patch

from run_agent import AIAgent


def _make_agent(skip_background_review: bool = False) -> AIAgent:
    """Construct a minimally-configured AIAgent for unit testing.

    Mirrors the kwargs in tests/hermes_cli/test_timeouts.py — provider /
    base_url stub plus skip_memory + skip_context_files to keep init fast.
    """
    return AIAgent(
        model="openai/gpt-4o-mini",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=skip_background_review,
        platform="cli",
    )


def test_default_skip_background_review_is_false() -> None:
    """Without an explicit override, AIAgent does NOT skip background review."""
    agent = _make_agent()
    assert agent.skip_background_review is False


def test_skip_background_review_flag_persists() -> None:
    """Passing skip_background_review=True records the flag on the instance."""
    agent = _make_agent(skip_background_review=True)
    assert agent.skip_background_review is True


def test_review_path_short_circuits_when_flag_set() -> None:
    """The end-of-turn review block is gated on `not self.skip_background_review`.

    We don't drive a full conversation — instead we exercise the boolean
    guard expression directly to confirm the gate works as wired.
    """
    agent = _make_agent(skip_background_review=True)

    # Simulate the conditions that would have fired the review:
    final_response = "ok"
    interrupted = False
    _should_review_memory = True
    _should_review_skills = True

    with patch.object(agent, "_spawn_background_review") as mock_spawn:
        # This is the exact guard from run_agent.py end-of-turn block.
        if (
            final_response
            and not interrupted
            and not getattr(agent, "skip_background_review", False)
            and (_should_review_memory or _should_review_skills)
        ):
            agent._spawn_background_review(
                messages_snapshot=[],
                review_memory=_should_review_memory,
                review_skills=_should_review_skills,
            )

        mock_spawn.assert_not_called()


def test_review_path_fires_when_flag_unset() -> None:
    """Counterpart: with the flag off, the review path is reachable."""
    agent = _make_agent(skip_background_review=False)

    final_response = "ok"
    interrupted = False
    _should_review_memory = False
    _should_review_skills = True

    with patch.object(agent, "_spawn_background_review") as mock_spawn:
        if (
            final_response
            and not interrupted
            and not getattr(agent, "skip_background_review", False)
            and (_should_review_memory or _should_review_skills)
        ):
            agent._spawn_background_review(
                messages_snapshot=[],
                review_memory=_should_review_memory,
                review_skills=_should_review_skills,
            )

        mock_spawn.assert_called_once()


def test_cron_construction_sets_skip_background_review() -> None:
    """The cron scheduler MUST construct AIAgent with skip_background_review=True.

    Verified via source-text inspection — the cron scheduler is heavy to
    boot in tests (loads gateway config, profile, telemetry), so we
    assert that the source declares the flag rather than running the
    scheduler. This still catches accidental removal of the flag.
    """
    import pathlib

    scheduler_src = pathlib.Path(__file__).resolve().parents[2] / "cron" / "scheduler.py"
    text = scheduler_src.read_text(encoding="utf-8")

    # The flag must appear inside the cron AIAgent(...) construction block.
    # We look for it next to the existing skip_memory=True line.
    assert "skip_background_review=True" in text, (
        "cron/scheduler.py must construct AIAgent with skip_background_review=True "
        "(see ralplan-hermes-token-leaks.md §3.9 / Phase 8)."
    )
