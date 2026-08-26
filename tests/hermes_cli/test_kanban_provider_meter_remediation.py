from __future__ import annotations

from datetime import datetime, timezone

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from hermes_cli.kanban_provider_meter import (
    ProviderMeterCandidate,
    choose_available_candidate,
)


def _snapshot(provider: str, used: float | None) -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider=provider,
        source="test",
        fetched_at=datetime.now(timezone.utc),
        windows=(AccountUsageWindow(label="window", used_percent=used),),
    )


def test_chooses_claude_when_both_providers_have_headroom():
    candidates = (
        ProviderMeterCandidate("anthropic", "claude-sonnet-4-6"),
        ProviderMeterCandidate("openai-codex", "gpt-5.6-luna"),
    )
    result = choose_available_candidate(
        candidates,
        {"anthropic": _snapshot("anthropic", 20), "openai-codex": _snapshot("openai-codex", 10)},
        minimum_remaining_percent=10,
    )
    assert result == candidates[0]


def test_skips_exhausted_claude_and_chooses_codex():
    candidates = (
        ProviderMeterCandidate("anthropic", "claude-sonnet-4-6"),
        ProviderMeterCandidate("openai-codex", "gpt-5.6-luna"),
    )
    result = choose_available_candidate(
        candidates,
        {"anthropic": _snapshot("anthropic", 95), "openai-codex": _snapshot("openai-codex", 20)},
        minimum_remaining_percent=10,
    )
    assert result == candidates[1]


def test_excludes_nous_and_fails_closed_on_missing_or_unknown_usage():
    candidates = (
        ProviderMeterCandidate("nous", "nous-model"),
        ProviderMeterCandidate("anthropic", "claude-sonnet-4-6"),
    )
    assert choose_available_candidate(candidates, {"anthropic": None}, minimum_remaining_percent=10) is None
    assert choose_available_candidate(
        candidates,
        {"anthropic": _snapshot("anthropic", None)},
        minimum_remaining_percent=10,
    ) is None


def test_candidate_order_is_not_allowed_to_override_claude_priority():
    candidates = (
        ProviderMeterCandidate("openai-codex", "gpt-5.6-luna"),
        ProviderMeterCandidate("anthropic", "claude-sonnet-4-6"),
    )
    result = choose_available_candidate(
        candidates,
        {"anthropic": _snapshot("anthropic", 20), "openai-codex": _snapshot("openai-codex", 20)},
        minimum_remaining_percent=10,
    )
    assert result == candidates[1]


def test_codex_requires_headroom_in_every_hard_quota_window():
    candidate = ProviderMeterCandidate("openai-codex", "gpt-5.6-luna")
    snapshot = AccountUsageSnapshot(
        provider="openai-codex",
        source="test",
        fetched_at=datetime.now(timezone.utc),
        windows=(
            AccountUsageWindow(label="Session", used_percent=100),
            AccountUsageWindow(label="Weekly", used_percent=20),
        ),
    )
    assert choose_available_candidate(
        [candidate], {"openai-codex": snapshot}, minimum_remaining_percent=10
    ) is None
