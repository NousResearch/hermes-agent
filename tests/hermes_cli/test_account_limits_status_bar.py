from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from cli import HermesCLI


def _snapshot(provider: str = "openai-codex") -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider=provider,
        source="test",
        fetched_at=datetime.now(timezone.utc),
        windows=(
            AccountUsageWindow(label="Session", used_percent=14.4),
            AccountUsageWindow(label="Weekly", used_percent=2.1),
        ),
    )


def test_format_account_limits_badge_uses_remaining_percentages() -> None:
    assert HermesCLI._format_account_limits_badge(_snapshot()) == "Acct S86% W98%"


def test_format_account_limits_badge_falls_back_for_generic_provider() -> None:
    assert HermesCLI._format_account_limits_badge(_snapshot("anthropic")) == "Acct S86% W98%"


def test_format_account_limits_badge_ignores_missing_usage() -> None:
    snapshot = AccountUsageSnapshot(
        provider="openai-codex",
        source="test",
        fetched_at=datetime.now(timezone.utc),
        windows=(AccountUsageWindow(label="Session", used_percent=None),),
    )

    assert HermesCLI._format_account_limits_badge(snapshot) == ""


def test_status_bar_text_includes_cached_account_limits(monkeypatch) -> None:
    shell = HermesCLI.__new__(HermesCLI)
    shell.model = "openai-codex/gpt-5.1-codex"
    shell.session_start = datetime.now()
    shell._prompt_start_time = None
    shell._prompt_duration = 0.0
    shell._background_tasks = {}
    shell.agent = SimpleNamespace(
        model="openai-codex/gpt-5.1-codex",
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_api_calls=0,
        context_compressor=None,
    )
    monkeypatch.setattr(shell, "_maybe_refresh_account_limits_badge", lambda agent: "Acct S86% W98%")

    text = shell._build_status_bar_text(width=100)

    assert "Acct S86% W98%" in text
