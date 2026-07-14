from datetime import datetime, timezone
from unittest.mock import MagicMock

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from cli import HermesCLI


def _snapshot() -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime.now(timezone.utc),
        plan="Pro",
        windows=(
            AccountUsageWindow(
                label="Session",
                used_percent=15,
                reset_at=datetime(2026, 5, 28, 16, 30, tzinfo=timezone.utc),
            ),
        ),
    )


def test_codex_usage_command_renders_subscription_usage(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli._app = None
    cli._console_print = MagicMock()
    monkeypatch.setattr(
        "agent.account_usage.fetch_account_usage",
        lambda provider, **kwargs: _snapshot(),
    )

    cli._handle_codex_usage_command("/codex-usage")

    rendered = "\n".join(
        str(call.args[0]) if call.args else ""
        for call in cli._console_print.call_args_list
    )
    assert "openai-codex (Pro)" in rendered
    assert "Session: 85% remaining (15% used)" in rendered
    assert "resets" in rendered


def test_codex_usage_command_dispatches_through_registry():
    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_resume_sessions = None
    cli._handle_codex_usage_command = MagicMock()

    assert cli.process_command("/codex-usage") is True
    cli._handle_codex_usage_command.assert_called_once_with("/codex-usage")
