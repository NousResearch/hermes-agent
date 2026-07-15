from argparse import Namespace
from datetime import datetime, timezone
import json

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from hermes_cli import account_usage_command as command


def test_usage_command_writes_machine_readable_snapshot_atomically(tmp_path, monkeypatch):
    snapshot = AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc),
        windows=(AccountUsageWindow(label="Weekly", used_percent=17),),
    )
    monkeypatch.setattr(command, "_effective_provider", lambda _explicit: "openai-codex")
    monkeypatch.setattr(command, "fetch_account_usage", lambda _provider: snapshot)
    output = tmp_path / "nested" / "account-usage.json"

    result = command.account_usage_command(
        Namespace(provider=None, json=False, output=str(output))
    )

    assert result == 0
    assert json.loads(output.read_text())["windows"][0]["usedPercent"] == 17
    assert not list(output.parent.glob("*.tmp"))


def test_usage_command_reports_unsupported_provider_as_json(monkeypatch, capsys):
    monkeypatch.setattr(command, "_effective_provider", lambda _explicit: "zai")
    monkeypatch.setattr(command, "fetch_account_usage", lambda _provider: None)

    result = command.account_usage_command(Namespace(provider="zai", json=True, output=None))

    assert result == 2
    assert json.loads(capsys.readouterr().out)["provider"] == "zai"
