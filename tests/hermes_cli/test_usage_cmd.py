from types import SimpleNamespace

from agent.account_usage import (
    AccountUsageSnapshot,
    AccountUsageWindow,
    compact_account_usage_line,
    normalize_usage_provider,
    snapshot_to_dict,
)
from hermes_cli.usage_cmd import collect_usage_report, format_usage_text, usage_command


def test_normalize_usage_provider_aliases():
    assert normalize_usage_provider("codex") == "openai-codex"
    assert normalize_usage_provider("claude") == "anthropic"
    assert normalize_usage_provider("grok") == "xai-oauth"


def test_collect_usage_report_grok_is_unsupported():
    report = collect_usage_report("xai-oauth")
    assert report["status"] == "unsupported"
    assert "official" in report["reason"]


def test_collect_usage_report_unknown_provider():
    report = collect_usage_report("gemini")
    assert report["status"] == "unknown"


def test_collect_usage_report_codex_ok(monkeypatch):
    from datetime import datetime, timezone

    snap = AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime(2026, 8, 23, tzinfo=timezone.utc),
        plan="Plus",
        windows=(
            AccountUsageWindow(
                label="Session",
                used_percent=100.0,
                reset_at=datetime(2026, 8, 27, 9, 24, tzinfo=timezone.utc),
            ),
            AccountUsageWindow(label="Weekly", used_percent=40.0),
        ),
        details=("You have 1 reset banked - use /usage reset to activate",),
    )
    monkeypatch.setattr("hermes_cli.usage_cmd.fetch_account_usage", lambda provider: snap)
    report = collect_usage_report("codex")
    assert report["status"] == "ok"
    assert report["plan"] == "Plus"
    assert report["windows"][0]["remaining_percent"] == 0
    assert report["windows"][1]["remaining_percent"] == 60


def test_collect_usage_report_unavailable_on_none(monkeypatch):
    monkeypatch.setattr("hermes_cli.usage_cmd.fetch_account_usage", lambda provider: None)
    report = collect_usage_report("anthropic")
    assert report["status"] == "unavailable"


def test_usage_command_json_single(monkeypatch, capsys):
    monkeypatch.setattr(
        "hermes_cli.usage_cmd.collect_usage_report",
        lambda provider: {"provider": "xai-oauth", "status": "unsupported", "reason": "no official quota API"},
    )
    code = usage_command(SimpleNamespace(provider="grok", json=True))
    assert code == 0
    payload = capsys.readouterr().out
    assert "unsupported" in payload
    assert "xai-oauth" in payload


def test_usage_command_empty_install_is_explicit(monkeypatch, capsys):
    monkeypatch.setattr(
        "hermes_cli.usage_cmd.collect_usage_report",
        lambda provider: {"provider": provider, "status": "unavailable", "reason": "quota endpoint unavailable"},
    )
    code = usage_command(SimpleNamespace(provider="", json=False))
    assert code == 0
    out = capsys.readouterr().out
    assert "No official quota sources are signed in." in out
    assert "unavailable" in out
    assert "unsupported" in out


def test_usage_command_unknown_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(
        "hermes_cli.usage_cmd.collect_usage_report",
        lambda provider: {"provider": "gemini", "status": "unknown", "reason": "no official quota source"},
    )
    code = usage_command(SimpleNamespace(provider="gemini", json=False))
    assert code == 2
    assert "unknown" in capsys.readouterr().out


def test_compact_account_usage_line_omits_local_clock():
    from datetime import datetime, timezone, timedelta

    snap = AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime.now(timezone.utc),
        windows=(
            AccountUsageWindow(
                label="Session",
                used_percent=80.0,
                reset_at=datetime.now(timezone.utc) + timedelta(hours=2, minutes=10),
            ),
        ),
    )
    line = compact_account_usage_line(snap)
    assert "Session 20% left" in line
    assert "resets in" in line
    assert "(" not in line


def test_snapshot_to_dict_roundtrip():
    from datetime import datetime, timezone

    snap = AccountUsageSnapshot(
        provider="openrouter",
        source="credits_api",
        fetched_at=datetime(2026, 8, 23, tzinfo=timezone.utc),
        windows=(AccountUsageWindow(label="API key quota", used_percent=50.0, detail="$25 of $50"),),
        details=("Credits balance: $7.54",),
    )
    data = snapshot_to_dict(snap)
    assert data["status"] == "ok"
    assert data["windows"][0]["remaining_percent"] == 50
    text = format_usage_text(data)
    assert "OpenRouter" not in text or "openrouter" in text.lower() or "Provider: openrouter" in text
    assert "50% remaining" in text


def test_auth_status_prints_usage_line(monkeypatch, capsys):
    from hermes_cli import auth_commands

    monkeypatch.setattr(
        auth_commands.auth_mod,
        "get_auth_status",
        lambda provider: {"logged_in": True, "auth_type": "oauth"},
    )
    monkeypatch.setattr(
        "hermes_cli.usage_cmd.compact_usage_line",
        lambda provider: "Session 12% left, resets in 1h 4m",
    )
    auth_commands.auth_status_command(SimpleNamespace(provider="openai-codex"))
    out = capsys.readouterr().out
    assert "openai-codex: logged in" in out
    assert "usage: Session 12% left, resets in 1h 4m" in out
