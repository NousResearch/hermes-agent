"""Tests for the realtime gateway activity ledger."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.hooks import HookRegistry
import gateway.activity_ledger as activity_ledger


def _hermes_home() -> Path:
    return Path(os.environ["HERMES_HOME"])


def _write_config(body: str) -> None:
    (_hermes_home() / "config.yaml").write_text(body, encoding="utf-8")


def _read_turns(date: str) -> list[dict]:
    path = _hermes_home() / "activity-ledger" / date / "turns.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(autouse=True)
def _clear_ledger_state():
    activity_ledger._clear_state_for_tests()
    yield
    activity_ledger._clear_state_for_tests()


@pytest.mark.asyncio
async def test_enabled_gateway_turn_writes_one_redacted_record(monkeypatch):
    fixed_now = datetime(2026, 5, 6, 12, 34, 56, tzinfo=timezone.utc)
    monkeypatch.setattr(activity_ledger, "hermes_now", lambda: fixed_now)
    _write_config(
        "activity_ledger:\n"
        "  enabled: true\n"
        "  capture_turns: true\n"
        "  max_preview_chars: 500\n"
    )

    registry = HookRegistry()
    with patch("gateway.hooks.HOOKS_DIR", _hermes_home() / "missing-hooks"):
        registry.discover_and_load()

    await registry.emit(
        "agent:start",
        {
            "session_id": "sess-1",
            "platform": "discord",
            "message": "deploy with api_key=plain-secret " + ("x" * 700),
        },
    )
    await registry.emit(
        "agent:step",
        {
            "session_id": "sess-1",
            "platform": "discord",
            "tool_names": ["read_file"],
            "tools": [
                {
                    "name": "terminal",
                    "result": "RAW TOOL OUTPUT sk-proj-abcdefghijklmnop123456",
                }
            ],
        },
    )
    await registry.emit(
        "agent:end",
        {
            "session_id": "sess-1",
            "platform": "discord",
            "response": "done token=response-secret",
            "tool_names": ["write_file"],
        },
    )

    turns = _read_turns("2026-05-06")
    assert len(turns) == 1

    record = turns[0]
    assert record["schema_version"] == 1
    assert record["type"] == "turn"
    assert record["session_id"] == "sess-1"
    assert record["platform"] == "discord"
    assert record["date"] == "2026-05-06"
    assert record["time"] == "2026-05-06T12:34:56+00:00"
    assert record["tool_names"] == ["read_file", "terminal", "write_file"]
    assert len(record["message_preview"]) <= 500
    assert len(record["response_preview"]) <= 500
    assert "plain-secret" not in record["message_preview"]
    assert "response-secret" not in record["response_preview"]
    assert "RAW TOOL OUTPUT" not in json.dumps(record)


@pytest.mark.asyncio
async def test_disabled_ledger_does_not_write_turn_record(monkeypatch):
    fixed_now = datetime(2026, 5, 6, 12, 34, 56, tzinfo=timezone.utc)
    monkeypatch.setattr(activity_ledger, "hermes_now", lambda: fixed_now)
    _write_config(
        "activity_ledger:\n"
        "  enabled: false\n"
        "  capture_turns: true\n"
    )

    registry = HookRegistry()
    with patch("gateway.hooks.HOOKS_DIR", _hermes_home() / "missing-hooks"):
        registry.discover_and_load()

    await registry.emit(
        "agent:start",
        {"session_id": "sess-disabled", "platform": "telegram", "message": "hello"},
    )
    await registry.emit(
        "agent:end",
        {"session_id": "sess-disabled", "platform": "telegram", "response": "hi"},
    )

    assert not (_hermes_home() / "activity-ledger").exists()


def test_redaction_covers_common_secret_shapes():
    github_pat = "github_pat_" + ("a" * 32)
    preview = activity_ledger.make_preview(
        "api_key=plain-secret "
        "client_secret: client-secret-value "
        "Cookie: sessionid=abc123 "
        "Bearer bearer-token-1234567890 "
        "postgres://user:db-password@example.com/app "
        f"{github_pat}",
        500,
    )

    assert "plain-secret" not in preview
    assert "client-secret-value" not in preview
    assert "sessionid=abc123" not in preview
    assert "bearer-token-1234567890" not in preview
    assert "db-password" not in preview
    assert github_pat not in preview
    assert "***" in preview


def test_ledger_write_failure_is_isolated(monkeypatch):
    _write_config(
        "activity_ledger:\n"
        "  enabled: true\n"
        "  capture_turns: true\n"
    )

    def _raise(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(activity_ledger, "_append_jsonl", _raise)

    activity_ledger.handle_gateway_hook(
        "agent:start",
        {"session_id": "sess-fail", "platform": "slack", "message": "hello"},
    )
    activity_ledger.handle_gateway_hook(
        "agent:end",
        {"session_id": "sess-fail", "platform": "slack", "response": "hi"},
    )

    assert not (_hermes_home() / "activity-ledger").exists()


def test_default_config_is_opt_in():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["activity_ledger"]["enabled"] is False
    assert DEFAULT_CONFIG["activity_ledger"]["capture_turns"] is True
