"""Tests for the /session-check CLI handoff command."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from tests.cli.test_cli_init import _make_cli


def test_session_check_registered_in_command_registry():
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("session-check")
    assert cmd is not None
    assert cmd.name == "session-check"
    assert cmd.category == "Session"


def test_session_check_writes_receipt_and_prints_bounded_notice(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    shell = _make_cli(
        config_overrides={
            "context_handoff": {
                "enabled": True,
                "threshold": 0.65,
                "critical_threshold": 0.82,
                "max_chars": 1800,
            },
        }
    )
    shell.session_id = "sess-session-check"
    shell.conversation_history = [
        {"role": "user", "content": "Build feature A"},
        {"role": "assistant", "content": "Changed foo.py and ran a focused test."},
        {"role": "user", "content": "Finish carefully and do not leak sk-pro...7890"},
    ]
    shell.agent = SimpleNamespace(
        session_id=shell.session_id,
        context_compressor=SimpleNamespace(
            context_length=1000,
            last_prompt_tokens=700,
        ),
        _cached_system_prompt="",
        tools=None,
        _todo_store=SimpleNamespace(
            read=lambda: [
                {"id": "final-tests", "content": "Run final tests", "status": "pending"},
            ]
        ),
    )

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=700):
        assert shell.process_command("/session-check") is True

    output = capsys.readouterr().out
    assert "Session handoff ready:" in output
    assert "Context: 70%" in output
    assert "Continue from handoff" in output
    assert "sk-pro...7890" not in output

    receipts = list((tmp_path / "session_handoffs").glob("**/*.json"))
    assert len(receipts) == 1
    persisted = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert persisted["schema"] == "hermes.session_handoff.v1"
    assert persisted["session_id"] == "sess-session-check"
    assert "Finish carefully" in persisted["active_user_request"]
    assert "Run final tests" in persisted["resume_prompt"]
    assert "sk-pro...7890" not in json.dumps(persisted, sort_keys=True)
