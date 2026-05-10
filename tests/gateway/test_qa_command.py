"""Tests for the gateway /qa Quick Actions control-plane command."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest


def _qa_dir(tmp_path):
    path = tmp_path / "telegram_quick_actions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_candidate(tmp_path, **overrides):
    row = {
        "token": "tok123",
        "action": "save",
        "status": "candidate",
        "title": "Useful saved result",
        "content": "content",
        "recommended_targets": ["cortex_memory"],
        "captured_at": "2026-05-09T00:00:00+00:00",
    }
    row.update(overrides)
    with (_qa_dir(tmp_path) / "routing_candidates.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def _runner_and_event(args: str):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    event = MagicMock()
    event.get_command_args.return_value = args
    event.source = MagicMock(user_name="Joohyun", user_id="u123")
    return runner, event


@pytest.mark.asyncio
async def test_qa_list_shows_candidates(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_candidate(tmp_path, token="tok123", title="Candidate title")
    runner, event = _runner_and_event("list --limit 5")

    output = await runner._handle_qa_command(event)

    assert "Quick Actions review" in output
    assert "`tok123` · save · memory" in output
    assert "Candidate title" in output
    assert "/qa promote tok123 --to cortex_memory" in output
    assert "id\tstatus\taction" not in output


@pytest.mark.asyncio
async def test_qa_promote_marks_candidate_without_downstream_mutation(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_candidate(tmp_path, token="tok123")
    runner, event = _runner_and_event("promote tok123 --to cortex_memory")

    output = await runner._handle_qa_command(event)

    assert "Promoted `tok123`" in output
    assert "pending_execution" in output
    rows = [json.loads(line) for line in (_qa_dir(tmp_path) / "routing_candidates.jsonl").read_text().splitlines()]
    assert rows[0]["status"] == "promoted"
    assert rows[0]["promoted_by"] == "gateway:Joohyun"
    promotions = [json.loads(line) for line in (_qa_dir(tmp_path) / "promotions.jsonl").read_text().splitlines()]
    assert promotions[0]["candidate_id"] == "tok123"
    assert promotions[0]["target"] == "cortex_memory"
    assert promotions[0]["status"] == "pending_execution"


@pytest.mark.asyncio
async def test_qa_prune_active(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = datetime.now(timezone.utc)
    (_qa_dir(tmp_path) / "active_actions.json").write_text(
        json.dumps(
            {
                "fresh": {"created_at": now.isoformat()},
                "old": {"created_at": (now - timedelta(days=30)).isoformat()},
            }
        ),
        encoding="utf-8",
    )
    runner, event = _runner_and_event("prune-active --older-than-days 14")

    output = await runner._handle_qa_command(event)

    assert '"kept": 1' in output
    assert '"removed": 1' in output
    kept = json.loads((_qa_dir(tmp_path) / "active_actions.json").read_text())
    assert set(kept) == {"fresh"}


def test_qa_is_gateway_known_and_bypasses_active_session():
    from hermes_cli.commands import is_gateway_known_command, should_bypass_active_session

    assert is_gateway_known_command("qa")
    assert is_gateway_known_command("quick-actions")
    assert should_bypass_active_session("qa")
