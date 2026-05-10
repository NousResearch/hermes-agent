"""Tests for Telegram Quick Actions local control plane."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

from hermes_cli.quick_actions import (
    discard_candidate,
    format_candidate_digest,
    list_candidates,
    promote_candidate,
    prune_active_actions,
)


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


def test_list_candidates_filters_candidate_status(tmp_path):
    _write_candidate(tmp_path, token="a", status="candidate")
    _write_candidate(tmp_path, token="b", status="discarded")

    rows = list_candidates(home=tmp_path)

    assert [r["token"] for r in rows] == ["a"]


def test_format_candidate_digest_is_telegram_friendly(tmp_path):
    _write_candidate(
        tmp_path,
        token="tok123",
        action="todo",
        title="A very useful follow-up that should be short enough to scan in Telegram",
        recommended_targets=["cortex_todo", "kanban_candidate"],
    )

    rows = list_candidates(home=tmp_path)
    output = format_candidate_digest(rows, status="candidate", limit=5)

    assert "Quick Actions review" in output
    assert "id\tstatus\taction" not in output
    assert "`tok123` · todo · todo, kanban" in output
    assert "state: candidate" in output
    assert "/qa promote tok123 --to cortex_todo" in output
    assert "/qa show <id>" in output


def test_promote_candidate_marks_row_and_appends_promotion(tmp_path):
    _write_candidate(tmp_path, token="tok123")

    updated = promote_candidate("tok123", target="cortex", home=tmp_path, actor="test")

    assert updated["status"] == "promoted"
    assert updated["promoted_to"] == "cortex"
    rows = [json.loads(line) for line in (_qa_dir(tmp_path) / "routing_candidates.jsonl").read_text().splitlines()]
    assert rows[0]["status"] == "promoted"
    events = [json.loads(line) for line in (_qa_dir(tmp_path) / "promotions.jsonl").read_text().splitlines()]
    assert events[0]["candidate_id"] == "tok123"
    assert events[0]["target"] == "cortex"
    assert events[0]["status"] == "pending_execution"


def test_discard_candidate_marks_row_and_appends_discard(tmp_path):
    _write_candidate(tmp_path, token="tok123")

    updated = discard_candidate("tok123", reason="noise", home=tmp_path, actor="test")

    assert updated["status"] == "discarded"
    assert updated["discard_reason"] == "noise"
    events = [json.loads(line) for line in (_qa_dir(tmp_path) / "discards.jsonl").read_text().splitlines()]
    assert events[0]["candidate_id"] == "tok123"
    assert events[0]["reason"] == "noise"


def test_prune_active_actions_removes_stale_and_can_drop_undated(tmp_path):
    now = datetime.now(timezone.utc)
    active = {
        "fresh": {"created_at": now.isoformat(), "content": "keep"},
        "old": {"created_at": (now - timedelta(days=30)).isoformat(), "content": "drop"},
        "legacy": {"content": "undated"},
    }
    (_qa_dir(tmp_path) / "active_actions.json").write_text(json.dumps(active), encoding="utf-8")

    result = prune_active_actions(older_than_days=14, drop_undated=False, home=tmp_path)

    assert result == {"kept": 2, "removed": 1, "total": 3}
    kept = json.loads((_qa_dir(tmp_path) / "active_actions.json").read_text())
    assert set(kept) == {"fresh", "legacy"}

    result = prune_active_actions(older_than_days=14, drop_undated=True, home=tmp_path)

    assert result == {"kept": 1, "removed": 1, "total": 2}
    kept = json.loads((_qa_dir(tmp_path) / "active_actions.json").read_text())
    assert set(kept) == {"fresh"}
