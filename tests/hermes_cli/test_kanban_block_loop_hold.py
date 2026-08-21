"""Regression coverage for the durable block-loop triage hold."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_specify as spec


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _held_triage_task(conn) -> str:
    tid = kb.create_task(conn, title="needs a human", assignee="worker")
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    assert kb.block_task(conn, tid, reason="credentials", kind="needs_input")
    assert kb.unblock_task(conn, tid)
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    assert kb.block_task(conn, tid, reason="credentials again", kind="needs_input")
    assert kb.get_task(conn, tid).status == "triage"
    return tid


def _fake_aux_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(payload)
    return response


def _event_count(conn, tid: str, *kinds: str) -> int:
    marks = ",".join("?" for _ in kinds)
    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind IN ({marks})",
            (tid, *kinds),
        ).fetchone()[0]
    )


def test_repeated_auto_decomposer_ticks_ignore_held_task(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _held_triage_task(conn)
        before = _event_count(conn, tid, "specified", "promoted", "claimed")

    attempted: list[str] = []
    for _ in range(3):
        for selected in decomp.list_triage_ids():
            attempted.append(selected)
            decomp.decompose_task(selected, author="auto-decomposer")

    with kb.connect_closing() as conn:
        assert attempted == []
        assert kb.get_task(conn, tid).status == "triage"
        assert _event_count(conn, tid, "specified", "promoted", "claimed") == before


def test_ordinary_triage_still_decomposes_normally(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="ordinary", triage=True)

    payload = {"fanout": False, "title": "specified", "body": "ready to run"}
    with patch("agent.auxiliary_client.call_llm", return_value=_fake_aux_response(payload)):
        outcome = decomp.decompose_task(tid, author="auto-decomposer")

    assert outcome.ok, outcome.reason
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "ready"
        assert _event_count(conn, tid, "specified", "promoted") == 2


def test_comments_do_not_clear_block_loop_hold(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _held_triage_task(conn)
        assert kb.has_block_loop_hold(conn, tid)
        kb.add_comment(conn, tid, author="operator", body="please try again")
        assert kb.has_block_loop_hold(conn, tid)

    assert tid not in decomp.list_triage_ids()


def test_atomic_promotion_and_claim_guards_reject_held_task(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _held_triage_task(conn)
        assert not kb.specify_triage_task(conn, tid, title="auto specified")
        assert kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="default",
            children=[{"title": "child", "body": "", "assignee": "worker", "parents": []}],
        ) is None
        assert kb.get_task(conn, tid).status == "triage"

        # Simulate a stale/racy writer promoting the held row directly. The
        # final ready -> running CAS must still enforce the durable hold.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        assert kb.claim_task(conn, tid, claimer="worker") is None
        assert kb.get_task(conn, tid).status == "triage"
        assert _event_count(conn, tid, "claimed") == 2  # only the pre-hold runs


def test_explicit_specify_recovers_hold_exactly_once(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _held_triage_task(conn)

    payload = {"title": "human recovered", "body": "approved recovery"}
    with patch("agent.auxiliary_client.call_llm", return_value=_fake_aux_response(payload)):
        first = spec.specify_task(tid, author="operator")
        second = spec.specify_task(tid, author="operator")

    assert first.ok
    assert not second.ok
    with kb.connect_closing() as conn:
        assert not kb.has_block_loop_hold(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"
        assert _event_count(conn, tid, "block_loop_recovered") == 1
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        assert kb.claim_task(conn, tid, claimer="worker") is None
        assert _event_count(conn, tid, "claimed") == 3
