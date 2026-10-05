"""N-fixes (audit dupla 2026-10-04): N-1/N-2/N-5 (human-gate multi-gate + artefact required),
N-4 (strip no compare de contrato), N-6 (sem evento espúrio em done/archived)."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


import hashlib as _hashlib_mod


def _dig(_p) -> str:
    """sha256 of the file at _p (bytes) — P2b-close digest binding."""
    from pathlib import Path as _P
    return _hashlib_mod.sha256(_P(_p).read_bytes()).hexdigest()

@pytest.fixture()
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    kb.init_db()
    c = kbc.connect()
    yield c
    c.close()


def _comment(conn, tid, body, author="operator", ts=100):
    conn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?,?,?,?)",
        (tid, author, body, ts),
    )
    conn.commit()


def _artifact(tmp_path, name="approval.md"):
    p = tmp_path / name
    p.write_text("ok", encoding="utf-8")
    return p


# --- N-1: multi-gate fail-open closed ---
def test_complete_refused_when_any_earlier_gate_still_armed(conn, tmp_path):
    tid = kb.create_task(conn, title="n1")
    art = _artifact(tmp_path, "g2.md")
    _comment(conn, tid, "HUMAN_GATE_PENDING: g1", ts=100)
    _comment(conn, tid, "HUMAN_GATE_PENDING: g2", ts=101)
    _comment(conn, tid, f"HUMAN_GATE_APPROVAL: g2 artifact={art} digest={_dig(art)}", ts=102)
    with pytest.raises(kb.HumanGatePendingError):
        kb.complete_task(conn, tid, force=True, result="ev", summary="s")
    assert kb.get_task(conn, tid).status != "done"


def test_complete_allowed_when_all_gates_released(conn, tmp_path):
    tid = kb.create_task(conn, title="n1ok")
    a1 = _artifact(tmp_path, "g1.md")
    a2 = _artifact(tmp_path, "g2.md")
    _comment(conn, tid, "HUMAN_GATE_PENDING: g1", ts=100)
    _comment(conn, tid, "HUMAN_GATE_PENDING: g2", ts=101)
    _comment(conn, tid, f"HUMAN_GATE_APPROVAL: g2 artifact={a2} digest={_dig(a2)}", ts=102)
    _comment(conn, tid, f"HUMAN_GATE_APPROVAL: g1 artifact={a1} digest={_dig(a1)}", ts=103)
    assert kb.complete_task(conn, tid, force=True, result="ev", summary="s") is True


# --- N-2: release exige artefacto em disco ---
def test_approval_without_artifact_does_not_release(conn):
    tid = kb.create_task(conn, title="n2")
    _comment(conn, tid, "HUMAN_GATE_PENDING: g1", ts=100)
    _comment(conn, tid, "HUMAN_GATE_APPROVAL: g1", ts=101)  # sem artifact=
    with pytest.raises(kb.HumanGatePendingError):
        kb.complete_task(conn, tid, force=True, result="ev", summary="s")


def test_approval_with_missing_artifact_file_does_not_release(conn, tmp_path):
    tid = kb.create_task(conn, title="n2b")
    _comment(conn, tid, "HUMAN_GATE_PENDING: g1", ts=100)
    _comment(conn, tid, f"HUMAN_GATE_APPROVAL: g1 artifact={tmp_path / 'nope.md'}", ts=101)
    with pytest.raises(kb.HumanGatePendingError):
        kb.complete_task(conn, tid, force=True, result="ev", summary="s")


# --- N-5: parser/lookup sem divergência (leading space invisible ao LIKE) ---
def test_leading_space_pending_marker_still_gates_completion(conn, tmp_path):
    tid = kb.create_task(conn, title="n5")
    _comment(conn, tid, " HUMAN_GATE_PENDING: g1", ts=100)
    # approval com artefacto real: o trail Scout via parser ACHA o pending (id g1)
    art = _artifact(tmp_path)
    _comment(conn, tid, f"HUMAN_GATE_APPROVAL: g1 artifact={art} digest={_dig(art)}", ts=101)
    # com o N-5 fixed, o approval do MESMO trail fecha o gate (parser-based lookup)
    assert kb.complete_task(conn, tid, force=True, result="ev", summary="s") is True


# --- N-4: newline no fim do contrato não gera falsa refusal ---
def test_trailing_newline_contract_same_text_no_false_refusal(conn, tmp_path):
    tid = kb.create_task(conn, title="n4")
    ws = Path(tempfile.mkdtemp(prefix="n4ws_"))
    art = ws / "a.md"
    art.write_text("x", encoding="utf-8")
    from hermes_cli import kanban_db_workspace as kbw
    kbw.set_workspace_path(conn, tid, str(ws))
    contract = json.dumps({"required_artifacts": [str(art)]}, separators=(",", ":")) + "\n"
    conn.execute("UPDATE tasks SET completion_contract=? WHERE id=?", (contract, tid))
    conn.commit()
    tid_claimed = kb.claim_task(conn, tid, claimer="host:mock")
    run_id = kb.get_task(conn, tid).current_run_id
    # complete: gate lê o mesmo texto (com \n). re-check strips both sides now.
    assert kb.complete_task(conn, tid, result="ev", expected_run_id=run_id) is True


# --- N-6: complete_task em card já done → False SEM evento novo ---
def test_complete_on_done_card_false_no_spurious_event(conn):
    tid = kb.create_task(conn, title="done")
    assert kb.complete_task(conn, tid, force=True, result="ev", summary="s") is True
    _comment(conn, tid, "HUMAN_GATE_PENDING: gX", ts=100)  # gate armado DEPOIS de done
    before = conn.execute("SELECT count(*) FROM task_events WHERE task_id=?", (tid,)).fetchone()[0]
    assert kb.complete_task(conn, tid, force=True, result="ev2", summary="s2") is False
    after = conn.execute("SELECT count(*) FROM task_events WHERE task_id=?", (tid,)).fetchone()[0]
    assert after == before
