"""Chain break — successor-guard extension (ADR-0020 §2.1/§3.1, card t_d5708f00).

G-55 only gated on the explicit ``SUCCESSOR: required`` declaration. These tests
pin the completion-time extension: a completion whose summary/result/metadata
*promises* follow-on work while discharging nothing stamps a ``chain_broke``
event and auto-files a ``CHAIN-BROKE:`` triage card (warn phase), the
``intent_gate`` knob promotes that to a rejection, and the phrase-level
deny-list keeps the ``wall_mid_turn_successor`` data-word class silent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

# ---------- fixtures / helpers ----------


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "k.db"))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _cfg(monkeypatch, **over):
    """Stub ``kanban.successor_guard`` (shared-root read) with chain-break defaults."""
    cfg = {
        "gate": True, "wake": False, "wake_require_idle_queue": True,
        "fresh_window_s": 21600, "wake_patterns": [], "routers": {},
        "chain_break": True, "intent_gate": False, "structural_break_check": False,
        "intent_patterns": None, "intent_denylist": None,
    }
    cfg.update(over)
    monkeypatch.setattr(kb, "_successor_guard_config", lambda: dict(cfg))
    return cfg


def _events(conn, tid):
    return kb.list_events(conn, tid)


def _kinds(conn, tid):
    return [e.kind for e in _events(conn, tid)]


def _one(conn, tid, kind):
    rows = [e for e in _events(conn, tid) if e.kind == kind]
    assert len(rows) == 1, f"expected exactly one {kind} event for {tid}, got {_kinds(conn, tid)}"
    assert rows[0].payload is not None
    return rows[0]


def _triage_count(conn):
    """How many CHAIN-BROKE triage cards the guard has minted."""
    return conn.execute(
        "SELECT count(*) c FROM tasks WHERE idempotency_key LIKE 'chain-broke:%'"
    ).fetchone()["c"]


def _status(conn, tid):
    return conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()["status"]


Q8_SUMMARY = "preparing handoff successor — filing t_next tomorrow"

# ---------- 1. intent matching: phrase-level, verb-ordered ----------


@pytest.mark.parametrize("text", [
    Q8_SUMMARY,
    "successor next",
    "follow-on card will handle the rollout",
    "next step is a bisect card",
    "shipped the fix; continuing in t_deadbeef12",
    "handoff successor card to be created",
])
def test_intent_phrases_match(text):
    assert kb._successor_intent_match(text) is not None


@pytest.mark.parametrize("text", [
    "fixed wall sample wall_mid_turn_successor label",
    "renamed successor_labels.csv and handoff_next_step fixtures",
    "the successor field is unused",
    "no successor needed; the chain legitimately ends here",
    "shipped POTTRACK felt theme; 23 tests green",
    "wall_mid_turn_successor",
    "dataset successor_predictions.parquet written",
    "result: 5/5 byte-match, exhibit banked P9-device/, hl-smi clean 672MiB",
])
def test_data_words_and_nouns_never_match(text):
    """The wall_mid_turn_successor lesson: identifiers and bare nouns are not promises."""
    assert kb._successor_intent_match(text) is None


def test_intent_denylist_is_config_extendable():
    assert kb._successor_intent_match("successor next") is not None
    assert kb._successor_intent_match("successor next", denylist=[r"successor"]) is None


def test_intent_patterns_config_override_replaces_defaults():
    assert kb._successor_intent_match("done", patterns=[r"done"]) == "done"
    assert kb._successor_intent_match(Q8_SUMMARY, patterns=[r"^nothing$"]) is None


# ---------- 2. warn phase: event + CHAIN-BROKE triage card ----------


def test_warn_phase_stamps_event_and_files_triage_card(conn, monkeypatch):
    _cfg(monkeypatch)
    tid = kb.create_task(conn, title="Q8 bisect", body="bisect the divergence", assignee="dev")
    assert kb.complete_task(conn, tid, summary=Q8_SUMMARY, created_cards=[]) is True
    assert _status(conn, tid) == "done"  # phase 1 never rejects the completion
    assert _one(conn, tid, "completed")  # the completion itself still lands

    ev = _one(conn, tid, "chain_broke")
    payload = dict(ev.payload or {})
    assert payload["matched_phrase"]
    assert payload["completing_task"] == tid
    assert payload["structural"] is False

    triage = kb.get_task(conn, str(payload["triage_task"]))
    assert triage is not None
    assert triage.status == "triage"
    assert triage.title.startswith("CHAIN-BROKE:")
    assert "Q8 bisect" in triage.title
    assert "preparing handoff successor" in triage.body
    assert tid in triage.body


def test_triage_card_is_idempotent_per_completing_task(conn, monkeypatch):
    _cfg(monkeypatch)
    tid = kb.create_task(conn, title="dup", body="b", assignee="dev")
    kb.complete_task(conn, tid, summary=Q8_SUMMARY)
    first = _one(conn, tid, "chain_broke").payload["triage_task"]
    # A second completion attempt must not mint a second triage card.
    again = kb.create_task(conn, title="dup", body="b", assignee="dev",
                           idempotency_key=f"chain-broke:{tid}")
    assert again == first


def test_created_cards_discharges_the_chain_break(conn, monkeypatch):
    _cfg(monkeypatch)
    tid = kb.create_task(conn, title="parent", body="b", assignee="dev")
    child = kb.create_task(conn, title="child", body="b", assignee="dev", parents=[tid],
                           created_by="dev")
    assert kb.complete_task(conn, tid, summary=Q8_SUMMARY, created_cards=[child]) is True
    assert "chain_broke" not in _kinds(conn, tid)


def test_live_child_discharges_without_created_cards(conn, monkeypatch):
    _cfg(monkeypatch)
    tid = kb.create_task(conn, title="parent", body="b", assignee="dev")
    kb.create_task(conn, title="child", body="b", assignee="dev", parents=[tid], created_by="dev")
    assert kb.complete_task(conn, tid, summary=Q8_SUMMARY) is True
    assert "chain_broke" not in _kinds(conn, tid)


def test_waiver_discharges_the_chain_break(conn, monkeypatch):
    _cfg(monkeypatch)
    tid = kb.create_task(conn, title="parent", body="b", assignee="dev")
    assert kb.complete_task(
        conn, tid, summary=Q8_SUMMARY,
        metadata={"no_successor": "GREEN: chain verified complete, nothing downstream"},
    ) is True
    assert "chain_broke" not in _kinds(conn, tid)


def test_clean_completion_is_silent(conn, monkeypatch):
    """Acceptance arm 3: no intent + no open chain → no event, no triage card."""
    _cfg(monkeypatch)
    tid = kb.create_task(conn, title="leaf", body="b", assignee="dev")
    assert kb.complete_task(
        conn, tid, summary="fixed the contrast gate; 15.70:1 measured, tests green",
        result="restyled Undo/Reset to OutlinedButton",
    ) is True
    assert "chain_broke" not in _kinds(conn, tid)
    assert _triage_count(conn) == 0


def test_recursion_guard_covers_the_reject_leg(conn, monkeypatch):
    """A CHAIN-BROKE card must not mint another one when it completes.

    The warn-phase guard is reachable on triage cards; the reject leg runs
    pre-txn and cannot see triage-status cards at all (``complete_task`` only
    accepts running/ready/blocked/review), so this pins the leg that has no
    second line of defence.
    """
    _cfg(monkeypatch, intent_gate=True)
    tid = kb.create_task(conn, title="CHAIN-BROKE: dup — promised successor, created none",
                         body="b", assignee="dev", created_by=kb._CHAIN_BREAK_CREATOR)
    assert kb.complete_task(conn, tid, summary="successor next, routing it now") is True
    assert "chain_broke" not in _kinds(conn, tid)
    assert _triage_count(conn) == 0


# ---------- 3. phase-2 intent_gate: reject instead of warn ----------


def test_intent_gate_rejects_the_completion(conn, monkeypatch):
    _cfg(monkeypatch, intent_gate=True)
    tid = kb.create_task(conn, title="gated", body="b", assignee="dev")
    with pytest.raises(kb.MissingSuccessorError):
        kb.complete_task(conn, tid, summary=Q8_SUMMARY)
    assert _status(conn, tid) != "done"
    assert _one(conn, tid, "chain_broke").payload["gate"] == "reject"


def test_intent_gate_leaves_clean_completions_alone(conn, monkeypatch):
    _cfg(monkeypatch, intent_gate=True)
    tid = kb.create_task(conn, title="clean", body="b", assignee="dev")
    assert kb.complete_task(conn, tid, summary="fixed the parser, tests green") is True
    assert "chain_broke" not in _kinds(conn, tid)


def test_chain_break_master_switch_off(conn, monkeypatch):
    _cfg(monkeypatch, chain_break=False)
    tid = kb.create_task(conn, title="off", body="b", assignee="dev")
    assert kb.complete_task(conn, tid, summary=Q8_SUMMARY) is True
    assert "chain_broke" not in _kinds(conn, tid)


# ---------- 4. structural arm (open goals + nothing running) ----------


def _board_with_dangling_goal(conn):
    """The gargantua01 shape: a completable card plus an open sibling goal,
    nothing running. Returns (completing_id, sibling_id)."""
    tid = kb.create_task(conn, title="leaf", body="b", assignee="dev")
    sibling = kb.create_task(conn, title="other goal", body="b", assignee="dev")
    return tid, sibling


def test_structural_arm_off_by_default(conn, monkeypatch):
    _cfg(monkeypatch)
    tid, _sibling = _board_with_dangling_goal(conn)
    assert kb.complete_task(conn, tid, summary="step landed, tests green") is True
    assert "chain_broke" not in _kinds(conn, tid)


def test_structural_arm_fires_when_enabled(conn, monkeypatch):
    _cfg(monkeypatch, structural_break_check=True)
    tid, _sibling = _board_with_dangling_goal(conn)
    assert kb.complete_task(conn, tid, summary="step landed, tests green") is True
    ev = _one(conn, tid, "chain_broke")
    payload = dict(ev.payload or {})
    assert payload["structural"] is True
    assert payload["matched_phrase"] is None
    triage = kb.get_task(conn, str(payload["triage_task"]))
    assert triage is not None and triage.status == "triage"


def test_structural_arm_quiet_while_work_is_running(conn, monkeypatch):
    """Quiet-by-design (§3.4): live work on the board means nothing broke."""
    _cfg(monkeypatch, structural_break_check=True)
    tid, sibling = _board_with_dangling_goal(conn)
    kb.claim_task(conn, sibling, claimer="dev:1", ttl_seconds=600)
    assert kb.get_task(conn, sibling).status == "running"
    assert kb.complete_task(conn, tid, summary="step landed, tests green") is True
    assert "chain_broke" not in _kinds(conn, tid)


def test_structural_arm_quiet_when_the_chain_continues(conn, monkeypatch):
    """A successor card minted by this completion keeps the chain alive."""
    _cfg(monkeypatch, structural_break_check=True)
    tid, _sibling = _board_with_dangling_goal(conn)
    child = kb.create_task(conn, title="next peel", body="b", assignee="dev",
                           parents=[tid], created_by="dev")
    assert kb.complete_task(conn, tid, summary="step landed, tests green",
                            created_cards=[child]) is True
    assert "chain_broke" not in _kinds(conn, tid)


# ---------- 5. never breaks a real completion ----------


def test_chain_break_failure_does_not_undo_completion(conn, monkeypatch):
    _cfg(monkeypatch)
    tid = kb.create_task(conn, title="boom", body="b", assignee="dev")

    def _boom(*a, **k):
        raise RuntimeError("triage minting exploded")

    monkeypatch.setattr(kb, "create_task", _boom)
    assert kb.complete_task(conn, tid, summary=Q8_SUMMARY) is True
    assert _status(conn, tid) == "done"
    assert _one(conn, tid, "chain_break_skipped").payload["reason"] == "error"
