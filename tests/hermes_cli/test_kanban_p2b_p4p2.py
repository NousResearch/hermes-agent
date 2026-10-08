"""P2b human-gate enforcement at completion + P4 PARTE 2 off-board served_model.

ALL of the behaviour lives in ``hermes_cli/kanban_db.py`` (product side,
self-contained — the marker parser helpers are COPIED from the autopilot
skill script, never imported from it).

P2b: ``complete_task`` refuses a card whose comment trail still carries an
ARMED ``HUMAN_GATE_PENDING: <gate_id>`` marker — scanned newest-first, the
first marker comment of {pending, approval} decides. Refusal is typed
(:class:`HumanGatePendingError`, a ``ValueError``) and audible
(``completion_blocked_human_gate``, own txn, run_id NULL = task-scoped).
A trailing matching approval (post-approval flow) completes normally.
``force=True`` does NOT bypass this gate (same rule as ``blocked``).

P4 PARTE 2: off-board completions (dispatcher never spawned the worker) are
opted in by the CALLER via the new keyword-only ``off_board=True``; the gate
then REQUIRES a non-blank string ``metadata['served_model']`` — refusal is
typed (:class:`OffBoardServedModelError`) + audible
(``completion_blocked_served_model``); success records ``route_served_model``
(payload: served_model) on the closing run. The on-board path (``off_board``
falsy) must be byte-for-byte unchanged: no new events, ever.

Sandbox: HERMES_HOME/HERMES_KANBAN_HOME pinned at tmp_path, delegated-child
marker dropped, Path.home monkeypatched (sibling fixture conventions).
Boards reais READ-ONLY.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

import tempfile as _tf
_ART = _tf.NamedTemporaryFile(prefix="p2b_art_", delete=False).name  # N-2: approvals need a real artifact on disk


import hashlib as _hashlib_mod


def _dig(_p) -> str:
    """sha256 of the file at _p (bytes) — P2b-close digest binding."""
    from pathlib import Path as _P
    return _hashlib_mod.sha256(_P(_p).read_bytes()).hexdigest()

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME/kanban home with an empty kanban DB."""
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_BUSY_TIMEOUT_MS", "2000")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect() as c:
        yield c


def _add_comment(conn, task_id, author, body):
    """Raw marker insert (the autopilot skill writes exactly this row shape)."""
    conn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) "
        "VALUES (?, ?, ?, ?)",
        (task_id, author, body, int(time.time())),
    )
    conn.commit()


PENDING = "HUMAN_GATE_PENDING:"
APPROVAL = "HUMAN_GATE_APPROVAL:"


def _status(conn, task_id):
    return conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()[0]


def _event_kinds(conn, task_id):
    return [
        r[0]
        for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
        ).fetchall()
    ]


def _event_payloads(conn, task_id, kind):
    return [
        json.loads(r[0]) if r[0] else {}
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
            (task_id, kind),
        ).fetchall()
    ]


def _event_run_ids(conn, task_id, kind):
    return [
        r[0]
        for r in conn.execute(
            "SELECT run_id FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
            (task_id, kind),
        ).fetchall()
    ]


def _claimed_task(conn):
    """A 'ready' card claimed by the dispatcher path (so a run row exists)."""
    tid = kb.create_task(conn, title="p2b card", assignee="coder")
    claimed = kb.claim_task(conn, tid, claimer="host:mock")
    assert claimed is not None, "claim must succeed for the completion path"
    return tid


class TestP2bHumanGateAtCompletion:
    """An armed HUMAN_GATE_PENDING marker blocks ``complete_task``."""

    def test_gate_pending_refuses_completion(self, conn):
        tid = _claimed_task(conn)
        _add_comment(conn, tid, "autopilot", f"{PENDING} g1")
        with pytest.raises(kb.HumanGatePendingError) as excinfo:
            kb.complete_task(conn, tid, result=" tried to close an armed gate")
        assert excinfo.value.gate_id == "g1"
        assert tid in str(excinfo.value)
        assert _status(conn, tid) == "running"  # status preserved
        kinds = _event_kinds(conn, tid)
        assert "completion_blocked_human_gate" in kinds
        assert "completed" not in kinds
        payload = _event_payloads(conn, tid, "completion_blocked_human_gate")
        assert payload == [{"gate_ids": ["g1"]}]

    def test_gate_refusal_event_is_task_scoped_run_id_null(self, conn):
        """The card may span runs before the human approves: the refusal event
        must be billed run_id=None (task-scoped), not to any one run."""
        tid = _claimed_task(conn)
        _add_comment(conn, tid, "autopilot", f"{PENDING} g1")
        with pytest.raises(kb.HumanGatePendingError):
            kb.complete_task(conn, tid, result="r")
        run_ids = _event_run_ids(conn, tid, "completion_blocked_human_gate")
        assert run_ids and run_ids[0] is None

    def test_gate_post_approval_flow_completes(self, conn):
        """The normal flow: pending marker then its matching approval (authored
        by anyone) must NOT trip the gate — completion proceeds."""
        tid = _claimed_task(conn)
        _add_comment(conn, tid, "autopilot", f"{PENDING} g1")
        _add_comment(conn, tid, "operator", f"{APPROVAL} g1 artifact={_ART} digest={_dig(_ART)}")
        assert kb.complete_task(conn, tid, result="approved by human") is True
        kinds = _event_kinds(conn, tid)
        assert "completed" in kinds
        assert "completion_blocked_human_gate" not in kinds
        assert _status(conn, tid) == "done"

    def test_gate_final_pending_rearms_after_approval(self, conn):
        """Newest-first semantics: an approval AFTER the pending disarms it,
        but a NEW pending after that approval re-arms the gate — the last
        marker of the trio decides."""
        tid = _claimed_task(conn)
        _add_comment(conn, tid, "autopilot", f"{PENDING} g1")
        _add_comment(conn, tid, "operator", f"{APPROVAL} g1 artifact={_ART} digest={_dig(_ART)}")
        _add_comment(conn, tid, "autopilot", f"{PENDING} g1")  # re-armed
        with pytest.raises(kb.HumanGatePendingError):
            kb.complete_task(conn, tid, result="r")
        assert _status(conn, tid) == "running"

    def test_gate_last_marker_approval_only_completes(self, conn):
        """A trailing approval marker completes the card ONLY when it closes
        a pending gate this trail actually armed; an approval for a gate id
        the trail never armed is inert noise, not a sign-off."""
        tid = _claimed_task(conn)
        _add_comment(conn, tid, "autopilot", "regular prose comment")
        _add_comment(conn, tid, "operator", f"{APPROVAL} gx artifact={_ART} digest={_dig(_ART)}")
        # gx never had a pending marker on its trail: nothing armed, nothing
        # signed off... the gate must not treat noise as approval. The card
        # completes (nothing is armed) — asserted explicitly so the gate's
        # 'not pending' reading of an inert approval stays pinned.
        assert kb.complete_task(conn, tid, result="r") is True
        assert _status(conn, tid) == "done"

    def test_gate_approval_without_matching_pending_completes(self, conn):
        """Ambiguity resolution for a lone-approval trail: with NO pending
        gate ever armed, no gate IS armed (None == 'not pending'), so the
        card completes — the assertion sibling of the uncoupled-gates test."""
        tid = _claimed_task(conn)
        _add_comment(conn, tid, "operator", f"{APPROVAL} gq artifact={_ART} digest={_dig(_ART)}")
        assert kb.complete_task(conn, tid, result="r") is True
        assert _status(conn, tid) == "done"

    def test_gate_approval_without_matching_pending_completes(self, conn):
        """Ambiguity resolution for a lone-approval trail: with NO pending
        gate ever armed, no gate IS armed, so the card completes."""
        tid = _claimed_task(conn)
        _add_comment(conn, tid, "operator", f"{APPROVAL} gq artifact={_ART} digest={_dig(_ART)}")
        assert kb.complete_task(conn, tid, result="r") is True
        assert _status(conn, tid) == "done"

    def test_gate_pending_after_approval_uncoupled_gates(self, conn):
        """Pending(g1) then APPROVAL for a DIFFERENT gate id (g2): the trail's
        last marker comment does not close g1 — newest-first only disarms
        when the approval id MATCHES the armed pending, so g1 stays armed."""
        tid = _claimed_task(conn)
        _add_comment(conn, tid, "autopilot", f"{PENDING} g1")
        _add_comment(conn, tid, "operator", f"{APPROVAL} g2 artifact={_ART} digest={_dig(_ART)}")
        with pytest.raises(kb.HumanGatePendingError) as excinfo:
            kb.complete_task(conn, tid, result="r")
        assert excinfo.value.gate_id == "g1"

    def test_gate_plain_comments_between_are_ignored(self, conn):
        tid = _claimed_task(conn)
        _add_comment(conn, tid, "autopilot", f"{PENDING} g1")
        _add_comment(conn, tid, "worker", "progress: half done")
        _add_comment(conn, tid, "operator", f"{APPROVAL} g1 artifact={_ART} digest={_dig(_ART)}")
        assert kb.complete_task(conn, tid, result="r") is True

    def test_gate_blank_pending_id_still_refused_with_none(self, conn):
        """A blank-gate-id pending marker cannot name what is being waited
        on; it is still armed and still refused (gate_id None)."""
        tid = _claimed_task(conn)
        _add_comment(conn, tid, "autopilot", f"{PENDING}   ")
        with pytest.raises(kb.HumanGatePendingError) as excinfo:
            kb.complete_task(conn, tid, result="r")
        assert excinfo.value.gate_id is None
        assert _event_payloads(conn, tid, "completion_blocked_human_gate") == [
            {"gate_ids": [None]}
        ]
        assert tid in str(excinfo.value)

    def test_gate_refusal_beats_hallucination_refusal(self, conn):
        """The parked-gate refusal is the SPECIFIC signal: it fires before
        the created-cards gate, so an armed gate + phantom created_cards
        reports the gate (not the hallucination)."""
        tid = _claimed_task(conn)
        _add_comment(conn, tid, "autopilot", f"{PENDING} g1")
        with pytest.raises(kb.HumanGatePendingError):
            kb.complete_task(
                conn, tid, result="r", created_cards=["t_ffffffffffff"],
            )
        kinds = _event_kinds(conn, tid)
        assert "completion_blocked_human_gate" in kinds
        assert "completion_blocked_hallucination" not in kinds
        assert "completed" not in kinds

    def test_gate_force_does_not_bypass(self, conn):
        """``force`` covers the live-claim fence only — same rule the blocked
        gate pins; an armed human gate is a human decision in flight."""
        tid = _claimed_task(conn)
        _add_comment(conn, tid, "autopilot", f"{PENDING} g1")
        with pytest.raises(kb.HumanGatePendingError):
            kb.complete_task(conn, tid, result="operator override", force=True)
        assert _status(conn, tid) == "running"
        assert "completed" not in _event_kinds(conn, tid)

    def test_gate_ignores_other_cards_comments(self, conn):
        tid = _claimed_task(conn)
        other = kb.create_task(conn, title="sibling", assignee="coder")
        _add_comment(conn, other, "autopilot", f"{PENDING} g1")
        assert kb.complete_task(conn, tid, result="r") is True
        assert _status(conn, tid) == "done"


class TestP4P2OffBoardServedModel:
    """off_board=True completions must name the model that served them."""

    def test_off_board_without_served_model_refused(self, conn):
        tid = _claimed_task(conn)
        with pytest.raises(kb.OffBoardServedModelError) as excinfo:
            kb.complete_task(conn, tid, result="r", off_board=True)
        assert tid in str(excinfo.value)
        assert _status(conn, tid) == "running"  # status preserved
        kinds = _event_kinds(conn, tid)
        assert "completion_blocked_served_model" in kinds
        assert "completed" not in kinds
        assert _event_run_ids(conn, tid, "completion_blocked_served_model") == [None]

    def test_off_board_blank_served_model_refused(self, conn):
        """A blank (or non-string) served_model is as good as none."""
        for blank in ("", "   "):
            tid = _claimed_task(conn)
            with pytest.raises(kb.OffBoardServedModelError):
                kb.complete_task(
                    conn, tid, result="r",
                    metadata={"served_model": blank}, off_board=True,
                )
            assert _status(conn, tid) == "running"
            assert "completed" not in _event_kinds(conn, tid)
        # A non-string (int/None/absent) also refuses: only a non-blank STR
        # names a model.
        for absent in (None, 42):
            tid = _claimed_task(conn)
            meta = {} if absent is None else {"served_model": absent}
            with pytest.raises(kb.OffBoardServedModelError):
                kb.complete_task(
                    conn, tid, result="r", metadata=meta, off_board=True,
                )
            assert _status(conn, tid) == "running"
            assert "completed" not in _event_kinds(conn, tid)

    def test_off_board_with_served_model_completes_and_records_route_event(
        self, conn,
    ):
        tid = _claimed_task(conn)
        run_id = kb.get_task(conn, tid).current_run_id
        assert run_id is not None
        ok = kb.complete_task(
            conn, tid, result="r", metadata={"served_model": "glm-5.3-flash"},
            off_board=True, require_recorded_origin=False,
        )
        assert ok is True
        assert _status(conn, tid) == "done"
        kinds = _event_kinds(conn, tid)
        assert "completion_blocked_served_model" not in kinds
        assert "route_served_model" in kinds
        rows = conn.execute(
            "SELECT run_id, payload FROM task_events "
            "WHERE task_id = ? AND kind = 'route_served_model' ORDER BY id",
            (tid,),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["run_id"] == run_id  # billed to the closing run
        assert json.loads(rows[0]["payload"]) == {"served_model": "glm-5.3-flash"}

    def test_no_route_event_when_off_board_argument_absent(self, conn):
        """A served_model in metadata without the caller's off_board=True
        stays the plain on-board path — no route event (the on-board gate is
        byte-for-byte unchanged)."""
        tid = _claimed_task(conn)
        assert kb.complete_task(
            conn, tid, result="r", metadata={"served_model": "glm-5.3-flash"},
        ) is True
        kinds = set(_event_kinds(conn, tid))
        assert "route_served_model" not in kinds

    def test_off_board_refusal_precedes_empty_evidence(self, conn):
        tid = _claimed_task(conn)
        with pytest.raises(kb.OffBoardServedModelError):
            kb.complete_task(conn, tid, off_board=True)  # no result/summary
        kinds = _event_kinds(conn, tid)
        assert "completion_blocked_served_model" in kinds
        assert "completion_blocked_empty_result" not in kinds
        assert "completed" not in kinds


def kbc_connect():
    from hermes_cli import kanban_db_connect as kbc

    return kbc.connect()


class TestP4P2OnBoardUnchanged:
    """The on-board path must stay byte-for-byte: no new event kinds."""

    def test_on_board_plain_completion_has_no_new_events(self, conn):
        tid = _claimed_task(conn)
        run_id = kb.get_task(conn, tid).current_run_id
        assert kb.complete_task(
            conn, tid, result="r", summary="s", expected_run_id=run_id,
        ) is True
        kinds = set(_event_kinds(conn, tid))
        # EXACTLY the pre-P2b baseline triple (created/claimed are
        # pre-existing); no new kind may appear on the on-board path.
        assert kinds == {"created", "claimed", "completed"}, kinds
        assert "route_served_model" not in kinds
        assert "completion_blocked_served_model" not in kinds
        assert "completion_blocked_human_gate" not in kinds

    def test_on_board_explicit_false_also_untouched(self, conn):
        tid = _claimed_task(conn)
        # served_model in metadata must NOT drip a route event on-board.
        assert kb.complete_task(
            conn, tid, result="r", metadata={"served_model": "glm-5.3-flash"},
            off_board=False,
        ) is True
        kinds = set(_event_kinds(conn, tid))
        assert "route_served_model" not in kinds
        assert kinds == {"created", "claimed", "completed"}, kinds

    def test_off_board_unclaimed_task_route_event_rides_synth_run(self, conn):
        """Never claimed: completion SYNTHESIZES a closed run (handoff
        survival, result present) — the route event is billed to THAT run
        (run_id NOT NULL), and the run's metadata carries served_model."""
        tid = kb.create_task(conn, title="offboard unclaimed", assignee="coder")
        ok = kb.complete_task(
            conn, tid, result="r", metadata={"served_model": "qwen-max"},
            off_board=True, require_recorded_origin=False,
        )
        assert ok is True
        run_row = conn.execute(
            "SELECT id, metadata, ended_at FROM task_runs WHERE task_id = ? "
            "ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()
        assert run_row is not None and run_row["ended_at"] is not None
        assert json.loads(run_row["metadata"]).get("served_model") == "qwen-max"
        ev = conn.execute(
            "SELECT run_id, payload FROM task_events "
            "WHERE task_id = ? AND kind = 'route_served_model'", (tid,),
        ).fetchall()
        assert len(ev) == 1
        assert ev[0]["run_id"] == run_row["id"]
        assert json.loads(ev[0]["payload"]) == {"served_model": "qwen-max"}

    def test_off_board_force_does_not_bypass(self, conn):
        tid = _claimed_task(conn)
        with pytest.raises(kb.OffBoardServedModelError):
            kb.complete_task(conn, tid, result="r", off_board=True, force=True)
        assert _status(conn, tid) == "running"
        assert "completed" not in _event_kinds(conn, tid)

    def test_metadata_unchanged_by_gate(self, conn):
        """The gate READS metadata['served_model']; it must not mutate or
        drop it on either exit."""
        tid = _claimed_task(conn)
        ok = kb.complete_task(
            conn, tid, result="r", metadata={"served_model": "glm-5.3-flash"},
            off_board=True, require_recorded_origin=False,
        )
        assert ok is True
        row = conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ("
            " SELECT current_run_id FROM tasks WHERE id = ?)",
            (tid,),
        ).fetchone()
        # current_run_id was cleared by completion; fall back to newest run.
        if row is None or row[0] is None:
            row = conn.execute(
                "SELECT metadata FROM task_runs WHERE task_id = ? "
                "ORDER BY id DESC LIMIT 1", (tid,),
            ).fetchone()
        assert json.loads(row[0]).get("served_model") == "glm-5.3-flash"


class TestP2bP4P2Ordering:
    """Both new gates precede the evidence gate (specific signal first)."""

    def test_human_gate_shadows_empty_evidence(self, conn):
        tid = _claimed_task(conn)
        _add_comment(conn, tid, "autopilot", f"{PENDING} g1")
        # No result/summary at all: the gate refuse must win over
        # EmptyCompletionError.
        with pytest.raises(kb.HumanGatePendingError):
            kb.complete_task(conn, tid)
        kinds = _event_kinds(conn, tid)
        assert "completion_blocked_human_gate" in kinds
        assert "completion_blocked_empty_result" not in kinds
        assert "completed" not in kinds

    def test_served_model_shadows_empty_evidence(self, conn):
        tid = _claimed_task(conn)
        with pytest.raises(kb.OffBoardServedModelError):
            kb.complete_task(conn, tid, off_board=True)
        kinds = _event_kinds(conn, tid)
        assert "completion_blocked_served_model" in kinds
        assert "completion_blocked_empty_result" not in kinds
        assert "completed" not in kinds
