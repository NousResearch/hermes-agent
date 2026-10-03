"""Acceptance tests for the kanban dispatcher dead-letter visibility lane.

Red-team suite (TDD red): every expectation below is derived ONLY from the
design contract (`## 契约规约` / `## 验收场景` in the session state.md).
No implementation source was read; these tests are expected to fail until
the lane lands.

Coverage map (design predicate -> test):
- P1  marking invariant            -> test_p1_aged_gave_up_and_blocked_cards_marked_with_contract_payload
- P2  idempotency + visibility-only-> test_p2_second_tick_is_idempotent_and_visibility_only
- P3  disabled lane                -> test_p3_zero_threshold_disables_the_lane
- threshold boundary (>= marks)    -> test_threshold_boundary_just_under_not_marked_and_at_threshold_marked
- anchor by id, anti-backfill      -> test_backfilled_boundary_event_anchors_by_event_id_not_timestamp
- idempotency by id, anti-backfill -> (inside the P2 test)
- anchor fallback tasks.created_at -> test_no_boundary_event_anchors_task_created_at_...
- reason priority + truncation     -> test_reason_is_first_line_... / ...falls_back_to_last_failure_error / ...none_reason
- revived non-sticky blocked card  -> test_raw_blocked_card_revived_by_promotion_is_not_dead_lettered
- configured_* contract            -> test_configured_dead_letter_after_hours_contract_table
- dispatch_once(None) plumbing     -> test_dispatch_once_none_threshold_consults_configured_...
- CLI surface (json + text)        -> test_cli_dispatch_surfaces_dead_lettered_...

Safety: this machine may export HERMES_KANBAN_DB pinned to the operator's
production board. The fixture strips every kanban path pin and asserts the
resolved board DB lives under the per-test temp HERMES_HOME; env overrides
here only ever point at mktemp/tmp dirs.
"""
from __future__ import annotations

import collections
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

HOUR = 3600


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Fresh HERMES_HOME with a clean kanban DB (paradigm of
    tests/hermes_cli/test_kanban_default_assignee.py), plus the red line for
    this card: strip every kanban path pin so nothing can resolve to the
    operator's real board."""
    test_home = tempfile.mkdtemp(prefix="kanban_dead_letter_acceptance_")
    monkeypatch.setenv("HERMES_HOME", test_home)
    for pin in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME",
                "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_LOGS_ROOT"):
        monkeypatch.delenv(pin, raising=False)
    # Force-reimport so the fresh HERMES_HOME is picked up.
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    # Safety gate: the board DB must resolve inside the temp home, never
    # at the production path.
    resolved_db = Path(kanban_db.kanban_db_path(board="default")).resolve()
    assert resolved_db.is_relative_to(Path(test_home).resolve()), (
        f"safety: board db {resolved_db} must live under temp HERMES_HOME {test_home}"
    )
    yield kanban_db, test_home


def _fake_spawn(*args, **kwargs):
    """Stand-in for the real worker spawn — no card in this suite is spawnable."""
    return 12345


# ---------------------------------------------------------------------------
# board / card builders (black-box data setup through public kanban_db APIs)
# ---------------------------------------------------------------------------


def _make_board(kanban_db):
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        kanban_db.create_board(slug="default", name="Test")


def _mk_gave_up_card(kb, conn, *, title, error, age_seconds, last_failure_error=None):
    """Terminal gave_up card: raw status flip + a backdated gave_up boundary
    event (gave_up is terminal, recompute_ready never touches it)."""
    tid = kb.create_task(conn, title=title)
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'gave_up', ?, ?)",
            (tid, json.dumps({"error": error}) if error is not None else None,
             now - age_seconds),
        )
        if last_failure_error is not None:
            conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (last_failure_error, tid))
        conn.execute("UPDATE tasks SET status='gave_up' WHERE id=?", (tid,))
    return tid


def _mk_eventless_gave_up_card(kb, conn, *, title, age_seconds, last_failure_error=None):
    """gave_up status with NO boundary event at all -> anchor must fall back
    to tasks.created_at for aging."""
    tid = kb.create_task(conn, title=title)
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='gave_up', created_at=? WHERE id=?",
            (now - age_seconds, tid),
        )
        if last_failure_error is not None:
            conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (last_failure_error, tid))
    return tid


def _mk_sticky_blocked_card(kb, conn, *, title, reason, kind="needs_input", age_seconds):
    """Sticky worker-initiated blocked card (must go through block_task — a
    raw SQL blocked flip would be revived by recompute_ready), then the
    blocked boundary event is backdated to set the card's age."""
    tid = kb.create_task(conn, title=title)
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    assert kb.block_task(conn, tid, reason=reason, kind=kind) is True
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_events SET created_at=? WHERE task_id=? AND kind='blocked'",
            (int(time.time()) - age_seconds, tid),
        )
    return tid


# ---------------------------------------------------------------------------
# observation helpers (own connections, black-box SQL on the temp board)
# ---------------------------------------------------------------------------


def _tick(**overrides):
    """One real dispatch_once tick against the temp board."""
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    kwargs = dict(spawn_fn=_fake_spawn, dry_run=False)
    kwargs.update(overrides)
    with kbc.connect_closing() as conn:
        return kbd.dispatch_once(conn, **kwargs)


def _dead_letter_rows(tid):
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        return list(conn.execute(
            "SELECT id, payload, run_id FROM task_events "
            "WHERE task_id=? AND kind='dead_letter' ORDER BY id",
            (tid,),
        ))


def _event_kind_counts(tid):
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        return collections.Counter(
            row[0] for row in conn.execute(
                "SELECT kind FROM task_events WHERE task_id=?", (tid,))
        )


def _status(tid):
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        return conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()[0]


def _payload(row):
    return json.loads(row["payload"])


# ---------------------------------------------------------------------------
# P1 — marking invariant
# ---------------------------------------------------------------------------


def test_p1_aged_gave_up_and_blocked_cards_marked_with_contract_payload(isolated_kanban_home):
    """P1: aged gave_up (multi-line error) + aged sticky blocked card, one
    real dispatch_once(dead_letter_after_hours=1) tick -> exactly one
    dead_letter event per card with the exact contract payload, and the
    tick roster carries both ids."""
    kb, _home = isolated_kanban_home
    _make_board(kb)
    from hermes_cli import kanban_db_connect as kbc

    giveup_error = "UPS unreachable: connection timed out\nretrying did not help\nsee run log"
    with kbc.connect_closing() as conn:
        t_giveup = _mk_gave_up_card(
            kb, conn, title="aged give-up", error=giveup_error, age_seconds=2 * HOUR)
        t_block = _mk_sticky_blocked_card(
            kb, conn, title="aged blocked", reason="waiting on human review",
            kind="needs_input", age_seconds=2 * HOUR)
    before = {tid: _event_kind_counts(tid) for tid in (t_giveup, t_block)}

    res = _tick(dead_letter_after_hours=1)

    # roster: set equality — the sweep SQL has no ORDER BY, never pin order.
    assert set(res.dead_lettered) == {t_giveup, t_block}

    for tid, want_status, want_block_kind, want_reason in (
        (t_giveup, "gave_up", None, "UPS unreachable: connection timed out"),
        (t_block, "blocked", "needs_input", "waiting on human review"),
    ):
        rows = _dead_letter_rows(tid)
        assert len(rows) == 1, f"exactly one dead_letter event for {tid}"
        row = rows[0]
        assert row["run_id"] is None, "contract: dead_letter events carry run_id=NULL"
        payload = _payload(row)
        # contract: payload JSON key set is exactly these four keys.
        assert set(payload.keys()) == {"age_hours", "status", "block_kind", "reason"}
        assert payload["status"] == want_status
        assert payload["block_kind"] == want_block_kind
        assert payload["reason"] == want_reason, "reason = first line of the cause, verbatim"
        assert isinstance(payload["age_hours"], float)
        assert 2.0 <= payload["age_hours"] < 3.0
        # visibility-only: the ONLY new event kind on the card is dead_letter.
        assert _event_kind_counts(tid) == before[tid] + collections.Counter({"dead_letter": 1})
        # no state change, no requeue.
        assert _status(tid) == want_status


# ---------------------------------------------------------------------------
# P2 — idempotency + visibility-only across a second tick
# ---------------------------------------------------------------------------


def test_p2_second_tick_is_idempotent_and_visibility_only(isolated_kanban_home):
    """P2: same board, second dispatch_once(dead_letter_after_hours=1) ->
    empty roster, dead_letter count still exactly 1 per card, statuses
    untouched. The dead_letter events are backdated BEFORE their anchors'
    timestamps to pin the contract 'idempotency compares autoincrement ids,
    not timestamps (backfill-proof)': a timestamp-comparing implementation
    would re-emit here."""
    kb, _home = isolated_kanban_home
    _make_board(kb)
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        t_giveup = _mk_gave_up_card(
            kb, conn, title="give-up", error="boom\nline2", age_seconds=2 * HOUR)
        t_block = _mk_sticky_blocked_card(
            kb, conn, title="blocked", reason="needs a human", age_seconds=2 * HOUR)

    res1 = _tick(dead_letter_after_hours=1)
    assert set(res1.dead_lettered) == {t_giveup, t_block}
    counts_after_first = {tid: len(_dead_letter_rows(tid)) for tid in (t_giveup, t_block)}
    assert counts_after_first == {t_giveup: 1, t_block: 1}

    # Anti-backfill: make the emitted events' timestamps older than the
    # anchors' (ids stay higher). Only an id-based idempotency check survives.
    from hermes_cli import kanban_db as kdb
    with kbc.connect_closing() as conn:
        with kdb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET created_at=? WHERE kind='dead_letter'",
                (int(time.time()) - 4 * HOUR,),
            )

    res2 = _tick(dead_letter_after_hours=1)
    assert res2.dead_lettered == [], "second tick must mark nothing (idempotent)"
    for tid, want_status in ((t_giveup, "gave_up"), (t_block, "blocked")):
        assert len(_dead_letter_rows(tid)) == 1, "idempotency: event count is exactly 1"
        assert _status(tid) == want_status, "visibility-only: terminal state unchanged"
    assert _status(t_giveup) != "ready" and _status(t_block) != "ready"


# ---------------------------------------------------------------------------
# P3 — disabled lane
# ---------------------------------------------------------------------------


def test_p3_zero_threshold_disables_the_lane(isolated_kanban_home):
    """P3: dead_letter_after_hours=0 -> no dead_letter events at all and an
    empty roster; dispatch otherwise behaves like the base tick."""
    kb, _home = isolated_kanban_home
    _make_board(kb)
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        tid = _mk_gave_up_card(
            kb, conn, title="aged give-up", error="boom", age_seconds=2 * HOUR)

    res = _tick(dead_letter_after_hours=0)

    assert res.dead_lettered == []
    assert _dead_letter_rows(tid) == []
    assert _status(tid) == "gave_up"


# ---------------------------------------------------------------------------
# threshold boundary — mark iff (now - anchor_time) >= after_hours*3600
# ---------------------------------------------------------------------------


def test_threshold_boundary_just_under_not_marked_and_at_threshold_marked(isolated_kanban_home):
    """A card aged just under the threshold stays unmarked; a card aged at
    the threshold is marked (predicate: skip on `<`, mark on `>=`)."""
    kb, _home = isolated_kanban_home
    _make_board(kb)
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        just_under = _mk_gave_up_card(
            kb, conn, title="just under", error="young-ish", age_seconds=HOUR - 120)
        at_threshold = _mk_gave_up_card(
            kb, conn, title="at threshold", error="old enough", age_seconds=HOUR)

    res = _tick(dead_letter_after_hours=1)

    assert set(res.dead_lettered) == {at_threshold}
    assert _dead_letter_rows(just_under) == []
    assert len(_dead_letter_rows(at_threshold)) == 1
    assert _status(just_under) == "gave_up"
    assert _status(at_threshold) == "gave_up"


# ---------------------------------------------------------------------------
# anchor = latest boundary event ORDER BY id DESC — backfill-proof
# ---------------------------------------------------------------------------


def test_backfilled_boundary_event_anchors_by_event_id_not_timestamp(isolated_kanban_home):
    """A backfilled boundary event (highest id, old timestamp) must become
    the anchor: the card ages from ITS timestamp and gets marked. An
    implementation anchoring by created_at DESC would pick the recent
    gave_up event and skip the card."""
    kb, _home = isolated_kanban_home
    _make_board(kb)
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="backfilled boundary")
        now = int(time.time())
        with kb.write_txn(conn):
            # recent gave_up boundary, inserted FIRST (lower id)
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'gave_up', ?, ?)",
                (tid, json.dumps({"error": "recent-but-ignored"}), now - 600),
            )
            # backfilled status boundary, inserted SECOND (highest id), old ts
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'status', ?, ?)",
                (tid, json.dumps({"status": "gave_up"}), now - 2 * HOUR),
            )
            conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?",
                         ("status lane says why", tid))
            conn.execute("UPDATE tasks SET status='gave_up' WHERE id=?", (tid,))

    res = _tick(dead_letter_after_hours=1)

    assert set(res.dead_lettered) == {tid}, "anchor must be the highest-id boundary event"
    rows = _dead_letter_rows(tid)
    assert len(rows) == 1
    payload = _payload(rows[0])
    # anchor kind is 'status' -> reason falls back to tasks.last_failure_error
    assert payload["reason"] == "status lane says why"
    assert payload["status"] == "gave_up"


# ---------------------------------------------------------------------------
# anchor fallback + reason fallback
# ---------------------------------------------------------------------------


def test_no_boundary_event_anchors_task_created_at_and_reason_falls_back_to_last_failure_error(isolated_kanban_home):
    """No boundary event at all -> anchor_id=0 and anchor_time =
    tasks.created_at; with a gave_up anchor absent the reason falls back to
    tasks.last_failure_error."""
    kb, _home = isolated_kanban_home
    _make_board(kb)
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        tid = _mk_eventless_gave_up_card(
            kb, conn, title="eventless give-up", age_seconds=2 * HOUR,
            last_failure_error="fallback: last failure says it all")

    res = _tick(dead_letter_after_hours=1)

    assert set(res.dead_lettered) == {tid}
    rows = _dead_letter_rows(tid)
    assert len(rows) == 1
    payload = _payload(rows[0])
    assert set(payload.keys()) == {"age_hours", "status", "block_kind", "reason"}
    assert payload["reason"] == "fallback: last failure says it all"
    assert payload["status"] == "gave_up"
    assert payload["block_kind"] is None
    assert 2.0 <= payload["age_hours"] < 3.0


# ---------------------------------------------------------------------------
# reason derivation
# ---------------------------------------------------------------------------


def test_reason_is_first_line_of_cause_truncated_to_200_chars(isolated_kanban_home):
    """reason = first line of the cause, truncated to 200 characters; later
    lines must not leak."""
    kb, _home = isolated_kanban_home
    _make_board(kb)
    from hermes_cli import kanban_db_connect as kbc
    long_first_line = "A" * 260
    with kbc.connect_closing() as conn:
        tid = _mk_gave_up_card(
            kb, conn, title="long error",
            error=long_first_line + "\nsecond line must not leak", age_seconds=2 * HOUR)

    res = _tick(dead_letter_after_hours=1)

    assert set(res.dead_lettered) == {tid}
    payload = _payload(_dead_letter_rows(tid)[0])
    assert payload["reason"] == "A" * 200
    assert len(payload["reason"]) == 200
    assert "second line" not in payload["reason"]


def test_all_empty_causes_yield_none_reason(isolated_kanban_home):
    """blocked anchor with no reason in the payload and nothing in
    tasks.last_failure_error -> payload reason is None (never a string like
    'None' or '')."""
    kb, _home = isolated_kanban_home
    _make_board(kb)
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        tid = _mk_sticky_blocked_card(
            kb, conn, title="silent block", reason=None, age_seconds=2 * HOUR)

    res = _tick(dead_letter_after_hours=1)

    assert set(res.dead_lettered) == {tid}
    payload = _payload(_dead_letter_rows(tid)[0])
    assert payload["reason"] is None


# ---------------------------------------------------------------------------
# anchor resets / revived cards are not dead
# ---------------------------------------------------------------------------


def test_latest_boundary_unblocked_event_resets_the_dead_clock(isolated_kanban_home):
    """The anchor is the LATEST boundary event: an old blocked event followed
    by a recent unblocked boundary means the segment is young -> no marking."""
    kb, _home = isolated_kanban_home
    _make_board(kb)
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="re-aged card")
        now = int(time.time())
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'blocked', ?, ?)",
                (tid, json.dumps({"reason": "old block"}), now - 2 * HOUR),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'unblocked', ?, ?)",
                (tid, json.dumps({"actor": "operator"}), now - 600),
            )
            conn.execute("UPDATE tasks SET status='gave_up' WHERE id=?", (tid,))

    res = _tick(dead_letter_after_hours=1)

    assert res.dead_lettered == [], "recent unblocked boundary resets the age"
    assert _dead_letter_rows(tid) == []


def test_raw_blocked_card_revived_by_promotion_is_not_dead_lettered(isolated_kanban_home):
    """Design decision: a non-sticky blocked card that promotion revives to
    ready is not a dead card — only cards still gave_up/blocked when the
    sweep runs may be marked."""
    kb, _home = isolated_kanban_home
    _make_board(kb)
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="raw blocked, will be revived")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='blocked', created_at=? WHERE id=?",
                (int(time.time()) - 4 * HOUR, tid),
            )

    res = _tick(dead_letter_after_hours=1)

    assert tid not in res.dead_lettered
    assert _dead_letter_rows(tid) == []
    assert _status(tid) == "ready", "promotion must revive the non-sticky blocked card"


# ---------------------------------------------------------------------------
# configured_dead_letter_after_hours contract (config-key parsing)
# ---------------------------------------------------------------------------


def test_configured_dead_letter_after_hours_contract_table(monkeypatch):
    """Contract: >=1 int (numeric strings included) -> value; 0 / negative /
    unparseable / missing key / config read failure -> 0. Never raises."""
    import hermes_cli.config as cfgmod
    from hermes_cli import kanban_db_dispatch as kbd

    cases = [
        ({"kanban": {"dead_letter_after_hours": 2}}, 2),
        ({"kanban": {"dead_letter_after_hours": "3"}}, 3),
        ({"kanban": {"dead_letter_after_hours": 0}}, 0),
        ({"kanban": {"dead_letter_after_hours": "0"}}, 0),
        ({"kanban": {"dead_letter_after_hours": -5}}, 0),
        ({"kanban": {"dead_letter_after_hours": "soon"}}, 0),
        ({"kanban": {}}, 0),
        ({}, 0),
    ]
    for config, expected in cases:
        monkeypatch.setattr(cfgmod, "load_config_readonly", lambda c=config: c)
        assert kbd.configured_dead_letter_after_hours() == expected, config

    def _boom():
        raise RuntimeError("config backend down")

    monkeypatch.setattr(cfgmod, "load_config_readonly", _boom)
    assert kbd.configured_dead_letter_after_hours() == 0, "read failure degrades to disabled"


def test_dispatch_once_none_threshold_consults_configured_and_explicit_value_wins(isolated_kanban_home, monkeypatch):
    """dispatch_once without an explicit threshold consults
    configured_dead_letter_after_hours per call; an explicit value takes
    priority over the configured one."""
    kb, _home = isolated_kanban_home
    _make_board(kb)
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        tid = _mk_gave_up_card(kb, conn, title="aged give-up", error="boom", age_seconds=2 * HOUR)

    from hermes_cli import kanban_db_dispatch as kbd
    consults = []
    monkeypatch.setattr(kbd, "configured_dead_letter_after_hours", lambda: consults.append(1) or 1)

    res = _tick()  # no dead_letter_after_hours kwarg -> must read the config lane
    assert set(res.dead_lettered) == {tid}
    assert consults, "None threshold must consult configured_dead_letter_after_hours"
    assert len(_dead_letter_rows(tid)) == 1

    res2 = _tick(dead_letter_after_hours=0)  # explicit 0 beats configured 1
    assert res2.dead_lettered == []
    assert len(_dead_letter_rows(tid)) == 1


# ---------------------------------------------------------------------------
# CLI surface — real dispatch path, dry_run computes but writes nothing
# ---------------------------------------------------------------------------


def test_cli_dispatch_surfaces_dead_lettered_json_and_text_dry_run_writes_nothing(isolated_kanban_home, monkeypatch, capsys):
    """`hermes kanban dispatch` --json gains the `dead_lettered` key and the
    text output gains the `Dead-lettered:` line; with dry_run the roster is
    computed but zero dead_letter events are written."""
    kb, _home = isolated_kanban_home
    _make_board(kb)
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        tid = _mk_gave_up_card(kb, conn, title="cli give-up", error="boom", age_seconds=2 * HOUR)

    lane_on = {"kanban": {"dead_letter_after_hours": 1}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: lane_on)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: lane_on)

    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_ops

    args = SimpleNamespace(dry_run=True, max=None,
                           failure_limit=kbd.DEFAULT_FAILURE_LIMIT, json=True)
    assert kanban_ops._cmd_dispatch(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["dead_lettered"]) == {tid}

    args_text = SimpleNamespace(dry_run=True, max=None,
                                failure_limit=kbd.DEFAULT_FAILURE_LIMIT, json=False)
    assert kanban_ops._cmd_dispatch(args_text) == 0
    out = capsys.readouterr().out
    assert "Dead-lettered:" in out
    assert tid in out

    # dry_run visibility-only: computing the roster must not write events.
    assert _dead_letter_rows(tid) == []
    assert _status(tid) == "gave_up"
