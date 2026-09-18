"""The two dispatcher dead ends that leave a graph unable to move by itself.

1. **The breaker abandons a child.** ``_record_task_failure`` parks it in
   ``blocked`` with ``gave_up``. It is terminal but not successful, so it
   satisfies nothing: ``recompute_ready`` promotes a waiter only when every
   parent is ``done``/``archived``, ``promote_task`` refuses by design, and the
   orchestrator root sits in ``todo`` -- a lane no dispatcher reads.
2. **A card reaches ``review`` owned by its own implementer.**
   ``request_review`` leaves the assignee untouched when no reviewer is named,
   and the review lane spawns ``row["assignee"]`` with the ``sdlc-review``
   skill, i.e. the implementer certifying its own work.

Both are fixed by handing the orchestrator one durable, schedulable card while
the unsuccessful work stays unsuccessful. These tests pin the lifecycle facts
that must NOT change (a failed child never satisfies a success dependency, a
card in review is never spawned to its implementer) and the reconciliation
card that makes each actionable.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile

import pytest


FAILURE_LIMIT = 2
PROFILES = ("foreman", "default", "gauge")


def _home(orchestrator: str | None) -> str:
    home = tempfile.mkdtemp(prefix="kanban_reconcile_test_")
    for profile in PROFILES:
        profile_dir = os.path.join(home, "profiles", profile)
        os.makedirs(profile_dir, exist_ok=True)
        # `profile_exists` resolves through `named_profile_is_live`, which requires an identity
        # marker: a bare dir is a ghost shell and never spawnable, so a reconciliation card
        # assigned to a marker-less `foreman` would not read as dispatchable work.
        with open(os.path.join(profile_dir, "config.yaml"), "w", encoding="utf-8") as fh:
            fh.write("{}\n")
    body = "kanban:\n  review_dispatch: true\n"
    if orchestrator:
        body += f"  orchestrator_profile: {orchestrator}\n"
    with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as fh:
        fh.write(body)
    return home


@contextlib.contextmanager
def _fresh_modules():
    """Re-import hermes_cli against this test's HERMES_HOME, then put the
    original module objects back. Leaving them evicted hands every later test
    file in the process a second copy of hermes_cli.* while its collection-time
    bindings still point at the first."""
    def _owned():
        return [
            n for n in list(sys.modules)
            if n.startswith(("hermes_cli", "hermes_state")) or n == "hermes_constants"
        ]

    evicted = {name: sys.modules[name] for name in _owned()}
    for name in evicted:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in _owned():
            del sys.modules[name]
        sys.modules.update(evicted)


@pytest.fixture()
def board(monkeypatch):
    """Fresh HERMES_HOME whose board names ``foreman`` as the orchestrator."""
    monkeypatch.setenv("HERMES_HOME", _home("foreman"))
    with _fresh_modules():
        from hermes_cli import kanban_db
        kanban_db.init_db()
        yield kanban_db


@pytest.fixture()
def board_without_orchestrator(monkeypatch):
    monkeypatch.setenv("HERMES_HOME", _home(None))
    with _fresh_modules():
        from hermes_cli import kanban_db
        kanban_db.init_db()
        yield kanban_db


def _fake_spawn(*args, **kwargs):
    return 12345


def _fanout(kb, conn):
    """Child first, then the root linked under it -- how ``decompose_triage_task``
    wires a fan-out, and the only shape that yields a ``todo`` root."""
    child = kb.create_task(conn, title="implement the exporter", assignee="default")
    root = kb.create_task(conn, title="ship the exporter", assignee="foreman", parents=[child])
    assert kb.get_task(conn, root).status == "todo"
    return root, child


def _fail_once(kb, kbd, conn, task_id, *, limit=FAILURE_LIMIT):
    kb.claim_task(conn, task_id)
    return kbd._record_task_failure(
        conn, task_id, error="ImportError: no module named 'pyarrow'",
        outcome="crashed", failure_limit=limit, release_claim=True, end_run=True,
    )


def _exhaust(kb, kbd, conn, task_id, *, limit=FAILURE_LIMIT):
    for _ in range(limit):
        tripped = _fail_once(kb, kbd, conn, task_id, limit=limit)
    assert tripped, "the breaker must trip on the last permitted attempt"


def _reconcile_cards(kb, conn):
    from hermes_cli.kanban_reconcile import KEY_PREFIX
    return [
        row["id"] for row in conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key LIKE ? ORDER BY created_at, id",
            (KEY_PREFIX + ":%",),
        ).fetchall()
    ]


# ---------------------------------------------------------------------------
# The breaker abandons a child
# ---------------------------------------------------------------------------


def test_successful_child_leaves_no_reconciliation_card(board):
    """Happy path is untouched: the root promotes itself, nothing is raised."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        root, child = _fanout(kb, conn)
        kb.claim_task(conn, child)
        assert kb.complete_task(conn, child, result="shipped")
        assert kb.get_task(conn, root).status == "ready"
        assert _reconcile_cards(kb, conn) == []


def test_failure_below_the_limit_leaves_no_reconciliation_card(board):
    """Negative control: a retry still in budget is not a dead end."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        root, child = _fanout(kb, conn)
        assert _fail_once(kb, kbd, conn, child) is False
        assert kb.get_task(conn, child).status == "ready"
        assert _reconcile_cards(kb, conn) == []


def test_breaker_trip_hands_the_orchestrator_a_schedulable_card(board):
    """The fix: the abandoned child stays unsuccessful, and the orchestrator
    gets one card the dispatcher will actually spawn."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        root, child = _fanout(kb, conn)
        _exhaust(kb, kbd, conn, child)

        # Nothing about success semantics moved.
        assert kb.get_task(conn, child).status == "blocked"
        assert kb.get_task(conn, root).status == "todo"
        assert kb.recompute_ready(conn, failure_limit=FAILURE_LIMIT) == 0
        ok, err = kb.promote_task(conn, root, actor="operator")
        assert ok is False and child in (err or "")

        cards = _reconcile_cards(kb, conn)
        assert len(cards) == 1
        card = kb.get_task(conn, cards[0])
        assert card.assignee == "foreman"
        assert card.status == "ready"
        # Self-contained: the worker must be able to act without this session.
        assert child in card.body and root in card.body
        assert "pyarrow" in card.body
        assert kb.parent_ids(conn, card.id) == []

        # Externally observable: it is now in a dispatch lane.
        assert kbd.has_spawnable_ready(conn) is True
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True, max_in_progress=5)
        assert (card.id, "foreman") in [(t, a) for t, a, _ in res.spawned]


def test_the_failed_child_is_still_not_a_satisfied_dependency(board):
    """I6: reconciliation must never be bought by marking failure as success."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        root, child = _fanout(kb, conn)
        _exhaust(kb, kbd, conn, child)
        assert len(_reconcile_cards(kb, conn)) == 1
        c = kb.get_task(conn, child)
        assert c.status == "blocked" and c.status != "done"
        assert kb._parents_satisfied(conn, root) is False
        assert kb.claim_task(conn, root) is None


def test_reconciliation_card_is_created_once_across_repeated_ticks(board):
    """I8: repeated reconciliation over unchanged state converges."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        root, child = _fanout(kb, conn)
        _exhaust(kb, kbd, conn, child)
        first = _reconcile_cards(kb, conn)
        assert len(first) == 1
        for _ in range(5):
            kb.recompute_ready(conn, failure_limit=FAILURE_LIMIT)
            kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True, max_in_progress=5)
            assert _reconcile_cards(kb, conn) == first
        events = [e.kind for e in kb.list_events(conn, child)]
        assert events.count("reconcile_requested") == 1


def test_repeated_unblock_and_re_exhaust_files_exactly_one_open_card(board):
    """I7. Every unblock resets ``consecutive_failures``, so a cron or operator
    unblock loop re-trips the breaker again and again. While the orchestrator's
    card is still open, that is the SAME unresolved situation and must not mint
    a card per cycle."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        root, child = _fanout(kb, conn)
        for _ in range(6):
            _exhaust(kb, kbd, conn, child)
            kb.recompute_ready(conn, failure_limit=FAILURE_LIMIT)
            assert kb.unblock_task(conn, child)
        cards = _reconcile_cards(kb, conn)
        assert len(cards) == 1, f"one open opportunity, got {len(cards)}"
        assert kb.get_task(conn, cards[0]).assignee == "foreman"


def test_resolving_the_card_without_acting_re_files_it_then_stops(board):
    """The opportunity must not be silently consumable: completing the card
    without fixing anything leaves the root just as stranded, so the board asks
    again -- but a finite number of times."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli.kanban_reconcile import MAX_CARDS_PER_TASK
    with kbc.connect_closing() as conn:
        root, child = _fanout(kb, conn)
        _exhaust(kb, kbd, conn, child)
        for _ in range(MAX_CARDS_PER_TASK + 3):
            open_cards = [
                cid for cid in _reconcile_cards(kb, conn)
                if kb.get_task(conn, cid).status not in ("done", "archived")
            ]
            if not open_cards:
                # Nothing open: another exhaustion of the same still-failing
                # child is what a live board would produce next.
                assert kb.unblock_task(conn, child)
                _exhaust(kb, kbd, conn, child)
                continue
            assert len(open_cards) == 1
            kb.claim_task(conn, open_cards[0])
            assert kb.complete_task(conn, open_cards[0], result="closed without acting")
        filed = _reconcile_cards(kb, conn)
        assert len(filed) == MAX_CARDS_PER_TASK, (
            f"bounded at {MAX_CARDS_PER_TASK}, got {len(filed)}"
        )
        # The root is still stranded, and the diagnostic still says so.
        assert kb.get_task(conn, root).status == "todo"


def test_a_case_mismatched_orchestrator_profile_still_finds_the_root(board, monkeypatch):
    """``create_task`` canonicalizes assignees, but config is free text. A board
    configured as ``Foreman`` must not silently strand every root."""
    kb = board
    import hermes_cli.config as cfgmod
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"orchestrator_profile": "Foreman", "failure_limit": 2}},
    )
    with kbc.connect_closing() as conn:
        root, child = _fanout(kb, conn)
        assert kb.get_task(conn, root).assignee == "foreman"
        _exhaust(kb, kbd, conn, child)
        cards = _reconcile_cards(kb, conn)
        assert len(cards) == 1, "a casing difference must not strand the root"
        assert kb.get_task(conn, cards[0]).assignee == "foreman"


def test_a_reconcile_failure_does_not_take_the_dispatcher_tick_down(board, monkeypatch):
    """Reconciliation is a courtesy on top of the breaker. If it breaks, the
    breaker trip and the rest of the tick must still land."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_reconcile

    def _boom(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(kanban_reconcile, "waiting_orchestrator_roots", _boom)
    with kbc.connect_closing() as conn:
        root, child = _fanout(kb, conn)
        _exhaust(kb, kbd, conn, child)
        assert kb.get_task(conn, child).status == "blocked"
        assert _reconcile_cards(kb, conn) == []
        # And a full tick over the same board still completes.
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True, max_in_progress=5)
        assert res.skipped_locked is False


def test_concurrent_dispatch_ticks_still_file_one_card(board):
    """I8 under concurrency. The check-then-create is not atomic by itself; it
    does not have to be, because every caller reaches it from inside
    ``dispatch_once``, which holds the board's single-writer tick lock. This
    pins that the real path converges -- if the call ever moves outside the
    lock, this is what fails."""
    import threading

    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        root, child = _fanout(kb, conn)
        _exhaust(kb, kbd, conn, child)
        assert len(_reconcile_cards(kb, conn)) == 1
        # A second generation of the same failure, filed by racing ticks.
        assert kb.unblock_task(conn, child)
        _exhaust(kb, kbd, conn, child)

    errors: list[str] = []

    def tick():
        try:
            with kbc.connect_closing() as own:
                kbd.dispatch_once(own, spawn_fn=_fake_spawn, dry_run=True, max_in_progress=5)
        except Exception as exc:  # pragma: no cover - a raise here is the failure
            errors.append(repr(exc))

    threads = [threading.Thread(target=tick) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    with kbc.connect_closing() as conn:
        cards = _reconcile_cards(kb, conn)
        assert len(cards) == len(set(cards))
        per_generation = {kb.get_task(conn, cid).idempotency_key for cid in cards}
        assert len(per_generation) == len(cards), "one card per failure generation"


def test_reconciliation_card_survives_reopening_the_board(board):
    """I9/I14: the durable state is the card, not anyone's context."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        root, child = _fanout(kb, conn)
        _exhaust(kb, kbd, conn, child)
        before = _reconcile_cards(kb, conn)
    with kbc.connect_closing() as conn:
        assert _reconcile_cards(kb, conn) == before
        card = kb.get_task(conn, before[0])
        assert card.status == "ready" and card.assignee == "foreman"
        assert kb.get_task(conn, child).last_failure_error


def test_without_an_orchestrator_nothing_is_created(board_without_orchestrator):
    """Negative control: no orchestrator profile means no owner to hand it to."""
    kb = board_without_orchestrator
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        root, child = _fanout(kb, conn)
        _exhaust(kb, kbd, conn, child)
        assert kb.get_task(conn, child).status == "blocked"
        assert _reconcile_cards(kb, conn) == []


def test_a_standalone_failure_creates_no_card(board):
    """Negative control: nothing is stranded, so nothing needs reconciling --
    the failed card raises ``repeated_failures`` on its own."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        solo = kb.create_task(conn, title="standalone", assignee="default")
        _exhaust(kb, kbd, conn, solo)
        assert kb.get_task(conn, solo).status == "blocked"
        assert _reconcile_cards(kb, conn) == []


def test_a_failing_reconciliation_card_does_not_chain(board):
    """I7: the loop guard. A reconciliation card that exhausts its own retries
    must not produce another one."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        root, child = _fanout(kb, conn)
        _exhaust(kb, kbd, conn, child)
        cards = _reconcile_cards(kb, conn)
        assert len(cards) == 1
        # Park a root behind the reconciliation card so the only thing stopping
        # a chain is the guard itself.
        waiter = kb.create_task(conn, title="waits on the reconcile card",
                                assignee="foreman", parents=[cards[0]])
        assert kb.get_task(conn, waiter).status == "todo"
        _exhaust(kb, kbd, conn, cards[0])
        assert kb.get_task(conn, cards[0]).status == "blocked"
        assert _reconcile_cards(kb, conn) == cards


# ---------------------------------------------------------------------------
# A card in review owned by its own implementer
# ---------------------------------------------------------------------------


def _request_review(kb, conn, *, assignee="default", reviewer=None):
    tid = kb.create_task(conn, title="add the retry guard", assignee=assignee)
    kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    ok = kb.request_review(conn, tid, summary="implemented and unit-tested",
                           reviewer=reviewer, expected_run_id=run_id)
    assert ok
    assert kb.get_task(conn, tid).status == "review"
    return tid


def test_review_without_a_named_reviewer_is_not_dispatched_to_its_implementer(board):
    """Known-red: the card stays owned by the implementer, and the review lane
    would spawn exactly that profile to certify its own work."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        tid = _request_review(kb, conn)
        # The lifecycle fact that makes this possible.
        assert kb.get_task(conn, tid).assignee == "default"

        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True, max_in_progress=5)
        assert (tid, "default") in res.skipped_self_review
        assert tid not in [t for t, _, _ in res.spawned]
        assert kb.get_task(conn, tid).status == "review"


def test_an_explicit_self_reviewer_is_refused_too(board):
    """Naming yourself is the same conflict, stated out loud."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        tid = _request_review(kb, conn, assignee="default", reviewer="default")
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True, max_in_progress=5)
        assert (tid, "default") in res.skipped_self_review
        assert tid not in [t for t, _, _ in res.spawned]


def test_missing_implementer_provenance_also_fails_closed(board):
    """A card in ``review`` whose implementer cannot be read is not shown to
    have an independent reviewer. "Cannot be shown" is not "is"."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli.kanban_reconcile import IMPLEMENTER_UNKNOWN
    with kbc.connect_closing() as conn:
        tid = _request_review(kb, conn, assignee="default", reviewer="gauge")
        # Drop the provenance the review handoff wrote (a truncated event log, a
        # hand-edited row, a card dragged into review by some other path).
        with kb.write_txn(conn):
            conn.execute(
                "DELETE FROM task_events WHERE task_id = ? AND kind = 'review_requested'",
                (tid,),
            )
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True, max_in_progress=5)
        assert (tid, IMPLEMENTER_UNKNOWN) in res.skipped_self_review
        assert tid not in [t for t, _, _ in res.spawned]

        # The card handed to the orchestrator must say what is actually wrong,
        # not name a profile called "implementer_unknown".
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn, max_in_progress=5)
        cards = _reconcile_cards(kb, conn)
        assert len(cards) == 1
        body = kb.get_task(conn, cards[0]).body
        assert "provenance is missing or malformed" in body
        assert IMPLEMENTER_UNKNOWN not in body


def test_an_independent_reviewer_still_dispatches(board):
    """Negative control: the review lane must keep working."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        tid = _request_review(kb, conn, assignee="default", reviewer="gauge")
        assert kb.get_task(conn, tid).assignee == "gauge"
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True, max_in_progress=5)
        assert res.skipped_self_review == []
        assert (tid, "gauge") in [(t, a) for t, a, _ in res.spawned]


def test_self_review_hands_the_orchestrator_a_routing_card(board):
    """Fail-closed must still be actionable: somebody has to pick a reviewer."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        tid = _request_review(kb, conn)
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn, max_in_progress=5)
        cards = _reconcile_cards(kb, conn)
        assert len(cards) == 1
        card = kb.get_task(conn, cards[0])
        assert card.assignee == "foreman" and card.status == "ready"
        assert tid in card.body and "default" in card.body
        # The reviewed card is untouched: it needs a reviewer, not a rerun.
        assert kb.get_task(conn, tid).status == "review"


def test_the_self_review_refusal_is_idempotent(board):
    """I8: ticking over an unresolved conflict converges."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        tid = _request_review(kb, conn)
        for _ in range(3):
            res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, max_in_progress=5)
            assert (tid, "default") in res.skipped_self_review
        assert len(_reconcile_cards(kb, conn)) == 1
        assert [e.kind for e in kb.list_events(conn, tid)].count("reconcile_requested") == 1


def test_a_dry_run_records_the_conflict_without_writing(board):
    """A dry run reports; it must not create cards."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        tid = _request_review(kb, conn)
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True, max_in_progress=5)
        assert (tid, "default") in res.skipped_self_review
        assert _reconcile_cards(kb, conn) == []


def test_an_unreviewable_card_does_not_read_as_pending_review_work(board):
    """Health probes use ``has_spawnable_review`` to tell a stuck board from a
    correctly idle one. A card whose only candidate reviewer is its own
    implementer is never claimed, so reporting it as pending review work would
    make the board read "stuck" forever -- the exact distinction that predicate
    exists to make."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        tid = _request_review(kb, conn)  # implementer is still the assignee
        assert kb.get_task(conn, tid).status == "review"
        assert kbd.has_spawnable_review(conn) is False

        # Routing it to someone who did not write it makes it real review work.
        assert kb.reassign_task(conn, tid, "gauge")
        assert kbd.has_spawnable_review(conn) is True


def test_an_unspawnable_review_row_does_not_starve_the_ready_lane(board):
    """Regression: the review reservation holds a slot back whenever spawnable
    review work exists. A card that can never be spawned must not hold it."""
    kb = board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        _request_review(kb, conn)
        work = kb.create_task(conn, title="unrelated ready work", assignee="gauge")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True, max_spawn=1, max_in_progress=5,
        )
        assert work in [t for t, _, _ in res.spawned]
