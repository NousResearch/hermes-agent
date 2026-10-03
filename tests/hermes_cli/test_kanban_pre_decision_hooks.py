"""Tests for the Kanban pre-decision hooks.

``pre_kanban_dispatch`` and ``pre_kanban_task_create`` are the Kanban analogue of
``pre_tool_call``: they fire BEFORE the worker spawn / the card insert and their return value is
HONOURED. Every other ``kanban_*`` / ``on_kanban_*`` hook fires after the commit with its return
discarded.

The contract these tests pin down:

* ``pre_kanban_dispatch`` runs at the single pre-spawn chokepoint both lanes share and a
  ``{"action": "hold", "reason"}`` refuses that candidate exactly as the built-in
  ``check_respawn_guard`` does: a ``respawn_guarded`` event, the card stays where it was, no spawn.
* ``pre_kanban_task_create`` runs in ``create_task`` before the write transaction opens and a
  ``{"action": "suppress", ...}`` skips the insert.
* Backward compatibility: with no subscriber the hooks are never invoked (short-circuit on
  ``has_hook``), a ``None`` return changes nothing, a reasonless directive is ignored, and a
  callback that raises cannot wedge the dispatcher.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def hooks():
    """Register stub callbacks for a hook name; the registry is restored on teardown."""
    mgr = get_plugin_manager()

    class _Registry:
        def __init__(self) -> None:
            self._saved = {k: list(v) for k, v in mgr._hooks.items()}

        def add(self, name: str, callback):
            mgr._hooks.setdefault(name, []).append(callback)
            return callback

    reg = _Registry()
    try:
        yield reg
    finally:
        mgr._hooks = reg._saved


def _events(conn, tid: str, kind: str) -> list[dict]:
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
    ).fetchall()
    return [json.loads(r["payload"]) if r["payload"] else None
            for r in rows if r["kind"] == kind]


def _count(conn) -> int:
    return conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()["c"]


# ---------------------------------------------------------------------------
# pre_kanban_dispatch
# ---------------------------------------------------------------------------


def test_dispatch_hold_refuses_the_spawn_and_records_respawn_guarded(
    kanban_home, all_assignees_spawnable, hooks,
):
    """A hold is recorded exactly as the built-in guard records one, and nothing spawns."""
    hooks.add("pre_kanban_dispatch",
              lambda **kw: {"action": "hold", "reason": "completion_pending"})
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="alice")
        spawned: list[str] = []
        result = kbd.dispatch_once(conn, spawn_fn=lambda task, ws, **k: spawned.append(task.id) or 4242)
        assert spawned == [], "a held candidate must not be spawned"
        assert result.spawned == []
        assert (tid, "completion_pending") in result.respawn_guarded
        assert kb.get_task(conn, tid).status == "ready", "ready stays ready"
        assert kb.get_task(conn, tid).claim_lock is None
        assert _events(conn, tid, "respawn_guarded") == [{"reason": "completion_pending"}]
    finally:
        conn.close()


def test_dispatch_hold_receives_the_candidate_payload(
    kanban_home, all_assignees_spawnable, hooks,
):
    """Kwargs are the documented read-only snapshot of the candidate the hook rules on."""
    seen: list[dict] = []

    def _capture(**kw):
        seen.append(kw)
        return None

    hooks.add("pre_kanban_dispatch", _capture)
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=lambda task, ws, **k: 4242)
    finally:
        conn.close()
    assert seen, "the hook must fire for the candidate"
    kw = seen[0]
    assert kw["task_id"] == tid
    assert kw["assignee"] == "alice"
    assert kw["lane"] == "ready"
    assert kw["board"] is None
    assert kw["task"] is not None and kw["task"].id == tid


def test_dispatch_hold_still_leaves_other_candidates_dispatchable(
    kanban_home, all_assignees_spawnable, hooks,
):
    """A hold is per-candidate: the next ready card in the same tick still dispatches."""
    conn = kbc.connect()
    try:
        held = kb.create_task(conn, title="held", assignee="alice", priority=10)
        other = kb.create_task(conn, title="other", assignee="bob")
        hooks.add("pre_kanban_dispatch", lambda task_id, **kw: (
            {"action": "hold", "reason": "duplicate"} if task_id == held else None))
        spawned: list[str] = []
        result = kbd.dispatch_once(
            conn, spawn_fn=lambda task, ws, **k: spawned.append(task.id) or 4242,
        )
        assert spawned == [other]
        assert [row[0] for row in result.respawn_guarded] == [held]
    finally:
        conn.close()


def test_dispatch_hold_leaves_a_review_candidate_in_review(
    kanban_home, all_assignees_spawnable, hooks,
):
    """The reviewer lane holds too — a held review candidate stays in ``review``."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="impl", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        assert kb.request_review(
            conn, tid, summary="done", reviewer="reviewer", expected_run_id=run_id,
        ) is True
        hooks.add("pre_kanban_dispatch", lambda lane, **kw: (
            {"action": "hold", "reason": "review_churn"} if lane == "review" else None))
        spawned: list[str] = []
        result = kbd.dispatch_once(conn, spawn_fn=lambda task, ws, **k: spawned.append(task.id) or 4242)
        assert spawned == []
        assert (tid, "review_churn") in result.respawn_guarded
        assert kb.get_task(conn, tid).status == "review"
    finally:
        conn.close()


def test_dispatch_without_subscriber_is_unchanged(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """No subscriber: the hook is never invoked and dispatch is untouched (backward compat)."""
    from hermes_cli import lifecycle

    invoked: list[str] = []
    real_invoke = lifecycle.invoke_hook

    def _spy(hook_name, **kw):
        invoked.append(hook_name)
        return real_invoke(hook_name, **kw)

    monkeypatch.setattr(lifecycle, "invoke_hook", _spy)
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="alice")
        spawned: list[str] = []
        result = kbd.dispatch_once(
            conn, spawn_fn=lambda task, ws, **k: spawned.append(task.id) or 4242,
        )
    finally:
        conn.close()
    assert "pre_kanban_dispatch" not in invoked
    assert spawned == [tid]
    assert result.respawn_guarded == []


def test_dispatch_ignores_reasonless_and_malformed_holds(
    kanban_home, all_assignees_spawnable, hooks,
):
    """Only a well-formed directive holds. Anything else must dispatch as before."""
    hooks.add("pre_kanban_dispatch", lambda **kw: {"action": "hold"})            # no reason
    hooks.add("pre_kanban_dispatch", lambda **kw: {"reason": "no action key"})
    hooks.add("pre_kanban_dispatch", lambda **kw: "hold")                        # not a dict
    hooks.add("pre_kanban_dispatch", lambda **kw: {"action": "allow"})
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="alice")
        spawned: list[str] = []
        result = kbd.dispatch_once(
            conn, spawn_fn=lambda task, ws, **k: spawned.append(task.id) or 4242,
        )
    finally:
        conn.close()
    assert spawned == [tid]
    assert result.respawn_guarded == []


def test_misbehaving_dispatch_guard_does_not_wedge_the_dispatcher(
    kanban_home, all_assignees_spawnable, hooks,
):
    """A raising callback falls through: the tick still spawns and returns a DispatchResult."""
    def _boom(**kw):
        raise RuntimeError("guard exploded")

    hooks.add("pre_kanban_dispatch", _boom)
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="alice")
        spawned: list[str] = []
        result = kbd.dispatch_once(
            conn, spawn_fn=lambda task, ws, **k: spawned.append(task.id) or 4242,
        )
        assert isinstance(result, kb.DispatchResult)
    finally:
        conn.close()
    assert spawned == [tid]


# ---------------------------------------------------------------------------
# pre_kanban_task_create
# ---------------------------------------------------------------------------


def test_create_suppress_returns_the_named_existing_card(
    kanban_home, hooks,
):
    """Suppression skips the insert and hands back the card the directive named."""
    conn = kbc.connect()
    try:
        existing = kb.create_task(conn, title="original")
        hooks.add("pre_kanban_task_create", lambda **kw: {
            "action": "suppress", "reason": "equivalent card open",
            "existing_task_id": existing,
        })
        assert kb.create_task(conn, title="duplicate") == existing
        assert _count(conn) == 1, "no duplicate row may be inserted"
    finally:
        conn.close()


def test_create_suppress_without_a_card_creates_nothing(
    kanban_home, hooks,
):
    """A suppression that names no card returns the empty-id sentinel and writes nothing."""
    conn = kbc.connect()
    try:
        hooks.add("pre_kanban_task_create",
                  lambda **kw: {"action": "suppress", "reason": "policy"})
        assert kb.create_task(conn, title="blocked") == ""
        assert _count(conn) == 0
    finally:
        conn.close()


def test_create_suppress_with_a_dangling_id_does_not_leak_it(
    kanban_home, hooks,
):
    """A suppressed create must never hand a caller an id that is not a real card."""
    conn = kbc.connect()
    try:
        hooks.add("pre_kanban_task_create", lambda **kw: {
            "action": "suppress", "reason": "dupe", "existing_task_id": "t_deadbeefcafe",
        })
        assert kb.create_task(conn, title="blocked") == ""
        assert _count(conn) == 0
    finally:
        conn.close()


def test_create_hook_receives_the_documented_payload(kanban_home, hooks):
    """Kwargs are the pre-write snapshot ``create_task`` was called with."""
    seen: list[dict] = []
    hooks.add("pre_kanban_task_create", lambda **kw: seen.append(kw))
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn, title="t", body="b", assignee="alice", session_id="s1",
            idempotency_key="k1",
        )
        assert tid and _count(conn) == 1
    finally:
        conn.close()
    assert seen, "the gate must fire before the insert"
    assert {k: seen[0][k] for k in (
        "title", "body", "board", "assignee", "session_id", "idempotency_key",
    )} == {
        "title": "t", "body": "b", "board": None, "assignee": "alice",
        "session_id": "s1", "idempotency_key": "k1",
    }


def test_create_without_subscriber_is_unchanged(kanban_home, monkeypatch):
    """No subscriber: the hook is never invoked and the card is created as before."""
    from hermes_cli import lifecycle

    invoked: list[str] = []
    real_invoke = lifecycle.invoke_hook

    def _spy(hook_name, **kw):
        invoked.append(hook_name)
        return real_invoke(hook_name, **kw)

    monkeypatch.setattr(lifecycle, "invoke_hook", _spy)
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="normal", assignee="alice")
        assert tid and _count(conn) == 1
    finally:
        conn.close()
    assert "pre_kanban_task_create" not in invoked


def test_create_ignores_reasonless_and_malformed_suppressions(kanban_home, hooks):
    """A suppress directive without a reason, or a non-dict return, creates the card."""
    hooks.add("pre_kanban_task_create", lambda **kw: {"action": "suppress"})
    hooks.add("pre_kanban_task_create", lambda **kw: {"action": "suppress", "reason": "  "})
    hooks.add("pre_kanban_task_create", lambda **kw: None)
    hooks.add("pre_kanban_task_create", lambda **kw: "suppress")
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="kept")
        assert tid and kb.get_task(conn, tid) is not None
        assert _count(conn) == 1
    finally:
        conn.close()


def test_create_treats_a_non_string_existing_id_as_no_card(kanban_home, hooks):
    """A malformed ``existing_task_id`` still suppresses, but yields the empty-id sentinel."""
    conn = kbc.connect()
    try:
        hooks.add("pre_kanban_task_create", lambda **kw: {
            "action": "suppress", "reason": "dupe", "existing_task_id": 7,
        })
        assert kb.create_task(conn, title="blocked") == ""
        assert _count(conn) == 0
    finally:
        conn.close()


def test_misbehaving_create_gate_does_not_break_creation(kanban_home, hooks):
    """A raising callback falls through; the card is created."""
    def _boom(**kw):
        raise RuntimeError("gate exploded")

    hooks.add("pre_kanban_task_create", _boom)
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="kept", assignee="alice")
        assert tid and _count(conn) == 1
    finally:
        conn.close()


def test_pre_decision_hooks_are_registered():
    """Both hook names are part of the public VALID_HOOKS surface."""
    assert {"pre_kanban_dispatch", "pre_kanban_task_create"} <= VALID_HOOKS
