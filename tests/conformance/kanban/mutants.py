"""Broken local kanban backends the conformance suite must reject (``test_mutants.py``).

Each passed an earlier version of the suite. ``tenant_blind`` drops the pin; ``ambient_refused``
over-pins the ambient caller; ``reads_on_ambient_root``, ``boards_on_ambient_root``,
``notify_on_ambient_root``, ``dispatch_on_ambient_root`` and ``sweep_on_ambient_root`` re-resolve
their operation's kanban root from the environment at call time, so an operation drifts onto
whatever root the ambient environment names (another agent's boards); ``claim_without_compare``
claims without the compare-and-set, so two replicas both win; ``refusal_commits_first`` refuses
only after the write has landed; ``tenant_only_pin`` drops the store's declaration of its one
shared board root while the pin keeps checking the tenant alone, so a backend that claims agent
isolation serves another same-tenant agent's scope on the same handle. Each mutant is applied by
the ``backend`` fixture when ``HERMES_KANBAN_MUTANT`` names it.
"""

import os
import time


def _tenant_blind(monkeypatch):
    """The pin is a no-op: a scope of any tenant is served."""
    from hermes_kanban_port_local import LocalKanbanStore
    monkeypatch.setattr(LocalKanbanStore, "_pin", lambda self: None)


def _ambient_refused(monkeypatch):
    """The pin refuses the ambient caller with no scope bound (today's CLI)."""
    from hermes_kanban_port_local import LocalKanbanStore
    from hermes_kanban_port import KanbanPinRefused
    from hermes_scope import current_scope

    def pin(self):
        if current_scope() is None:
            raise KanbanPinRefused("no scope bound")
    monkeypatch.setattr(LocalKanbanStore, "_pin", pin)


def _on_ambient_root(monkeypatch, method_names):
    """The named operations resolve the kanban root from the environment at call time, not
    from the root the store was opened for."""
    from contextlib import nullcontext
    from hermes_kanban_port_local import LocalKanbanStore

    for name in method_names:
        original = getattr(LocalKanbanStore, name)

        def ambient(self, *args, _original=original, **kwargs):
            had_own_root = "_root" in self.__dict__
            pinned = self.__dict__.get("_root")
            self._root = lambda: nullcontext()  # today's env resolution, per call
            try:
                return _original(self, *args, **kwargs)
            finally:
                if had_own_root:
                    self._root = pinned
                else:
                    del self.__dict__["_root"]
        monkeypatch.setattr(LocalKanbanStore, name, ambient)


def _claim_without_compare(monkeypatch):
    """claim_task flips the row without the status compare-and-set: a second claimant wins too."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing
    from hermes_kanban_port_local import LocalKanbanStore

    def claim(self, task_id, *, board=None, ttl_seconds=None, claimer=None):
        self._pin()
        with self._root(), connect_closing(board=board) as conn:
            now = int(time.time())
            lock = claimer or kb._claimer_id()
            expires = now + kb._resolve_claim_ttl_seconds(ttl_seconds)
            with kb.write_txn(conn):
                # The real claim's CAS guards on the status; this one flips whatever is there.
                cur = conn.execute(
                    "UPDATE tasks SET status = 'running', claim_lock = ?, claim_expires = ?, "
                    "started_at = COALESCE(started_at, ?) WHERE id = ?",
                    (lock, expires, now, task_id),
                )
                if cur.rowcount != 1:
                    return None
                run_id = kb._claim_and_open_run(
                    conn, task_id, "running", lock, expires, now, event_extra={"unchecked": True})
                if run_id is None:  # its CAS cannot match a row already flipped to running
                    trow = conn.execute(
                        "SELECT assignee, max_runtime_seconds, current_step_key FROM tasks WHERE id = ?",
                        (task_id,)).fetchone()
                    run_cur = conn.execute(
                        "INSERT INTO task_runs (task_id, profile, step_key, status, claim_lock, "
                        "claim_expires, max_runtime_seconds, started_at) "
                        "VALUES (?, ?, ?, 'running', ?, ?, ?, ?)",
                        (task_id, trow["assignee"] if trow else None,
                         trow["current_step_key"] if trow else None,
                         lock, expires, trow["max_runtime_seconds"] if trow else None, now),
                    )
                    run_id = run_cur.lastrowid
                    conn.execute("UPDATE tasks SET current_run_id = ? WHERE id = ?", (run_id, task_id))
                    kb._append_event(conn, task_id, "claimed",
                                     {"lock": lock, "expires": expires, "run_id": run_id,
                                      "unchecked": True}, run_id=run_id)
            return kb.get_task(conn, task_id)
    monkeypatch.setattr(LocalKanbanStore, "claim_task", claim)


def _refusal_commits_first(monkeypatch):
    """The pin runs after the operation: a refused write has already landed."""
    from hermes_kanban_port_local import LocalKanbanStore
    real_pin = LocalKanbanStore._pin
    monkeypatch.setattr(LocalKanbanStore, "_pin", lambda self: None)
    from hermes_conformance.kanban.suite import WRITES
    for name in [name for name, _call, _ok in WRITES]:
        original = getattr(LocalKanbanStore, name)

        def after(self, *args, _original=original, **kwargs):
            result = _original(self, *args, **kwargs)
            real_pin(self)
            return result
        monkeypatch.setattr(LocalKanbanStore, name, after)


def _tenant_only_pin(monkeypatch):
    """The store stops declaring its one shared board root while the pin keeps checking the
    tenant alone: a backend that claims agent isolation its pin does not enforce, serving
    another same-tenant agent's scope on the same handle (reading and claiming its rows)."""
    from hermes_kanban_port_local import LocalKanbanStore
    monkeypatch.delattr(LocalKanbanStore, "LOCAL_SHARED_BOARD")


_READS = ("get_task", "list_tasks", "list_events", "list_comments", "list_boards",
          "list_notify_subs", "latest_run", "list_runs", "unseen_events_for_sub")
_NOTIFY = ("list_notify_subs", "unseen_events_for_sub", "claim_unseen_events_for_sub",
           "remove_notify_sub")
_DISPATCH = ("claim_task", "claim_review_task", "heartbeat_claim")
_SWEEP = ("release_stale_claims", "reclaim_task", "reassign_task", "recompute_ready")

MUTANTS = {
    "tenant_blind": _tenant_blind,
    "ambient_refused": _ambient_refused,
    "reads_on_ambient_root": lambda m: _on_ambient_root(m, _READS),
    "boards_on_ambient_root": lambda m: _on_ambient_root(m, ["list_boards"]),
    "notify_on_ambient_root": lambda m: _on_ambient_root(m, _NOTIFY),
    "dispatch_on_ambient_root": lambda m: _on_ambient_root(m, _DISPATCH),
    "sweep_on_ambient_root": lambda m: _on_ambient_root(m, _SWEEP),
    "claim_without_compare": _claim_without_compare,
    "refusal_commits_first": _refusal_commits_first,
    "tenant_only_pin": _tenant_only_pin,
}


def apply_mutant(monkeypatch):
    """Break the local backend the way ``HERMES_KANBAN_MUTANT`` names (a no-op when unset)."""
    name = os.environ.get("HERMES_KANBAN_MUTANT", "").strip()
    if name:
        MUTANTS[name](monkeypatch)
