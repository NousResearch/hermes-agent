"""Dashboard SessionDB handles are POOLED: their lifecycle belongs to the registry.

Regression coverage for #82919 (dashboard read/write session handles closed with
``db.close()`` instead of being returned to the pool):

1. ACCOUNTING — a dashboard open returns a registry-owned handle (the "pooled"
   marker is propagated), and a returned handle must drop both the live
   generation and its refcount: repeatedly polling with the old
   ``open + db.close()`` pattern must not leave generations/borrows stacked up
   (the reported symptom: ``db_in_use`` stuck, borrow never returned, fds
   growing ~0.7/h while the pool's steady state should be 3 per profile).
2. ORPHANED MARKER — when a handle still carries the pooled marker but the
   registry no longer tracks it (a pool entry was retired out from under the
   handle), ``close()`` must still close the connection: returning silently
   strands the descriptor for the process lifetime, which is the leak the
   report measured.
3. CALL SITES — every dashboard call site returns the handle through the
   registry drop-in (``release_or_close``), never a bare ``db.close()``.
"""

import ast
from pathlib import Path

import pytest

import hermes_state_registry as registry
import hermes_cli.web_routers.analytics as rt_analytics
import hermes_cli.web_routers.cron as rt_cron
import hermes_cli.web_routers.sessions as rt_sessions
import hermes_cli.web_routers.status as rt_status
import hermes_cli.web_server_chat as web_chat
import hermes_cli.web_server_sessions as web_sessions

# Modules whose call sites open a dashboard session handle.
DASHBOARD_MODULES = (rt_sessions, rt_analytics, rt_cron, rt_status, web_sessions, web_chat)

OPENERS = {"_open_session_db_for_profile", "_open_session_db_at_path"}
POOLED_RETURNS = {"release_or_close"}


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate the process-global registry between tests."""
    registry.close_all()
    registry._generations.clear()
    registry._retired.clear()
    registry._opening.clear()
    registry._tearing_down.clear()
    registry._path_lifecycle_locks.clear()
    yield
    registry.close_all()
    registry._generations.clear()
    registry._retired.clear()
    registry._opening.clear()
    registry._tearing_down.clear()


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_dashboard_pooled_open_propagates_registry_ownership(tmp_path):
    """The marker that makes close() a release must survive the dashboard opener."""
    db_path = tmp_path / "state.db"
    db = web_sessions._open_session_db_at_path(db_path, read_only=False)
    try:
        assert registry.stats()["live_generations"] == 1
        assert getattr(db, "_shared_registry_owned", False) is True, (
            "the pooled (registry-owned) marker was dropped on the way out of "
            "_open_session_db_at_path")
    finally:
        registry.release_or_close(db)
    assert registry.stats() == {"live_generations": 0, "retired_generations": 0, "total_refcounts": 0}


def test_dashboard_poll_cycle_balances_registry_accounting(tmp_path):
    """N dashboard polls (read-only and pooled write opens) leave no borrow stacked."""
    db_path = tmp_path / "state.db"
    baseline = registry.stats()
    for _ in range(20):
        read_db = web_sessions._open_session_db_at_path(db_path, read_only=True)
        registry.release_or_close(read_db)
        write_db = web_sessions._open_session_db_at_path(db_path, read_only=False)
        registry.release_or_close(write_db)
    assert registry.stats() == baseline, (
        "dashboard poll cycles stacked registry generations/borrows — the pool "
        "entry would stay db_in_use and the connection would never be returned")
    assert read_db._conn is None, "read handle kept its connection after release"
    assert write_db._conn is None, "pooled write handle kept its connection after release"


def test_close_on_orphaned_pooled_handle_releases_the_connection(tmp_path):
    """A marker whose registry record is gone must not strand the descriptor.

    The report's mechanism in one line: the handle still believes it is pooled,
    the pool no longer lists it, and ``close()`` returns without closing — so
    nothing ever closes that connection.
    """
    db_path = tmp_path / "state.db"
    from hermes_state import SessionDB

    db = SessionDB(db_path=db_path)
    assert db._conn is not None
    db._shared_registry_owned = True  # pooled marker set, no registry record
    db.close()
    assert db._conn is None, (
        "close() on a pooled handle the registry does not own left the "
        "connection open: every fd it holds is leaked until process exit")


def _returned_opener_calls(node: ast.AST) -> set[int]:
    """Line numbers of opener calls a function only hands back to its caller."""
    handed_back: set[int] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Return) and sub.value is not None:
            for call in ast.walk(sub.value):
                if isinstance(call, ast.Call) and _call_name(call) in OPENERS:
                    handed_back.add(call.lineno)
    return handed_back


def test_dashboard_call_sites_return_pooled_handles_via_registry():
    """AST guard: a handle opened by the dashboard must go back through release_or_close."""
    offenders = []
    for mod in DASHBOARD_MODULES:
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)]
            opener_calls = [c for c in calls if _call_name(c) in OPENERS]
            if not opener_calls:
                continue
            # A thin resolver (``_open_session_db_for_profile``) only returns the
            # handle; ownership stays with ITS caller, which this guard checks.
            if all(call.lineno in _returned_opener_calls(node) for call in opener_calls):
                continue
            if not any(_call_name(c) in POOLED_RETURNS for c in calls):
                offenders.append(f"{mod.__name__}:{node.lineno} {node.name}()")
    assert not offenders, (
        "dashboard call sites close a pooled SessionDB handle instead of "
        "returning it to the registry: " + ", ".join(offenders))
