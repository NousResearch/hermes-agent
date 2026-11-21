"""Every reader of a board database crosses the guarded connect, hosted or not.

The shared board's hosted refusal sits at the one opener of its database
(``hermes_cli.kanban_db_connect.connect``) and at the root resolution
(``hermes_cli.kanban_db.kanban_home`` — see ``test_shared_board_surfaces.py``).
The surfaces that reach the board through helpers of their own used to open the
database raw: the notifier's subscription counter, the worker's claim-liveness
proof, the watcher's bound-claim check, the worker-result read and the repair
probe each accepted an explicit path or a ``HERMES_KANBAN_DB`` pin and read the
first agent's private cards under the second agent's bound scope in a hosted
process, without crossing either refusal. Those opens now go through the one
guarded connect (read-only where the surface only reads), so each probe below
refuses before any board byte is read: no subscription count, no claim state, no
worker output, no integrity read. On a single-user machine nothing changes — the
same readers serve the shared root across profiles exactly as today.
"""
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

from agent.runtime_scope import bind_profile_scope

#: Both refusals say what they are; the root's own says it in its own words.
_REFUSAL = "refuses to open in a hosted process"


def _serve_hosted_tenants(monkeypatch):
    """This process reports itself hosted the way the boot profile does
    (``hermes_cli.hosted_mode.is_hosted`` — absent from a checkout without that
    module, which is why the shared-board boundary's guarded import answers
    False there)."""
    module = types.ModuleType("hermes_cli.hosted_mode")
    module.is_hosted = lambda: True
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.fixture
def seeded_shared_board(tmp_path, monkeypatch):
    """One agent's recognizable private state in the machine's one shared root,
    seeded while the process is not hosted: a claimed card with a worker
    binding, a recorded worker result and one private notification subscription.
    A second profile reading it is today's single-user handoff, and what a
    hosted process must never serve through any of its board readers."""
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "board"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    for name in ("home-a", "home-b"):
        (tmp_path / name).mkdir()
    with bind_profile_scope(tmp_path / "home-a", hydrate_secrets=False):
        with kbc.connect_closing() as conn:
            ident = kb.create_task(conn, title="TENANT-A-PRIVATE", assignee="default")
            task = kb.claim_task(conn, ident, claimer="test:owner", ttl_seconds=3600)
            kb._append_event(conn, ident, "worker_bound",
                             {"pid": os.getpid(), "claim_lock": task.claim_lock},
                             run_id=task.current_run_id)
            kb._append_event(conn, ident, "worker_result",
                             {"claim_lock": task.claim_lock, "last_output": "TENANT-A-PRIVATE",
                              "exit_code": 0},
                             run_id=task.current_run_id)
            conn.execute(
                "INSERT INTO kanban_notify_subs (task_id, platform, chat_id, thread_id, created_at) "
                "VALUES (?, 'tui', 'private', '', 1)", (ident,))
        board_db = kb.kanban_db_path()
    return tmp_path, board_db, task


@pytest.mark.parametrize("surface", [
    "notify_explicit", "notify_env",
    "claim_explicit", "claim_env",
    "worker_result", "bound_claim",
    "repair_explicit", "repair_env",
])
def test_hosted_every_board_reader_refuses(seeded_shared_board, monkeypatch, surface):
    """The readers that reach a board through their own helper — with an
    explicit path or through the worker's environment pin — must cross the same
    refusal as a plain connect: the first agent's subscription count, claim
    state and worker output stay unreadable under the second agent's scope."""
    root, board_db, task = seeded_shared_board
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli.kanban_db_notify import count_notify_subs
    from agent.kanban_turn_recovery import worker_claim_is_live
    from gateway.session_kanban import worker_result, _bound_claim_is_live
    context = {"db": str(board_db), "task_id": task.id, "run_id": task.current_run_id,
               "claim_lock": task.claim_lock}
    _serve_hosted_tenants(monkeypatch)
    with bind_profile_scope(root / "home-b", hydrate_secrets=False):
        # The guarded open refuses first, whatever path form it arrives in.
        assert kb._refuse_shared_board_if_hosted
        with pytest.raises(LookupError, match=_REFUSAL):
            kbc.connect(board_db)
        if surface.endswith("_env"):
            monkeypatch.setenv("HERMES_KANBAN_DB", str(board_db))
            monkeypatch.setenv("HERMES_KANBAN_TASK", task.id)
            monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
            monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", task.claim_lock)
        if surface.startswith("notify"):
            with pytest.raises(LookupError, match=_REFUSAL):
                count_notify_subs(**({"db_path": board_db} if surface.endswith("explicit") else {}))
        elif surface.startswith("claim"):
            live = worker_claim_is_live(**(
                {"task_id": task.id, "db_path": str(board_db), "run_id": str(task.current_run_id),
                 "claim_lock": task.claim_lock} if surface.endswith("explicit") else {}))
            assert live is False, "a hosted process proved a live claim on the shared board"
        elif surface == "worker_result":
            with pytest.raises(LookupError, match=_REFUSAL):
                worker_result(str(board_db), context)
        elif surface == "bound_claim":
            assert _bound_claim_is_live(context) is False, \
                "a hosted process proved a live worker binding on the shared board"
        else:
            with pytest.raises(LookupError, match=_REFUSAL):
                kbc.repair_db(**({"db_path": board_db} if surface.endswith("explicit") else {}))


@pytest.mark.parametrize("surface", ["transfer", "workspace", "watcher", "dashboard", "repair_default"])
def test_hosted_other_board_surfaces_refuse_also(seeded_shared_board, monkeypatch, surface):
    """The surfaces that already refused stay refused: an export cannot snapshot
    the shared board, sibling boards cannot be enumerated, the watcher cannot
    list boards, the dashboard cannot open one and a plain repair cannot resolve
    the default board."""
    root, _board_db, _task = seeded_shared_board
    from hermes_cli import kanban_db_connect as kbc
    _serve_hosted_tenants(monkeypatch)

    def _dashboard_open():
        spec = importlib.util.spec_from_file_location(
            "review_dashboard", Path.cwd() / "plugins" / "kanban" / "dashboard" / "plugin_api.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module._conn("default")

    surfaces = {
        "transfer": lambda: _export_scan(root),
        "workspace": lambda: _sibling_scan(),
        "watcher": lambda: _list_boards_scan(),
        "dashboard": _dashboard_open,
        "repair_default": lambda: kbc.repair_db(),
    }
    with bind_profile_scope(root / "home-b", hydrate_secrets=False):
        with pytest.raises(LookupError):
            surfaces[surface]()


def _export_scan(root):
    from hermes_cli.kanban_transfer import export_board
    export_board("default", str(root / "out"))


def _sibling_scan():
    import sqlite3
    from hermes_cli.kanban_db_workspace import _sibling_board_db_files
    with sqlite3.connect(":memory:") as conn:
        _sibling_board_db_files(conn)


def _list_boards_scan():
    from gateway.kanban_watchers_common import _list_boards
    from hermes_cli import kanban_db as kb
    _list_boards(kb)


def test_locally_the_same_readers_serve_the_shared_root(seeded_shared_board):
    """The control: on a single-user machine — not hosted — the same readers
    serve the one shared root across profiles exactly as today, the dispatcher's
    handoff included, and the repair probe still reads its board."""
    root, board_db, task = seeded_shared_board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_db_notify import count_notify_subs
    from agent.kanban_turn_recovery import worker_claim_is_live
    from gateway.session_kanban import worker_result
    with bind_profile_scope(root / "home-b", hydrate_secrets=False):
        assert count_notify_subs(db_path=board_db) == 1
        assert worker_claim_is_live(task_id=task.id, db_path=str(board_db),
                                    run_id=str(task.current_run_id),
                                    claim_lock=task.claim_lock) is True
        result = worker_result(str(board_db), {"task_id": task.id, "run_id": task.current_run_id,
                                               "claim_lock": task.claim_lock})
        assert result.get("last_output") == "TENANT-A-PRIVATE"
        assert json.dumps(result)
        assert kbc.repair_db(db_path=board_db).status == "ok"
