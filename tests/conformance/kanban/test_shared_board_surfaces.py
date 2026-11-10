"""Every existing surface of the shared board refuses a hosted process alike.

The hosted refusal sits where every path to the machine's one shared board crosses:
resolving the shared root (``hermes_cli.kanban_db.kanban_home`` — the storage port's
local store resolves its frozen root there, the board metadata lives under it, and
every ``kanban_db_path`` resolution lands in it) and opening the board database
(``hermes_cli.kanban_db_connect.connect`` — including opens that arrive with an
explicit or a pinned path and never resolve the root). A hosted process serves
tenants that must not see each other's boards, so no card of one agent's board is
readable or writable under another agent's scope, whatever surface asks: the raw DB
helpers, the model tools, the CLI's JSON output and the assignee API alike. Locally
nothing changes: no hosted boot profile, no refusal.
"""
import argparse
import sys
import types

import pytest

from agent.runtime_scope import bind_profile_scope
from hermes_kanban_port import resolve_kanban_storage
from hermes_kanban_port_local import LocalKanbanStore

#: Both refusals say what they are; the resolver's own says it in its own words.
_REFUSAL = "refuses to open in a hosted process"


def _serve_hosted_tenants(monkeypatch):
    """This process reports itself hosted the way the boot profile does
    (``hermes_cli.hosted_mode.is_hosted`` — absent from a checkout without that module,
    which is why the shared-board boundary's guarded import answers False there)."""
    module = types.ModuleType("hermes_cli.hosted_mode")
    module.is_hosted = lambda: True
    monkeypatch.setitem(sys.modules, "hermes_cli.hosted_mode", module)


def _seed_a_private_card(tmp_path, monkeypatch):
    """One agent's recognizable card in the machine's one shared root, seeded while the
    process is not hosted: a second profile reading it is today's single-user handoff,
    and what a hosted process must never serve."""
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "board"))
    (tmp_path / "home-a").mkdir()
    (tmp_path / "home-b").mkdir()
    with bind_profile_scope(tmp_path / "home-a", hydrate_secrets=False):
        with kbc.connect_closing() as conn:
            return kb.create_task(conn, title="TENANT-A-PRIVATE", assignee="a")


@pytest.mark.parametrize("surface", ["db", "tools", "cli", "profiles"])
def test_hosted_the_shared_board_is_unreachable_from_every_existing_surface(
        tmp_path, monkeypatch, capsys, surface):
    """The surfaces that never construct the port's local store: the DB helpers a
    connection is handed to, the model tools, the CLI's JSON output and the assignee
    API. Under the second agent's bound scope each must refuse before any board is
    read or written — the first agent's card is neither served nor writable."""
    ident = _seed_a_private_card(tmp_path, monkeypatch)
    _serve_hosted_tenants(monkeypatch)
    with bind_profile_scope(tmp_path / "home-b", hydrate_secrets=False):
        if surface == "db":
            from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
            with pytest.raises(LookupError, match=_REFUSAL):
                with kbc.connect_closing() as conn:
                    kb.get_task(conn, ident)
                    kb.create_task(conn, title="TENANT-B-WRITES-A-BOARD")
        elif surface == "tools":
            from tools.kanban_tools import _board
            with pytest.raises(LookupError, match=_REFUSAL):
                with _board("default") as (kb, conn):
                    kb.get_task(conn, ident)
                    kb.create_task(conn, title="TENANT-B-WRITES-A-BOARD")
        elif surface == "cli":
            from hermes_cli.kanban import kanban_command
            args = argparse.Namespace(
                kanban_action="list", board=None, status=None, assignee=None,
                json=True, mine=False, tenant=None, session=None, archived=False,
                workflow_template_id=None, current_step_key=None)
            rc = kanban_command(args)
            out = capsys.readouterr().out
            assert "TENANT-A-PRIVATE" not in out, out
            assert rc != 0
        else:
            from hermes_cli import kanban_db_connect as kbc
            from hermes_cli.kanban_db_profiles import known_assignees
            with pytest.raises(LookupError, match=_REFUSAL):
                with kbc.connect_closing() as conn:
                    known_assignees(conn)


def test_an_explicit_not_hosted_answer_cannot_downgrade_a_hosted_boot_verdict(
        tmp_path, monkeypatch):
    """The resolver's explicit ``hosted=`` answer may only force the refusal, never
    lift it: a caller that answers not-hosted while the boot profile says hosted gets
    the same refusal, so the shared board cannot be opened through the resolver."""
    _seed_a_private_card(tmp_path, monkeypatch)
    _serve_hosted_tenants(monkeypatch)
    with bind_profile_scope(tmp_path / "home-b", hydrate_secrets=False):
        with pytest.raises(LookupError):
            store = resolve_kanban_storage("local", hosted=False).open().kanban
            store.get_task("whatever", board="default")


def test_hosted_an_open_with_an_explicit_board_path_refuses_also(
        tmp_path, monkeypatch):
    """Opens that never resolve the shared root — an explicit ``db_path`` argument, or
    a worker's ``HERMES_KANBAN_DB`` pin — cross the same refusal at the open itself,
    so a pinned path cannot carry a hosted process onto the shared board."""
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    _seed_a_private_card(tmp_path, monkeypatch)
    board_db = kb.kanban_db_path()  # resolved while the process is still local
    _serve_hosted_tenants(monkeypatch)
    with pytest.raises(LookupError, match=_REFUSAL):
        kbc.connect(db_path=board_db)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board_db))
    with pytest.raises(LookupError, match=_REFUSAL):
        kbc.connect()


def test_hosted_board_metadata_is_unreachable_too(tmp_path, monkeypatch):
    """Refusing the opens is not enough: board names and metadata live under the same
    shared root without opening a database, and the root's resolution refuses them
    alike — for a direct resolution and for a store that froze its root while the
    process was still local."""
    from hermes_cli import kanban_db as kb
    _seed_a_private_card(tmp_path, monkeypatch)
    store = LocalKanbanStore()  # opened while local: the root freezes here
    assert store.list_boards()
    _serve_hosted_tenants(monkeypatch)
    with pytest.raises(LookupError, match=_REFUSAL):
        kb.kanban_home()
    with pytest.raises(LookupError, match=_REFUSAL):
        store.list_boards()


def test_locally_every_surface_still_serves_the_shared_root(tmp_path, monkeypatch, capsys):
    """The control: on a single-user machine — not hosted — the same surfaces serve the
    one shared root exactly as today, across profiles (the dispatcher's handoff)."""
    import json
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli.kanban_db_profiles import known_assignees
    ident = _seed_a_private_card(tmp_path, monkeypatch)
    with bind_profile_scope(tmp_path / "home-b", hydrate_secrets=False):
        with kbc.connect_closing() as conn:
            task = kb.get_task(conn, ident)
            assert task is not None and task.title == "TENANT-A-PRIVATE"
            data = known_assignees(conn)
        assert any(row["name"] == "a" for row in data)
        from hermes_cli.kanban import kanban_command
        args = argparse.Namespace(
            kanban_action="list", board=None, status=None, assignee=None,
            json=True, mine=False, tenant=None, session=None, archived=False,
            workflow_template_id=None, current_step_key=None)
        assert kanban_command(args) == 0
        rows = json.loads(capsys.readouterr().out)
        assert any(row.get("id") == ident for row in rows)
