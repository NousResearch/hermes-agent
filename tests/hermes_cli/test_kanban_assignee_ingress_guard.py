"""Assignee-existence guard at the agent ingress surfaces.

``create_task``/``assign_task`` only lowercase-normalized their assignee — every
agent-facing surface could park a card on a profile that does not exist on this
host. The dispatcher then refuses those cards every tick as
``skipped_nonspawnable`` and the card rusts (real case: 20.4h stuck on
``devon``, created via ``kanban_create`` by an orchestration).

Fix shape: the **DB layer stays permissive** (restore flows, ~300 test
fixtures, external orchestrators), the guard lives at the ingress surfaces
(``kanban_create`` tool + CLI ``create``/``assign``/``reassign``/``swarm``).
The reviewer check introduced the same contract at request-review time
(#106163); this extends it to card creation and (re)assignment.

Accepted: ``None`` (unassigned), ``default``, live on-disk profiles, and
``bot_peers`` entries (cross-host workers behind a remote gateway). Operators
can force a known-good ghost through with ``HERMES_KANBAN_ALLOW_ANY_ASSIGNEE=1``
for a single command.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (same shape as test_kanban_db)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _make_profile(home: Path, name: str) -> None:
    """A live profile dir carries an identity marker (``named_profile_is_live``)."""
    (home / "profiles" / name).mkdir(parents=True)
    (home / "profiles" / name / "config.yaml").write_text("{}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Helper semantics
# ---------------------------------------------------------------------------


def test_validate_accepts_none_default_and_live_profile(kanban_home):
    from hermes_cli.kanban_db import validate_assignee_exists

    _make_profile(kanban_home, "vera")
    assert validate_assignee_exists(None) is None  # unassigned stays allowed
    assert validate_assignee_exists("default") == "default"
    assert validate_assignee_exists("vera") == "vera"


def test_validate_refuses_ghost_with_roster_and_override_hint(kanban_home):
    from hermes_cli.kanban_db import validate_assignee_exists

    with pytest.raises(ValueError) as exc:
        validate_assignee_exists("devon")
    msg = str(exc.value)
    assert "devon" in msg
    assert "default" in msg  # roster of installed profiles is in the message
    assert "HERMES_KANBAN_ALLOW_ANY_ASSIGNEE" in msg  # exact oversteer knob


def test_validate_normalizes_case(kanban_home):
    from hermes_cli.kanban_db import validate_assignee_exists

    _make_profile(kanban_home, "vera")
    assert validate_assignee_exists("Vera") == "vera"


def test_validate_tombstoned_profile_is_refused(kanban_home):
    """A deleted (tombstoned) profile directory must not pass ``profile_exists``."""
    from hermes_cli.kanban_db import validate_assignee_exists

    _make_profile(kanban_home, "ghosty")
    marker = kanban_home / "profiles" / ".deleted" / "ghosty"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("deleted\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_assignee_exists("ghosty")


def test_validate_accepts_bot_peer_as_cross_host_worker(kanban_home):
    """bot_peers are remote gateways, not local profiles — a card for such a
    worker is delivered via the peer transport, never spawned locally."""
    from hermes_cli.kanban_db import validate_assignee_exists

    (kanban_home / "config.yaml").write_text(
        "bot_peers:\n"
        "  naya:\n"
        "    url: http://100.114.57.14:8377\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        validate_assignee_exists("not-in-peers")
    assert validate_assignee_exists("naya") == "naya"


def test_validate_oversteer_env_allows_known_ghost(kanban_home, monkeypatch):
    from hermes_cli.kanban_db import validate_assignee_exists

    monkeypatch.setenv("HERMES_KANBAN_ALLOW_ANY_ASSIGNEE", "1")
    assert validate_assignee_exists("devon") == "devon"


# ---------------------------------------------------------------------------
# CLI verb surfaces (same entry the gateway /kanban route uses)
# ---------------------------------------------------------------------------


def _kanban_args(argv: list[str]) -> argparse.Namespace:
    from hermes_cli import kanban as kc

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    return parser.parse_args(["kanban", *argv])


def test_cli_create_refuses_ghost_assignee(kanban_home):
    from hermes_cli import kanban as kc
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    rc = kc.kanban_command(_kanban_args(["create", "rust card", "--assignee", "devon"]))
    assert rc == 2
    with kbc.connect_closing() as conn:
        assert kb.list_tasks(conn, limit=50) == []


def test_cli_create_accepts_live_profile(kanban_home):
    from hermes_cli import kanban as kc
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    _make_profile(kanban_home, "vera")
    rc = kc.kanban_command(_kanban_args(["create", "real card", "--assignee", "vera"]))
    assert rc == 0
    with kbc.connect_closing() as conn:
        (task,) = kb.list_tasks(conn, limit=50)
        assert task.assignee == "vera"


def test_cli_create_accepts_unassigned(kanban_home):
    from hermes_cli import kanban as kc

    assert kc.kanban_command(_kanban_args(["create", "no owner"])) == 0


def test_cli_assign_refuses_ghost_and_accepts_live(kanban_home):
    from hermes_cli import kanban as kc
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    _make_profile(kanban_home, "vera")
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="ownerless")
    assert kc.kanban_command(_kanban_args(["assign", tid, "devon"])) == 2
    assert kc.kanban_command(_kanban_args(["assign", tid, "vera"])) == 0
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).assignee == "vera"


def test_cli_assign_still_allows_unassign(kanban_home):
    from hermes_cli import kanban as kc
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    _make_profile(kanban_home, "vera")
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="moving", assignee="vera")
    assert kc.kanban_command(_kanban_args(["assign", tid, "none"])) == 0
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).assignee is None


def test_cli_reassign_refuses_ghost(kanban_home):
    from hermes_cli import kanban as kc
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    _make_profile(kanban_home, "vera")
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="handoff", assignee="vera")
    assert kc.kanban_command(_kanban_args(["reassign", tid, "devon"])) == 2
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).assignee == "vera"  # unchanged


def test_cli_swarm_refuses_ghost_worker(kanban_home):
    from hermes_cli import kanban as kc
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    _make_profile(kanban_home, "vera")
    rc = kc.kanban_command(_kanban_args([
        "swarm", "prove the guard", "--worker", "devon:Rust away",
        "--verifier", "vera", "--synthesizer", "vera",
    ]))
    assert rc == 2
    with kbc.connect_closing() as conn:
        assert kb.list_tasks(conn, limit=50) == []