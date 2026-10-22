"""Per-project handover record: ``projects_db`` storage and the ``hermes project state`` CLI.

The record answers "where is project X at?" after compaction, a restart or a new session, so the
contracts below are the ones a resuming reader depends on: a partial write never loses the other
fields, history is bounded, refused writes leave nothing behind, and the record is per-profile.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import projects_cmd
from hermes_cli import projects_db as pdb


@pytest.fixture
def conn(tmp_path):
    c = pdb.connect(db_path=tmp_path / "projects.db")
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def pid(conn):
    return pdb.create_project(conn, name="Demo")


def _fields(state: dict) -> dict:
    return {k: state[k] for k in ("goal", "now", "next", "blockers", "updated_by")}


# --- storage -------------------------------------------------------------------------------------


def test_no_handover_reads_as_none(conn, pid):
    assert pdb.get_project_state(conn, pid) is None
    assert pdb.project_state_history(conn, pid) == []


def test_set_then_get_round_trips(conn, pid):
    saved = pdb.set_project_state(
        conn, pid, goal="Ship v1", now="writing tests", next="review", blockers="CI red",
        updated_by="agent")

    state = pdb.get_project_state(conn, pid)
    assert state == saved
    assert _fields(state) == {
        "goal": "Ship v1", "now": "writing tests", "next": "review", "blockers": "CI red",
        "updated_by": "agent"}
    assert state["updated_at"] > 0


def test_partial_update_keeps_unspecified_fields(conn, pid):
    pdb.set_project_state(conn, pid, goal="Ship v1", now="tests", blockers="CI red", updated_by="agent")
    pdb.set_project_state(conn, pid, now="implementation", updated_by="user")

    assert _fields(pdb.get_project_state(conn, pid)) == {
        "goal": "Ship v1", "now": "implementation", "next": None, "blockers": "CI red",
        "updated_by": "user"}


def test_empty_string_clears_one_field(conn, pid):
    pdb.set_project_state(conn, pid, goal="Ship v1", blockers="CI red", updated_by="agent")
    pdb.set_project_state(conn, pid, blockers="", updated_by="agent")

    state = pdb.get_project_state(conn, pid)
    assert state["blockers"] is None
    assert state["goal"] == "Ship v1"


def test_history_is_newest_first_and_capped_per_project(conn, pid):
    other = pdb.create_project(conn, name="Other")
    pdb.set_project_state(conn, other, goal="untouched", updated_by="user")
    cap = pdb.STATE_HISTORY_LIMIT
    total = cap + 5
    for i in range(total):
        pdb.set_project_state(conn, pid, now=f"step {i}", updated_by="agent")

    stored = conn.execute("SELECT COUNT(*) FROM project_state WHERE project_id = ?", (pid,)).fetchone()[0]
    assert stored == cap  # pruned in the write, not merely clamped on read
    history = pdb.project_state_history(conn, pid, limit=total)
    assert len(history) == cap
    assert [h["now"] for h in history[:2]] == [f"step {total - 1}", f"step {total - 2}"]
    assert history[-1]["now"] == f"step {total - cap}"
    assert [h["now"] for h in pdb.project_state_history(conn, pid, limit=2)] == [
        f"step {total - 1}", f"step {total - 2}"]
    # Retention is per project: the neighbour's single record survives.
    assert pdb.get_project_state(conn, other)["goal"] == "untouched"


def test_history_limit_must_be_positive(conn, pid):
    with pytest.raises(ValueError, match="limit"):
        pdb.project_state_history(conn, pid, limit=0)


def test_field_size_cap_names_field_and_limit(conn, pid):
    cap = pdb.STATE_FIELD_MAX_CHARS
    pdb.set_project_state(conn, pid, goal="x" * cap, updated_by="agent")  # exactly at the cap is fine

    with pytest.raises(ValueError, match=rf"next.*{cap}"):
        pdb.set_project_state(conn, pid, next="x" * (cap + 1), updated_by="agent")
    assert len(pdb.project_state_history(conn, pid)) == 1  # the refused write appended nothing


def test_archived_project_refuses_writes_but_stays_readable(conn, pid):
    pdb.set_project_state(conn, pid, goal="Ship v1", updated_by="agent")
    pdb.archive_project(conn, pid)

    with pytest.raises(ValueError, match="archived"):
        pdb.set_project_state(conn, pid, now="sneaky", updated_by="agent")
    assert pdb.get_project_state(conn, pid)["now"] is None

    pdb.restore_project(conn, pid)
    pdb.set_project_state(conn, pid, now="back", updated_by="agent")
    assert pdb.get_project_state(conn, pid)["now"] == "back"


@pytest.mark.parametrize("by", ["", "system", "User", None])
def test_invalid_updated_by_is_refused(conn, pid, by):
    with pytest.raises(ValueError, match="updated_by"):
        pdb.set_project_state(conn, pid, goal="Ship v1", updated_by=by)
    assert pdb.get_project_state(conn, pid) is None


def test_all_empty_write_is_refused(conn, pid):
    with pytest.raises(ValueError):
        pdb.set_project_state(conn, pid, updated_by="agent")
    with pytest.raises(ValueError):
        pdb.set_project_state(conn, pid, goal="   ", now="", updated_by="agent")
    assert pdb.get_project_state(conn, pid) is None


def test_clearing_the_last_field_is_refused(conn, pid):
    pdb.set_project_state(conn, pid, goal="Ship v1", updated_by="agent")
    with pytest.raises(ValueError):
        pdb.set_project_state(conn, pid, goal="", updated_by="agent")
    assert pdb.get_project_state(conn, pid)["goal"] == "Ship v1"


def test_unknown_project_is_refused(conn):
    with pytest.raises(ValueError, match="no such project"):
        pdb.set_project_state(conn, "p_missing", goal="g", updated_by="agent")


def test_deleting_a_project_cascades_its_state(conn, pid):
    pdb.set_project_state(conn, pid, goal="g", updated_by="agent")
    pdb.set_project_state(conn, pid, now="n", updated_by="agent")

    pdb.delete_project(conn, pid)

    assert conn.execute("SELECT COUNT(*) FROM project_state WHERE project_id = ?", (pid,)).fetchone()[0] == 0


def test_legacy_db_without_state_table_upgrades_on_connect(tmp_path):
    path = tmp_path / "legacy" / "projects.db"
    path.parent.mkdir()
    raw = sqlite3.connect(path)
    raw.executescript(
        """
        CREATE TABLE projects (
            id TEXT PRIMARY KEY, slug TEXT NOT NULL UNIQUE, name TEXT NOT NULL, description TEXT,
            created_at INTEGER NOT NULL, archived INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE project_folders (
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE, path TEXT NOT NULL,
            label TEXT, is_primary INTEGER NOT NULL DEFAULT 0, added_at INTEGER NOT NULL,
            PRIMARY KEY (project_id, path));
        CREATE TABLE project_meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO projects (id, slug, name, created_at) VALUES ('p_legacy', 'legacy', 'Legacy', 1);
        """
    )
    raw.commit()
    raw.close()

    conn = pdb.connect(db_path=path)
    try:
        assert pdb.get_project_state(conn, "p_legacy") is None
        pdb.set_project_state(conn, "p_legacy", goal="survive the upgrade", updated_by="user")
        assert pdb.get_project_state(conn, "p_legacy")["goal"] == "survive the upgrade"
    finally:
        conn.close()


# --- CLI -----------------------------------------------------------------------------------------


def _run(argv):
    """Build the project subparser, parse argv, and dispatch. Returns rc."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    projects_cmd.build_parser(sub)
    return projects_cmd.projects_command(parser.parse_args(["project", *argv]))


def _stored_state(slug: str):
    with pdb.connect_closing() as c:
        return pdb.get_project_state(c, pdb.get_project(c, slug).id)


@pytest.fixture
def app(tmp_path, capsys):
    assert _run(["create", "My App", str(tmp_path / "repo")]) == 0
    capsys.readouterr()
    return "my-app"


def test_cli_show_without_handover(app, capsys):
    assert _run(["state", app]) == 0
    assert "no handover recorded" in capsys.readouterr().out

    assert _run(["state", app, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] is None
    assert payload["project"]["slug"] == app


def test_cli_set_then_show_text_and_json(app, capsys):
    assert _run(["state", app, "--set", "--goal", "Ship v1", "--now", "writing tests"]) == 0
    capsys.readouterr()

    assert _run(["state", app]) == 0
    out = capsys.readouterr().out
    assert "Ship v1" in out and "writing tests" in out

    assert _run(["state", app, "--json"]) == 0
    state = json.loads(capsys.readouterr().out)["state"]
    assert _fields(state) == {
        "goal": "Ship v1", "now": "writing tests", "next": None, "blockers": None, "updated_by": "user"}


def test_cli_by_agent_records_agent(app):
    assert _run(["state", app, "--set", "--next", "review", "--by", "agent"]) == 0
    assert _stored_state(app)["updated_by"] == "agent"


def test_cli_history_text_json_and_limit(app, capsys):
    for i in range(3):
        assert _run(["state", app, "--set", "--now", f"step {i}"]) == 0
    capsys.readouterr()

    assert _run(["state", app, "--history", "--limit", "2", "--json"]) == 0
    history = json.loads(capsys.readouterr().out)["history"]
    assert [h["now"] for h in history] == ["step 2", "step 1"]

    assert _run(["state", app, "--history"]) == 0
    out = capsys.readouterr().out
    assert out.index("step 2") < out.index("step 0")


def test_cli_history_limit_beyond_sqlite_integer_range_is_clamped(app, capsys):
    for i in range(2):
        assert _run(["state", app, "--set", "--now", f"step {i}"]) == 0
    capsys.readouterr()

    assert _run(["state", app, "--history", "--limit", str(2**63), "--json"]) == 0
    captured = capsys.readouterr()
    history = json.loads(captured.out)["history"]
    assert [h["now"] for h in history] == ["step 1", "step 0"]
    assert len(history) <= pdb.STATE_HISTORY_LIMIT
    assert "Traceback" not in captured.err


def test_cli_unknown_project_is_rc1(capsys):
    assert _run(["state", "nope"]) == 1
    assert _run(["state", "nope", "--set", "--goal", "g"]) == 1
    assert _run(["state", "nope", "--history"]) == 1
    assert "no such project" in capsys.readouterr().err


def test_cli_refused_writes_are_rc2_and_record_nothing(app, capsys):
    assert _run(["state", app, "--set"]) == 2
    assert _run(["state", app, "--set", "--goal", "x" * (pdb.STATE_FIELD_MAX_CHARS + 1)]) == 2
    # Field flags without --set are a mistake, not a silent read.
    assert _run(["state", app, "--goal", "forgot --set"]) == 2
    assert _run(["archive", app]) == 0
    assert _run(["state", app, "--set", "--goal", "archived"]) == 2
    assert "project:" in capsys.readouterr().err
    assert _stored_state(app) is None


@pytest.mark.parametrize("argv", [
    ["--limit", "1"],                       # --limit outside --history
    ["--set", "--goal", "g", "--limit", "1"],
    ["--by", "agent"],                      # --by outside --set
    ["--history", "--by", "agent"],
])
def test_cli_mode_flags_outside_their_mode_are_rc2(app, capsys, argv):
    assert _run(["state", app, *argv]) == 2
    assert "project:" in capsys.readouterr().err
    assert _stored_state(app) is None


def test_cli_every_state_option_has_help_text():
    parser = argparse.ArgumentParser()
    top = projects_cmd.build_parser(parser.add_subparsers(dest="command"))
    state = next(a for a in top._actions if isinstance(a, argparse._SubParsersAction)).choices["state"]
    assert [a.option_strings for a in state._actions if a.option_strings and not a.help] == []


def test_cli_set_and_history_are_exclusive(app, capsys):
    with pytest.raises(SystemExit):
        _run(["state", app, "--set", "--history"])
    assert "not allowed with argument" in capsys.readouterr().err
    assert _stored_state(app) is None


# --- profile isolation (E2E, real imports, two temp HERMES_HOMEs) ---------------------------------


def _use_home(monkeypatch, tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    home = root / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: root)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_handover_is_isolated_per_profile_home(monkeypatch, tmp_path, capsys):
    home_a = _use_home(monkeypatch, tmp_path, "a")
    assert _run(["create", "Shared", str(tmp_path / "repo")]) == 0
    assert _run(["state", "shared", "--set", "--goal", "A's goal", "--by", "agent"]) == 0

    home_b = _use_home(monkeypatch, tmp_path, "b")
    assert _run(["state", "shared"]) == 1  # the project itself lives only in A
    assert _run(["create", "Shared", str(tmp_path / "repo")]) == 0
    capsys.readouterr()
    assert _run(["state", "shared", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] is None
    assert _run(["state", "shared", "--set", "--goal", "B's goal"]) == 0

    _use_home(monkeypatch, tmp_path, "a")
    capsys.readouterr()
    assert _run(["state", "shared", "--json"]) == 0
    state = json.loads(capsys.readouterr().out)["state"]
    assert (state["goal"], state["updated_by"]) == ("A's goal", "agent")
    assert (home_a / "projects.db").is_file() and (home_b / "projects.db").is_file()
