"""Tests for per-project default model routes (`hermes project set-model`).

A project may pin a default model route (``model`` + optional ``provider``); sessions whose working
directory falls inside one of the project's folders start on that route unless the caller asked for
a model explicitly. Unpinned projects follow the profile default.
"""

from __future__ import annotations

import argparse
import os
import sqlite3

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


# ── store ───────────────────────────────────────────────────────────────────────────────────


def test_pinned_model_resolves_for_sessions_under_the_project_folder(conn):
    pid = pdb.create_project(conn, name="Mesa Alta", folders=["/www/mesa-alta"])
    pdb.set_project_model(conn, pid, "claude-opus-4-8", "anthropic")

    assert pdb.project_model_for_path(conn, "/www/mesa-alta") == ("claude-opus-4-8", "anthropic", "mesa-alta")
    assert pdb.project_model_for_path(conn, "/www/mesa-alta/sub/dir")[0] == "claude-opus-4-8"
    # Outside every project folder there is no pin; the profile default applies.
    assert pdb.project_model_for_path(conn, "/www/elsewhere") is None


def test_innermost_project_wins_and_sibling_prefix_is_not_a_match(conn):
    outer = pdb.create_project(conn, name="Outer", folders=["/www/outer"])
    inner = pdb.create_project(conn, name="Inner", folders=["/www/outer/inner"])
    sibling = pdb.create_project(conn, name="Sibling", folders=["/www/outer-extra"])
    pdb.set_project_model(conn, outer, "outer-model")
    pdb.set_project_model(conn, inner, "inner-model")
    pdb.set_project_model(conn, sibling, "sibling-model")

    assert pdb.project_model_for_path(conn, "/www/outer/inner/deep")[0] == "inner-model"
    assert pdb.project_model_for_path(conn, "/www/outer/other")[0] == "outer-model"
    # "/www/outer-extra" must not be read as a child of "/www/outer".
    assert pdb.project_model_for_path(conn, "/www/outer-extra")[0] == "sibling-model"


def test_provider_without_model_is_refused(conn):
    pid = pdb.create_project(conn, name="Pin", folders=["/www/pin"])
    with pytest.raises(ValueError):
        pdb.set_project_model(conn, pid, "", "anthropic")
    assert pdb.get_project(conn, pid).model is None


def test_unpinned_and_cleared_projects_follow_the_profile_default(conn):
    pid = pdb.create_project(conn, name="Follow", folders=["/www/follow"])
    assert pdb.project_model_for_path(conn, "/www/follow") is None

    pdb.set_project_model(conn, pid, "deepseek-v4-flash", "opencode-go")
    assert pdb.project_model_for_path(conn, "/www/follow")[0] == "deepseek-v4-flash"

    pdb.set_project_model(conn, pid, "", "")
    assert pdb.project_model_for_path(conn, "/www/follow") is None


def test_project_route_survives_dict_serialization(conn):
    """The dashboard/gateway read projects as dicts, so the pin must be in ``to_dict()``."""
    pid = pdb.create_project(conn, name="Serialized", folders=["/www/s"])
    pdb.set_project_model(conn, pid, "some-model", "some-provider")

    assert pdb.get_project(conn, pid).to_dict()["model"] == "some-model"
    assert pdb.get_project(conn, pid).to_dict()["provider"] == "some-provider"


def test_legacy_db_without_model_columns_upgrades_in_place(tmp_path):
    """A pre-feature projects.db keeps its rows and gains the columns on open."""
    db = tmp_path / "projects.db"
    legacy = sqlite3.connect(db)
    legacy.executescript(
        "CREATE TABLE projects (id TEXT PRIMARY KEY, slug TEXT NOT NULL UNIQUE, name TEXT NOT NULL,"
        " description TEXT, icon TEXT, color TEXT, board_slug TEXT, primary_path TEXT,"
        " created_at INTEGER NOT NULL, archived INTEGER NOT NULL DEFAULT 0);"
        "CREATE TABLE project_folders (project_id TEXT NOT NULL, path TEXT NOT NULL, label TEXT,"
        " is_primary INTEGER NOT NULL DEFAULT 0, added_at INTEGER NOT NULL,"
        " PRIMARY KEY (project_id, path));"
    )
    legacy.execute("INSERT INTO projects (id, slug, name, created_at, archived) VALUES ('p_old','old','Old',1,0)")
    legacy.execute("INSERT INTO project_folders (project_id, path, is_primary, added_at) VALUES ('p_old','/www/old',1,1)")
    legacy.commit()
    legacy.close()

    c = pdb.connect(db_path=db)
    try:
        proj = pdb.get_project(c, "old")
        assert proj is not None and proj.name == "Old" and proj.model is None
        assert pdb.project_model_for_path(c, "/www/old") is None  # migrated, not pinned
        assert pdb.set_project_model(c, "p_old", "gpt-5-mini", "openai") == ("gpt-5-mini", "openai")
        assert pdb.project_model_for_path(c, "/www/old/sub")[0] == "gpt-5-mini"
    finally:
        c.close()


def test_resolver_is_fail_open(tmp_path):
    """Startup must degrade to 'no pin' rather than fail on a missing/corrupt projects DB."""
    assert pdb.resolve_default_model_for_path("", db_path=tmp_path / "projects.db") is None
    (tmp_path / "corrupt.db").write_bytes(b"not a sqlite database")
    assert pdb.resolve_default_model_for_path("/www/anything", db_path=tmp_path / "corrupt.db") is None


def test_pin_resolves_through_a_symlinked_spelling_of_the_folder(conn, tmp_path):
    """macOS ``/tmp`` -> ``/private/tmp``: a session cwd and a stored folder may differ only by symlink."""
    real = tmp_path / "real-project"
    real.mkdir()
    link = tmp_path / "link-project"
    link.symlink_to(real, target_is_directory=True)

    pid = pdb.create_project(conn, name="Linked", folders=[str(real)])
    pdb.set_project_model(conn, pid, "linked-model", "linked-provider")

    assert pdb.project_model_for_path(conn, str(link))[0] == "linked-model"
    assert pdb.project_model_for_path(conn, str(link / "deep" / "dir"))[0] == "linked-model"


# ── startup resolution (CLI) ────────────────────────────────────────────────────────────────


def _pin_in_sandboxed_home(folder, model: str, provider: str) -> None:
    """Pin a route on a project owning ``folder`` in the test home's projects DB."""
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(conn, name=os.path.basename(str(folder)), folders=[str(folder)])
        pdb.set_project_model(conn, pid, model, provider)


def _startup_route(monkeypatch, cwd, *, model=None, provider=None):
    """Run the real CLI startup model resolution with ``cwd`` as the working directory.

    ``cli`` is stubbed and route resolution is pinned to None: importing the real ``cli`` loads the
    checkout's manifest out of the live ``~/.hermes`` (the suite's I/O guard refuses that), and the
    alias/qualified-string path is covered by ``test_startup_model_routing.py``. What is under test
    here is the project pin's precedence inside the mixin.
    """
    import sys
    import types

    from hermes_cli import model_switch
    from hermes_cli.cli_init_mixin import CLIInitMixin

    stub = types.ModuleType("cli")
    stub.CLI_CONFIG = {"model": {"default": "profile-default-model", "provider": "anthropic"}}
    stub._normalize_moa_model = lambda value: ("", value)
    stub._split_model_config_default = lambda raw: (raw, "")
    monkeypatch.setitem(sys.modules, "cli", stub)
    monkeypatch.setattr(model_switch, "resolve_startup_model_route", lambda *a, **k: None)
    monkeypatch.chdir(cwd)

    probe = CLIInitMixin.__new__(CLIInitMixin)  # the method under test touches only init attrs
    probe._init_model_and_provider(model, provider, None, None)
    return probe


def test_cli_startup_adopts_the_project_pin_for_its_working_directory(tmp_path, monkeypatch):
    project_dir = tmp_path / "awaken"
    project_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _pin_in_sandboxed_home(project_dir, "deepseek-v4-flash", "opencode-go")

    inside = _startup_route(monkeypatch, project_dir)
    assert inside.model == "deepseek-v4-flash"
    assert inside.requested_provider == "opencode-go"
    assert inside._project_model_pin == ("deepseek-v4-flash", "opencode-go", "awaken")

    # An unpinned directory still gets the profile's configured default.
    outside = _startup_route(monkeypatch, elsewhere)
    assert outside.model == "profile-default-model"
    assert outside._project_model_pin is None


def test_cli_startup_explicit_model_beats_the_project_pin(tmp_path, monkeypatch):
    project_dir = tmp_path / "awaken"
    project_dir.mkdir()
    _pin_in_sandboxed_home(project_dir, "deepseek-v4-flash", "opencode-go")

    explicit = _startup_route(monkeypatch, project_dir, model="claude-opus-4-8")
    assert explicit.model == "claude-opus-4-8"
    assert explicit.requested_provider == "anthropic"
    assert explicit._project_model_pin is None


# ── startup resolution (oneshot) ────────────────────────────────────────────────────────────


def _oneshot_choice(monkeypatch, cwd, *, model=None, provider=None):
    from hermes_cli.oneshot import _resolve_model_and_provider

    monkeypatch.chdir(cwd)
    return _resolve_model_and_provider({"model": {"default": "profile-default-model", "provider": "anthropic"}},
                                       model, provider)


def test_oneshot_uses_the_project_pin_then_falls_back_to_the_profile_default(tmp_path, monkeypatch):
    project_dir = tmp_path / "awaken"
    project_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _pin_in_sandboxed_home(project_dir, "deepseek-v4-flash", "opencode-go")

    inside = _oneshot_choice(monkeypatch, project_dir)
    assert (inside.model, inside.provider) == ("deepseek-v4-flash", "opencode-go")

    outside = _oneshot_choice(monkeypatch, elsewhere)
    # No pin and nothing explicit: the configured default, with the provider left to resolution.
    assert outside.model == "profile-default-model"
    assert outside.provider is None

    explicit = _oneshot_choice(monkeypatch, project_dir, model="claude-opus-4-8", provider="anthropic")
    assert (explicit.model, explicit.provider) == ("claude-opus-4-8", "anthropic")


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────

def _run(argv):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    p = projects_cmd.build_parser(sub)
    p.set_defaults(func=projects_cmd.projects_command)
    args = parser.parse_args(["project", *argv])
    return projects_cmd.projects_command(args)


def test_cli_set_model_shows_in_show_and_list_then_clears(capsys, tmp_path):
    assert _run(["create", "Pinned", str(tmp_path)]) == 0
    capsys.readouterr()

    assert _run(["set-model", "pinned", "deepseek-v4-flash", "--provider", "opencode-go"]) == 0
    assert "deepseek-v4-flash" in capsys.readouterr().out

    assert _run(["show", "pinned"]) == 0
    shown = capsys.readouterr().out
    assert "model:" in shown and "opencode-go" in shown

    assert _run(["list"]) == 0
    assert "deepseek-v4-flash" in capsys.readouterr().out

    assert _run(["set-model", "pinned", "--clear"]) == 0
    assert "Cleared model" in capsys.readouterr().out

    assert _run(["show", "pinned"]) == 0
    assert "model:" not in capsys.readouterr().out

    with pdb.connect_closing() as conn:
        assert pdb.get_project(conn, "pinned").model is None


def test_cli_provider_without_model_is_an_error(capsys, tmp_path):
    assert _run(["create", "Err", str(tmp_path)]) == 0
    capsys.readouterr()

    assert _run(["set-model", "err", "--provider", "anthropic"]) == 2
    assert "needs a model" in capsys.readouterr().err


def test_cli_bare_set_model_reports_instead_of_wiping(capsys, tmp_path):
    assert _run(["create", "Report", str(tmp_path)]) == 0
    assert _run(["set-model", "report", "m1"]) == 0
    capsys.readouterr()

    assert _run(["set-model", "report"]) == 0
    assert "m1" in capsys.readouterr().out

    with pdb.connect_closing() as conn:
        assert pdb.get_project(conn, "report").model == "m1"
