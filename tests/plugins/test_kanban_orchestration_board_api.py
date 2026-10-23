"""Board-scoped orchestration API: ``/orchestration?board=<slug>``.

The two profile knobs (``orchestrator_profile`` / ``default_assignee``) gain a
board-first read and a board-scoped write under ``kanban.boards.<slug>``; an
explicit empty string clears the board override so it inherits the global. The
response adds ``board`` / ``board_*`` fields alongside the unchanged global
fields so a client can render "overridden here" vs "inherited".
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_orch_test", plugin_file,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return hermes_home


@pytest.fixture
def client(home, monkeypatch):
    from hermes_cli import profiles as profiles_mod

    monkeypatch.setattr(
        profiles_mod, "profile_exists", lambda n: n in {"default", "planner", "worker"},
    )
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _write_config(home, text):
    (home / "config.yaml").write_text(text, encoding="utf-8")


def _read_config(home):
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}


def _make_board(slug):
    kb.create_board(slug=slug, name=slug)


def test_get_board_inherits_global(client, home):
    _make_board("tsa-mgmt")
    _write_config(home, "kanban:\n  default_assignee: worker\n")
    r = client.get("/api/plugins/kanban/orchestration?board=tsa-mgmt")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["board"] == "tsa-mgmt"
    assert data["default_assignee"] == "worker"  # inherited from global
    assert data["board_default_assignee"] == ""  # no board override


def test_get_board_unknown_override_falls_through_to_global(client, home, monkeypatch):
    """Review #135651: a board override naming an unknown profile must not
    swallow a valid global when resolving the effective value."""
    from hermes_cli import profiles as profiles_mod
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")
    _make_board("tsa-mgmt")
    _write_config(
        home,
        "kanban:\n  default_assignee: worker\n"
        "  boards:\n    tsa-mgmt:\n      default_assignee: ghost\n",
    )
    data = client.get("/api/plugins/kanban/orchestration?board=tsa-mgmt").json()
    # Raw fields keep the pre-fix semantics (board wins the merge verbatim).
    assert data["board_default_assignee"] == "ghost"
    assert data["default_assignee"] == "ghost"
    # The resolved value falls through to the valid global.
    assert data["resolved_default_assignee"] == "worker"


def test_get_board_and_global_unknown_resolves_to_active(client, home, monkeypatch):
    from hermes_cli import profiles as profiles_mod
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")
    _make_board("tsa-mgmt")
    _write_config(
        home,
        "kanban:\n  default_assignee: ghost-global\n"
        "  boards:\n    tsa-mgmt:\n      default_assignee: ghost-board\n",
    )
    data = client.get("/api/plugins/kanban/orchestration?board=tsa-mgmt").json()
    assert data["resolved_default_assignee"] == "default"


def test_get_pin_collapsed_board_suppresses_override(client, home, monkeypatch):
    """With HERMES_KANBAN_DB pinned to this board's own db the boards are
    indistinguishable (design R1): the GET must report no board override and
    resolve through the global, matching the dispatcher/decomposer readers."""
    from hermes_cli import kanban_db
    from hermes_cli import profiles as profiles_mod
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")
    _make_board("tsa-mgmt")
    # Point the pin at this board's own kanban.db, the pin-collapse topology.
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_db.kanban_db_path(board="tsa-mgmt")))
    _write_config(
        home,
        "kanban:\n  default_assignee: worker\n"
        "  boards:\n    tsa-mgmt:\n      default_assignee: planner\n",
    )
    data = client.get("/api/plugins/kanban/orchestration?board=tsa-mgmt").json()
    assert data["board"] == "tsa-mgmt"  # requested slug still echoed
    assert data["board_default_assignee"] == ""  # override suppressed
    assert data["resolved_default_assignee"] == "worker"  # global instead


def test_put_board_writes_board_scope(client, home):
    _make_board("tsa-mgmt")
    r = client.put(
        "/api/plugins/kanban/orchestration?board=tsa-mgmt",
        json={"orchestrator_profile": "planner", "default_assignee": "worker"},
    )
    assert r.status_code == 200, r.text
    cfg = _read_config(home)
    assert cfg["kanban"]["boards"]["tsa-mgmt"] == {
        "orchestrator_profile": "planner", "default_assignee": "worker",
    }
    data = r.json()
    assert data["board"] == "tsa-mgmt"
    assert data["board_orchestrator_profile"] == "planner"
    assert data["orchestrator_profile"] == "planner"


def test_put_board_empty_string_clears_override(client, home):
    _make_board("tsa-mgmt")
    _write_config(home, "kanban:\n  boards:\n    tsa-mgmt:\n      default_assignee: worker\n")
    r = client.put(
        "/api/plugins/kanban/orchestration?board=tsa-mgmt",
        json={"default_assignee": ""},
    )
    assert r.status_code == 200, r.text
    cfg = _read_config(home)
    assert "default_assignee" not in (cfg["kanban"]["boards"].get("tsa-mgmt") or {})
    assert r.json()["board_default_assignee"] == ""


def test_put_board_unknown_profile_is_400(client):
    _make_board("tsa-mgmt")
    r = client.put(
        "/api/plugins/kanban/orchestration?board=tsa-mgmt",
        json={"default_assignee": "ghost"},
    )
    assert r.status_code == 400


def test_put_board_reuses_existing_case_variant_slug(client, home):
    """A hand-written ``TSA-Mgmt`` config key must be reused, not duplicated by
    a new lowercase sibling that would read back as a different board."""
    _make_board("tsa-mgmt")
    _write_config(home, "kanban:\n  boards:\n    TSA-Mgmt:\n      default_assignee: planner\n")
    r = client.put(
        "/api/plugins/kanban/orchestration?board=tsa-mgmt",
        json={"default_assignee": "worker"},
    )
    assert r.status_code == 200, r.text
    boards = _read_config(home)["kanban"]["boards"]
    assert list(boards) == ["TSA-Mgmt"]  # no duplicate slug key
    assert boards["TSA-Mgmt"]["default_assignee"] == "worker"
    assert r.json()["board_default_assignee"] == "worker"


def test_put_board_canonicalizes_setting_key_variant(client, home):
    """An entry holding ``Default_Assignee`` is replaced by the canonical key so
    the entry cannot accumulate two spellings."""
    _make_board("tsa-mgmt")
    _write_config(home, "kanban:\n  boards:\n    tsa-mgmt:\n      Default_Assignee: planner\n")
    r = client.put(
        "/api/plugins/kanban/orchestration?board=tsa-mgmt",
        json={"default_assignee": "worker"},
    )
    assert r.status_code == 200, r.text
    entry = _read_config(home)["kanban"]["boards"]["tsa-mgmt"]
    assert entry == {"default_assignee": "worker"}


def test_put_board_clear_removes_case_variants(client, home):
    _make_board("tsa-mgmt")
    _write_config(home, "kanban:\n  boards:\n    TSA-Mgmt:\n      Default_Assignee: planner\n")
    r = client.put(
        "/api/plugins/kanban/orchestration?board=tsa-mgmt",
        json={"default_assignee": ""},
    )
    assert r.status_code == 200, r.text
    entry = _read_config(home)["kanban"]["boards"]["TSA-Mgmt"]
    assert "Default_Assignee" not in entry
    assert "default_assignee" not in entry
    assert r.json()["board_default_assignee"] == ""


def test_put_board_rebuilds_non_dict_entry(client, home):
    _make_board("tsa-mgmt")
    _write_config(home, "kanban:\n  boards:\n    tsa-mgmt: not-a-mapping\n")
    r = client.put(
        "/api/plugins/kanban/orchestration?board=tsa-mgmt",
        json={"default_assignee": "worker"},
    )
    assert r.status_code == 200, r.text
    assert _read_config(home)["kanban"]["boards"]["tsa-mgmt"] == {"default_assignee": "worker"}


def test_unknown_board_is_404(client):
    assert client.get("/api/plugins/kanban/orchestration?board=ghost").status_code == 404
    r = client.put(
        "/api/plugins/kanban/orchestration?board=ghost",
        json={"default_assignee": "default"},
    )
    assert r.status_code == 404


def test_put_without_board_writes_global(client, home):
    r = client.put("/api/plugins/kanban/orchestration", json={"default_assignee": "worker"})
    assert r.status_code == 200, r.text
    cfg = _read_config(home)
    assert cfg["kanban"]["default_assignee"] == "worker"
    assert not cfg["kanban"].get("boards")


def test_board_scope_does_not_write_global_booleans(client, home):
    _make_board("tsa-mgmt")
    r = client.put(
        "/api/plugins/kanban/orchestration?board=tsa-mgmt",
        json={"auto_decompose": False},
    )
    assert r.status_code == 200, r.text
    cfg = _read_config(home)
    assert cfg.get("kanban", {}).get("auto_decompose") is None
    assert cfg.get("kanban", {}).get("boards", {}).get("tsa-mgmt") in (None, {})
