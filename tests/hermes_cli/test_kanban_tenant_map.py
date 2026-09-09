"""W1 (2026-09-06): a tenant's code card is a worktree under the tenant's repo
whoever mints it, via the fleet-level tenant map — never an empty scratch dir."""
from __future__ import annotations

import json
import os

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_conn(tmp_path):
    c = kbc.connect(db_path=tmp_path / "kanban.db")
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def tenant_map(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    m = tmp_path / "kanban-tenants.json"
    m.write_text(json.dumps({"acme": {"id": "p_acme", "slug": "acme", "name": "Acme",
                                      "primary_path": str(repo)}}))
    monkeypatch.setenv("HERMES_KANBAN_TENANTS", str(m))
    return repo


def test_tenant_scratch_becomes_worktree_under_tenant_repo(kanban_conn, tenant_map):
    tid = kb.create_task(kanban_conn, title="Add search", tenant="acme")
    t = kb.get_task(kanban_conn, tid)
    assert t.workspace_kind == "worktree"
    assert t.workspace_path == os.path.join(str(tenant_map), ".worktrees", tid)
    assert t.project_id == "p_acme"
    assert t.branch_name and t.branch_name.startswith("acme/")


def test_explicit_dir_is_left_alone(kanban_conn, tenant_map, tmp_path):
    d = tmp_path / "shared"
    d.mkdir()
    tid = kb.create_task(kanban_conn, title="x", tenant="acme",
                         workspace_kind="dir", workspace_path=str(d))
    t = kb.get_task(kanban_conn, tid)
    assert t.workspace_kind == "dir" and t.workspace_path == str(d)


def test_unknown_tenant_stays_scratch(kanban_conn, tenant_map):
    tid = kb.create_task(kanban_conn, title="x", tenant="nobody")
    assert kb.get_task(kanban_conn, tid).workspace_kind == "scratch"


def test_negative_control_no_map_means_old_behaviour(kanban_conn, tmp_path, monkeypatch):
    # The control: with the map absent the card is scratch, proving the
    # first test measured the map and not a coincidence.
    monkeypatch.setenv("HERMES_KANBAN_TENANTS", str(tmp_path / "missing.json"))
    tid = kb.create_task(kanban_conn, title="x", tenant="acme")
    assert kb.get_task(kanban_conn, tid).workspace_kind == "scratch"


def test_malformed_map_fails_open(kanban_conn, tmp_path, monkeypatch):
    m = tmp_path / "bad.json"
    m.write_text("{not json")
    monkeypatch.setenv("HERMES_KANBAN_TENANTS", str(m))
    tid = kb.create_task(kanban_conn, title="x", tenant="acme")
    assert kb.get_task(kanban_conn, tid).workspace_kind == "scratch"
