"""Explicit session -> project binding on the API server (``sessions.project_id``).

POST /api/sessions binds a project atomically; GET/list/fork carry the ``project`` object; the
list filters by project; /v1/capabilities advertises ``session_projects``.
"""

import os

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import projects_db as pdb
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "otto-repo"
    folder.mkdir()
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(conn, name="Otto", slug="otto", folders=[str(folder)], icon="sparkles",
                                 color="#ff8800", board_slug="otto")
        return pdb.get_project(conn, pid)


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
    app.router.add_patch("/api/sessions/{session_id}", adapter._handle_patch_session)
    app.router.add_post("/api/sessions/{session_id}/fork", adapter._handle_fork_session)
    return app


def _expected_ref(project) -> dict:
    return {"id": project.id, "slug": "otto", "name": "Otto", "icon": "sparkles", "color": "#ff8800",
            "folder": project.primary_path, "board_slug": "otto"}


@pytest.mark.asyncio
async def test_capabilities_advertise_session_projects(adapter):
    async with TestClient(TestServer(_app(adapter))) as cli:
        data = await (await cli.get("/v1/capabilities")).json()
    assert data["features"]["session_projects"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["project_id", "project"])
async def test_create_binds_project_and_its_folder_in_one_write(adapter, session_db, project, field):
    ref = project.id if field == "project_id" else project.slug
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/api/sessions", json={"id": "s-proj", "title": "Hi", field: ref})
        assert resp.status == 201, await resp.text()
        session = (await resp.json())["session"]

    assert session["project_id"] == project.id
    assert session["project"] == _expected_ref(project)
    row = session_db.get_session("s-proj")
    assert row["project_id"] == project.id
    assert row["cwd"] == project.primary_path
    assert row["title"] == "Hi"


@pytest.mark.asyncio
async def test_create_records_the_project_folder_git_identity(adapter, session_db, project, monkeypatch):
    from tui_gateway import git_probe
    monkeypatch.setattr(git_probe, "branch", lambda cwd: "main" if cwd == project.primary_path else "")
    monkeypatch.setattr(git_probe, "common_repo_root", lambda cwd: cwd if cwd == project.primary_path else "")
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/api/sessions", json={"id": "s-git", "project_id": project.id})
        assert resp.status == 201
    row = session_db.get_session("s-git")
    assert (row["git_branch"], row["git_repo_root"]) == ("main", project.primary_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    {"project_id": "p_missing"},
    {"project": "nope"},
    {"project_id": ""},
    {"project_id": 7},
])
async def test_create_refuses_unknown_project_without_writing_a_row(adapter, session_db, project, body):
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/api/sessions", json={"id": "s-bad", **body})
        payload = await resp.json()
    assert resp.status == 400
    assert payload["error"]["code"] == "invalid_project"
    assert session_db.get_session("s-bad") is None


@pytest.mark.asyncio
async def test_create_refuses_archived_project(adapter, session_db, project):
    with pdb.connect_closing() as conn:
        pdb.archive_project(conn, project.id)
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/api/sessions", json={"id": "s-arch", "project_id": project.id})
        payload = await resp.json()
    assert resp.status == 400
    assert payload["error"]["code"] == "invalid_project"
    assert session_db.get_session("s-arch") is None


@pytest.mark.asyncio
async def test_create_refuses_contradicting_id_and_slug(adapter, session_db, project, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    with pdb.connect_closing() as conn:
        pdb.create_project(conn, name="Other", slug="other", folders=[str(other)])
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/api/sessions", json={"id": "s-x", "project_id": project.id, "project": "other"})
    assert resp.status == 400
    assert session_db.get_session("s-x") is None


@pytest.mark.asyncio
async def test_unbound_session_has_null_project(adapter):
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/api/sessions", json={"id": "s-plain"})
        session = (await resp.json())["session"]
        fetched = (await (await cli.get("/api/sessions/s-plain")).json())["session"]
    assert session["project"] is None
    assert fetched["project"] is None


@pytest.mark.asyncio
async def test_get_list_and_patch_return_project(adapter, session_db, project):
    async with TestClient(TestServer(_app(adapter))) as cli:
        await cli.post("/api/sessions", json={"id": "s-bound", "project_id": project.id})
        session_db.append_message("s-bound", "user", "hello")
        fetched = (await (await cli.get("/api/sessions/s-bound")).json())["session"]
        listed = (await (await cli.get("/api/sessions")).json())["data"]
        patched = (await (await cli.patch("/api/sessions/s-bound", json={"pinned": True})).json())["session"]
    assert fetched["project"] == _expected_ref(project)
    assert [s["project"] for s in listed if s["id"] == "s-bound"] == [_expected_ref(project)]
    assert patched["project"] == _expected_ref(project)


@pytest.mark.asyncio
@pytest.mark.parametrize("by", ["id", "slug"])
async def test_list_filters_by_project(adapter, session_db, project, by):
    async with TestClient(TestServer(_app(adapter))) as cli:
        await cli.post("/api/sessions", json={"id": "s-in", "project_id": project.id})
        await cli.post("/api/sessions", json={"id": "s-out"})
        for sid in ("s-in", "s-out"):
            session_db.append_message(sid, "user", "hello")
        ref = project.id if by == "id" else project.slug
        filtered = (await (await cli.get(f"/api/sessions?project_id={ref}")).json())["data"]
        unknown = (await (await cli.get("/api/sessions?project_id=p_nobody")).json())["data"]
        everything = (await (await cli.get("/api/sessions")).json())["data"]
    assert [s["id"] for s in filtered] == ["s-in"]
    assert unknown == []
    assert {s["id"] for s in everything} >= {"s-in", "s-out"}


@pytest.mark.asyncio
async def test_fork_inherits_project(adapter, session_db, project):
    async with TestClient(TestServer(_app(adapter))) as cli:
        await cli.post("/api/sessions", json={"id": "s-src", "project_id": project.id, "title": "Source"})
        session_db.append_message("s-src", "user", "hello")
        resp = await cli.post("/api/sessions/s-src/fork", json={"id": "s-fork"})
        assert resp.status == 201
        fork = (await resp.json())["session"]
    assert fork["project"] == _expected_ref(project)
    row = session_db.get_session("s-fork")
    assert (row["project_id"], row["cwd"]) == (project.id, project.primary_path)


def test_project_folder_survives_a_missing_directory(project, tmp_path):
    """A project whose folder is gone on this host still binds (path kept, no git identity)."""
    from gateway.platforms.api_server_session_projects import project_workspace
    os.rmdir(project.primary_path)
    assert project_workspace(project) == {
        "project_id": project.id, "cwd": project.primary_path, "git_branch": None, "git_repo_root": None}
