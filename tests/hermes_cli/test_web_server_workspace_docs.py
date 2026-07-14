from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import web_server

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def client(monkeypatch):
    previous_auth_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        if previous_auth_required is None:
            try:
                delattr(web_server.app.state, "auth_required")
            except AttributeError:
                pass
        else:
            web_server.app.state.auth_required = previous_auth_required


def _with_hermes_home(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return set_hermes_home_override(path)


VALID_DOC = """---
type: hermes-workspace-document
doc_type: runbook
title: Oncall runbook
workspace_id: /tmp/project
created_at: '2026-07-14T00:00:00+00:00'
updated_at: '2026-07-14T00:00:00+00:00'
status: draft
apply_state: unapplied
description: Reusable draft
tags:
  - ops
---
# Body

Some notes.
"""

INVALID_DOC = "# no frontmatter here\n"


def _write_doc(workspace: Path, relative: str, text: str = VALID_DOC) -> Path:
    target = workspace / ".hermes" / "docs" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


class TestWorkspaceDocsList:
    def test_list_returns_only_markdown_sorted_with_metadata(self, client, tmp_path):
        workspace = tmp_path / "project"
        _write_doc(workspace, "a.md")
        _write_doc(workspace, "z.md")
        _write_doc(workspace, "subdir/b.md")
        _write_doc(workspace, "invalid.md", INVALID_DOC)
        (workspace / ".hermes" / "docs" / "note.txt").write_text("not markdown")

        response = client.get("/api/workspace-docs/list", params={"workspaceRoot": str(workspace)})

        assert response.status_code == 200
        documents = response.json()["documents"]
        paths = [doc["path"] for doc in documents]
        assert paths == ["a.md", "invalid.md", "subdir/b.md", "z.md"]

        valid_doc = next(doc for doc in documents if doc["path"] == "a.md")
        assert valid_doc["valid"] is True
        assert valid_doc["title"] == "Oncall runbook"
        assert valid_doc["docType"] == "runbook"
        assert valid_doc["status"] == "draft"
        assert valid_doc["tags"] == ["ops"]

        invalid_doc = next(doc for doc in documents if doc["path"] == "invalid.md")
        assert invalid_doc["valid"] is False
        assert "error" in invalid_doc

    def test_list_missing_docs_root_returns_empty(self, client, tmp_path):
        workspace = tmp_path / "empty-project"
        workspace.mkdir()

        response = client.get("/api/workspace-docs/list", params={"workspaceRoot": str(workspace)})

        assert response.status_code == 200
        assert response.json() == {"documents": []}


class TestWorkspaceDocsRead:
    def test_read_returns_content_metadata_and_path(self, client, tmp_path):
        workspace = tmp_path / "project"
        _write_doc(workspace, "runbooks/oncall.md")

        response = client.get(
            "/api/workspace-docs/read",
            params={"workspaceRoot": str(workspace), "path": "runbooks/oncall.md"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["path"] == "runbooks/oncall.md"
        assert body["frontmatter"]["title"] == "Oncall runbook"
        assert body["frontmatter"]["docType"] == "runbook"
        assert body["frontmatter"]["tags"] == ["ops"]
        assert "# Body" in body["body"]
        assert body["content"] == VALID_DOC

    def test_read_missing_document_returns_404(self, client, tmp_path):
        workspace = tmp_path / "project"
        (workspace / ".hermes" / "docs").mkdir(parents=True)

        response = client.get(
            "/api/workspace-docs/read",
            params={"workspaceRoot": str(workspace), "path": "missing.md"},
        )

        assert response.status_code == 404

    def test_read_invalid_frontmatter_rejected(self, client, tmp_path):
        workspace = tmp_path / "project"
        _write_doc(workspace, "broken.md", INVALID_DOC)

        response = client.get(
            "/api/workspace-docs/read",
            params={"workspaceRoot": str(workspace), "path": "broken.md"},
        )

        assert response.status_code == 422


class TestWorkspaceDocsWrite:
    def test_create_writes_atomically_and_validates_frontmatter(self, client, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()

        response = client.post(
            "/api/workspace-docs",
            json={
                "workspaceRoot": str(workspace),
                "path": "notes/new.md",
                "frontmatter": {"doc_type": "generic-md", "title": "New note"},
                "body": "# New note\n\nHello.",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["frontmatter"]["title"] == "New note"
        assert body["frontmatter"]["docType"] == "generic-md"
        assert body["frontmatter"]["status"] == "draft"

        on_disk = workspace / ".hermes" / "docs" / "notes" / "new.md"
        assert on_disk.is_file()
        content = on_disk.read_text()
        assert "type: hermes-workspace-document" in content
        assert "# New note" in content

    def test_update_existing_document_overwrites_content(self, client, tmp_path):
        workspace = tmp_path / "project"
        _write_doc(workspace, "notes/existing.md")

        response = client.post(
            "/api/workspace-docs",
            json={
                "workspaceRoot": str(workspace),
                "path": "notes/existing.md",
                "frontmatter": {"doc_type": "runbook", "title": "Updated title"},
                "body": "# Updated body",
            },
        )

        assert response.status_code == 200
        on_disk = workspace / ".hermes" / "docs" / "notes" / "existing.md"
        content = on_disk.read_text()
        assert "Updated title" in content
        assert "# Updated body" in content
        assert "Some notes." not in content

    def test_create_rejects_invalid_frontmatter(self, client, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()

        response = client.post(
            "/api/workspace-docs",
            json={
                "workspaceRoot": str(workspace),
                "path": "notes/bad.md",
                "frontmatter": {"title": "Missing doc_type"},
                "body": "content",
            },
        )

        assert response.status_code == 422

    def test_create_rejects_path_traversal(self, client, tmp_path):
        workspace = tmp_path / "project"
        (workspace / ".hermes" / "docs").mkdir(parents=True)

        response = client.post(
            "/api/workspace-docs",
            json={
                "workspaceRoot": str(workspace),
                "path": "../outside.md",
                "frontmatter": {"doc_type": "generic-md", "title": "Escape"},
                "body": "content",
            },
        )

        assert response.status_code == 400

    def test_create_rejects_oversized_body(self, client, tmp_path, monkeypatch):
        from agent import workspace_docs as workspace_docs_module

        monkeypatch.setattr(workspace_docs_module, "WORKSPACE_DOC_MAX_BYTES", 8)
        workspace = tmp_path / "project"
        workspace.mkdir()

        response = client.post(
            "/api/workspace-docs",
            json={
                "workspaceRoot": str(workspace),
                "path": "notes/huge.md",
                "frontmatter": {"doc_type": "generic-md", "title": "Huge"},
                "body": "way more than eight bytes of body content",
            },
        )

        assert response.status_code == 422


class TestWorkspaceDocsReadPathTraversal:
    def test_read_rejects_path_traversal(self, client, tmp_path):
        workspace = tmp_path / "project"
        (workspace / ".hermes" / "docs").mkdir(parents=True)

        response = client.get(
            "/api/workspace-docs/read",
            params={"workspaceRoot": str(workspace), "path": "../../etc/passwd"},
        )

        assert response.status_code == 400


class TestWorkspaceDocsArchive:
    def test_archive_marks_doc_archived_and_list_read_reflect_state(self, client, tmp_path):
        workspace = tmp_path / "project"
        _write_doc(workspace, "runbooks/oncall.md")

        archive_response = client.post(
            "/api/workspace-docs/archive",
            json={"workspaceRoot": str(workspace), "path": "runbooks/oncall.md"},
        )

        assert archive_response.status_code == 200
        assert archive_response.json()["frontmatter"]["status"] == "archived"

        read_response = client.get(
            "/api/workspace-docs/read",
            params={"workspaceRoot": str(workspace), "path": "runbooks/oncall.md"},
        )
        assert read_response.json()["frontmatter"]["status"] == "archived"

        list_response = client.get("/api/workspace-docs/list", params={"workspaceRoot": str(workspace)})
        listed = next(doc for doc in list_response.json()["documents"] if doc["path"] == "runbooks/oncall.md")
        assert listed["status"] == "archived"

    def test_archive_is_idempotent(self, client, tmp_path):
        workspace = tmp_path / "project"
        _write_doc(workspace, "runbooks/oncall.md")

        first = client.post(
            "/api/workspace-docs/archive",
            json={"workspaceRoot": str(workspace), "path": "runbooks/oncall.md"},
        )
        second = client.post(
            "/api/workspace-docs/archive",
            json={"workspaceRoot": str(workspace), "path": "runbooks/oncall.md"},
        )

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["frontmatter"]["status"] == "archived"

    def test_archive_missing_document_returns_404(self, client, tmp_path):
        workspace = tmp_path / "project"
        (workspace / ".hermes" / "docs").mkdir(parents=True)

        response = client.post(
            "/api/workspace-docs/archive",
            json={"workspaceRoot": str(workspace), "path": "missing.md"},
        )

        assert response.status_code == 404


class TestWorkspaceDocsSafety:
    def test_workspace_inside_hermes_home_is_accepted(self, client, tmp_path):
        hermes_home = tmp_path / ".hermes"
        workspace = hermes_home / "hermes-agent"
        _write_doc(workspace, "notes/plan.md")

        token = _with_hermes_home(hermes_home)
        try:
            response = client.get("/api/workspace-docs/list", params={"workspaceRoot": str(workspace)})
        finally:
            reset_hermes_home_override(token)

        assert response.status_code == 200
        assert response.json()["documents"][0]["path"] == "notes/plan.md"

    def test_cross_profile_workspace_rejected(self, client, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        other_profile_workspace = hermes_home / "profiles" / "other" / "skills" / "draft-repo"
        _write_doc(other_profile_workspace, "x.md")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        token = _with_hermes_home(hermes_home)
        try:
            response = client.get(
                "/api/workspace-docs/list",
                params={"workspaceRoot": str(other_profile_workspace)},
            )
        finally:
            reset_hermes_home_override(token)

        assert response.status_code == 403

    def test_sandbox_mirror_workspace_rejected(self, client, tmp_path):
        workspace = tmp_path / "profiles" / "group1" / "sandboxes" / "docker" / "default" / "home"
        _write_doc(workspace, "x.md")

        response = client.get("/api/workspace-docs/list", params={"workspaceRoot": str(workspace)})

        assert response.status_code == 403

    def test_container_mirror_safety_error_maps_to_403(self, client, tmp_path, monkeypatch):
        # The REST layer has no active container/sandbox task context to derive
        # a mirror prefix from (unlike the in-process file-editing tool), so
        # this slice never triggers classify_container_mirror_target end-to-end.
        # Verify the endpoint's error-mapping still turns that classifier into a
        # 403 by simulating the safety error directly.
        from agent import workspace_docs as workspace_docs_module

        def _raise_container_mirror(*args, **kwargs):
            raise workspace_docs_module.WorkspaceDocSafetyError(
                "container-mirror target", classifier="container_mirror"
            )

        monkeypatch.setattr(workspace_docs_module, "list_workspace_docs", _raise_container_mirror)
        monkeypatch.setattr(web_server._workspace_docs, "list_workspace_docs", _raise_container_mirror)

        response = client.get("/api/workspace-docs/list", params={"workspaceRoot": str(tmp_path)})

        assert response.status_code == 403


class TestWorkspaceDocsEndpointsRequireAuth:
    def test_workspace_docs_endpoints_require_auth(self, tmp_path):
        test_client = TestClient(web_server.app)
        workspace = tmp_path / "project"
        _write_doc(workspace, "a.md")

        list_response = test_client.get(
            "/api/workspace-docs/list", params={"workspaceRoot": str(workspace)}
        )
        read_response = test_client.get(
            "/api/workspace-docs/read",
            params={"workspaceRoot": str(workspace), "path": "a.md"},
        )
        write_response = test_client.post(
            "/api/workspace-docs",
            json={
                "workspaceRoot": str(workspace),
                "path": "a.md",
                "frontmatter": {"doc_type": "generic-md", "title": "X"},
                "body": "x",
            },
        )
        archive_response = test_client.post(
            "/api/workspace-docs/archive",
            json={"workspaceRoot": str(workspace), "path": "a.md"},
        )

        assert list_response.status_code == 401
        assert read_response.status_code == 401
        assert write_response.status_code == 401
        assert archive_response.status_code == 401
