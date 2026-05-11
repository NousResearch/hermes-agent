"""Tests for the Hermes Docs dashboard plugin backend.

Mirrors the pattern established by tests/plugins/test_kanban_dashboard_plugin.py:
  - Load plugin_api.py dynamically with importlib so tests can run without
    the full dashboard startup.
  - Isolate HERMES_HOME per test via the monkeypatch fixture.
  - Mount the router onto a bare FastAPI instance and use TestClient.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Loader + fixtures
# ---------------------------------------------------------------------------


def _load_plugin_router():
    """Dynamically load plugins/hermes-docs/dashboard/plugin_api.py."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = (
        repo_root / "plugins" / "hermes-docs" / "dashboard" / "plugin_api.py"
    )
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    mod_name = "hermes_dashboard_plugin_hermes_docs_test"
    # Re-load on each test collection — avoid stale cached module between runs
    sys.modules.pop(mod_name, None)

    spec = importlib.util.spec_from_file_location(mod_name, plugin_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def docs_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME pointing to a temp directory."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


@pytest.fixture
def client(docs_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/hermes-docs")
    return TestClient(app)


@pytest.fixture
def sample_folder(tmp_path):
    """A real local folder to register as a workspace."""
    folder = tmp_path / "my-docs"
    folder.mkdir()
    (folder / "README.md").write_text("# Hello\n\nThis is a test workspace.", encoding="utf-8")
    (folder / "notes.md").write_text("## Notes\n\nSome notes here.", encoding="utf-8")
    subdir = folder / "sub"
    subdir.mkdir()
    (subdir / "child.md").write_text("child file", encoding="utf-8")
    return folder


# ---------------------------------------------------------------------------
# GET /status — available with empty workspace list
# ---------------------------------------------------------------------------


def test_status_empty(client):
    r = client.get("/api/plugins/hermes-docs/status")
    assert r.status_code == 200
    data = r.json()
    assert data["available"] is True
    assert data["workspace_count"] == 0
    assert data["recent"] == []


# ---------------------------------------------------------------------------
# Workspace registry: create / list / remove
# ---------------------------------------------------------------------------


def test_create_workspace(client, sample_folder):
    r = client.post("/api/plugins/hermes-docs/workspaces", json={
        "name": "My Docs",
        "path": str(sample_folder),
    })
    assert r.status_code == 201
    ws = r.json()
    assert ws["name"] == "My Docs"
    assert ws["path"] == str(sample_folder.resolve())
    assert "id" in ws
    assert "created_at" in ws


def test_create_workspace_defaults_name_to_folder(client, sample_folder):
    r = client.post("/api/plugins/hermes-docs/workspaces", json={
        "name": "",
        "path": str(sample_folder),
    })
    assert r.status_code == 201
    assert r.json()["name"] == sample_folder.name


def test_create_workspace_missing_folder(client, tmp_path):
    r = client.post("/api/plugins/hermes-docs/workspaces", json={
        "name": "Ghost",
        "path": str(tmp_path / "nonexistent"),
    })
    assert r.status_code == 400


def test_create_workspace_duplicate(client, sample_folder):
    client.post("/api/plugins/hermes-docs/workspaces", json={"name": "A", "path": str(sample_folder)})
    r = client.post("/api/plugins/hermes-docs/workspaces", json={"name": "B", "path": str(sample_folder)})
    assert r.status_code == 409


def test_list_workspaces(client, sample_folder):
    client.post("/api/plugins/hermes-docs/workspaces", json={"name": "WS", "path": str(sample_folder)})
    r = client.get("/api/plugins/hermes-docs/workspaces")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 1
    assert items[0]["folder_exists"] is True


def test_remove_workspace(client, sample_folder):
    create = client.post("/api/plugins/hermes-docs/workspaces", json={"name": "WS", "path": str(sample_folder)})
    ws_id = create.json()["id"]

    r = client.delete(f"/api/plugins/hermes-docs/workspaces/{ws_id}")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

    r2 = client.get("/api/plugins/hermes-docs/workspaces")
    assert r2.json() == []


def test_remove_workspace_not_found(client):
    r = client.delete("/api/plugins/hermes-docs/workspaces/does-not-exist")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Storage isolation — metadata stays in HERMES_HOME, not workspace folder
# ---------------------------------------------------------------------------


def test_metadata_stored_in_hermes_home_not_workspace(client, docs_home, sample_folder):
    r = client.post("/api/plugins/hermes-docs/workspaces", json={"name": "WS", "path": str(sample_folder)})
    ws_id = r.json()["id"]

    # Registry lives under hermes home
    registry = docs_home / "docs-workspaces" / "registry.json"
    assert registry.exists()
    data = json.loads(registry.read_text())
    assert len(data) == 1

    # Per-workspace metadata dir exists under hermes home
    ws_meta = docs_home / "docs-workspaces" / ws_id
    assert ws_meta.is_dir()
    assert (ws_meta / "preferences.json").exists()

    # Source folder is untouched (no hidden .hermes dir injected)
    assert not (sample_folder / ".hermes").exists()


# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------


def test_list_files_root(client, sample_folder):
    r = client.post("/api/plugins/hermes-docs/workspaces", json={"name": "WS", "path": str(sample_folder)})
    ws_id = r.json()["id"]

    r2 = client.get(f"/api/plugins/hermes-docs/workspaces/{ws_id}/files")
    assert r2.status_code == 200
    names = [e["name"] for e in r2.json()]
    assert "README.md" in names
    assert "notes.md" in names
    assert "sub" in names


def test_list_files_hidden_entries_excluded(client, sample_folder, tmp_path):
    """Dot-prefixed files/dirs must not appear in the listing."""
    hidden = sample_folder / ".hidden"
    hidden.write_text("secret")
    r = client.post("/api/plugins/hermes-docs/workspaces", json={"name": "WS", "path": str(sample_folder)})
    ws_id = r.json()["id"]
    r2 = client.get(f"/api/plugins/hermes-docs/workspaces/{ws_id}/files")
    names = [e["name"] for e in r2.json()]
    assert ".hidden" not in names


def test_list_files_path_traversal_blocked(client, sample_folder):
    r = client.post("/api/plugins/hermes-docs/workspaces", json={"name": "WS", "path": str(sample_folder)})
    ws_id = r.json()["id"]
    r2 = client.get(f"/api/plugins/hermes-docs/workspaces/{ws_id}/files?rel=../../etc")
    assert r2.status_code == 403


# ---------------------------------------------------------------------------
# File read / write / preview
# ---------------------------------------------------------------------------


def test_read_file(client, sample_folder):
    r = client.post("/api/plugins/hermes-docs/workspaces", json={"name": "WS", "path": str(sample_folder)})
    ws_id = r.json()["id"]
    r2 = client.get(f"/api/plugins/hermes-docs/workspaces/{ws_id}/file?rel=README.md")
    assert r2.status_code == 200
    assert "Hello" in r2.json()["content"]


def test_read_file_not_found(client, sample_folder):
    r = client.post("/api/plugins/hermes-docs/workspaces", json={"name": "WS", "path": str(sample_folder)})
    ws_id = r.json()["id"]
    r2 = client.get(f"/api/plugins/hermes-docs/workspaces/{ws_id}/file?rel=nope.md")
    assert r2.status_code == 404


def test_read_file_sibling_prefix_escape_blocked(client, sample_folder):
    """A sibling path whose prefix matches the workspace name must still be blocked."""
    sibling = sample_folder.parent / f"{sample_folder.name}-evil"
    sibling.mkdir()
    (sibling / "secret.md").write_text("secret", encoding="utf-8")

    r = client.post("/api/plugins/hermes-docs/workspaces", json={"name": "WS", "path": str(sample_folder)})
    ws_id = r.json()["id"]
    r2 = client.get(
        f"/api/plugins/hermes-docs/workspaces/{ws_id}/file"
        f"?rel=../{sibling.name}/secret.md"
    )
    assert r2.status_code == 403


def test_write_file_preview(client, sample_folder):
    r = client.post("/api/plugins/hermes-docs/workspaces", json={"name": "WS", "path": str(sample_folder)})
    ws_id = r.json()["id"]
    r2 = client.put(
        f"/api/plugins/hermes-docs/workspaces/{ws_id}/file?rel=README.md",
        json={"content": "# New content", "preview": True},
    )
    assert r2.status_code == 200
    body = r2.json()
    assert body["preview"] is True
    assert "proposed" in body
    # Source file must NOT be modified when preview=True
    assert "Hello" in (sample_folder / "README.md").read_text()


def test_write_file_actual(client, sample_folder):
    r = client.post("/api/plugins/hermes-docs/workspaces", json={"name": "WS", "path": str(sample_folder)})
    ws_id = r.json()["id"]
    client.put(
        f"/api/plugins/hermes-docs/workspaces/{ws_id}/file?rel=README.md",
        json={"content": "# Updated", "preview": False},
    )
    assert (sample_folder / "README.md").read_text() == "# Updated"


def test_write_file_path_traversal_blocked(client, sample_folder):
    r = client.post("/api/plugins/hermes-docs/workspaces", json={"name": "WS", "path": str(sample_folder)})
    ws_id = r.json()["id"]
    r2 = client.put(
        f"/api/plugins/hermes-docs/workspaces/{ws_id}/file?rel=../../evil.sh",
        json={"content": "rm -rf /", "preview": False},
    )
    assert r2.status_code == 403


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------


def _make_ws(client, sample_folder):
    r = client.post("/api/plugins/hermes-docs/workspaces", json={"name": "WS", "path": str(sample_folder)})
    return r.json()["id"]


def test_create_comment(client, sample_folder):
    ws_id = _make_ws(client, sample_folder)
    r = client.post(f"/api/plugins/hermes-docs/workspaces/{ws_id}/comments", json={
        "document": "README.md",
        "text": "Clarify this sentence.",
        "anchor_start": 5,
        "anchor_end": 20,
        "anchor_text": "Hello",
    })
    assert r.status_code == 201
    c = r.json()
    assert c["text"] == "Clarify this sentence."
    assert c["resolved"] is False


def test_list_comments(client, sample_folder):
    ws_id = _make_ws(client, sample_folder)
    client.post(f"/api/plugins/hermes-docs/workspaces/{ws_id}/comments", json={
        "document": "README.md", "text": "First", "anchor_start": 0, "anchor_end": 1, "anchor_text": "H",
    })
    client.post(f"/api/plugins/hermes-docs/workspaces/{ws_id}/comments", json={
        "document": "notes.md", "text": "Second", "anchor_start": 0, "anchor_end": 1, "anchor_text": "H",
    })
    r = client.get(f"/api/plugins/hermes-docs/workspaces/{ws_id}/comments")
    assert len(r.json()) == 2


def test_list_comments_filtered_by_document(client, sample_folder):
    ws_id = _make_ws(client, sample_folder)
    client.post(f"/api/plugins/hermes-docs/workspaces/{ws_id}/comments", json={
        "document": "README.md", "text": "A", "anchor_start": 0, "anchor_end": 1, "anchor_text": "x",
    })
    client.post(f"/api/plugins/hermes-docs/workspaces/{ws_id}/comments", json={
        "document": "notes.md", "text": "B", "anchor_start": 0, "anchor_end": 1, "anchor_text": "x",
    })
    r = client.get(f"/api/plugins/hermes-docs/workspaces/{ws_id}/comments?document=README.md")
    items = r.json()
    assert len(items) == 1
    assert items[0]["document"] == "README.md"


def test_resolve_comment(client, sample_folder):
    ws_id = _make_ws(client, sample_folder)
    c = client.post(f"/api/plugins/hermes-docs/workspaces/{ws_id}/comments", json={
        "document": "README.md", "text": "Fix", "anchor_start": 0, "anchor_end": 1, "anchor_text": "H",
    }).json()
    r = client.patch(
        f"/api/plugins/hermes-docs/workspaces/{ws_id}/comments/{c['id']}",
        json={"resolved": True},
    )
    assert r.status_code == 200
    assert r.json()["resolved"] is True
    assert r.json()["resolved_at"] is not None


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------


def test_preferences_defaults(client, sample_folder):
    ws_id = _make_ws(client, sample_folder)
    r = client.get(f"/api/plugins/hermes-docs/workspaces/{ws_id}/preferences")
    assert r.status_code == 200
    assert r.json()["drawer_pinned"] is False


def test_update_preferences(client, sample_folder):
    ws_id = _make_ws(client, sample_folder)
    client.put(
        f"/api/plugins/hermes-docs/workspaces/{ws_id}/preferences",
        json={"drawer_pinned": True},
    )
    r = client.get(f"/api/plugins/hermes-docs/workspaces/{ws_id}/preferences")
    assert r.json()["drawer_pinned"] is True


# ---------------------------------------------------------------------------
# Side chat stub (broker unavailable — default)
# ---------------------------------------------------------------------------


def test_sidechat_stub(client, sample_folder):
    ws_id = _make_ws(client, sample_folder)
    r = client.post(f"/api/plugins/hermes-docs/workspaces/{ws_id}/sidechat", json={
        "content": "Summarise this document.",
        "document": "README.md",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "assistant"
    assert body["content"]
    assert body["context"]["workspace"] == "WS"
    # No broker configured → brokered must be False and content must mention docs persona
    assert body["brokered"] is False
    assert "docs" in body["content"].lower() or "persona" in body["content"].lower()


def test_sidechat_persists_session_entry(client, docs_home, sample_folder):
    ws_id = _make_ws(client, sample_folder)
    client.post(f"/api/plugins/hermes-docs/workspaces/{ws_id}/sidechat", json={
        "content": "Hello",
    })
    sessions_dir = docs_home / "docs-workspaces" / ws_id / "sessions"
    assert sessions_dir.exists()
    entries = list(sessions_dir.glob("*.json"))
    assert len(entries) == 1
    data = json.loads(entries[0].read_text())
    assert data["user"] == "Hello"
    # Response is persisted too
    assert "assistant" in data
    assert "brokered" in data


# ---------------------------------------------------------------------------
# Side chat — broker available path
# ---------------------------------------------------------------------------


def _load_plugin_module():
    """Return the loaded plugin module (not just the router) for override access."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = (
        repo_root / "plugins" / "hermes-docs" / "dashboard" / "plugin_api.py"
    )
    mod_name = "hermes_dashboard_plugin_hermes_docs_test_mod"
    import importlib.util
    import sys
    sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location(mod_name, plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_sidechat_brokered_when_override_set(docs_home, sample_folder):
    """When _broker_override is set, sidechat response must come from it."""
    mod = _load_plugin_module()

    original = mod._broker_override
    try:
        mod._broker_override = lambda msg, ctx: f"Brokered: {msg}"

        app = FastAPI()
        app.include_router(mod.router, prefix="/api/plugins/hermes-docs")
        c = TestClient(app)

        ws_id = c.post("/api/plugins/hermes-docs/workspaces", json={
            "name": "BrokerWS", "path": str(sample_folder)
        }).json()["id"]

        r = c.post(f"/api/plugins/hermes-docs/workspaces/{ws_id}/sidechat", json={
            "content": "What is this?",
            "document": "README.md",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["brokered"] is True
        assert "Brokered: What is this?" in body["content"]
        assert body["context"]["workspace"] == "BrokerWS"
    finally:
        mod._broker_override = original


def test_sidechat_brokered_persists_assistant_response(docs_home, sample_folder):
    """Session entry must include the assistant response when brokered."""
    mod = _load_plugin_module()

    original = mod._broker_override
    try:
        mod._broker_override = lambda msg, ctx: "Agent says hello"

        app = FastAPI()
        app.include_router(mod.router, prefix="/api/plugins/hermes-docs")
        c = TestClient(app)

        ws_id = c.post("/api/plugins/hermes-docs/workspaces", json={
            "name": "PersistWS", "path": str(sample_folder)
        }).json()["id"]

        c.post(f"/api/plugins/hermes-docs/workspaces/{ws_id}/sidechat", json={
            "content": "Persist me",
        })

        sessions_dir = docs_home / "docs-workspaces" / ws_id / "sessions"
        entries = list(sessions_dir.glob("*.json"))
        assert len(entries) == 1
        data = json.loads(entries[0].read_text())
        assert data["assistant"] == "Agent says hello"
        assert data["brokered"] is True
    finally:
        mod._broker_override = original


def test_call_docs_agent_uses_docs_profile_cli(monkeypatch, docs_home):
    """The real broker path should invoke Hermes with the docs profile."""
    mod = _load_plugin_module()
    docs_profile = docs_home / "profiles" / "docs"
    docs_profile.mkdir(parents=True)
    (docs_profile / "config.yaml").write_text("model: test\n", encoding="utf-8")

    calls = {}

    def _fake_run(args, capture_output, text, timeout, check):
        calls["args"] = args
        calls["capture_output"] = capture_output
        calls["text"] = text
        calls["timeout"] = timeout
        calls["check"] = check
        return SimpleNamespace(returncode=0, stdout="Docs response\n", stderr="")

    monkeypatch.setattr(mod.sys, "executable", str(docs_home / "bin" / "python"))
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/local/bin/hermes")
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    result = mod._call_docs_agent("Hello", {"workspace": "WS", "document": "a.md"})

    assert result == "Docs response"
    assert calls["args"][:5] == ["/usr/local/bin/hermes", "-p", "docs", "chat", "-q"]
    assert "Workspace: WS" in calls["args"][5]
    assert "Document: a.md" in calls["args"][5]
    assert calls["capture_output"] is True
    assert calls["text"] is True
    assert calls["timeout"] == 90
    assert calls["check"] is False


def test_sidechat_fallback_when_broker_raises(docs_home, sample_folder):
    """If the broker raises, the endpoint must return a deterministic stub (not 500)."""
    mod = _load_plugin_module()

    original = mod._broker_override
    try:
        def _failing_broker(msg, ctx):
            raise RuntimeError("no model configured")

        mod._broker_override = _failing_broker

        app = FastAPI()
        app.include_router(mod.router, prefix="/api/plugins/hermes-docs")
        c = TestClient(app)

        ws_id = c.post("/api/plugins/hermes-docs/workspaces", json={
            "name": "FallbackWS", "path": str(sample_folder)
        }).json()["id"]

        r = c.post(f"/api/plugins/hermes-docs/workspaces/{ws_id}/sidechat", json={
            "content": "Will this explode?",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["brokered"] is False
        assert body["content"]  # stub message returned, not an exception
    finally:
        mod._broker_override = original


# ---------------------------------------------------------------------------
# Status reflects workspace count
# ---------------------------------------------------------------------------


def test_status_with_workspaces(client, sample_folder, tmp_path):
    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    client.post("/api/plugins/hermes-docs/workspaces", json={"name": "A", "path": str(sample_folder)})
    client.post("/api/plugins/hermes-docs/workspaces", json={"name": "B", "path": str(ws2)})
    r = client.get("/api/plugins/hermes-docs/status")
    data = r.json()
    assert data["workspace_count"] == 2
    assert len(data["recent"]) == 2
