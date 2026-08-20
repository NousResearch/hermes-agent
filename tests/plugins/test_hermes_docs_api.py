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


# ---------------------------------------------------------------------------
# Docs persona profile status and bootstrap
# ---------------------------------------------------------------------------


def test_profile_status_missing(client):
    """GET /profile/status returns installed=False when docs profile is absent."""
    r = client.get("/api/plugins/hermes-docs/profile/status")
    assert r.status_code == 200
    data = r.json()
    assert data["installed"] is False
    assert data["profile_dir"] is None
    assert data["has_soul"] is False
    assert data["has_config"] is False


def test_profile_bootstrap_creates_profile(client, docs_home):
    """POST /profile/bootstrap creates the docs profile when it is absent."""
    r = client.post("/api/plugins/hermes-docs/profile/bootstrap")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "created"
    assert data["profile_dir"] is not None

    profile_dir = docs_home / "profiles" / "docs"
    assert profile_dir.is_dir()
    assert (profile_dir / "config.yaml").exists()
    assert (profile_dir / "SOUL.md").exists()
    for subdir in ("memories", "sessions", "skills", "logs"):
        assert (profile_dir / subdir).is_dir()


def test_profile_bootstrap_idempotent(client, docs_home):
    """POST /profile/bootstrap is idempotent — second call returns already_exists."""
    r1 = client.post("/api/plugins/hermes-docs/profile/bootstrap")
    assert r1.json()["status"] == "created"

    r2 = client.post("/api/plugins/hermes-docs/profile/bootstrap")
    assert r2.status_code == 200
    data = r2.json()
    assert data["status"] == "already_exists"
    assert data["created_files"] == []


def test_profile_bootstrap_preserves_existing_config(client, docs_home):
    """Bootstrap must not overwrite a pre-existing config.yaml."""
    profile_dir = docs_home / "profiles" / "docs"
    profile_dir.mkdir(parents=True)
    custom_config = "model:\n  provider: anthropic\n  default: claude-sonnet-4-6\n"
    (profile_dir / "config.yaml").write_text(custom_config, encoding="utf-8")

    r = client.post("/api/plugins/hermes-docs/profile/bootstrap")
    assert r.status_code == 200
    assert r.json()["status"] == "already_exists"

    # config.yaml content must be unchanged
    assert (profile_dir / "config.yaml").read_text(encoding="utf-8") == custom_config
    assert (profile_dir / "SOUL.md").exists()


def test_profile_bootstrap_preserves_existing_soul(client, docs_home):
    """Bootstrap must not overwrite a pre-existing SOUL.md."""
    profile_dir = docs_home / "profiles" / "docs"
    profile_dir.mkdir(parents=True)
    (profile_dir / "config.yaml").write_text("model:\n  provider: auto\n", encoding="utf-8")
    custom_soul = "# My custom docs persona\n\nDo exactly what I say.\n"
    (profile_dir / "SOUL.md").write_text(custom_soul, encoding="utf-8")

    r = client.post("/api/plugins/hermes-docs/profile/bootstrap")
    assert r.status_code == 200
    assert r.json()["status"] == "already_exists"

    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == custom_soul


def test_profile_status_installed_after_bootstrap(client, docs_home):
    """GET /profile/status returns installed=True after a successful bootstrap."""
    client.post("/api/plugins/hermes-docs/profile/bootstrap")

    r = client.get("/api/plugins/hermes-docs/profile/status")
    assert r.status_code == 200
    data = r.json()
    assert data["installed"] is True
    assert data["has_config"] is True
    assert data["has_soul"] is True
    assert data["profile_dir"] is not None


# ---------------------------------------------------------------------------
# Kordoc detection endpoint
# ---------------------------------------------------------------------------


def _load_plugin_module_fresh(mod_key="hermes_docs_plugin_api_kordoc_test"):
    """Load plugin_api.py as a fresh module (for override access)."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "hermes-docs" / "dashboard" / "plugin_api.py"
    sys.modules.pop(mod_key, None)
    spec = importlib.util.spec_from_file_location(mod_key, plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    spec.loader.exec_module(mod)
    return mod


def test_kordoc_status_available(docs_home):
    """When kordoc_helper reports available, the endpoint echoes that."""
    mod = _load_plugin_module_fresh()
    mod.kordoc_helper._detect_override = {
        "available": True,
        "version": "2.7.1",
        "detail": "kordoc available",
    }
    try:
        app = FastAPI()
        app.include_router(mod.router, prefix="/api/plugins/hermes-docs")
        c = TestClient(app)
        r = c.get("/api/plugins/hermes-docs/kordoc/status")
        assert r.status_code == 200
        data = r.json()
        assert data["available"] is True
        assert data["version"] == "2.7.1"
    finally:
        mod.kordoc_helper._detect_override = None


def test_kordoc_status_unavailable(docs_home):
    """When no local Kordoc command is available, the endpoint reports False."""
    mod = _load_plugin_module_fresh()
    mod.kordoc_helper._detect_override = {
        "available": False,
        "version": None,
        "detail": "kordoc executable not found on PATH",
    }
    try:
        app = FastAPI()
        app.include_router(mod.router, prefix="/api/plugins/hermes-docs")
        c = TestClient(app)
        r = c.get("/api/plugins/hermes-docs/kordoc/status")
        assert r.status_code == 200
        data = r.json()
        assert data["available"] is False
        assert data["version"] is None
        assert "kordoc" in data["detail"]
    finally:
        mod.kordoc_helper._detect_override = None


# ---------------------------------------------------------------------------
# Kordoc conversion preview endpoint
# ---------------------------------------------------------------------------


def test_kordoc_preview_unavailable_returns_stub(docs_home, sample_folder):
    """When kordoc is unavailable, preview returns 200 with available=False."""
    mod = _load_plugin_module_fresh()
    mod.kordoc_helper._detect_override = {
        "available": False,
        "version": None,
        "detail": "kordoc executable not found on PATH",
    }
    try:
        app = FastAPI()
        app.include_router(mod.router, prefix="/api/plugins/hermes-docs")
        c = TestClient(app)
        ws_id = c.post("/api/plugins/hermes-docs/workspaces", json={
            "name": "KD", "path": str(sample_folder)
        }).json()["id"]

        r = c.post(
            f"/api/plugins/hermes-docs/workspaces/{ws_id}/kordoc/preview",
            json={"rel": "README.md", "target_format": "markdown"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["available"] is False
        assert data["content"] is None
        assert data["rel"] == "README.md"
        assert data["target_format"] == "markdown"
    finally:
        mod.kordoc_helper._detect_override = None


def test_kordoc_preview_available_calls_subprocess(docs_home, sample_folder):
    """When kordoc is available, preview invokes the local command."""
    mod = _load_plugin_module_fresh()
    mod.kordoc_helper._detect_override = {
        "available": True,
        "version": "2.7.1",
        "detail": "kordoc available",
    }

    def _fake_run(cmd, *, capture_output, text, timeout, check):
        # Record what was called; return synthetic markdown
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0, stdout="# Hello\n\nConverted.", stderr="")

    mod.kordoc_helper._subprocess_run_override = _fake_run
    try:
        app = FastAPI()
        app.include_router(mod.router, prefix="/api/plugins/hermes-docs")
        c = TestClient(app)
        ws_id = c.post("/api/plugins/hermes-docs/workspaces", json={
            "name": "KD2", "path": str(sample_folder)
        }).json()["id"]

        r = c.post(
            f"/api/plugins/hermes-docs/workspaces/{ws_id}/kordoc/preview",
            json={"rel": "README.md", "target_format": "markdown"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["available"] is True
        assert "Converted" in (data["content"] or "")
        assert data["message"] == "ok"
        # Source file must not be mutated
        assert "Hello" in (sample_folder / "README.md").read_text()
    finally:
        mod.kordoc_helper._detect_override = None
        mod.kordoc_helper._subprocess_run_override = None


def test_kordoc_preview_path_traversal_blocked(docs_home, sample_folder):
    """Traversal attempts in rel must return HTTP 403."""
    mod = _load_plugin_module_fresh()
    mod.kordoc_helper._detect_override = {
        "available": True,
        "version": "2.7.1",
        "detail": "kordoc available",
    }
    try:
        app = FastAPI()
        app.include_router(mod.router, prefix="/api/plugins/hermes-docs")
        c = TestClient(app)
        ws_id = c.post("/api/plugins/hermes-docs/workspaces", json={
            "name": "KDTRV", "path": str(sample_folder)
        }).json()["id"]

        r = c.post(
            f"/api/plugins/hermes-docs/workspaces/{ws_id}/kordoc/preview",
            json={"rel": "../../etc/passwd", "target_format": "markdown"},
        )
        assert r.status_code == 403
    finally:
        mod.kordoc_helper._detect_override = None


def test_kordoc_preview_bad_format_returns_400(docs_home, sample_folder):
    """An unsupported target_format must return HTTP 400."""
    mod = _load_plugin_module_fresh()
    mod.kordoc_helper._detect_override = {
        "available": True,
        "version": "2.7.1",
        "detail": "kordoc available",
    }
    try:
        app = FastAPI()
        app.include_router(mod.router, prefix="/api/plugins/hermes-docs")
        c = TestClient(app)
        ws_id = c.post("/api/plugins/hermes-docs/workspaces", json={
            "name": "KDFMT", "path": str(sample_folder)
        }).json()["id"]

        r = c.post(
            f"/api/plugins/hermes-docs/workspaces/{ws_id}/kordoc/preview",
            json={"rel": "README.md", "target_format": "docx"},
        )
        assert r.status_code == 400
    finally:
        mod.kordoc_helper._detect_override = None


def test_kordoc_preview_missing_file_returns_404(docs_home, sample_folder):
    """A rel path to a non-existent file must return 404."""
    mod = _load_plugin_module_fresh()
    mod.kordoc_helper._detect_override = {
        "available": True,
        "version": "2.7.1",
        "detail": "kordoc available",
    }
    try:
        app = FastAPI()
        app.include_router(mod.router, prefix="/api/plugins/hermes-docs")
        c = TestClient(app)
        ws_id = c.post("/api/plugins/hermes-docs/workspaces", json={
            "name": "KDMIS", "path": str(sample_folder)
        }).json()["id"]

        r = c.post(
            f"/api/plugins/hermes-docs/workspaces/{ws_id}/kordoc/preview",
            json={"rel": "no-such-file.hwpx", "target_format": "markdown"},
        )
        assert r.status_code == 404
    finally:
        mod.kordoc_helper._detect_override = None


# ---------------------------------------------------------------------------
# GET /auth/codex/status — Codex auth readiness
# ---------------------------------------------------------------------------


def _make_codex_auth_client(docs_home_fixture, status_override: dict):
    """Load a fresh module, inject _status_override, return (mod, TestClient)."""
    mod = _load_plugin_module_fresh(mod_key="hermes_docs_plugin_api_codex_auth_test")
    mod.codex_auth_helper._status_override = status_override
    app = FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/hermes-docs")
    return mod, TestClient(app)


def test_codex_auth_status_configured(docs_home):
    """When credentials are present the endpoint reports configured=True."""
    override = {
        "provider_id": "openai-codex",
        "configured": True,
        "available": True,
        "cli_command": "hermes auth add openai-codex",
        "token_exposed": False,
        "detail": "Authenticated (hermes-auth-store; store: /tmp/.hermes/auth.json)",
        "next_action": None,
    }
    mod, c = _make_codex_auth_client(docs_home, override)
    try:
        r = c.get("/api/plugins/hermes-docs/auth/codex/status")
        assert r.status_code == 200
        data = r.json()
        assert data["provider_id"] == "openai-codex"
        assert data["configured"] is True
        assert data["available"] is True
        assert data["cli_command"] == "hermes auth add openai-codex"
        assert data["token_exposed"] is False
        assert data["next_action"] is None
    finally:
        mod.codex_auth_helper._status_override = None


def test_codex_auth_status_not_configured(docs_home):
    """When credentials are absent the endpoint reports configured=False."""
    override = {
        "provider_id": "openai-codex",
        "configured": False,
        "available": False,
        "cli_command": "hermes auth add openai-codex",
        "token_exposed": False,
        "detail": "Not configured",
        "next_action": "Run: hermes auth add openai-codex",
    }
    mod, c = _make_codex_auth_client(docs_home, override)
    try:
        r = c.get("/api/plugins/hermes-docs/auth/codex/status")
        assert r.status_code == 200
        data = r.json()
        assert data["configured"] is False
        assert data["available"] is False
        assert data["next_action"] is not None
        # next_action must mention the CLI command
        assert "hermes auth add openai-codex" in data["next_action"]
    finally:
        mod.codex_auth_helper._status_override = None


def test_codex_auth_status_no_token_exposure(docs_home):
    """Raw auth fields from Hermes auth status must be stripped."""
    mod = _load_plugin_module_fresh(mod_key="hermes_docs_plugin_api_codex_auth_real_test")

    def fake_auth_status():
        return {
            "logged_in": True,
            "source": "pool:primary",
            "auth_store": "/tmp/.hermes/auth.json",
            "api_key": "sk-sensitive-value",
            "access_token": "token-sensitive-value",
            "refresh_token": "refresh-sensitive-value",
        }

    mod.codex_auth_helper._read_auth_status = fake_auth_status
    app = FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/hermes-docs")
    c = TestClient(app)

    r = c.get("/api/plugins/hermes-docs/auth/codex/status")
    assert r.status_code == 200
    data = r.json()
    encoded = json.dumps(data)

    assert data["configured"] is True
    assert data["token_exposed"] is False
    for key in ("api_key", "access_token", "refresh_token", "token", "secret", "password"):
        assert key not in data, f"Sensitive key {key!r} must not appear in response"
    assert "sensitive-value" not in encoded
