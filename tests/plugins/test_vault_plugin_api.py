import shutil
import tempfile
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from workstation.vault import VaultManager
import plugins.vault.dashboard.plugin_api as v_api


@pytest.fixture
def client(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    mgr = VaultManager(vault_dir=temp_dir)
    monkeypatch.setattr(v_api, "_vault_manager", mgr)
    app = FastAPI()
    app.include_router(v_api.router)
    yield TestClient(app)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_vault_plugin_api_crud_and_links(client):
    # 1. Initially empty
    resp = client.get("/notes")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0

    # 2. Create note 1
    resp1 = client.post(
        "/notes",
        json={
            "title": "Obsidian Integration",
            "content": "Building local-first PKM with [[Hermes Work]] links.",
            "tags": ["pkm", "obsidian"],
            "frontmatter": {"author": "User"},
        },
    )
    assert resp1.status_code == 200
    assert resp1.json()["note"]["title"] == "Obsidian Integration"

    # 3. Create note 2
    resp2 = client.post(
        "/notes",
        json={
            "title": "Hermes Work",
            "content": "Hermes Workstation agentic operating layer.",
            "tags": ["agent"],
        },
    )
    assert resp2.status_code == 200

    # 4. Read note 2 and check backlinks
    resp_read = client.get("/notes/Hermes Work")
    assert resp_read.status_code == 200
    data = resp_read.json()["note"]
    assert "Obsidian Integration" in data["backlinks"]

    # 5. Search
    resp_search = client.get("/search?q=pkm")
    assert resp_search.status_code == 200
    assert resp_search.json()["count"] >= 1

    # 6. Graph
    resp_graph = client.get("/graph")
    assert resp_graph.status_code == 200
    graph = resp_graph.json()["graph"]
    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1

    # 7. Suggestions
    resp_sug = client.get("/suggest?prefix=Hermes")
    assert resp_sug.status_code == 200
    assert any(s["title"] == "Hermes Work" for s in resp_sug.json()["suggestions"])

    # 8. Delete
    resp_del = client.delete("/notes/Hermes Work")
    assert resp_del.status_code == 200
    resp_read_after = client.get("/notes/Hermes Work")
    assert resp_read_after.status_code == 404
