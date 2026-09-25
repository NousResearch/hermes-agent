"""HTTP contract for the merged multi-profile journey graph (``GET /api/learning/graph?profiles=a,b``)."""

import contextlib

import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient

from hermes_cli import web_server_profiles
from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

_MEM = "memory:memory:0:0123456789ab"


def _graph(profile, provider=None):
    return {
        "nodes": [
            {"id": "shared-skill", "label": "shared-skill", "kind": "skill", "category": "tools"},
            {"id": _MEM, "label": f"{profile} fact", "kind": "memory", "category": "memory"},
        ],
        "edges": [{"source": _MEM, "target": "shared-skill"}],
        "clusters": [{"category": "tools", "count": 1}, {"category": "memory", "count": 1}],
        "memory": [{"source": "memory", "title": f"{profile} fact", "body": f"{profile} fact body"}],
        "memoryProvider": provider,
        "stats": {"nodes": 1, "memory_nodes": 1, "isolated_pct": 0.0},
    }


@pytest.fixture()
def client(monkeypatch):
    """``alpha`` is the current profile; each graph is built inside its profile's scope."""
    import agent.learning_graph as lg

    graphs = {"alpha": _graph("alpha"), "beta": _graph("beta", provider="Honcho")}
    active = []

    @contextlib.contextmanager
    def fake_profile_scope(profile):
        name = profile or "alpha"
        if name not in graphs:
            raise HTTPException(status_code=404, detail=f"Profile '{name}' does not exist.")
        active.append(name)
        try:
            yield None
        finally:
            active.pop()

    monkeypatch.setattr(web_server_profiles, "_profile_scope", fake_profile_scope)
    monkeypatch.setattr(lg, "build_learning_graph", lambda: graphs[active[-1]])
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_profiles_param_merges_graphs_with_profile_prefixed_ids(client):
    resp = client.get("/api/learning/graph", params={"profiles": "alpha,beta"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["multiProfile"] is True
    assert body["profiles"] == ["alpha", "beta"]
    assert {n["id"] for n in body["nodes"]} == {
        "alpha:shared-skill", "beta:shared-skill", f"alpha:{_MEM}", f"beta:{_MEM}"}
    for node in body["nodes"]:
        assert node["id"] == f"{node['profile']}:{node['_originalId']}"
    assert {(e["source"], e["target"], e["profile"]) for e in body["edges"]} == {
        (f"alpha:{_MEM}", "alpha:shared-skill", "alpha"),
        (f"beta:{_MEM}", "beta:shared-skill", "beta"),
    }
    assert [(c["profile"], c["body"]) for c in body["memory"]] == [
        ("alpha", "alpha fact body"), ("beta", "beta fact body")]
    assert {c["category"]: c["count"] for c in body["clusters"]} == {"tools": 2, "memory": 2}
    assert body["stats"]["nodes"] == 2
    assert body["stats"]["memory_nodes"] == 2
    assert body["memoryProvider"] == "Honcho"


def test_unknown_and_repeated_profiles_are_skipped(client):
    body = client.get("/api/learning/graph", params={"profiles": "alpha,missing,alpha"}).json()

    assert body["profiles"] == ["alpha"]
    assert {n["profile"] for n in body["nodes"]} == {"alpha"}
    assert body["memoryProvider"] is None


def test_without_profiles_the_single_profile_graph_is_unchanged(client):
    for params in ({}, {"profiles": " , "}):
        body = client.get("/api/learning/graph", params=params).json()
        assert "multiProfile" not in body
        assert {n["id"] for n in body["nodes"]} == {"shared-skill", _MEM}


def test_too_many_profiles_is_rejected(client):
    names = ",".join(f"p{i}" for i in range(17))

    assert client.get("/api/learning/graph", params={"profiles": names}).status_code == 400
