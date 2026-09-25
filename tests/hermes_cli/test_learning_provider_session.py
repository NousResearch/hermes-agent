"""HTTP contracts for Journey provider-session provenance."""

from starlette.testclient import TestClient

from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app


def _client():
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def test_recall_draft_route_returns_mutation_result(monkeypatch):
    import agent.learning_mutations as mutations

    expected = {
        "connected_count": 0,
        "findings": [],
        "id": "memory:memory:0:abcdef123456",
        "kind": "memory",
        "label": "remember this",
        "ok": True,
        "text": "reference context",
        "truncated": False,
    }
    monkeypatch.setattr(mutations, "build_recall_draft", lambda node_id: expected if node_id == expected["id"] else {})

    response = _client().get(
        "/api/learning/recall-draft",
        params={"id": expected["id"]},
    )

    assert response.status_code == 200
    assert response.json() == expected


def test_provider_session_route_preserves_provider_role(monkeypatch):
    """The source-corpus endpoint must not discard role attribution supplied
    by a provider; desktop and web clients type and consume this field."""

    class Provider:
        def journey_session_messages(self, session_id, limit=500):
            assert session_id == "source-session"
            return [
                {
                    "content": "question",
                    "peer": "human",
                    "role": "user",
                    "timestamp": 1_700_000_000,
                },
                {
                    "content": "answer",
                    "peer": "agent",
                    "role": "assistant",
                    "timestamp": 1_700_000_100,
                },
            ][:limit]

    import plugins.memory as pm

    monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: "fakemem")
    monkeypatch.setattr(pm, "load_memory_provider", lambda _name: Provider())

    client = _client()
    response = client.get(
        "/api/learning/provider-session",
        params={"session_id": "source-session"},
    )

    assert response.status_code == 200
    assert [m["role"] for m in response.json()["messages"]] == [
        "user",
        "assistant",
    ]
