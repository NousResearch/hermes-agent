"""Conversation lists leave internal runs out (#156048, #148148).

One-shot, kanban and tool runs are executions, not conversations. Every picker
hides them and one-shots carry no title (#151820), so ``GET /api/sessions`` hides them unless a caller opts in. A comparison
client sends ``exclude_sources=cron``; that still hides internal runs.
"""
from starlette.testclient import TestClient

from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session

_ROWS = {"chat-a": "cli", "run-1": "oneshot", "kb-1": "kanban", "tool-1": "tool", "cron-1": "cron"}
_INTERNAL = {"run-1", "kb-1", "tool-1"}


def _seed(store):
    for sid, source in _ROWS.items():
        seed_session(store, sid, source)
        seed_messages(store, sid, {"role": "user", "content": "hi"})


def _ids(client, query=""):
    response = client.get(f"/api/sessions?profile=default&limit=50{query}")
    assert response.status_code == 200, response.text
    body = response.json()
    ids = {row["id"] for row in body["sessions"]}
    assert body["total"] == len(ids)
    return ids


def test_api_session_list_hides_internal_runs_unless_named(_isolate_hermes_home):
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    with open_persistence(profile_scope()) as store:
        _seed(store)

    assert _ids(client) == {"chat-a", "cron-1"}
    # A caller that excludes another source still gets no internal runs.
    assert _ids(client, "&exclude_sources=cron") == {"chat-a"}
    # Naming a source is an explicit ask, so it is honoured.
    assert _ids(client, "&source=oneshot") == {"run-1"}
    assert _ids(client, "&sources=oneshot,cli") == {"run-1", "chat-a"}
    # The dashboard Sessions page lists every source.
    assert _ids(client, "&include_internal=1") == set(_ROWS)
    assert _ids(client, "&include_internal=1&exclude_sources=cron,cli") == _INTERNAL

    # Apply the default exclusion before LIMIT/OFFSET, including total.
    for offset in range(2):
        response = client.get(f"/api/sessions?profile=default&limit=1&offset={offset}")
        assert response.status_code == 200, response.text
        if offset == 0:
            first = response.json()
            assert first["total"] == 2
            assert len(first["sessions"]) == 1
        else:
            second = response.json()
            assert second["total"] == 2
            assert len(second["sessions"]) == 1
    assert {first["sessions"][0]["id"], second["sessions"][0]["id"]} == {"chat-a", "cron-1"}


def test_api_session_list_keeps_subagents_when_configured(_isolate_hermes_home):
    """``sessions.show_subagents`` reads an empty exclusion list as the recents shape, so the
    internal exclusion must not turn delegate runs off."""
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text("sessions:\n  show_subagents: true\n")
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    with open_persistence(profile_scope()) as store:
        _seed(store)
        seed_session(store, "delegate-run", "subagent", parent_session_id="chat-a",
                     model_config={"_delegate_from": "chat-a"})
        seed_messages(store, "delegate-run", {"role": "user", "content": "hi"})

    ids = _ids(client)
    assert "delegate-run" in ids
    assert not ids & _INTERNAL
