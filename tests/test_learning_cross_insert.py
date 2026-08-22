"""Regression test: POST /api/learning/node/cross-insert must write to the
SAME file the learning graph reads — ``<home>/memories/MEMORY.md`` — not the
profile root ``<home>/MEMORY.md``.

Bug fixed 2026-08-21: the endpoint wrote to ``get_hermes_home()/MEMORY.md``
and returned ok=True, but ``agent.learning_graph._memory_cards`` reads
``get_hermes_home()/memories/MEMORY.md`` — so the insert silently never
appeared on the target profile's star map.
"""

import pathlib

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app


@pytest.fixture()
def client():
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_cross_insert_writes_graph_visible_memories_file(
    client, tmp_path, monkeypatch
):
    source_home = tmp_path / "profiles" / "src-bot"
    target_home = tmp_path / "profiles" / "dst-bot"
    (source_home / "memories").mkdir(parents=True)
    (target_home / "memories").mkdir(parents=True)
    (target_home / "memories" / "MEMORY.md").write_text(
        "existing target entry\n", encoding="utf-8"
    )

    homes = {"src-bot": source_home, "dst-bot": target_home}

    import contextlib

    @contextlib.contextmanager
    def fake_profile_scope(profile):
        import hermes_constants

        token = hermes_constants.set_hermes_home_override(str(homes[profile]))
        try:
            yield homes[profile]
        finally:
            hermes_constants.reset_hermes_home_override(token)

    monkeypatch.setattr(web_server, "_profile_scope", fake_profile_scope)

    import agent.learning_mutations as lm

    def fake_node_detail(node_id):
        assert node_id == "memory:memory:0"
        return {
            "ok": True,
            "kind": "memory",
            "content": "cross-inserted fact body",
        }

    monkeypatch.setattr(lm, "node_detail", fake_node_detail)

    resp = client.post(
        "/api/learning/node/cross-insert",
        json={
            "id": "memory:memory:0",
            "source_profile": "src-bot",
            "target_profile": "dst-bot",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True, body

    # THE regression: content must land in memories/MEMORY.md (graph-visible)
    graph_file = target_home / "memories" / "MEMORY.md"
    text = graph_file.read_text(encoding="utf-8")
    assert "cross-inserted fact body" in text
    assert "[Imported from profile: src-bot]" in text
    assert text.startswith("existing target entry")  # append, never clobber

    # and must NOT create the dead profile-root file the bug wrote to
    assert not (target_home / "MEMORY.md").exists()


def test_cross_insert_refuses_skill_nodes(client, tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "any-bot"
    (home / "memories").mkdir(parents=True)

    import contextlib

    @contextlib.contextmanager
    def fake_profile_scope(profile):
        import hermes_constants

        token = hermes_constants.set_hermes_home_override(str(home))
        try:
            yield home
        finally:
            hermes_constants.reset_hermes_home_override(token)

    monkeypatch.setattr(web_server, "_profile_scope", fake_profile_scope)

    import agent.learning_mutations as lm

    monkeypatch.setattr(
        lm,
        "node_detail",
        lambda node_id: {"ok": True, "kind": "skill", "content": "skill body"},
    )

    resp = client.post(
        "/api/learning/node/cross-insert",
        json={
            "id": "some-skill",
            "source_profile": "any-bot",
            "target_profile": "any-bot",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is False
    assert "not supported" in resp.json()["message"]
