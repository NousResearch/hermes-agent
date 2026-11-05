from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.mark.parametrize("serving_profile", ["work", None])
def test_message_pages_identify_the_serving_profile(tmp_path, monkeypatch, serving_profile):
    from persistence.conversation.messages import NewMessage
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_session

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    work_home = default_home / "profiles" / "work" if serving_profile else default_home / "custom-home"
    default_home.mkdir(parents=True)
    work_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(work_home))

    for home, count in ((default_home, 1), (work_home, 199)):
        with open_persistence(profile_scope(home=home)) as store:
            seed_session(store, "same-id", "desktop")
            store.messages.append_batch("same-id", [
                NewMessage.from_live({"role": "user", "content": f"message-{index}"})
                for index in range(count)
            ])

    from hermes_cli.web_routers.sessions import manage_router

    app = FastAPI()
    app.include_router(manage_router)
    with TestClient(app) as client:
        query = "limit=120&order=latest&include_compacted=true"
        tail = client.get(f"/api/sessions/same-id/messages?{query}").json()
        older = client.get(f"/api/sessions/same-id/messages?{query}&offset=120").json()
        default = client.get(f"/api/sessions/same-id/messages?{query}&profile=default").json()

    assert tail["profile"] == older["profile"] == (serving_profile or "default")
    assert default["profile"] == "default"
    assert len(tail["messages"]) == 120
    assert len(older["messages"]) == 79
    assert len(default["messages"]) == 1
    assert [row["content"] for row in older["messages"] + tail["messages"]] == [
        f"message-{index}" for index in range(199)
    ]


def test_message_pages_type_untyped_failed_turn_rows(tmp_path, monkeypatch):
    """Desktop cold-loads and pages through REST, not ``session.resume``: a failed-turn boundary
    written before the closers typed it must reach it as ``failed_turn``, not model text."""
    from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND, FAILED_TURN_NOTICE, PARTIAL_FAILED_TURN_NOTICE
    from persistence.conversation.messages import NewMessage
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_session

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with open_persistence(profile_scope(home=home)) as store:
        seed_session(store, "s", "desktop")
        store.messages.append_batch("s", [NewMessage.from_live(message) for message in (
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": FAILED_TURN_NOTICE},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": PARTIAL_FAILED_TURN_NOTICE},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": f"Quoting Hermes: {FAILED_TURN_NOTICE}"},
        )])

    from hermes_cli.web_routers.sessions import manage_router

    app = FastAPI()
    app.include_router(manage_router)
    with TestClient(app) as client:
        rows = client.get("/api/sessions/s/messages").json()["messages"]

    assert [row.get("display_kind") for row in rows] == [
        None, FAILED_TURN_DISPLAY_KIND, None, FAILED_TURN_DISPLAY_KIND, None, None,
    ]
