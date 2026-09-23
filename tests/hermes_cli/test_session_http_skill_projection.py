"""HTTP session history must project skill invocations like the WebSocket history does.

``/api/sessions/{id}/messages`` and ``/messages/around`` return the stored user
row of a slash-skill-expanded turn as-is: the full scaffold body the model needs
on replay. The WebSocket history already ships the typed invocation instead
(``display_kind="skill_invocation"``); reading the same rows over HTTP used to
expose the internal scaffold. These drive the real routes through a real
SessionDB, plus unit rows for the explicit-display-metadata precedence.
"""

from copy import deepcopy
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import agent.skill_commands as skill_commands
import tools.skills_tool as skills_tool
from hermes_state import SessionDB


@pytest.fixture
def http_store(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")

    skills_dir = home / "skills"
    for name in ("check-change", "format-change"):
        directory = skills_dir / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Test skill\n---\nPrivate skill instructions.\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)

    single = skill_commands.build_skill_invocation_message("/check-change", "Check this change")
    stacked = skill_commands.build_stacked_skill_invocation_message(
        ["/check-change", "/format-change"], "Check both"
    )
    assert single and stacked
    contents = [single, stacked[0]]

    db = SessionDB(db_path=home / "state.db")
    db.create_session("skill-history", source="cli")
    row_ids = []
    for content in contents:
        row_ids.append(db.append_message("skill-history", role="user", content=content))
        db.append_message("skill-history", role="assistant", content="Done")

    from hermes_cli.web_routers.sessions import manage_router

    app = FastAPI()
    app.include_router(manage_router)
    with TestClient(app) as client:
        yield db, client, contents, row_ids, home
    db.close()


def test_http_messages_project_stored_skill_invocations(http_store):
    db, client, contents, row_ids, home = http_store
    for suffix in ("messages", f"messages/around?row_id={row_ids[0]}"):
        response = client.get(f"/api/sessions/skill-history/{suffix}")
        assert response.status_code == 200
        displayed = [row for row in response.json()["messages"] if row["role"] == "user"]
        assert [row["display_content"] for row in displayed] == [
            skill_commands.describe_skill_invocation(content, separator=" ") for content in contents
        ]
        assert all(row["display_kind"] == "skill_invocation" for row in displayed)
        assert [row["content"] for row in displayed] == contents

    # The projection is display-only: stored rows keep the full scaffold, untyped.
    with SessionDB(db_path=home / "state.db", read_only=True) as db:
        stored = [row for row in db.get_messages("skill-history") if row["role"] == "user"]
    assert [row["content"] for row in stored] == contents
    assert all(not row.get("display_kind") for row in stored)


def test_projection_preserves_explicit_display_metadata_and_plain_text(http_store):
    from hermes_cli.web_routers.sessions import _project_for_display

    scaffold = http_store[2][0]
    rows = [
        {"role": "user", "content": "/unknown keep this text"},
        {"role": "user", "content": scaffold, "display_kind": "hidden"},
        {"role": "user", "content": scaffold, "display_kind": "model_switch"},
        {"role": "user", "content": scaffold, "display_content": "Explicit projection"},
        {"role": "user", "content": scaffold, "display_content": ""},
        {"role": "user", "content": scaffold, "display_content": None},
        {"role": "assistant", "content": scaffold},
    ]
    before = deepcopy(rows)
    assert _project_for_display(rows) == before
    assert rows == before
