"""Dashboard session-resume lineage regression tests."""

from __future__ import annotations

import asyncio

import pytest

from hermes_cli.web_server import _session_latest_descendant
from hermes_cli.web_routers import sessions as web_sessions
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


@pytest.mark.parametrize(
    ("child_id", "source", "model_config"),
    [
        ("branch", "webui", {"_branched_from": "parent"}),
        ("delegate", "subagent", {"_delegate_from": "parent"}),
        ("reset", "webui", {"_reset_from": "parent"}),
        ("tool", "tool", {}),
    ],
)
def test_latest_descendant_does_not_follow_non_continuation_children(
    db: SessionDB,
    child_id: str,
    source: str,
    model_config: dict,
) -> None:
    db.create_session("parent", source="webui")
    db.create_session(
        child_id,
        source=source,
        parent_session_id="parent",
        model_config=model_config,
    )

    assert _session_latest_descendant("parent", db) == ("parent", ["parent"])


def test_latest_descendant_does_not_follow_legacy_branch_child(
    db: SessionDB,
) -> None:
    db.create_session("parent", source="webui")
    db.end_session("parent", "branched")
    db.create_session(
        "legacy-branch",
        source="webui",
        parent_session_id="parent",
    )

    assert _session_latest_descendant("parent", db) == ("parent", ["parent"])


def test_latest_descendant_does_not_follow_legacy_reset_child(
    db: SessionDB,
) -> None:
    session_key = "agent:main:webui:dm:lane"
    db.create_session("parent", source="webui", session_key=session_key)
    db.end_session("parent", "session_reset")
    db.create_session(
        "legacy-reset",
        source="webui",
        session_key=session_key,
        parent_session_id="parent",
    )

    assert _session_latest_descendant("parent", db) == ("parent", ["parent"])


def test_latest_descendant_still_follows_model_child_with_inherited_marker(
    db: SessionDB,
) -> None:
    db.create_session("branch", source="webui")
    db.create_session(
        "model-child",
        source="webui",
        parent_session_id="branch",
        model_config={"_branched_from": "original-parent"},
    )

    assert _session_latest_descendant("branch", db) == (
        "model-child",
        ["branch", "model-child"],
    )


def test_messages_endpoint_keeps_legacy_branch_parent_transcript(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    try:
        seed.create_session("parent", source="webui")
        seed.append_message("parent", "user", "parent transcript")
        seed.end_session("parent", "branched")
        seed.create_session(
            "legacy-branch",
            source="webui",
            parent_session_id="parent",
        )
        seed.append_message("legacy-branch", "user", "branch transcript")
        seed.reopen_session("parent")

        marker = seed._conn.execute(
            "SELECT json_extract(model_config, '$._branched_from') "
            "FROM sessions WHERE id = 'legacy-branch'"
        ).fetchone()[0]
        assert marker == "parent"
        assert _session_latest_descendant("parent", seed) == (
            "parent",
            ["parent"],
        )
        assert seed.resolve_resume_session_id("parent") == "parent"
    finally:
        seed.close()

    def _open_db(profile=None, *, read_only):
        assert profile is None
        return SessionDB(db_path=db_path, read_only=read_only)

    monkeypatch.setattr(web_sessions, "_open_session_db_for_profile", _open_db)

    payload = asyncio.run(
        web_sessions.get_session_messages(
            "parent",
            limit=None,
            offset=0,
            order=None,
            include_compacted=False,
        )
    )

    assert payload["session_id"] == "parent"
    assert [message["content"] for message in payload["messages"]] == [
        "parent transcript"
    ]
