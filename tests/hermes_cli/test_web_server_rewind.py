"""HTTP test for POST /api/sessions/{id}/rewind.

Exposes SessionDB.rewind_to_message (the /undo + /rewind mechanic, issue
#21910) over the desktop-facing gateway API so an API client can clear
conversation context from a chosen user message onward. The primitive
previously had no HTTP route.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from hermes_state import SessionDB
from hermes_cli import web_server


@pytest.fixture()
def seeded_db(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    ids = []
    for i in range(1, 4):
        ids.append(db.append_message("s1", "user", f"q{i}"))
        ids.append(db.append_message("s1", "assistant", f"a{i}"))
    # Serve this exact db instance to the endpoint, regardless of profile.
    monkeypatch.setattr(web_server, "_open_session_db_for_profile", lambda profile: db)
    return db, ids


def test_rewind_to_user_message_clears_later_context(seeded_db):
    db, ids = seeded_db
    q2_id = ids[2]  # third seeded row is the 2nd user turn (q2)

    res = asyncio.run(
        web_server.rewind_session_endpoint(
            "s1", web_server.SessionRewind(target_message_id=q2_id)
        )
    )

    assert res["ok"] is True
    assert res["rewound_count"] == 4  # q2, a2, q3, a3
    assert res["target_message"]["content"] == "q2"
    # Only q1, a1 remain in the active transcript the model sees.
    active = db.get_messages("s1")
    assert [m["content"] for m in active] == ["q1", "a1"]
    # Rewound rows are soft-deleted (kept for audit), not destroyed.
    assert len(db.get_messages("s1", include_inactive=True)) == 6


def test_rewind_rejects_non_user_target(seeded_db):
    db, ids = seeded_db
    a1_id = ids[1]  # an assistant message — rewind targets must be 'user'

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            web_server.rewind_session_endpoint(
                "s1", web_server.SessionRewind(target_message_id=a1_id)
            )
        )
    assert exc.value.status_code == 400


def test_rewind_unknown_session_returns_404(seeded_db):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            web_server.rewind_session_endpoint(
                "does-not-exist", web_server.SessionRewind(target_message_id=1)
            )
        )
    assert exc.value.status_code == 404
