"""Every display surface shows a compaction's re-stated in-flight request exactly once (#131104).

When compaction leaves an unfinished request before its handoff, the compressor re-states it after
the summary, framed by ``_INFLIGHT_TASK_REPLAY_HEADER``: as a standalone user row, or merged onto a
user-role carrier. Committing the compaction archives the original row (``active=0, compacted=1``).
So a lineage read (``include_compacted=true``, resume history) carries the original AND the
restatement, while an active-only read (the default REST/API read, the live window) carries only
the restatement. Each surface must show the request once, never inside the model-only frame, and
never drop it.
"""

from __future__ import annotations

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    ContextCompressor,
    _INFLIGHT_TASK_REPLAY_HEADER,
    _SUMMARY_END_MARKER,
)
from gateway.platforms.api_server import _project_client_message
from hermes_cli.web_routers.sessions import _project_for_display
from hermes_state import SessionDB
from tui_gateway.server import _history_to_messages

ASK = "please migrate the billing tables and tell me when it is done"


def _committed_session(tmp_path, carrier_role: str) -> tuple[SessionDB, str]:
    """A session whose in-flight ask was re-stated by a committed compaction.

    An assistant carrier leaves the window ending on assistant, so the restatement is a standalone
    user row; a user carrier gets it merged after its end marker.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "s"
    db.create_session(session_id=sid, source="desktop")
    db.append_messages_batch(sid, [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": ASK},
    ])
    original = db.get_messages(sid)[-1]
    carrier = {"role": carrier_role, COMPRESSED_SUMMARY_METADATA_KEY: True,
               "content": f"{SUMMARY_PREFIX}\n## Summary\nran migrations.\n\n{_SUMMARY_END_MARKER}"}
    compressor = object.__new__(ContextCompressor)
    compressor.quiet_mode = True
    window = compressor._reappend_inflight_user_task([carrier], {**original})
    assert any(_INFLIGHT_TASK_REPLAY_HEADER in str(m.get("content")) for m in window)
    db.archive_and_compact(sid, window)
    return db, sid


def _user_texts_rest(rows: list[dict]) -> list[str]:
    return [str(m.get("display_content", m.get("content"))) for m in rows
            if m.get("role") == "user" and m.get("display_kind") != "hidden"]


def _surfaces(db: SessionDB, sid: str) -> dict[str, list[str]]:
    lineage = db.get_messages(sid, include_compacted=True, include_ancestors=True)
    active = db.get_messages(sid, include_ancestors=True)
    surfaces = {}
    for name, rows, is_lineage in (("lineage", lineage, True), ("active", active, False)):
        surfaces[f"rest-{name}"] = _user_texts_rest(_project_for_display(rows, lineage=is_lineage))
        surfaces[f"history-{name}"] = [str(m.get("text") or m.get("content"))
                                       for m in _history_to_messages(rows) if m.get("role") == "user"]
    # The API server's /v1 read projects row by row; a page cut between the original and its
    # restatement is still a lineage read.
    surfaces["api-active"] = [str(m["content"]) for m in map(_project_client_message, active)
                              if m.get("role") == "user" and m.get("display_kind") != "hidden"]
    return surfaces


@pytest.mark.parametrize("carrier_role", ["assistant", "user"], ids=["standalone", "merged-onto-carrier"])
def test_the_inflight_request_shows_once_unframed_on_every_surface(tmp_path, carrier_role):
    db, sid = _committed_session(tmp_path, carrier_role)
    for surface, texts in _surfaces(db, sid).items():
        assert sum(ASK in text for text in texts) == 1, (surface, texts)
        assert not any(_INFLIGHT_TASK_REPLAY_HEADER in text for text in texts), (surface, texts)


def test_a_lineage_page_holding_only_the_restatement_still_hides_it(tmp_path):
    # The Desktop pages include_compacted reads; the original can sit on the previous page.
    db, sid = _committed_session(tmp_path, "assistant")
    lineage = db.get_messages(sid, include_compacted=True, include_ancestors=True)
    page = [m for m in lineage if not (m.get("role") == "user" and m.get("content") == ASK)]
    assert not any(ASK in text for text in _user_texts_rest(_project_for_display(page, lineage=True)))
