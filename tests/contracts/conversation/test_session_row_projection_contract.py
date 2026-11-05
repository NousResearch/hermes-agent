"""Contract cases for the session row projections: ``SessionRecord.as_session_row()``, ``SessionListEntry.as_list_row()`` and ``CronRunEntry.as_cron_row()`` emit the store's row dicts (the detail, listing and job-runs shapes), with the same keys in the same order and the stored cells as values.

Backend cases seed through public protocol operations only; the expected dicts are literal apart from the creation instant, the activity instant and the owner stamp, which are read back from the record.
"""

from __future__ import annotations

import hashlib
import json

from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import (
    COMPACT_EXCLUDED_COLUMNS,
    NewSession,
    SessionLineageKind,
    SessionQuery,
    SessionWindow,
)

SID = SessionId("proj-a")
PIN = ["terminal"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cells(record, **cells):
    """The literal cells of a fresh row, overridden by ``cells``."""
    row = {
        "id": str(record.session_id), "runtime_revision": 0, "runtime_generation": 0, "source": "cli",
        "created_source": "cli", "user_id": None, "session_key": None, "chat_id": None, "chat_type": None,
        "thread_id": None, "display_name": None, "origin_json": None, "expiry_finalized": 0, "model": None,
        "model_config": None, "system_prompt": None, "system_prompt_hash": None, "parent_session_id": None,
        "started_at": record.started_at, "ended_at": None, "end_reason": None, "message_count": 0, "tool_call_count": 0,
        "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0, "reasoning_tokens": 0,
        "cwd": None, "git_branch": None, "git_repo_root": None, "git_metadata_generation": 0, "billing_provider": None,
        "billing_base_url": None, "billing_mode": None, "estimated_cost_usd": None, "actual_cost_usd": None,
        "cost_status": None, "cost_source": None, "pricing_version": None, "title": None, "title_source": None,
        "last_activity_at": None, "last_activity_description": None, "last_activity_provenance": None,
        "api_call_count": 0, "handoff_state": None, "handoff_platform": None, "handoff_error": None,
        "compression_failure_cooldown_until": None, "compression_failure_error": None, "compression_fallback_streak": 0,
        "compression_ineffective_count": 0, "compression_recovery_deadline": None, "compression_overload_streak": 0,
        "profile_name": record.profile_name, "transport_profile": None, "rewind_count": 0, "archived": 0,
        "auto_archived": 0, "pinned": 0, "hidden": 0, "last_read_at": None, "tool_names": None, "transcript_revision": 0,
    }
    assert set(cells) <= set(row)
    return {**row, **cells}


def _seed(store):
    store.sessions.create(NewSession(
        session_id=SID, source="telegram", model="m", model_config={"a": 1}, system_prompt="sp",
        origin={"platform": "telegram"},
    ))
    store.sessions.update_tool_pin(SID, PIN)
    store.sessions.set_pinned(SID, True)
    return store.sessions.get(SID)


def _seeded_cells(record, *, tool_names):
    return _cells(
        record, source="telegram", created_source="telegram", origin_json=json.dumps({"platform": "telegram"}),
        model="m", model_config=json.dumps({"a": 1}), system_prompt="sp", system_prompt_hash=_sha("sp"), pinned=1,
        tool_names=tool_names,
    )


def test_detail_row_is_the_stored_row(backend, scope_a):
    record = _seed(backend.open(scope_a))
    row = record.as_session_row()
    # CONTRACT: every column in table order with its stored cell; get resolves the pin to its text.
    assert row == _seeded_cells(record, tool_names=json.dumps(PIN))
    assert list(row) == list(_seeded_cells(record, tool_names=None))


def test_listing_rows_carry_the_listing_keys(backend, scope_a):
    store = backend.open(scope_a)
    record = _seed(store)
    [entry] = store.sessions.list_sessions(SessionQuery(), SessionWindow(limit=10))
    row = entry.as_list_row()
    # CONTRACT: a listing read leaves the pin as its content hash; last_active, preview and unread follow the cells.
    expected = {**_seeded_cells(record, tool_names=_sha(json.dumps(PIN))), "last_active": entry.last_active, "preview": "", "unread": False}
    assert row == expected and list(row) == list(expected)
    [compact] = store.sessions.list_sessions(SessionQuery(), SessionWindow(limit=10), compact_rows=True)
    compact_row = compact.as_list_row()
    # CONTRACT: a compact row has no prompt columns and no metadata generation.
    assert list(compact_row) == [key for key in expected if key not in COMPACT_EXCLUDED_COLUMNS]
    assert compact_row == {key: value for key, value in expected.items() if key not in COMPACT_EXCLUDED_COLUMNS}


def test_markers_follow_the_cells(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SessionId("proj-p"), source="cli"))
    store.sessions.create(NewSession(
        session_id=SessionId("proj-b"), source="cli", parent_session_id=SessionId("proj-p"), lineage_kind=SessionLineageKind.BRANCH,
    ))
    child = store.sessions.get(SessionId("proj-b"))
    marker = json.dumps({"_branched_from": "proj-p"})
    assert child.as_session_row() == {**_cells(child, parent_session_id="proj-p", model_config=marker), "_branched_from": "proj-p"}
    listed = {e.session.session_id: e for e in store.sessions.list_sessions(SessionQuery(), SessionWindow(limit=10))}
    row = listed["proj-b"].as_list_row()
    assert list(row)[-4:] == ["last_active", "_branched_from", "preview", "unread"]
    assert row["_branched_from"] == "proj-p"


def test_projected_root_names_its_chain_last(backend, scope_a):
    store = backend.open(scope_a)
    root, tip = SessionId("proj-root"), SessionId("proj-tip")
    store.sessions.create(NewSession(session_id=root, source="cli"))
    store.messages.append(root, NewMessage(role="user", content="start"))
    store.sessions.end(root, "compression")
    store.sessions.create(NewSession(session_id=tip, source="cli", parent_session_id=root))
    [entry] = store.sessions.list_sessions(SessionQuery(), SessionWindow(limit=10))
    row = entry.as_list_row()
    # CONTRACT: the projected row is the tip's id and preview, with the chain keys after the preview and before unread.
    assert row["id"] == "proj-tip" and row["preview"] == ""
    assert list(row)[-5:] == ["preview", "_lineage_root_id", "_lineage_ids", "continuation_kind", "unread"]
    assert (row["_lineage_root_id"], row["_lineage_ids"], row["continuation_kind"]) == ("proj-root", ["proj-root", "proj-tip"], "compression")


def test_cron_row_carries_the_job_keys(backend, scope_a):
    store = backend.open(scope_a)
    sid = SessionId("cron_pj_20260923_120000")
    store.sessions.create(NewSession(session_id=sid, source="cron"))
    [run] = store.sessions.list_cron_job_runs("pj", SessionWindow(limit=5))
    record = run.entry.session
    row = run.as_cron_row()
    # CONTRACT: the job-runs row has the execution id before last_active and the job id last, and no unread.
    expected = {**_cells(record, source="cron", created_source="cron"), "cron_execution_id": None,
                "last_active": run.entry.last_active, "preview": "", "cron_job_id": "pj"}
    assert row == expected and list(row) == list(expected)
