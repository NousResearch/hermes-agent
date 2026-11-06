"""Keyset cursor contracts for the transcript list route.

Why this exists: ``order=latest&offset=N`` counts backwards from the newest row, so every append
shifts the window under a paging client — a reader that keeps polling can repeat or skip rows
mid-read. ``messages.id`` is an AUTOINCREMENT row id (durable, monotonic), and the store already
pages by it, so the route surfaces it as ``after_row_id``: "everything after the last row I hold".

A client proves support from the answer itself — the param is echoed back in ``pagination`` — so a
server that ignores it is detectable without version sniffing.

The merge modes (a compression lineage, compacted display generations) cannot be combined with a
cursor in the store, so the route rejects those combinations instead of returning a quieter,
different set of messages.
"""
import asyncio
import json

import pytest

from gateway.platforms.api_server import APIServerAdapter, _coerce_after_row_id
from hermes_state import SessionDB


def test_coerce_after_row_id_accepts_non_negative_integers():
    assert _coerce_after_row_id(None) == (None, None)
    assert _coerce_after_row_id("0") == (0, None)
    assert _coerce_after_row_id("42") == (42, None)
    assert _coerce_after_row_id(" 7 ") == (7, None)


def test_coerce_after_row_id_rejects_everything_else():
    for bad in ("-1", "abc", "", "1.5", "1e3", "None"):
        value, err = _coerce_after_row_id(bad)
        assert value is None, bad
        assert err == "after_row_id must be a non-negative integer", bad


def _db_with_rows(tmp_path, count=5):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="cursor-test", source="cli")
    for index in range(count):
        db.append_message(session_id="cursor-test", role="user", content=f"m{index}")
    return db


def test_store_keyset_returns_only_rows_after_the_cursor(tmp_path):
    db = _db_with_rows(tmp_path)
    all_rows = db.get_messages("cursor-test")
    cursor = all_rows[1]["id"]
    after = db.get_messages("cursor-test", after_id=cursor)
    assert [row["id"] for row in after] == [row["id"] for row in all_rows[2:]]
    assert all(row["id"] > cursor for row in after)


def test_store_refuses_cursor_with_the_merge_modes_the_route_must_reject(tmp_path):
    db = _db_with_rows(tmp_path)
    with pytest.raises(ValueError):
        db.get_messages("cursor-test", after_id=1, latest=True)
    with pytest.raises(ValueError):
        db.get_messages("cursor-test", after_id=1, offset=2)
    with pytest.raises(ValueError):
        db.get_messages("cursor-test", after_id=1, include_compacted=True)


class _FakeRequest:
    """Only the two attributes this handler reads."""
    def __init__(self, session_id, params):
        self.match_info = {"session_id": session_id}
        self.query = params


def _call_route(db, params):
    adapter = object.__new__(APIServerAdapter)

    async def _existing(_session_id):
        return {"id": "cursor-test"}, None

    async def _db():
        return db

    adapter._get_existing_session_or_404 = _existing
    adapter._ensure_session_db_async = _db
    handler = APIServerAdapter._handle_session_messages.__wrapped__
    response = asyncio.run(handler(adapter, _FakeRequest("cursor-test", params)))
    return response.status, json.loads(response.text)


def test_route_pages_by_cursor_and_echoes_it(tmp_path):
    db = _db_with_rows(tmp_path)
    status, body = _call_route(db, {"after_row_id": "0", "limit": "2"})
    assert status == 200
    assert body["pagination"]["after_row_id"] == 0
    assert body["pagination"]["order"] == "oldest"
    assert body["pagination"]["returned"] == 2
    ids = [row["id"] for row in body["data"]]
    assert ids == sorted(ids) and all(row_id > 0 for row_id in ids)

    # A follow-up read from the last id the client holds returns exactly the rows it lacks.
    status, body2 = _call_route(db, {"after_row_id": str(ids[-1])})
    assert status == 200
    assert body2["pagination"]["returned"] == 3
    assert all(row["id"] > ids[-1] for row in body2["data"])


def test_route_defaults_to_a_full_forward_page_and_never_newest_first(tmp_path):
    db = _db_with_rows(tmp_path)
    status, body = _call_route(db, {"after_row_id": "0"})
    assert status == 200
    assert body["pagination"]["limit"] == 500
    assert body["pagination"]["order"] == "oldest"
    assert body["pagination"]["returned"] == 5


def test_route_without_the_param_keeps_todays_shape(tmp_path):
    db = _db_with_rows(tmp_path)
    status, body = _call_route(db, {})
    assert status == 200
    assert body["pagination"]["after_row_id"] is None
    assert body["pagination"]["order"] == "latest"
    assert body["pagination"]["returned"] == 5


def test_route_rejects_a_bad_cursor_with_a_named_code(tmp_path):
    db = _db_with_rows(tmp_path)
    status, body = _call_route(db, {"after_row_id": "abc"})
    assert status == 400
    assert body["error"]["code"] == "invalid_cursor"


def test_route_rejects_cursor_combined_with_merge_modes(tmp_path):
    db = _db_with_rows(tmp_path)
    for params in (
        {"after_row_id": "1", "offset": "2"},
        {"after_row_id": "1", "include_compacted": "true"},
        {"after_row_id": "1", "order": "latest"},
    ):
        status, body = _call_route(db, params)
        assert status == 400, params
        assert body["error"]["code"] == "invalid_pagination", params
