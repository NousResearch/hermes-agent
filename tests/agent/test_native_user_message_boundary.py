"""Exact persisted current-row metadata for existing request middleware."""
from types import SimpleNamespace

import pytest

from agent.turn_api_request import _native_user_message
from hermes_state import SessionDB


@pytest.fixture
def native(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("parent", source="cli")
    db.create_session("child", source="cli", parent_session_id="parent")
    yield db
    db.close()


def export(db, session, messages, index, original="human request", wrapper="task wrapper"):
    return _native_user_message(SimpleNamespace(_session_db=db, session_id=session),
                                messages, index, wrapper, original)


def test_initial_override_and_compressed_copy_have_distinct_exact_anchors(native):
    parent_id = native.append_message("parent", "user", "human request")
    child_id = native.append_message("child", "user", "task wrapper")
    parent = {"role": "user", "content": "task wrapper", "_row_id": parent_id}
    child = {**parent, "_row_id": child_id}
    assert export(native, "parent", [parent], 0) == {
        "role": "user", "content": "human request", "_row_id": parent_id}
    assert export(native, "child", [child], 0) == {
        "role": "user", "content": "task wrapper", "_row_id": child_id}
    # Identical text in a parent never resolves the wrong-session coordinate.
    assert export(native, "child", [parent], 0) is None


def test_index_not_last_user_or_text_search_controls_anchor(native):
    earlier = native.append_message("parent", "user", "human request")
    current = native.append_message("parent", "user", "human request")
    steering = native.append_message("parent", "user", "steering instruction")
    messages = [{"role": "user", "content": "task wrapper", "_row_id": row_id}
                for row_id in (earlier, current)]
    messages.append({"role": "user", "content": "steering instruction", "_row_id": steering})
    assert export(native, "parent", messages, 1)["_row_id"] == current
    assert export(native, "parent", messages, 2) is None
    assert export(native, "parent", messages, -1) is None
    assert export(native, "parent", messages, True) is None


def test_changed_or_missing_native_preimage_is_not_exported(native):
    row_id = native.append_message("parent", "user", "changed unrelated input")
    row = {"role": "user", "content": "task wrapper", "_row_id": row_id}
    assert export(native, "parent", [row], 0) is None
    assert export(native, "parent", [{**row, "_row_id": row_id + 1}], 0) is None
    assert export(native, "parent", [{**row, "role": "assistant"}], 0) is None


def test_multimodal_descriptor_is_an_independent_copy(native):
    content = [{"type": "text", "text": "human request"},
               {"type": "image_url", "image_url": {"url": "asset://test"}}]
    row_id = native.append_message("parent", "user", content)
    row = {"role": "user", "content": "task wrapper", "_row_id": row_id}
    result = export(native, "parent", [row], 0, original=content)
    assert result["content"] == content
    result["content"][0]["text"] = "modified observer copy"
    assert content[0]["text"] == "human request"


def test_unavailable_storage_returns_no_descriptor(native, monkeypatch):
    row_id = native.append_message("parent", "user", "human request")
    row = {"role": "user", "content": "task wrapper", "_row_id": row_id}
    def unavailable(*args, **kwargs):
        raise OSError("storage unavailable")
    monkeypatch.setattr(native, "get_messages", unavailable)
    assert export(native, "parent", [row], 0) is None
