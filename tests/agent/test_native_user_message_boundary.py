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


@pytest.mark.parametrize("override", [None, "clean human request"])
@pytest.mark.parametrize("image_count", [1, 2])
def test_flushed_image_turn_exports_its_durable_projection(native, override, image_count):
    from copy import deepcopy
    from agent.session_persistence import _db_flush_row, _db_flush_write

    content = [{"type": "text", "text": "inspect the attached image"}] + [
        {"type": "image_url", "image_url": {"url": f"asset://current-{index}"}}
        for index in range(image_count)]
    row = {"role": "user", "content": content}
    agent = SimpleNamespace(_session_db=native, session_id="parent",
                            _persist_user_message_override=override)
    _db_flush_write(agent, [_db_flush_row(agent, row, True)], [row], [row])
    before = deepcopy(row)
    result = _native_user_message(agent, [row], 0, content, override or content)
    assert result == {"role": "user", "content": "inspect the attached image" + "\n[screenshot]" * image_count,
                      "_row_id": row["_row_id"]}
    assert row == before
    # Projection validates the turn-owned coordinate, never another session or input.
    agent.session_id = "child"
    assert _native_user_message(agent, [row], 0, content, override or content) is None
    agent.session_id = "parent"
    changed = [{"type": "text", "text": "unrelated input"}, content[1]]
    assert _native_user_message(agent, [{**row, "content": changed}], 0,
                                changed, changed) is None


def test_unrepresented_parts_do_not_become_an_empty_native_anchor(native):
    from agent.session_persistence import _db_flush_row, _db_flush_write

    content = [{"type": "unsupported", "data": "not retained"}]
    row = {"role": "user", "content": content}
    agent = SimpleNamespace(_session_db=native, session_id="parent")
    _db_flush_write(agent, [_db_flush_row(agent, row, True)], [row], [row])
    assert _native_user_message(agent, [row], 0, content, content) is None


def test_unavailable_storage_returns_no_descriptor(native, monkeypatch):
    row_id = native.append_message("parent", "user", "human request")
    row = {"role": "user", "content": "task wrapper", "_row_id": row_id}
    def unavailable(*args, **kwargs):
        raise OSError("storage unavailable")
    monkeypatch.setattr(native, "get_messages", unavailable)
    assert export(native, "parent", [row], 0) is None


def test_resumed_durable_users_keep_current_anchor_and_merge_only_on_wire(native):
    from copy import deepcopy
    from agent.agent_runtime_helpers import (
        drop_thinking_only_and_merge_users, repair_message_sequence,
    )

    old_id = native.append_message("parent", "user", "earlier task")
    current_id = native.append_message("parent", "user", "resume task")
    # Gateway replay omits historical row IDs; the current turn-start flush
    # supplies its exact newly persisted coordinate before sequence repair.
    messages = [
        {"role": "user", "content": "earlier task", "_db_persisted": True},
        {"role": "user", "content": "resume task", "_row_id": current_id,
         "_db_persisted": True},
    ]
    before = deepcopy(messages)
    assert repair_message_sequence(SimpleNamespace(), messages) == 0
    assert messages == before
    assert export(native, "parent", messages, 1, original="resume task", wrapper="resume task") == {
        "role": "user", "content": "resume task", "_row_id": current_id}
    assert [(row["id"], row["content"]) for row in native.get_messages("parent")] == [
        (old_id, "earlier task"), (current_id, "resume task")]
    wire = drop_thinking_only_and_merge_users([
        {"role": row["role"], "content": row["content"]} for row in messages])
    assert wire == [{"role": "user", "content": "earlier task\n\nresume task"}]
    assert messages == before
