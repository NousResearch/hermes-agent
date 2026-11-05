"""The seed helpers write sessions, transcripts and admission settlements through the storage interface."""
from tests.fixtures.storage_seed import seed_messages, seed_session, settle_claim
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

TOOL_CALL = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}


def test_seed_session_creates_the_row_then_titles_flags_and_ends_it(tmp_path):
    store = storage_handle(tmp_path)
    assert seed_session(store, "s", "telegram", model="m", origin_json='{"platform": "telegram"}', title="Title",
                        pinned=True, archived=True, hidden=True, end_reason="user_exit") == "s"
    record = store.sessions.get("s")
    assert (record.source, record.model, dict(record.origin), record.title) == ("telegram", "m", {"platform": "telegram"}, "Title")
    assert (record.pinned, record.archived, record.hidden, record.end_reason) == (True, True, True, "user_exit")


def test_seed_session_leaves_unset_columns_alone(tmp_path):
    store = storage_handle(tmp_path)
    seed_session(store, "s", "cli")
    record = store.sessions.get("s")
    assert (record.title, record.pinned, record.archived, record.hidden, record.ended_at) == (None, False, False, False, None)


def test_seed_messages_appends_each_dict_in_order(tmp_path):
    store = storage_handle(tmp_path)
    seed_session(store, "s", "cli")
    ids = seed_messages(store, "s", {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
                        {"role": "tool", "content": "out", "tool_call_id": "c1", "tool_name": "f"},
                        {"role": "assistant", "content": "summary", "_compressed_summary": True})
    rows = store.messages.read_transcript("s")
    assert [row.id for row in rows] == ids
    assert [(row.role, row.content) for row in rows] == [("user", "hi"), ("assistant", ""), ("tool", "out"), ("assistant", "summary")]
    assert (rows[1].tool_calls, rows[2].tool_call_id, rows[2].tool_name, rows[3].compressed_summary) == ([TOOL_CALL], "c1", "f", True)
    assert store.sessions.get("s").message_count == 4


def test_settle_claim_settles_the_claimed_admission(tmp_path):
    store = storage_handle(tmp_path)
    seed_session(store, "s", "cli")
    epoch = store.runtime.begin_epoch(instance_id="test")
    admitted, _ = store.admissions.admit(epoch=epoch, principal_id="owner", session_id="s", request_id="r1", payload={"text": "hi"})
    claimed = store.admissions.claim(epoch=epoch, session_id="s")
    assert claimed.admission_id == admitted.admission_id
    settled = settle_claim(store, claimed)
    assert (settled.admission_id, settled.status, settled.outcome) == (admitted.admission_id, "terminal", "completed")
    assert store.admissions.get(admission_id=admitted.admission_id).outcome == "completed"
