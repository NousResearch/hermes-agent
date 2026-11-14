"""A delivered reply folded into an older row must reach provider replay (#144546)."""

from copy import deepcopy

import pytest

from agent.anthropic_message_convert import _convert_assistant_message
from agent.bedrock_adapter import _assistant_blocks
from agent.context_compressor import MODEL_ONLY_DISPLAY_METADATA_KEY
from agent.conversation_compression_reply_anchor import _ensure_compressed_keeps_last_assistant_reply
from agent.message_metadata import ABSORBED_MESSAGE_UIDS, MESSAGE_UID
from agent.turn_context import _reset_per_turn_agent_state, build_api_messages
from tests.agent.test_compression_last_assistant_anchor import (
    FOLLOWER,
    REPLY,
    SUMMARY,
    _assert_no_same_role_adjacency,
    _compress,
    _seed,
    _tool_chain,
)


def _merge_case():
    older = {"role": "assistant", "content": "noted", MESSAGE_UID: "older-reply", "_row_id": 2}
    original = [
        {"role": "user", "content": "Audit the auth module."},
        older,
        {"role": "user", "content": "Go deeper." * 100},
        {"role": "assistant", "content": REPLY, MESSAGE_UID: "delivered-reply"},
        {"role": "user", "content": FOLLOWER},
    ]
    compressed = [
        {"role": "user", "content": SUMMARY},
        deepcopy(older),
        deepcopy(original[-1]),
    ]
    return original, compressed


def _request(agent, messages):
    _reset_per_turn_agent_state(agent)
    request, _ = build_api_messages(
        agent, messages, current_turn_user_idx=len(messages) - 1,
        ext_prefetch_cache="", plugin_user_context="", moa_config=None, active_system_prompt="sys",
    )
    return request


def test_merge_records_original_reply_uid_and_keeps_survivor_identity(caplog):
    original, compressed = _merge_case()
    original[-2][ABSORBED_MESSAGE_UIDS] = ["earlier-fold"]

    with caplog.at_level("INFO", logger="agent.conversation_compression_reply_anchor"):
        assert _ensure_compressed_keeps_last_assistant_reply(original, compressed) is None

    survivor = compressed[1]
    assert REPLY.strip() in survivor["content"]
    assert survivor["display_metadata"][MODEL_ONLY_DISPLAY_METADATA_KEY] is True
    assert survivor[MESSAGE_UID] == "older-reply"
    assert survivor["_row_id"] == 2
    assert survivor[ABSORBED_MESSAGE_UIDS] == ["delivered-reply", "earlier-fold"]
    assert any("merged it" in r.message and "reply=delivered-reply" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "carrier",
    [
        {"codex_message_items": [{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "noted"}]}]},
        {"codex_reasoning_items": [{"type": "reasoning", "encrypted_content": "fixture"}]},
        {"finish_reason": "incomplete"},
    ],
    ids=["message_items", "reasoning_items", "incomplete"],
)
def test_codex_carrier_is_unchanged_and_skip_identifies_reply(carrier, caplog):
    original, compressed = _merge_case()
    compressed[1].update(carrier)
    before = deepcopy(compressed[1])

    with caplog.at_level("WARNING", logger="agent.conversation_compression_reply_anchor"):
        assert _ensure_compressed_keeps_last_assistant_reply(original, compressed, session_id="sid") is None

    assert compressed[1] == before
    assert any(
        "neighbour=codex" in r.message and "reply=delivered-reply" in r.message and "session=sid" in r.message
        for r in caplog.records
    )


@pytest.mark.parametrize(
    ("kind", "neighbour_fields"),
    [
        ("summary", {"content": "[CONTEXT COMPACTION] summary", "_compressed_summary": True}),
        ("tool_call", _tool_chain("call_1")[0]),
        ("paraphrase", {"content": "engine-invented paraphrase"}),
        ("following", {}),
        ("empty", {"content": None}),
    ],
    ids=["summary", "tool_call", "paraphrase", "following", "empty"],
)
@pytest.mark.parametrize("reply_uid", ["delivered-reply", None], ids=["uid", "no_uid"])
def test_skip_warning_identifies_neighbour_and_reply(kind, neighbour_fields, reply_uid, caplog):
    original, compressed = _merge_case()
    if reply_uid is None:
        original[-2].pop(MESSAGE_UID)
    compressed[1].update(neighbour_fields)
    if kind == "following":
        original = original[:-1] + _tool_chain("call_1")
        compressed = [compressed[0], *_tool_chain("call_1")]
    before = deepcopy(compressed)

    with caplog.at_level("WARNING", logger="agent.conversation_compression_reply_anchor"):
        assert _ensure_compressed_keeps_last_assistant_reply(original, compressed) is None

    assert compressed == before
    assert any(
        f"neighbour={kind}" in r.message and f"reply={reply_uid or 'none'}" in r.message
        for r in caplog.records
    )


def test_merge_leaves_a_shared_live_original_as_stored():
    original, compressed = _merge_case()
    compressed[1] = original[1]  # an engine may keep the live dict itself
    assert _ensure_compressed_keeps_last_assistant_reply(original, compressed) is None
    assert original[1] == {"role": "assistant", "content": "noted", MESSAGE_UID: "older-reply", "_row_id": 2}
    assert compressed[1] is not original[1] and REPLY.strip() in compressed[1]["content"]


THINKING = [{"type": "thinking", "thinking": "plan", "signature": "sig"}]


@pytest.mark.parametrize(
    ("carrier", "replay"),
    [
        ({"anthropic_content_blocks": [{"type": "text", "text": "noted"}]},
         lambda m: _convert_assistant_message(m)["content"]),
        ({"bedrock_content_blocks": [{"text": "noted"}]}, lambda m: _assistant_blocks(m, m["content"])),
    ],
    ids=["anthropic", "bedrock"],
)
def test_native_ordered_carrier_replays_the_merged_reply(carrier, replay):
    original, compressed = _merge_case()
    compressed[1].update(carrier, reasoning_details=THINKING)
    assert _ensure_compressed_keeps_last_assistant_reply(original, compressed) is None
    merged = compressed[1]
    assert REPLY.strip() in str(replay(merged))
    assert merged["reasoning_details"] == THINKING  # the persisted carrier the replay falls back to


def test_merge_is_charged_its_growth_not_a_peer_row():
    """The delivered row's reasoning never moves into the merge, so it must not count against it."""
    original, compressed = _merge_case()
    original[-2]["reasoning_content"] = "r" * 4000
    compressed[0] = {"role": "user", "content": SUMMARY + " s" * 1000}
    assert _ensure_compressed_keeps_last_assistant_reply(original, compressed) is None
    assert REPLY.strip() in compressed[1]["content"]
    assert "reasoning_content" not in compressed[1]


def test_api_sidecar_merge_reaches_next_request_and_durable_active_set(tmp_path):
    fold = [
        {"role": "user", "content": SUMMARY},
        {"role": "assistant", "content": "noted"},
        {"role": "user", "content": FOLLOWER},
    ]
    db, session_id, messages, agent = _seed(
        tmp_path, [("user", "ok"), ("assistant", "noted"), ("user", "ok")], fold, FOLLOWER,
    )
    try:
        older, reply = messages[-4], messages[-2]
        db.set_message_api_content(session_id, older["_row_id"], older["content"], "stale provider text")
        older["api_content"] = "stale provider text"
        fold[1] = deepcopy(older)
        older_uid, reply_uid = older[MESSAGE_UID], reply[MESSAGE_UID]

        out_messages = _compress(agent, messages)

        assert agent._last_compaction_in_place is True
        live = db.get_messages_as_conversation(session_id)
        for view in (out_messages, live):
            _assert_no_same_role_adjacency(view)
            uids = [m[MESSAGE_UID] for m in view]
            assert len(uids) == len(set(uids)), "the commit cloned a carried row into the active set"
            assert sum(str(m.get("content", "")).count(REPLY.strip()) for m in view) == 1
            request = _request(agent, view)
            assert sum(str(m.get("content", "")).count(REPLY.strip()) for m in request) == 1
            survivor = next(m for m in view if m.get(MESSAGE_UID) == older_uid)
            assert "api_content" not in survivor
            assert reply_uid in survivor[ABSORBED_MESSAGE_UIDS]
            assert all(MESSAGE_UID not in m and ABSORBED_MESSAGE_UIDS not in m for m in request)
        # Full history (the warm-switch read) shows each source row once and not the model-only join.
        display = db.get_messages_as_conversation(session_id, include_ancestors=True, include_compacted=True)
        assert sum(str(m.get("content", "")).count(REPLY.strip()) for m in display) == 1
        assert sum(m.get("content") == "noted" for m in display) == 1
    finally:
        db.close()
