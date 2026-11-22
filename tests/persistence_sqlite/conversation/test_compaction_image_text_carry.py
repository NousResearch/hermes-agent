"""Text-projected image rows remain recognizable after compression (#155402)."""

def test_a_carry_whose_image_was_stripped_claims_its_text_row(tmp_path) -> None:
    """Rows hold the flush's text projection, where an image is a ``[screenshot]`` line (#148146): a carried
    copy whose image the compressor swapped for a placeholder is that row rewritten, not a second message."""
    from persistence_assembly import open_persistence, profile_scope
    from persistence.conversation.messages import NewMessage, TranscriptView
    from tests.fixtures.storage_seed import seed_session, seed_messages
    from tests.fixtures.storage_read import conversation, transcript_rows
    db = open_persistence(profile_scope(home=tmp_path))
    SID = "carry"
    seed_session(db, SID, source="test")
    watermark = db.messages.active_watermark(SID)
    call = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "vision_analyze", "arguments": "{}"}}]}
    result = {"role": "tool", "tool_call_id": "call_1", "content": "Image loaded.\n[screenshot]"}
    seed_messages(db, SID, call, result)
    stripped = {**result, "content": "Image loaded.\n[Attached image — stripped after compression]"}

    db.messages.compact(SID, [NewMessage.from_live(m) for m in [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"}, dict(call), stripped]],
        watermark=watermark, carried_tail_count=2)

    live = conversation(db, SID)
    assert [m["role"] for m in live] == ["user", "assistant", "tool"]
    assert live[-1]["content"] == stripped["content"]
    assert "Image loaded.\n[screenshot]" in [m["content"] for m in transcript_rows(db, SID, TranscriptView.ALL)]
    db.close()
