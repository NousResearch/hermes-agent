"""Rehydrating a batch summary does not absorb its carrier's live payload."""
import pytest

from agent.context_compressor import ContextCompressor, MICRO_COMPACT_MARKER_KEY


@pytest.mark.parametrize("kind", ["user", "assistant", "tool_call"])
@pytest.mark.parametrize("defrag", [False, True])
def test_micro_rehydration_preserves_merged_live_payload(kind, defrag, tmp_path):
    cc = ContextCompressor(model="test-model", provider="test", config_context_length=40960,
                           quiet_mode=True, protect_first_n=0, protect_last_n=2)
    cc._micro_compact_enabled = True
    cc._micro_compact_defrag_threshold_tokens = 1 if defrag else 100000
    cc._micro_summarize_one = lambda text: "Fresh rolling facts."
    role = "assistant" if kind == "tool_call" else kind
    carrier = {"role": role, "content": "" if kind == "tool_call" else "LIVE CARRIER PAYLOAD"}
    if kind == "tool_call":
        carrier["tool_calls"] = [{"id": "carried", "type": "function", "function": {
            "name": "read_file", "arguments": "{}"}}]
    cc._merge_summary_into_tail_row(carrier, cc._with_summary_prefix("Old summary facts."), role, False)
    messages = [{"role": "system", "content": "system"}, carrier]
    if kind == "tool_call":
        messages.append({"role": "tool", "tool_call_id": "carried", "content": "LIVE TOOL RESULT"})
    for i in range(8):
        messages.extend([{"role": "user", "content": f"question {i}"},
                         {"role": "assistant", "content": f"answer {i} " + "x" * 400}])
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("carrier", source="cli")
        db.append_messages_batch("carrier", messages)
        cc._session_db, cc._session_id = db, "carrier"
        messages = db.get_resume_conversations("carrier")[0]
        result = cc._micro_compact(messages)
        resumed = db.get_resume_conversations("carrier")[0]
    finally:
        db.close()
    assert "Fresh rolling facts." in str(result), "must run the selected micro path"
    for view in [result, resumed]:
        live = [cc._strip_context_summary_handoff_message(m) for m in view]
        if kind == "tool_call":
            assert any(m and m.get("tool_calls") == carrier["tool_calls"] for m in live)
            assert any(m.get("tool_call_id") == "carried" for m in view)
        else:
            assert any(m and "LIVE CARRIER PAYLOAD" in str(m.get("content")) for m in live)
    if not defrag:
        recovered = next(m for m in result if m.get("tool_calls") or "LIVE CARRIER PAYLOAD" in str(m.get("content")))
        assert not recovered.get(MICRO_COMPACT_MARKER_KEY)
