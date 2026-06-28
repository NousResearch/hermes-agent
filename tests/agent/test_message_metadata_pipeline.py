from types import SimpleNamespace

from agent.agent_runtime_helpers import sanitize_api_messages
from agent.chat_completion_helpers import build_assistant_message
from agent.tool_dispatch_helpers import make_tool_result_message
from agent.transports.chat_completions import ChatCompletionsTransport
from hermes_state import SessionDB


def test_session_db_persists_and_replays_message_metadata(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli")

    db.append_message(
        "s1",
        role="user",
        content="hello",
        turn_id="turn-1",
        compression_generation=2,
    )
    db.append_message(
        "s1",
        role="assistant",
        content="hi",
        turn_id="turn-1",
        compression_generation=2,
    )
    db.append_message(
        "s1",
        role="tool",
        content="ok",
        tool_call_id="call-1",
        tool_name="read_file",
        turn_id="turn-1",
        compression_generation=2,
    )

    rows = db.get_messages("s1")
    assert [row["turn_id"] for row in rows] == ["turn-1", "turn-1", "turn-1"]
    assert [row["compression_generation"] for row in rows] == [2, 2, 2]

    replay = db.get_messages_as_conversation("s1")
    assert [msg["turn_id"] for msg in replay] == ["turn-1", "turn-1", "turn-1"]
    assert [msg["compression_generation"] for msg in replay] == [2, 2, 2]


def test_archive_and_compact_stamps_next_compression_generation(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli")
    db.append_message(
        "s1",
        role="user",
        content="before",
        turn_id="turn-1",
        compression_generation=0,
    )

    db.archive_and_compact(
        "s1",
        [{"role": "assistant", "content": "summary", "turn_id": "compression:1"}],
    )

    active = db.get_messages("s1")
    archived = db.get_messages("s1", include_inactive=True)
    assert active[0]["compression_generation"] == 1
    assert active[0]["turn_id"] == "compression:1"
    assert any(row["content"] == "before" and row["compression_generation"] == 0 for row in archived)


def test_runtime_assistant_and_tool_messages_carry_metadata():
    agent = SimpleNamespace(
        _current_turn_id="turn-1",
        _current_compression_generation=3,
        verbose_logging=False,
        reasoning_callback=None,
        stream_delta_callback=None,
        _stream_callback=None,
        _extract_reasoning=lambda _msg: None,
        _strip_think_blocks=lambda text: text,
        _needs_thinking_reasoning_pad=lambda: False,
    )

    assistant = build_assistant_message(
        agent,
        SimpleNamespace(content="done", tool_calls=[]),
        "stop",
    )
    assert assistant["turn_id"] == "turn-1"
    assert assistant["compression_generation"] == 3

    tool = make_tool_result_message(
        "read_file",
        "ok",
        "call-1",
        turn_id="turn-1",
        compression_generation=3,
    )
    assert tool["turn_id"] == "turn-1"
    assert tool["compression_generation"] == 3


def test_api_sanitizers_strip_local_message_metadata():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "function": {"name": "read_file", "arguments": "{}"}}],
            "turn_id": "turn-1",
            "compression_generation": 4,
        },
        {
            "role": "tool",
            "content": "ok",
            "tool_call_id": "call-1",
            "turn_id": "turn-1",
            "compression_generation": 4,
        },
    ]

    sanitized = sanitize_api_messages(messages)
    assert all("turn_id" not in msg for msg in sanitized)
    assert all("compression_generation" not in msg for msg in sanitized)
    assert messages[0]["turn_id"] == "turn-1"

    transport = ChatCompletionsTransport()
    converted = transport.convert_messages(
        [
            {
                "role": "user",
                "content": "hello",
                "timestamp": 123.0,
                "turn_id": "turn-2",
                "compression_generation": 5,
            }
        ],
        model="test-model",
    )
    assert converted == [{"role": "user", "content": "hello"}]
