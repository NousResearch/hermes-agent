"""Unit tests for the AG-UI event bridge.

These drive the bridge's callback surface with synthetic data (exactly the
shapes the real Hermes agent produces) and assert the emitted AG-UI event
sequence. No provider, network, or agent is required.
"""

from ag_ui.core import EventType

from agui_adapter.events import AGUIEventBridge


def _bridge():
    events = []
    # Deterministic ids so sequences are easy to assert on.
    counter = {"n": 0}

    def id_factory(prefix: str) -> str:
        counter["n"] += 1
        return f"{prefix}{counter['n']}"

    bridge = AGUIEventBridge(events.append, id_factory=id_factory)
    return bridge, events


def _types(events):
    return [e.type for e in events]


def test_text_lifecycle_opens_and_closes_once():
    bridge, events = _bridge()
    bridge.on_text_delta("Hello")
    bridge.on_text_delta(", world")
    bridge.on_text_delta(None)  # flush

    assert _types(events) == [
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
    ]
    # Same message id throughout.
    assert events[0].message_id == events[1].message_id == events[3].message_id
    assert events[0].role == "assistant"
    assert events[1].delta == "Hello"


def test_reasoning_then_text_closes_thinking_first():
    bridge, events = _bridge()
    bridge.on_reasoning_delta("let me think")
    bridge.on_text_delta("answer")
    bridge.finish()

    assert _types(events) == [
        EventType.REASONING_MESSAGE_START,
        EventType.REASONING_MESSAGE_CONTENT,
        # text delta arriving closes the open reasoning message first
        EventType.REASONING_MESSAGE_END,
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
    ]
    # Reasoning message events share one message_id and carry role "reasoning".
    reasoning = [
        e
        for e in events
        if e.type
        in (
            EventType.REASONING_MESSAGE_START,
            EventType.REASONING_MESSAGE_CONTENT,
            EventType.REASONING_MESSAGE_END,
        )
    ]
    assert reasoning[0].message_id == reasoning[1].message_id == reasoning[2].message_id
    assert reasoning[0].role == "reasoning"
    assert reasoning[1].delta == "let me think"


def test_tool_start_then_completion_via_step():
    bridge, events = _bridge()
    bridge.on_tool_progress("tool.started", "read_file", None, {"path": "a.py"})
    # completion arrives on the next step as prev_tools
    bridge.on_step(2, [{"name": "read_file", "result": "file contents"}])

    assert _types(events) == [
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_ARGS,
        EventType.TOOL_CALL_END,
        EventType.TOOL_CALL_RESULT,
    ]
    start, args, end, result = events
    assert start.tool_call_name == "read_file"
    assert start.tool_call_id == args.tool_call_id == end.tool_call_id == result.tool_call_id
    assert args.delta == '{"path": "a.py"}'
    assert result.content == "file contents"


def test_parallel_same_name_tools_pair_fifo():
    bridge, events = _bridge()
    bridge.on_tool_progress("tool.started", "read_file", None, {"path": "a"})
    bridge.on_tool_progress("tool.started", "read_file", None, {"path": "b"})
    bridge.on_step(2, [
        {"name": "read_file", "result": "RA"},
        {"name": "read_file", "result": "RB"},
    ])

    starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
    results = [e for e in events if e.type == EventType.TOOL_CALL_RESULT]
    assert len(starts) == 2 and len(results) == 2
    # FIFO: first start pairs with first result.
    assert starts[0].tool_call_id == results[0].tool_call_id
    assert starts[1].tool_call_id == results[1].tool_call_id
    assert results[0].content == "RA"
    assert results[1].content == "RB"


def test_client_tool_id_override_matches_model_id():
    bridge, events = _bridge()
    # Adapter binds the model-issued id before the tool starts.
    bridge.bind_client_tool_id("show_dialog", "call_model_123")
    bridge.on_tool_progress("tool.started", "show_dialog", None, {"msg": "hi"})

    start = events[0]
    assert start.type == EventType.TOOL_CALL_START
    assert start.tool_call_id == "call_model_123"


def test_open_text_is_closed_before_tool_call():
    bridge, events = _bridge()
    bridge.on_text_delta("thinking out loud")
    bridge.on_tool_progress("tool.started", "terminal", None, {"command": "ls"})

    types = _types(events)
    # The text message must end before the tool call starts.
    assert types.index(EventType.TEXT_MESSAGE_END) < types.index(EventType.TOOL_CALL_START)


def test_text_streams_live_not_buffered():
    """Each text delta emits its own TEXT_MESSAGE_CONTENT the moment it arrives —
    text is never buffered into one chunk. This is the load-bearing property:
    the proper tool-ordering fix must NOT sacrifice live token streaming."""
    bridge, events = _bridge()
    bridge.on_text_delta("Hel")
    # As soon as the first delta lands, START + its CONTENT are already emitted
    # (not held until a later flush).
    assert _types(events) == [EventType.TEXT_MESSAGE_START, EventType.TEXT_MESSAGE_CONTENT]
    bridge.on_text_delta("lo ")
    bridge.on_text_delta("there")
    contents = [e for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT]
    # Three separate deltas => three separate content events (multi-chunk stream).
    assert [c.delta for c in contents] == ["Hel", "lo ", "there"]
    # All share one message id (one streaming assistant message).
    assert len({c.message_id for c in contents}) == 1


def test_server_tool_interleaves_text_card_text():
    """Server tool: preamble text streams, then the card streams live
    (START/ARGS on tool.started, END/RESULT on the next step), then the model's
    follow-up narration streams — all in DOM order, no buffering."""
    bridge, events = _bridge()
    # Iteration N: preamble narration streams live.
    bridge.on_text_delta("Let me check that.")
    # ...then the server tool starts (same iteration) -> live START/ARGS.
    bridge.on_tool_progress("tool.started", "read_file", None, {"path": "a.py"})
    # Iteration N+1: step_callback delivers the completion -> live END/RESULT.
    bridge.on_step(2, [{"name": "read_file", "result": "contents"}])
    # ...then the follow-up narration streams live.
    bridge.on_text_delta("The file says hello.")
    bridge.finish()

    types = _types(events)
    assert types == [
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,   # preamble
        EventType.TEXT_MESSAGE_END,       # closed before the tool card
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_ARGS,
        EventType.TOOL_CALL_END,
        EventType.TOOL_CALL_RESULT,
        EventType.TEXT_MESSAGE_START,     # follow-up narration, AFTER the card
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
    ]
    result = next(e for e in events if e.type == EventType.TOOL_CALL_RESULT)
    assert result.content == "contents"


def test_frontend_tool_skipped_in_live_path():
    """Frontend/client tools are owned by the server's post-run handoff (real
    model ids), so the live path emits NO TOOL_CALL_* events for them."""
    bridge, events = _bridge()
    bridge.set_tool_classes(frontend_names={"change_background"})
    bridge.on_tool_progress("tool.started", "change_background", None, {"background": "#fff"})
    # A completion for it arriving on the next step is likewise ignored.
    bridge.on_step(2, [{"name": "change_background", "result": "ok"}])
    assert events == []


def test_state_writer_tool_suppressed_in_live_path():
    """State-writer tools are represented by a StateSnapshotEvent (emitted by the
    server), not a chip, so the live path suppresses their TOOL_CALL_* events."""
    bridge, events = _bridge()
    bridge.set_tool_classes(state_writer_names={"set_notes"})
    bridge.on_tool_progress("tool.started", "set_notes", None, {"notes": ["a"]})
    bridge.on_step(2, [{"name": "set_notes", "result": "State updated."}])
    assert events == []


def test_mixed_batch_server_tool_flushed_on_interrupt():
    """When a server tool and a frontend tool share one batch, the frontend tool
    interrupts the run so step_callback never fires. The server tool's START/ARGS
    still streamed live; the server closes the dangling card via
    ``flush_pending_server_tools`` from the post-run message results."""
    bridge, events = _bridge()
    bridge.set_tool_classes(frontend_names={"change_background"})
    # Live: server tool starts (frontend tool is skipped in the live path).
    bridge.on_tool_progress("tool.started", "echo_server", None, {"text": "hi"})
    bridge.on_tool_progress("tool.started", "change_background", None, {"background": "#fff"})
    # No step_callback (run unwound via interrupt). Server flushes from messages.
    from collections import deque

    bridge.flush_pending_server_tools({"echo_server": deque(["SERVER_OK"])})

    types = _types(events)
    # Only the server tool produced events; the frontend tool was skipped.
    starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
    assert len(starts) == 1 and starts[0].tool_call_name == "echo_server"
    assert types == [
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_ARGS,
        EventType.TOOL_CALL_END,
        EventType.TOOL_CALL_RESULT,
    ]
    assert events[-1].content == "SERVER_OK"


def test_close_open_closes_dangling_text_on_error():
    """The common mid-stream failure: the LLM stream dies after some text
    deltas, leaving TEXT_MESSAGE open. On the error path the server calls
    ``close_open`` before RUN_ERROR so the client doesn't render the message
    as perpetually streaming."""
    bridge, events = _bridge()
    bridge.on_text_delta("partial ans")  # stream opens a text message, then dies
    events.clear()  # focus on what close_open emits

    bridge.close_open()

    assert _types(events) == [EventType.TEXT_MESSAGE_END]
    # Idempotent: a second call emits nothing (everything already closed).
    events.clear()
    bridge.close_open()
    assert events == []


def test_close_open_closes_multiple_dangling_tools_on_error():
    """A parallel server-tool batch that dies before on_step pairs any
    completion leaves several TOOL_CALL starts open; close_open force-closes
    each with a bare TOOL_CALL_END (no result — the run failed)."""
    bridge, events = _bridge()
    bridge.on_tool_progress("tool.started", "get_weather", None, {"city": "SF"})
    bridge.on_tool_progress("tool.started", "get_stock_price", None, {"ticker": "AAPL"})
    events.clear()

    bridge.close_open()

    assert _types(events) == [EventType.TOOL_CALL_END, EventType.TOOL_CALL_END]


def test_finish_leaves_tool_open_but_close_open_force_closes_it():
    """``finish`` (success path) closes only text/reasoning and intentionally
    leaves tool starts open for on_step/flush; ``close_open`` (error path) is
    the one that force-closes them."""
    bridge, events = _bridge()
    bridge.on_tool_progress("tool.started", "get_weather", None, {"city": "SF"})
    events.clear()

    bridge.finish()
    assert events == []  # finish does NOT close the open tool call

    bridge.close_open()
    assert _types(events) == [EventType.TOOL_CALL_END]
