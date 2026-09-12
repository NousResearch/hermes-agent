"""Tests for agent.replay_cleanup — shared replay-tail sanitizers.

These functions were extracted from gateway/run.py so every resume surface
(messaging gateway AND TUI/WebUI gateway) strips poisoned tool-call tails the
same way. Regression coverage for #29086 (WebUI session permanently stuck
because the dangling tool-call tail was replayed on every resume).
"""

from agent.replay_cleanup import (
    is_interrupted_tool_result,
    strip_dangling_tool_call_tail,
    strip_interrupted_tool_tails,
    sanitize_replay_history,
)


def _user(text):
    return {"role": "user", "content": text}


def _assistant_tc(name):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": name, "arguments": "{}"}}
        ],
    }


def _tool(content):
    return {"role": "tool", "tool_call_id": "c1", "content": content}










def test_mixed_dangling_batch_uses_truthful_per_call_wording():
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "read", "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "write", "function": {"name": "write_file", "arguments": "{}"}},
        ],
    }
    out = strip_dangling_tool_call_tail([_user("hi"), assistant])

    read_result, write_result = out[-2:]
    assert read_result["effect_disposition"] == "none"
    assert "no effect" in read_result["content"].lower()
    assert "unknown" not in read_result["content"].lower()
    assert write_result["effect_disposition"] == "unknown"
    assert "unknown" in write_result["content"].lower()












def test_sanitize_replay_history_combines_both():
    # interrupted block is removed; a dangling read-only call is safe to erase
    history = [
        _user("first"),
        _assistant_tc("terminal"), _tool("[Command interrupted]"),
        _user("second"),
        _assistant_tc("read_file"),  # dangling
    ]
    out = sanitize_replay_history(history)
    assert out[:2] == [
        _user("first"),
        _assistant_tc("terminal"),
    ]
    assert out[2]["effect_disposition"] == "unknown"
    assert out[-1] == _user("second")


def test_sanitize_replay_history_noop_on_clean_history():
    history = [_user("hi"), {"role": "assistant", "content": "hello"}]
    assert sanitize_replay_history(history) == history


def test_sanitize_replay_history_empty():
    assert sanitize_replay_history([]) == []


# --- Send/replay canonicalization parity (#105236 §6, salvage of #105308) ---

import copy
import json

from agent.replay_cleanup import canonicalize_replay_history
from agent.transports.chat_completions import ChatCompletionsTransport
from agent.turn_context import build_api_messages
from hermes_state import SessionDB


class _SendAgent:
    api_mode = "chat_completions"
    ephemeral_system_prompt = None
    _compression_warning = None
    _current_turn_timestamp = 10_000.0

    @staticmethod
    def _copy_reasoning_content_for_api(_source, _target):
        return None

    @staticmethod
    def _should_sanitize_tool_calls():
        return False


def _wire(messages):
    return json.dumps(ChatCompletionsTransport().convert_messages(list(messages)), sort_keys=True)


def _send(agent, history):
    request, _ = build_api_messages(
        agent, history, current_turn_user_idx=len(history) - 1, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )
    return request


def test_send_wire_matches_replay_wire_after_db_round_trip(tmp_path):
    """The bytes a resumed session replays and the bytes the live send path emits for the
    same persisted prefix are identical through the real transport, sidecars applied; the
    durable transcript is untouched; rows appended by the CURRENT turn are never rewritten."""
    now = 10_000.0
    db = SessionDB(db_path=tmp_path / "t.db")
    db.create_session(session_id="s1", source="cli")
    db.append_message("s1", role="user", content="hello", api_content="hello [with memory]", timestamp=now - 300)
    db.append_message("s1", role="assistant", content="hi", timestamp=now - 299)
    db.append_message("s1", role="user", content="confirm reboot", timestamp=now - 120)
    db.append_message("s1", role="assistant", content="", tool_calls=[
        {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}], timestamp=now - 119)
    db.append_message("s1", role="tool", content="[execution interrupted — user stop]", tool_call_id="c1", tool_name="read_file", timestamp=now - 118)
    # An ordinary result that merely mentions interrupts is NOT replay debris.
    grep_hit = '{"output": "loop.py:12: except KeyboardInterrupt:\\n@@ -1,4 +1,5 @@", "exit_code": 0}'
    db.append_message("s1", role="assistant", content="", tool_calls=[
        {"id": "c2", "type": "function", "function": {"name": "search_files", "arguments": "{}"}}], timestamp=now - 110)
    db.append_message("s1", role="tool", content=grep_hit, tool_call_id="c2", tool_name="search_files", timestamp=now - 109)
    persisted = db.get_messages_as_conversation("s1")
    db.close()
    assert persisted[0].get("api_content") == "hello [with memory]" and persisted[2].get("timestamp")

    # What every resume surface feeds the model, with the sidecar bytes the send path replays.
    replay = [{**m, "content": m.get("api_content") or m.get("content")}
              for m in canonicalize_replay_history(persisted, now=now)]
    live = copy.deepcopy(persisted) + [{"role": "user", "content": "now", "timestamp": now}]
    frozen = copy.deepcopy(live)
    request = _send(_SendAgent(), live)

    assert live == frozen
    assert _wire(request) == _wire(replay + [{"role": "user", "content": "now"}])
    assert "[with memory]" in request[0]["content"] and "EXPIRED" in request[2]["content"]
    assert [m["role"] for m in request] == ["user", "assistant", "user", "assistant", "tool", "user"]
    assert request[4]["content"] == grep_hit

    # Rows this turn appended stay verbatim even when they look like replay debris.
    live += [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c3", "type": "function", "function": {"name": "search_files", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c3", "content": "[Command interrupted]"},
    ]
    request2 = build_api_messages(
        _SendAgent(), live, current_turn_user_idx=len(persisted), ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )[0]
    assert _wire(request2[: len(request)]) == _wire(request)
    assert request2[-1]["content"] == "[Command interrupted]"


def test_confirmation_expiry_uses_frozen_admission_clock_and_fails_closed(monkeypatch):
    """Expiry is judged once per turn at admission (not the input's event stamp, not
    per-request wall time); a present-but-corrupt stamp is treated as expired."""
    from agent.turn_context import _reset_per_turn_agent_state

    agent = _SendAgent()
    agent._tool_guardrails = type("G", (), {"reset_for_turn": staticmethod(lambda: None)})()
    agent._memory_store = None
    agent.max_iterations = 4
    history = [
        {"role": "user", "content": "confirm reboot", "timestamp": 9_941.0},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "go", "timestamp": 10_000.0},  # platform event stamp: 59s old
    ]

    monkeypatch.setattr("agent.turn_context.time.time", lambda: 10_070.0)  # admitted 129s later
    _reset_per_turn_agent_state(agent)
    request = _send(agent, history)
    assert "EXPIRED" in request[0]["content"]
    assert _wire(request[:2]) == _wire(canonicalize_replay_history(history[:2], now=10_070.0))

    agent._current_turn_timestamp = 9_990.0  # admitted at 49s: fresh, and stays fresh...
    history += [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ready"}]
    monkeypatch.setattr("agent.turn_context.time.time", lambda: 10_500.0)  # ...however long the tools take
    late = build_api_messages(agent, history, current_turn_user_idx=2, ext_prefetch_cache="",
                              plugin_user_context="", moa_config=None, active_system_prompt="")[0]
    assert late[0]["content"] == "confirm reboot"

    corrupt = [{"role": "user", "content": "confirm reboot", "timestamp": "nan", "api_content": "confirm reboot"}]
    out = canonicalize_replay_history(corrupt, now=10_000.0)
    assert "EXPIRED" in out[0]["content"] and "api_content" not in out[0]
    assert canonicalize_replay_history([{"role": "user", "content": "confirm reboot"}], now=1e9)[0]["content"] == "confirm reboot"
