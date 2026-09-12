"""Tests for agent.replay_cleanup — shared replay-tail sanitizers.

These functions were extracted from gateway/run.py so every resume surface
(messaging gateway AND TUI/WebUI gateway) strips poisoned tool-call tails the
same way. Regression coverage for #29086 (WebUI session permanently stuck
because the dangling tool-call tail was replayed on every resume).
"""

import copy
import json
from pathlib import Path
import tempfile
import time

from agent.replay_cleanup import (
    canonicalize_history_for_send,
    canonicalize_replay_history,
    is_interrupted_tool_result,
    strip_dangling_tool_call_tail,
    strip_interrupted_tool_tails,
    strip_stale_dangerous_confirmations,
    sanitize_replay_history,
)
from agent.transports.chat_completions import ChatCompletionsTransport
from agent.turn_context import build_api_messages
from hermes_state import SessionDB


def _wire(messages):
    return ChatCompletionsTransport().convert_messages(list(messages))


def _canon(objs):
    return json.dumps(objs, sort_keys=True, separators=(",", ":"))


class _Agent:
    api_mode = "chat_completions"
    ephemeral_system_prompt = None
    _compression_warning = None
    max_iterations = 10

    @staticmethod
    def _copy_reasoning_content_for_api(_source, _target):
        return None

    @staticmethod
    def _should_sanitize_tool_calls():
        return False

    @staticmethod
    def _sanitize_tool_calls_for_strict_api(*_args, **_kwargs):
        return None


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


def test_canonicalize_replay_history_matches_all_resume_transforms():
    """Send and resume consumers must apply the same destructive transforms."""
    now = 10_000.0
    history = [
        _user("before"),
        _assistant_tc("read_file"), _tool("[command interrupted]"),
        {"role": "user", "content": "confirm forced restart", "timestamp": now - 120},
        {"role": "assistant", "content": "ack"},
    ]

    expected = strip_stale_dangerous_confirmations(
        sanitize_replay_history(copy.deepcopy(history)), now=now
    )
    actual = canonicalize_replay_history(copy.deepcopy(history), now=now)

    assert actual == expected


def test_canonicalize_history_for_send_alias():
    assert canonicalize_history_for_send is canonicalize_replay_history


def test_send_builder_uses_canonical_history_without_mutating_source():
    """The request copy must match replay cleanup while durable history stays intact."""
    from agent.turn_context import build_api_messages

    now = 10_000.0
    agent = _Agent()
    agent._current_turn_timestamp = now
    history = [
        _user("before"),
        {"role": "assistant", "content": "ack"},
        _assistant_tc("read_file"), _tool("[command interrupted]"),
        {"role": "user", "content": "confirm reboot", "timestamp": now - 120},
        {"role": "assistant", "content": "ack2"},
        {"role": "user", "content": "current", "api_content": "current-wire"},
    ]
    original = copy.deepcopy(history)

    request, _ = build_api_messages(
        agent, history, current_turn_user_idx=len(history) - 1, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )

    assert history == original
    assert [message["role"] for message in request] == [
        "user", "assistant", "user", "assistant", "user"
    ]
    assert "EXPIRED" in request[2]["content"]
    assert request[-1]["content"] == "current-wire"

    # Wire representation comparison: send wire matches replay wire with sidecar applied
    wire_request = _wire(request)
    expected_replay = canonicalize_replay_history(copy.deepcopy(history[:-1]), now=now) + [
        {"role": "user", "content": "current-wire"}
    ]
    assert _canon(wire_request) == _canon(_wire(expected_replay))


def test_send_byte_identity_with_tui_replay_interrupted_block():
    """Interrupted read-only assistant->tool block: send path wire bytes match TUI replay."""
    now = 10_000.0
    agent = _Agent()
    agent._current_turn_timestamp = now
    history = [
        _user("u1"),
        {"role": "assistant", "content": "a1"},
        _assistant_tc("read_file"), _tool("[command interrupted]"),
        _user("u2"),
    ]
    send_request, _ = build_api_messages(
        agent, history, current_turn_user_idx=len(history) - 1, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )
    wire_send = _wire(send_request)
    wire_replay = _wire(sanitize_replay_history(copy.deepcopy(history)))
    assert _canon(wire_send) == _canon(wire_replay)


def test_send_byte_identity_with_dangling_tool_call_tail():
    """Trailing unanswered assistant(tool_calls): send path wire bytes match TUI replay."""
    now = 10_000.0
    history = [
        _user("u1"),
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c9", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
    ]
    wire_replay = _wire(sanitize_replay_history(copy.deepcopy(history)))
    wire_canon = _wire(canonicalize_replay_history(copy.deepcopy(history), now=now))
    assert _canon(wire_canon) == _canon(wire_replay)


def test_send_byte_identity_with_stale_dangerous_confirmation():
    """Stale confirmation (>60s): send path wire bytes match gateway replay (redacted to sentinel)."""
    now = 10_000.0
    agent = _Agent()
    agent._current_turn_timestamp = now
    history = [
        {"role": "user", "content": "confirm forced restart", "timestamp": now - 120.0},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2", "timestamp": now},
    ]
    send_request, _ = build_api_messages(
        agent, history, current_turn_user_idx=len(history) - 1, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )
    wire_send = _wire(send_request)
    wire_replay = _wire(strip_stale_dangerous_confirmations(copy.deepcopy(history), now=now))
    assert _canon(wire_send) == _canon(wire_replay)
    assert any("EXPIRED" in (m.get("content") or "") for m in wire_send)


def test_send_byte_identity_with_fresh_confirmation():
    """Fresh confirmation (<60s): send path wire bytes match gateway replay (preserved verbatim)."""
    now = 10_000.0
    agent = _Agent()
    agent._current_turn_timestamp = now
    history = [
        {"role": "user", "content": "confirm reboot", "timestamp": now - 30.0},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2", "timestamp": now},
    ]
    send_request, _ = build_api_messages(
        agent, history, current_turn_user_idx=len(history) - 1, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )
    wire_send = _wire(send_request)
    wire_replay = _wire(strip_stale_dangerous_confirmations(copy.deepcopy(history), now=now))
    assert _canon(wire_send) == _canon(wire_replay)
    assert wire_send[0]["content"] == "confirm reboot"


def test_send_byte_identity_clean_history():
    """Clean history (no interrupted blocks, no stale confirmations): send matches replay."""
    now = 10_000.0
    agent = _Agent()
    agent._current_turn_timestamp = now
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    send_request, _ = build_api_messages(
        agent, history, current_turn_user_idx=len(history) - 1, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )
    wire_send = _wire(send_request)
    wire_replay = _wire(canonicalize_replay_history(copy.deepcopy(history), now=now))
    assert _canon(wire_send) == _canon(wire_replay) == _canon(_wire(history))


def test_send_byte_identity_with_sidecar():
    """Historical user turn with api_content sidecar: wire representation reproduces sidecar bytes."""
    now = 10_000.0
    agent = _Agent()
    agent._current_turn_timestamp = now
    history = [
        {"role": "user", "content": "hello", "api_content": "hello [with memory]", "timestamp": now - 100},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "current", "timestamp": now},
    ]
    send_request, _ = build_api_messages(
        agent, history, current_turn_user_idx=len(history) - 1, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )
    wire_send = _wire(send_request)
    assert wire_send[0]["content"] == "hello [with memory]"
    # Replay with sidecar intact matches
    replay = canonicalize_replay_history(copy.deepcopy(history), now=now)
    assert replay[0]["api_content"] == "hello [with memory]"


def test_active_turn_expiry_decision_frozen_across_tool_iterations(monkeypatch):
    """Deterministic wire-level probe (ehz0ah blocking defect):

    Confirmation timestamp: 9941.0.
    Active turn starts: 10000.0 (age = 59s <= 60s, fresh).
    Request 1 assembled at 10000.0 sends 'confirm reboot' on wire.
    Tool executes; request 2 assembled at 10002.0 (age = 61s > 60s).
    Because both requests belong to the same active turn, the expiry decision
    is frozen: request 2 does NOT rewrite the confirmation to EXPIRED, preserving
    the prompt cache prefix across tool iterations.
    Subsequent turn N+1 at 10070.0 DOES expire the confirmation and matches replay.
    """
    from agent.turn_context import _reset_per_turn_agent_state

    agent = _Agent()
    history = [
        {"role": "user", "content": "confirm reboot", "timestamp": 9941.0},
        {"role": "assistant", "content": "Preparing reboot..."},
        {"role": "user", "content": "proceed now", "timestamp": 10000.0},
    ]
    turn_user_idx = 2

    # Request 1 (first iteration of turn at t=10000.0):
    monkeypatch.setattr(time, "time", lambda: 10000.0)
    req1, _ = build_api_messages(
        agent, history, current_turn_user_idx=turn_user_idx, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )
    wire1 = _wire(req1)
    assert wire1[0]["content"] == "confirm reboot"

    # Tool executes during this turn; model response + tool result appended to history:
    history.append({
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "reboot_check", "arguments": "{}"}}],
    })
    history.append({"role": "tool", "tool_call_id": "c1", "content": "ready"})

    # Request 2 (second iteration of the SAME turn at t=10002.0 > 60s expiry threshold):
    monkeypatch.setattr(time, "time", lambda: 10002.0)
    req2, _ = build_api_messages(
        agent, history, current_turn_user_idx=turn_user_idx, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )
    wire2 = _wire(req2)
    # The confirmation row MUST NOT have mutated to EXPIRED mid-turn:
    assert wire2[0]["content"] == "confirm reboot"
    # The prefix (all messages prior to the new tool call) remains byte-identical:
    assert _canon(wire1) == _canon(wire2[:len(wire1)])

    # Turn N+1: user sends a new message at t=10070.0 (well past expiry):
    _reset_per_turn_agent_state(agent)
    history.append({"role": "user", "content": "system status", "timestamp": 10070.0})
    monkeypatch.setattr(time, "time", lambda: 10070.0)
    req3, _ = build_api_messages(
        agent, history, current_turn_user_idx=len(history) - 1, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )
    wire3 = _wire(req3)
    # In the new turn, the confirmation row IS expired on wire:
    assert "EXPIRED" in wire3[0]["content"]
    # Wire matches replay canonicalization at this turn boundary:
    wire_replay = _wire(canonicalize_replay_history(copy.deepcopy(history[:-1]), now=10070.0) + [history[-1]])
    assert _canon(wire3) == _canon(wire_replay)


def test_canonicalize_is_idempotent_and_non_mutating():
    """canonicalize_replay_history is idempotent and does not mutate source."""
    now = 10_000.0
    history = [
        _user("u1"),
        {"role": "assistant", "content": "a1"},
        _assistant_tc("read_file"), _tool("[command interrupted]"),
        {"role": "user", "content": "confirm forced restart", "timestamp": now - 120},
        {"role": "assistant", "content": "a2"},
        _user("u3"),
    ]
    original = copy.deepcopy(history)

    out1 = canonicalize_replay_history(copy.deepcopy(history), now=now)
    out2 = canonicalize_replay_history(copy.deepcopy(out1), now=now)

    assert _canon(_wire(out1)) == _canon(_wire(out2))
    assert history == original


def test_db_roundtrip_byte_identity():
    """SessionDB round-trip: stored messages read back and canonicalized match wire."""
    messages = [
        {"role": "user", "content": "check system"},
        {"role": "assistant", "content": "all ok"},
        {"role": "user", "content": "proceed"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "t.db")
        try:
            db.create_session(session_id="s1", source="cli")
            for m in messages:
                db.append_message("s1", role=m["role"], content=m["content"])
            conv = db.get_messages_as_conversation("s1")
            read_back = [
                {"role": m["role"], "content": m["content"]}
                for m in conv
                if m.get("content") is not None
            ]
            canon_read = canonicalize_replay_history(read_back)
            assert _canon(_wire(canon_read)) == _canon(_wire(messages))
        finally:
            db.close()


def test_canonicalize_replay_history_handles_malformed_timestamps():
    """Malformed or non-numeric timestamps must not raise exceptions and be safely preserved."""
    now = 10_000.0
    history = [
        {"role": "user", "content": "confirm reboot", "timestamp": "2026-09-08T00:00:00Z"},
        {"role": "user", "content": "confirm reboot", "timestamp": "not_a_number"},
        {"role": "user", "content": "confirm reboot", "timestamp": None},
        {"role": "user", "content": "confirm reboot", "timestamp": now - 120.0},
    ]
    agent = _Agent()
    # Should not raise TypeError or ValueError
    canon = canonicalize_replay_history(history, now=now)
    assert len(canon) == 4
    # String / None timestamps are left untouched (not expired)
    assert canon[0]["content"] == "confirm reboot"
    assert canon[1]["content"] == "confirm reboot"
    assert canon[2]["content"] == "confirm reboot"
    # The valid numeric timestamp older than 60s is expired cleanly
    assert "EXPIRED" in canon[3]["content"]

    # Also verify build_api_messages with string timestamp on current_turn_message
    res, _ = build_api_messages(
        agent,
        [{"role": "user", "content": "test", "timestamp": "2026-09-08T00:00:00Z"}],
        current_turn_user_idx=0,
        ext_prefetch_cache="",
        plugin_user_context="",
        moa_config=None,
        active_system_prompt="",
    )
    assert len(res) == 1
    assert res[0]["content"] == "test"

