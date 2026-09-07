"""Tests for agent.replay_cleanup — shared replay-tail sanitizers.

These functions were extracted from gateway/run.py so every resume surface
(messaging gateway AND TUI/WebUI gateway) strips poisoned tool-call tails the
same way. Regression coverage for #29086 (WebUI session permanently stuck
because the dangling tool-call tail was replayed on every resume).
"""

import copy

from agent.replay_cleanup import (
    canonicalize_replay_history,
    is_interrupted_tool_result,
    strip_dangling_tool_call_tail,
    strip_interrupted_tool_tails,
    strip_stale_dangerous_confirmations,
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


def test_send_builder_uses_canonical_history_without_mutating_source():
    """The request copy must match replay cleanup while durable history stays intact."""
    from agent.turn_context import build_api_messages

    class _Agent:
        api_mode = "chat_completions"
        ephemeral_system_prompt = None

        @staticmethod
        def _copy_reasoning_content_for_api(_source, _target):
            return None

        @staticmethod
        def _should_sanitize_tool_calls():
            return False

        @staticmethod
        def _sanitize_tool_calls_for_strict_api(*_args, **_kwargs):
            return None

    now = 10_000.0
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
        _Agent(), history, current_turn_user_idx=len(history) - 1, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )

    assert history == original
    assert [message["role"] for message in request] == [
        "user", "assistant", "user", "assistant", "user"
    ]
    assert "EXPIRED" in request[2]["content"]
    assert request[-1]["content"] == "current-wire"
