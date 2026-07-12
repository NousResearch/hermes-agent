from __future__ import annotations

from agent.tool_output_hygiene import HygieneConfig, apply_api_tool_output_hygiene


def _assistant(call_id: str, name: str, args: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}
        ],
    }


def _tool(call_id: str, name: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


def test_flag_off_returns_exact_same_api_message_list() -> None:
    messages = [
        {"role": "user", "content": "keep me"},
        _assistant("c1", "terminal", '{"command":"pwd"}'),
        _tool("c1", "terminal", "x" * 1000),
    ]

    out, stats = apply_api_tool_output_hygiene(
        messages,
        config=HygieneConfig(enabled=False),
        request_tokens=900,
        context_length=1000,
        session_id="s",
    )

    assert out is messages
    assert stats.total_pruned == 0
    assert messages[2]["content"] == "x" * 1000


def test_dedup_prunes_superseded_tool_result_but_keeps_latest_raw() -> None:
    old = "old-result" * 200
    latest = "latest-result" * 200
    messages = [
        {"role": "user", "content": "run pwd"},
        _assistant("c1", "terminal", '{"command":"pwd"}'),
        _tool("c1", "terminal", old),
        {"role": "user", "content": "run pwd again"},
        _assistant("c2", "terminal", '{"command":"pwd"}'),
        _tool("c2", "terminal", latest),
    ]

    out, stats = apply_api_tool_output_hygiene(
        messages,
        config=HygieneConfig(enabled=True, stale_context_ratio=0.99),
        request_tokens=100,
        context_length=1000,
        session_id="s",
    )

    assert out is not messages
    assert stats.dedup_pruned == 1
    assert "[pruned: superseded by call #2" in out[2]["content"]
    assert out[5]["content"] == latest
    # Original session/API input copy is not mutated.
    assert messages[2]["content"] == old


def test_failed_tool_output_prunes_only_after_configured_user_turn_age() -> None:
    error = '{"error":"boom","error_type":"RuntimeError","detail":"' + ("!" * 200) + '"}'
    messages = [
        {"role": "user", "content": "try"},
        _assistant("c1", "terminal", '{"command":"bad"}'),
        _tool("c1", "terminal", error),
        {"role": "user", "content": "turn1"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "turn2"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "turn3"},
    ]

    out, stats = apply_api_tool_output_hygiene(
        messages,
        config=HygieneConfig(enabled=True, failed_after_user_turns=3, stale_context_ratio=0.99),
        request_tokens=100,
        context_length=1000,
        session_id="s",
    )

    assert stats.failed_pruned == 1
    assert "[pruned: old failed tool output" in out[2]["content"]
    assert "error_type=RuntimeError" in out[2]["content"]

    young, young_stats = apply_api_tool_output_hygiene(
        messages[:-1],
        config=HygieneConfig(enabled=True, failed_after_user_turns=3, stale_context_ratio=0.99),
        request_tokens=100,
        context_length=1000,
        session_id="s",
    )
    assert young_stats.failed_pruned == 0
    assert young[2]["content"] == error


def test_stale_pruning_preserves_paths_and_respects_tail_window() -> None:
    old = "see /Users/automation/project/results/out.txt\n" + ("A" * 2000)
    recent = "recent tool result" * 100
    messages = [
        {"role": "user", "content": "old"},
        _assistant("c1", "read_file", '{"path":"/Users/automation/project/results/out.txt"}'),
        _tool("c1", "read_file", old),
        {"role": "user", "content": "recent"},
        _assistant("c2", "terminal", '{"command":"date"}'),
        _tool("c2", "terminal", recent),
    ]

    out, stats = apply_api_tool_output_hygiene(
        messages,
        config=HygieneConfig(
            enabled=True,
            stale_context_ratio=0.50,
            stale_protect_tail_tokens=1,
            protect_last_n=2,
        ),
        request_tokens=600,
        context_length=1000,
        session_id="s",
    )

    assert stats.stale_pruned == 1
    assert "[pruned: stale tool output" in out[2]["content"]
    assert "/Users/automation/project/results/out.txt" in out[2]["content"]
    assert out[5]["content"] == recent


def test_hygiene_never_prunes_user_or_assistant_content() -> None:
    user_text = "USER" * 1000
    assistant_text = "ASSISTANT" * 1000
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
        _assistant("c1", "terminal", '{"command":"date"}'),
        _tool("c1", "terminal", "old" * 1000),
        {"role": "user", "content": "tail"},
    ]

    out, stats = apply_api_tool_output_hygiene(
        messages,
        config=HygieneConfig(enabled=True, stale_context_ratio=0.1, stale_protect_tail_tokens=1, protect_last_n=1),
        request_tokens=900,
        context_length=1000,
        session_id="s",
    )

    assert stats.total_pruned == 1
    assert out[0]["content"] == user_text
    assert out[1]["content"] == assistant_text
