"""Digest advisory view: prose narration, substance artifacts, recency detail.

The legacy transcript rendering made 79-82% of every reference prompt
serialized "[called tool: ...]" / "[tool result: ...]" text (measured on a
5-turn trace). Small advisors next-token-predicted that dominant pattern —
fabricating tool transcripts, adopting the acting agent's first person,
inventing results. The digest view removes the imitable format while keeping
the content: operational tool activity becomes one-line prose narration,
substance results (file reads, search hits — e.g. the code under review)
stay visible as quoted artifacts, and the last few interactions are detailed
in the trailing synthetic user turn (which varies every iteration anyway, so
the stable prefix stays append-only for advisor prompt caching).
"""

import json

from agent.moa_loop import (
    _REFERENCE_DETAIL_RESULTS,
    _REFERENCE_SUBSTANCE_RESULT_BUDGET,
    _reference_messages,
)


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}


def test_operational_payloads_elided_from_digest():
    """A huge write_file payload must not ride into the advisory view."""
    payload = "x" * 30_000
    messages = [
        {"role": "user", "content": "write the file"},
        {
            "role": "assistant",
            "content": "writing now",
            "tool_calls": [_tool_call("c1", "write_file", {"path": "a.py", "content": payload})],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"bytes_written": 30000}'},
    ]

    view = _reference_messages(messages)
    joined = "\n".join(m["content"] for m in view)

    assert payload not in joined
    # The action itself is narrated.
    assert "- write_file:" in joined
    # Total view stays a small fraction of the raw payload.
    assert len(joined) < 10_000


def test_substance_results_preserved_as_quoted_artifacts():
    """File reads ARE the task material (code review) — content must survive."""
    code = "def hello():\n    return 'world'\n" * 50
    messages = [
        {"role": "user", "content": "review src/foo.py"},
        {
            "role": "assistant",
            "content": "reading the file",
            "tool_calls": [_tool_call("c1", "read_file", {"path": "src/foo.py"})],
        },
        {"role": "tool", "tool_call_id": "c1", "content": code},
        {"role": "assistant", "content": "analyzing"},
    ]

    view = _reference_messages(messages)
    joined = "\n".join(m["content"] for m in view)

    # The retrieved content is quoted in the stable digest, fenced.
    assert "def hello():" in joined
    assert "retrieved content quoted below:" in joined
    assert "~~~" in joined
    # But never as bracket transcript syntax.
    assert "[called tool:" not in joined
    assert "[tool result:" not in joined


def test_substance_result_head_tail_capped():
    """Substance artifacts are capped head+tail, not replayed verbatim."""
    code = "line\n" * 20_000  # 100K chars
    messages = [
        {"role": "user", "content": "review it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("c1", "read_file", {"path": "big.py"})],
        },
        {"role": "tool", "tool_call_id": "c1", "content": code},
    ]

    view = _reference_messages(messages)
    stable = "\n".join(m["content"] for m in view[:-1])

    assert "chars omitted" in stable
    # Stable frame stays within budget plus framing overhead.
    assert len(stable) < _REFERENCE_SUBSTANCE_RESULT_BUDGET + 2_000


def test_reference_detail_tools_override():
    """A preset can promote an operational tool to substance (and demote)."""
    messages = [
        {"role": "user", "content": "lint it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("c1", "terminal", {"command": "ruff check ."})],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"output": "E501 line too long", "exit_code": 1}'},
        {"role": "assistant", "content": "found issues"},
    ]

    default_view = _reference_messages(messages)
    default_stable = "\n".join(m["content"] for m in default_view[:-1])
    # terminal is operational by default: narration only, no quoted artifact.
    assert "retrieved content quoted below:" not in default_stable

    promoted_view = _reference_messages(messages, detail_tools=["terminal"])
    promoted_stable = "\n".join(m["content"] for m in promoted_view[:-1])
    assert "retrieved content quoted below:" in promoted_stable
    assert "E501 line too long" in promoted_stable


def test_detail_window_last_k_only_in_trailing_turn():
    """Recency detail: last K interactions, '>'-quoted, trailing turn only."""
    messages = [{"role": "user", "content": "go"}]
    for i in range(6):
        messages.append(
            {
                "role": "assistant",
                "content": f"step {i}",
                "tool_calls": [_tool_call(f"c{i}", "terminal", {"command": f"cmd{i}"})],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"result-{i}"})

    view = _reference_messages(messages)

    assert view[-1]["role"] == "user"
    tail = view[-1]["content"]
    assert "Recent tool interactions in detail" in tail
    # Only the last K appear in the detail block.
    kept = [i for i in range(6) if f"> result-{i}" in tail]
    assert kept == list(range(6 - _REFERENCE_DETAIL_RESULTS, 6))
    # Detail lives ONLY in the trailing synthetic turn.
    stable = "\n".join(m["content"] for m in view[:-1])
    assert "Recent tool interactions in detail" not in stable


def test_ends_on_real_user_turn_has_no_detail_section():
    """Turn start (fresh user prompt): real user turns are never modified."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "the current question"},
    ]

    view = _reference_messages(messages)

    assert view[-1]["role"] == "user"
    assert view[-1]["content"] == "the current question"


def test_digest_is_deterministic():
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "acting",
            "tool_calls": [_tool_call("c1", "terminal", {"command": "ls"})],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"output": "a b c", "exit_code": 0}'},
    ]
    assert _reference_messages(messages) == _reference_messages(messages)


def test_digest_prefix_is_append_only_across_iterations():
    """KV-cache invariant: iteration N's stable view is a prefix of N+1's.

    Advisor prompt caching (see the cache note in _run_reference) depends on
    the advisory view only ever APPENDING content as the tool loop advances —
    up to the trailing synthetic turn, which varies by design.
    """
    base = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "step one",
            "tool_calls": [_tool_call("c1", "terminal", {"command": "ls"})],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"output": "a", "exit_code": 0}'},
    ]
    grown = base + [
        {
            "role": "assistant",
            "content": "step two",
            "tool_calls": [_tool_call("c2", "terminal", {"command": "pwd"})],
        },
        {"role": "tool", "tool_call_id": "c2", "content": '{"output": "/x", "exit_code": 0}'},
    ]

    def _serialized_stable(msgs):
        view = _reference_messages(msgs)
        assert view[-1]["role"] == "user"  # trailing synthetic turn
        return "\x00".join(f"{m['role']}:{m['content']}" for m in view[:-1])

    assert _serialized_stable(grown).startswith(_serialized_stable(base))


def test_unmatched_pending_call_noted():
    """A call whose result never arrived (interrupt) is narrated, not lost."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "trying",
            "tool_calls": [_tool_call("c1", "terminal", {"command": "sleep 999"})],
        },
        {"role": "user", "content": "never mind, stop"},
    ]

    view = _reference_messages(messages)
    joined = "\n".join(m["content"] for m in view)
    assert "- terminal: called (no result recorded)" in joined
    assert view[-1]["content"] == "never mind, stop"


def test_assistant_prose_capped():
    """A pasted-file-sized assistant turn is head+tail capped in the digest."""
    prose = "p" * 30_000
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": prose},
        {"role": "user", "content": "next"},
    ]

    view = _reference_messages(messages)
    assistant = next(m for m in view if m["role"] == "assistant")
    assert len(assistant["content"]) < 6_000
    assert "chars omitted" in assistant["content"]


def test_reference_prose_budget_override():
    """A prose-centric preset can widen the assistant-prose cap so advisors
    see whole drafts instead of a head+tail excerpt."""
    draft = "d" * 10_000
    messages = [
        {"role": "user", "content": "write it"},
        {"role": "assistant", "content": draft},
        {"role": "user", "content": "revise"},
    ]

    capped = _reference_messages(messages)
    wide = _reference_messages(messages, prose_budget=16_000)

    capped_assistant = next(m for m in capped if m["role"] == "assistant")
    wide_assistant = next(m for m in wide if m["role"] == "assistant")
    assert "chars omitted" in capped_assistant["content"]
    assert wide_assistant["content"] == draft
