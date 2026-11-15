"""#137105: the pre-compression pressure pass must not lose a tool result the model has not read.

Two ways a still-unread round lost its body:

1. ``_pending_tool_round`` only recognised a round the transcript *ends* with, so any row appended after it
   (a background-process notice delivered as a user row, a queued prompt) made the round look answered and
   the newest result was demoted even when small.
2. When an unread result alone exceeds the tail's hard share it must still give way, but the replacement was
   a 1-line stub naming neither what was lost nor where it lives — so the model re-ran the call.

Plus a counting bug: the terminal/execute_code stub counted newlines on the JSON envelope, so every result
read "1 lines output".
"""

from __future__ import annotations

import json

import pytest

from agent.context_compressor import (
    ContextCompressor,
    _pending_tool_round,
    _summarize_tool_result,
    _tool_output_text,
)

CTX = 200_000
MAX_TOKENS = 32_000


def _compressor() -> ContextCompressor:
    comp = ContextCompressor(model="claude-opus-5-5", quiet_mode=True, config_context_length=CTX, max_tokens=MAX_TOKENS)
    comp.context_length = CTX
    comp.max_tokens = MAX_TOKENS
    return comp


def _call(i: int, name: str = "terminal", command: str = "cat big.json") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": f"c{i}", "type": "function",
                        "function": {"name": name, "arguments": json.dumps({"command": command})}}],
    }


def _result(i: int, chars: int = 30_000, body: str | None = None) -> dict:
    payload = body if body is not None else "x" * chars
    return {"role": "tool", "tool_call_id": f"c{i}",
            "content": json.dumps({"output": payload, "exit_code": 0})}


def _transcript(newest_chars: int = 30_000) -> list[dict]:
    """Older demotable rounds, then a fresh unread round."""
    msgs: list[dict] = []
    for i in range(12):
        msgs += [_call(i, command=f"cat old{i}.json"), _result(i, 24_000)]
    msgs += [_call(99), _result(99, newest_chars)]
    return msgs


# --- 1. a harness row after the round does not answer it -------------------------------------------------

def test_a_background_notice_after_the_round_keeps_it_pending():
    """A user row appended after the round (a background-process notice) is not the model's answer."""
    msgs = _transcript()
    msgs.append({"role": "user", "content": "[IMPORTANT: Background process proc_1 exited (exit code 0).]"})
    pending = _pending_tool_round(msgs)
    assert list(pending), "the round it has not read yet must still be pending"
    assert msgs[pending.start]["tool_call_id"] == "c99"


def test_a_steer_after_the_round_keeps_it_pending():
    msgs = _transcript()
    msgs.append({"role": "user", "content": "actually, check the last line too", "display_kind": "steer"})
    pending = _pending_tool_round(msgs)
    assert list(pending) and msgs[pending.start]["tool_call_id"] == "c99"


def test_an_assistant_row_after_the_round_settles_it():
    msgs = _transcript()
    msgs.append({"role": "assistant", "content": "I read it."})
    assert list(_pending_tool_round(msgs)) == []


def test_the_unread_round_survives_a_notice_row_in_the_pressure_pass():
    """Case C2 from the issue: the round the model asked for keeps its body.

    Uses the issue's own repro shape (``protect_tail_count=20``, 12 older 24k-char results, a 30k newest
    result): with the old detector the empty spared range hands the newest body to the last resort, which
    stubs it to 55 chars. A smaller tail budget or a smaller newest body passes under both detectors and
    proves nothing (mercant1982 on #156097).
    """
    comp = _compressor()
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(12):
        msgs += [_call(i, command=f"cat old{i}.json"), _result(i, 24_000)]
    msgs += [_call(999), _result(999, 30_000)]
    newest_before = msgs[-1]["content"]
    msgs.append({"role": "user", "content": "[IMPORTANT: Background process proc_1 exited unexpectedly.]"})

    pruned, count = comp._prune_old_tool_results(msgs, protect_tail_count=20, protect_tail_tokens=5_000)
    newest = next(m for m in pruned if m.get("tool_call_id") == "c999")["content"]
    assert newest == newest_before, f"demoted while still unread: {newest[:70]!r}"
    assert count > 0, "the older rounds must still give way"


def test_a_parallel_unread_round_keeps_evidence_for_every_result():
    """One assistant calling two tools: neither result has been read, so neither may become a bare stub.

    The first result of the batch is the one the old code sent through the one-line path (kvnloo on
    #156097): its markers vanished and the model could neither answer from it nor page it.
    """
    comp = _compressor()
    first = "FIRST_BATCH_HEAD\n" + ("a" * 91_000) + "\nFIRST_BATCH_TAIL"
    second = "SECOND_BATCH_HEAD\n" + ("b" * 91_000) + "\nSECOND_BATCH_TAIL"
    msgs: list[dict] = []
    for i in range(12):
        msgs += [_call(i, command=f"cat old{i}.json"), _result(i, 24_000)]
    msgs += [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "ca", "type": "function", "function": {"name": "terminal", "arguments": json.dumps({"command": "cat a"})}},
            {"id": "cb", "type": "function", "function": {"name": "terminal", "arguments": json.dumps({"command": "cat b"})}},
        ]},
        _result(0, body=first) | {"tool_call_id": "ca"},
        _result(1, body=second) | {"tool_call_id": "cb"},
    ]
    pruned, count = comp._prune_old_tool_results(msgs, protect_tail_count=10, protect_tail_tokens=5_000)
    assert count > 0
    bodies = {m["tool_call_id"]: m["content"] for m in pruned if m.get("tool_call_id") in ("ca", "cb")}
    for call_id, head, tail in (("ca", "FIRST_BATCH_HEAD", "FIRST_BATCH_TAIL"),
                                ("cb", "SECOND_BATCH_HEAD", "SECOND_BATCH_TAIL")):
        body = bodies[call_id]
        assert len(body) < 20_000, f"{call_id} must still shrink: {len(body):,} chars"
        assert head in body and tail in body, f"{call_id} lost its evidence to a one-line stub: {body[:80]!r}"
        assert "excerpt" in body.lower(), f"{call_id} does not say the body was cut"


# --- 2. an oversize unread result that must give way keeps an excerpt -----------------------------------

def test_an_oversize_unread_result_keeps_a_head_and_tail_excerpt():
    """Case B: the single 200 KB read must still give way, but not to a bare one-liner."""
    comp = _compressor()
    big = "FIRST_LINE_MARKER\n" + ("y" * 161_000) + "\nLAST_LINE_MARKER"
    msgs = _transcript()[:-1] + [_call(99), _result(99, body=big)]
    pruned, count = comp._prune_old_tool_results(msgs, protect_tail_count=10, protect_tail_tokens=5_000)
    body = next(m for m in pruned if m.get("tool_call_id") == "c99")["content"]
    assert count > 0
    assert len(body) < len(big) / 10, "it must still shrink"
    assert "FIRST_LINE_MARKER" in body, "the head of the output must survive"
    assert "LAST_LINE_MARKER" in body, "the tail of the output must survive"
    assert "excerpt" in body.lower(), "the stub must say the body was cut, not present it as the whole result"
    # Older rounds in the same pass still get plain one-line stubs; only the unread one keeps an excerpt.
    older = next(m for m in pruned if m.get("tool_call_id") == "c0")["content"]
    assert "excerpt" not in older.lower() and len(older) < 500, older


# --- 3. the line count comes from the tool's output, not the JSON envelope -------------------------------

def test_terminal_stub_counts_lines_in_the_output_not_the_envelope():
    output = "\n".join(f"line {i}" for i in range(2_000))
    content = json.dumps({"output": output, "exit_code": 0})
    assert content.count("\n") == 0, "the envelope escapes every newline: this is the bug"
    stub = _summarize_tool_result("terminal", json.dumps({"command": "cat big.json"}), content)
    assert "2,000 lines" in stub or "2000 lines" in stub, stub
    assert "1 lines" not in stub


def test_execute_code_stub_counts_lines_in_the_output():
    output = "\n".join(f"row {i}" for i in range(40))
    content = json.dumps({"output": output, "exit_code": 0})
    stub = _summarize_tool_result("execute_code", json.dumps({"code": "print(1)"}), content)
    assert "40 lines" in stub, stub


def test_a_bare_text_result_still_counts_its_own_lines():
    content = "alpha\nbeta\ngamma"
    assert _tool_output_text("read_file", content) == content
    stub = _summarize_tool_result("read_file", json.dumps({"path": "x.py"}), content)
    assert "chars" in stub, stub


def test_output_text_falls_back_when_the_envelope_is_malformed():
    broken = '{"output": "unterminated'
    assert _tool_output_text("terminal", broken) == broken
