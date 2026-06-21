"""Unit tests for leaked-tool-call detection (detect+retry guard).

Context: models sometimes emit a complete ``<invoke ...>...</invoke>`` tool
call as ordinary assistant TEXT instead of issuing it through the tool-call
channel. The turn then ends with ``finish_reason=stop`` and no ``tool_calls``.
The conversation loop responds by re-prompting for a real tool call (bounded
by a retry budget) rather than surfacing or deleting the XML.

These tests pin the DETECTOR contract — it is the false-positive-sensitive
piece. The bias is heavily toward NOT firing: suppressing a genuine answer is
far worse than missing a leak (the leak still gets re-prompted on the next
identical condition, and worst case is shown honestly).

The detector lives on ``AIAgent`` as two pure static methods plus one
think-block-aware instance wrapper:

- ``_strip_code_spans_for_invoke_scan`` — blanks fenced/inline code so
  documentation examples of ``<invoke>`` never count.
- ``_text_is_essentially_tool_call`` — core test on already-visible text.
- ``_looks_like_leaked_tool_call`` — strips think blocks first, then defers
  to the core. Tested via a duck-typed stub so no full agent is constructed.
"""

from __future__ import annotations

import re

import pytest

from run_agent import AIAgent


# ── True positives: the block IS essentially the whole message ─────────────

_LEAKS = [
    # bare invoke, no label
    '<invoke name="terminal"><parameter name="command">pwd</parameter></invoke>',
    # OpenClaw-style "course" label preamble
    'course\n<invoke name="terminal"><parameter name="command">pwd</parameter></invoke>',
    # "call" label preamble
    'call\n<invoke name="terminal">x</invoke>',
    # multiline / multi-param, realistic whitespace
    (
        '<invoke name="browser_navigate">\n'
        '<parameter name="url">https://example.com</parameter>\n'
        '</invoke>'
    ),
    # leaked wrapper tag around the invoke
    (
        'function_calls\n'
        '<invoke name="search_files">\n'
        '<parameter name="pattern">foo</parameter>\n'
        '</invoke>'
    ),
    # trailing whitespace / newline after close tag
    '<invoke name="read_file"><parameter name="path">/tmp/x</parameter></invoke>\n\n',
    # tiny incidental prose under the threshold ("okay")
    'okay\n<invoke name="terminal"><parameter name="command">ls</parameter></invoke>',
    # ── "preamble + leaked call" shape (Opus 4.x narration habit) ──────────
    # Substantial prose PRECEDES the block but the message still ENDS with a
    # complete, unfenced <invoke>…</invoke>. A real final answer never ends in
    # a bare </invoke>. This is the exact shape captured from a live Opus-4.8
    # leak (session c0717f698895, msg[105]): a couple sentences announcing the
    # next action, then the call emitted as prose.
    (
        "消息已发。现在用一个**自动捕获中途时刻**的探针：持续轮询，一旦发现 "
        "EventSource readyState=1（OPEN）且内容已有一定长度、且\"Done in\"尚未"
        "出现（即真正 streaming 中途），就立刻注入 error，然后观察 probe 行为。"
        "这样能精确命中我要验证的分支。\n\n"
        'call\n<invoke name="mcp_playwright_browser_run_code_unsafe">\n'
        '<parameter name="code">async (page) => { return 1; }</parameter>\n'
        "</invoke>"
    ),
    # English preamble + trailing leaked call, trailing newline after close
    (
        "Good — the message is sent. Now I'll probe the live stream to catch "
        "the mid-flight reconnect path and confirm the buffer survives.\n\n"
        '<invoke name="terminal"><parameter name="command">curl -s localhost:8701/health</parameter></invoke>\n'
    ),
]


# ── True positives that previously slipped the ≤24-char primary gate ───────
# These have LARGE residual prose (so the primary "block is the whole message"
# signal is False) but END in a complete unfenced invoke block (secondary
# signal). Kept as a separate list to document the regression class.
_PREAMBLE_LEAKS = [
    (
        "I traced it and the cleanest test is to inject mid-stream. Let me run "
        "the probe now and read back the EventSource state to see if the "
        "content survives the forced error.\n"
        '<invoke name="terminal"><parameter name="command">echo go</parameter></invoke>'
    ),
]


@pytest.mark.parametrize("text", _PREAMBLE_LEAKS)
def test_detects_preamble_then_leaked_call(text):
    assert AIAgent._text_is_essentially_tool_call(text) is True


@pytest.mark.parametrize("text", _LEAKS)
def test_detects_leaked_tool_call(text):
    assert AIAgent._text_is_essentially_tool_call(text) is True


# ── False positives we must NOT fire on ────────────────────────────────────

_SAFE = [
    # plain prose, no XML at all
    "Here's the summary of what I found in the logs.",
    # mentions the word invoke, no XML
    "I'll invoke the build script and report the result.",
    # mentions <invoke> as inline code — documentation
    "The `<invoke>` tag is how the model issues a real tool call.",
    # FULL invoke block shown inside a fenced code block — documentation
    (
        "Here's the leakage pattern to watch for:\n\n"
        "```\n"
        '<invoke name="terminal"><parameter name="command">pwd</parameter></invoke>\n'
        "```\n\n"
        "Notice there's no tool_calls field on that message."
    ),
    # fenced with a language tag
    (
        "Example:\n\n"
        "```xml\n"
        '<invoke name="read_file"><parameter name="path">/tmp/x</parameter></invoke>\n'
        "```\n"
    ),
    # substantial real answer that quotes the XML in a fence AND explains it
    # (this is the exact shape of the dev conversation that built this guard)
    (
        "I traced the agent loop and found the insertion point. The bug is "
        "that the model emits this as plain text:\n\n"
        "```\n"
        'course\n<invoke name="terminal"><parameter name="command">git status</parameter></invoke>\n'
        "```\n\n"
        "instead of issuing a real tool call, so finish_reason is stop with no "
        "tool_calls. The fix is a detect-and-retry guard, not a sanitizer."
    ),
    # empty / whitespace
    "",
    "   \n  ",
    # an opening invoke with NO closing tag (half-streamed / not a complete
    # block) — conservative: don't fire on incomplete markup
    '<invoke name="terminal"><parameter name="command">pwd</parameter>',
    # real answer that merely ends by referencing a tool name
    "Done. I ran terminal and the working tree is clean now.",
    # inline code invoke plus lots of prose
    (
        "When you see `<invoke name=\"terminal\">` rendered as text in the "
        "chat, that means the call leaked into the content field and never "
        "actually executed. Start a fresh session to clear the bad example."
    ),
    # ── Secondary-signal false-positive guards ────────────────────────────
    # Real answer whose LAST thing is a FENCED <invoke> example. The block is
    # at the very end, but it is inside ```code``` so the secondary
    # "ends-with-invoke" signal must NOT fire (fences are blanked before scan).
    (
        "Here is the leak shape, and it is the final thing I'll show you:\n\n"
        "```\n"
        '<invoke name="terminal"><parameter name="command">pwd</parameter></invoke>\n'
        "```"
    ),
    # Real answer ending in a fenced xml example with no trailing prose.
    (
        "The persisted bad message looks exactly like this — note it ends the "
        "message and there is no tool_calls field:\n\n"
        "```xml\n"
        '<invoke name="read_file"><parameter name="path">/tmp/x</parameter></invoke>\n'
        "```\n"
    ),
    # Long answer that mentions a complete invoke inline-coded at the very end.
    (
        "If the turn terminates with the literal text "
        '`<invoke name="terminal"></invoke>` and nothing else follows, the '
        "call leaked. Otherwise you are fine."
    ),
]


@pytest.mark.parametrize("text", _SAFE)
def test_does_not_fire_on_safe_text(text):
    assert AIAgent._text_is_essentially_tool_call(text) is False


# ── Code-span stripper behaves ─────────────────────────────────────────────

def test_code_span_stripper_removes_fenced_invoke():
    text = "```\n<invoke name=\"x\">y</invoke>\n```"
    out = AIAgent._strip_code_spans_for_invoke_scan(text)
    assert "<invoke" not in out


def test_code_span_stripper_removes_inline_invoke():
    text = "the `<invoke>` tag"
    out = AIAgent._strip_code_spans_for_invoke_scan(text)
    assert "<invoke" not in out


def test_code_span_stripper_drops_dangling_fence_tail():
    # An unterminated fence: everything from the opening fence onward is
    # dropped so a half-streamed example can't trip the scan.
    text = "intro text\n```\n<invoke name=\"x\">y</invoke>"
    out = AIAgent._strip_code_spans_for_invoke_scan(text)
    assert "<invoke" not in out
    assert "intro text" in out


# ── Think-block-aware instance wrapper ─────────────────────────────────────

class _ThinkStub:
    """Minimal stand-in exposing only what the wrapper touches."""

    def _strip_think_blocks(self, content: str) -> str:
        return re.sub(
            r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE
        )


def test_wrapper_catches_leak_hidden_after_think_block():
    stub = _ThinkStub()
    content = (
        "<think>The user wants the current directory, I'll call terminal."
        "</think>\n"
        '<invoke name="terminal"><parameter name="command">pwd</parameter></invoke>'
    )
    assert AIAgent._looks_like_leaked_tool_call(stub, content) is True


def test_wrapper_passes_normal_answer_with_think_block():
    stub = _ThinkStub()
    content = (
        "<think>Let me summarize the findings.</think>\n"
        "The working tree is clean and all tests pass."
    )
    assert AIAgent._looks_like_leaked_tool_call(stub, content) is False


def test_wrapper_rejects_non_string():
    stub = _ThinkStub()
    assert AIAgent._looks_like_leaked_tool_call(stub, None) is False
    assert AIAgent._looks_like_leaked_tool_call(stub, 123) is False


# ── Scaffolding lifecycle: the synthetic recovery pair must not persist ─────
#
# When the guard fires it appends a neutral placeholder assistant message and
# a re-prompt user message, both flagged `_leaked_tool_call_synthetic`. On the
# eventual real-text exit these are popped by the conversation loop's
# trailing-scaffolding `while`. On an interrupt/error persist path they are
# popped by `_drop_trailing_empty_response_scaffolding`'s dedicated pre-pass —
# WITHOUT engaging the destructive tool-result rewind that empty-response
# scaffolding triggers (real completed tool work before the leak is preserved).

class _ScaffoldStub:
    """Duck-typed stand-in for the persist-path scaffolding stripper."""

    # bind the real implementation under test
    _drop_trailing_empty_response_scaffolding = (
        AIAgent._drop_trailing_empty_response_scaffolding
    )


def test_leak_scaffolding_prepass_pops_only_synthetic_pair():
    stub = _ScaffoldStub()
    messages = [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "tool_calls": [{"id": "c1"}], "content": ""},
        {"role": "tool", "tool_call_id": "c1", "content": "real tool result"},
        {"role": "assistant", "content": "(tool call not executed)",
         "_leaked_tool_call_synthetic": True},
        {"role": "user", "content": "Re-issue it as a real tool call.",
         "_leaked_tool_call_synthetic": True},
    ]
    stub._drop_trailing_empty_response_scaffolding(messages)
    # The two synthetic recovery messages are gone …
    assert all(not m.get("_leaked_tool_call_synthetic") for m in messages)
    # … and the REAL completed tool work is fully preserved (no rewind).
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["content"] == "real tool result"
    assert len(messages) == 3


def test_leak_scaffolding_prepass_noop_without_flag():
    stub = _ScaffoldStub()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]
    before = [dict(m) for m in messages]
    stub._drop_trailing_empty_response_scaffolding(messages)
    assert messages == before

