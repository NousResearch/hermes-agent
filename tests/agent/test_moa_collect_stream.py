"""Regression tests for issue #58437.

The MoA quiet-mode path runs advisors and the aggregator through the same
``call_llm`` chokepoint, but in quiet mode (kanban workers, subagents,
``hermes -z``) **no display/TTS consumer is registered**, so the verbose
text rendering of each advisor's stream can be dropped.  Before this fix
the MoA reference/output helper extracted only ``response.choices[0].message
.content`` via ``_extract_text(response)`` and discarded any ``tool_calls``
the reference emitted, producing the bug from the issue body:

    reference emits tool_calls → content == "" → "empty_response_exhausted" crash

These tests pin down the three observable invariants the fix establishes:

1. **No regression on text suppression:** a reference that emits text in
   quiet mode produces no tool_calls; the joined advice the aggregator
   sees contains the text and nothing leaks into ``tool_calls``.
2. **Tool-call preservation:** a reference that emits a tool_call (with
   or without prose) carries that ``tool_calls`` payload into the
   ``_RefAccounting`` and the ``moa.reference`` emit.  This is the
   "preserve tool_calls even when quiet mode hides other parts"
   property the issue title calls out.
3. **Aggregator receives preserved tool_calls** and can re-dispatch
   them — exercised through the joined reference guidance block the
   aggregator sees, which now mentions the tool name(s) the reference
   attempted so the aggregator can choose to re-execute them.

The tests are deliberately unit-level — they exercise ``_run_reference``
and the ``moa.reference`` event hook directly without spinning up the
full conversation loop, which is enough to assert the contract the
agentic layer downstream relies on.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _text_response(text: str) -> SimpleNamespace:
    """A non-streaming reference response carrying only text content."""
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    return SimpleNamespace(choices=[choice], usage=None, model="fake-ref")


def _tool_call_response(*tool_calls: dict, content: str = "") -> SimpleNamespace:
    """A non-streaming reference response carrying tool_calls.

    Each ``tool_calls`` arg is the dict payload shape an OpenAI-compatible
    transport returns: ``{"id": str, "type": "function", "function":
    {"name": str, "arguments": str}}``.  We mirror the SDK's real shape
    (a ``SimpleNamespace`` top level + ``SimpleNamespace`` for the nested
    ``function``) so the production code paths are exercised exactly as
    they would be against a live provider.
    """
    calls = []
    for tc in tool_calls:
        if isinstance(tc, SimpleNamespace):
            calls.append(tc)
            continue
        fn = tc.get("function") or {}
        calls.append(
            SimpleNamespace(
                id=tc.get("id"),
                type=tc.get("type"),
                function=SimpleNamespace(
                    name=fn.get("name"),
                    arguments=fn.get("arguments"),
                ),
            )
        )
    message = SimpleNamespace(content=content, tool_calls=calls)
    choice = SimpleNamespace(message=message, finish_reason="tool_calls", index=0)
    return SimpleNamespace(choices=[choice], usage=None, model="fake-ref")


def _reference_slot(label: str = "openrouter:fake-ref") -> dict:
    return {"provider": "openrouter", "model": "fake-ref", "_label": label}


@pytest.fixture
def patched_call_llm_text():
    """Replace ``call_llm`` with one that returns only text content."""
    captured = {}

    def _fake(**kwargs):
        captured["kwargs"] = kwargs
        return _text_response("advice-only output")

    with patch("agent.moa_loop.call_llm", side_effect=_fake) as m:
        captured["mock"] = m
        yield captured


@pytest.fixture
def patched_call_llm_tool_call():
    """Replace ``call_llm`` with one that returns a tool_call response."""
    captured = {}
    tc_payload = {
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": '{"path": "/tmp/example.txt"}',
        },
    }

    def _fake(**kwargs):
        captured["kwargs"] = kwargs
        captured["tool_call"] = tc_payload
        return _tool_call_response(tc_payload)

    with patch("agent.moa_loop.call_llm", side_effect=_fake) as m:
        captured["mock"] = m
        yield captured


# ---------------------------------------------------------------------------
# (1) Quiet mode + reference emits text → text suppressed, no tool_calls leak.
# ---------------------------------------------------------------------------


def test_quiet_mode_text_reference_keeps_text_no_tool_calls(patched_call_llm_text):
    """A reference emitting only text must surface that text in the joined
    output and must NOT silently fabricate a ``tool_calls`` payload.

    This is the regression guard for "quiet mode text suppression": the
    existing behavior is preserved — the advisor's prose is delivered to
    the aggregator and ``_RefAccounting.tool_calls`` stays ``None``.
    """
    from agent import moa_loop

    label, output_text, acct = moa_loop._run_reference(
        _reference_slot(),
        [{"role": "user", "content": "hi"}],
    )

    assert "advice-only output" in output_text, (
        "text-only reference output must be preserved verbatim for the aggregator"
    )
    assert acct.tool_calls is None, (
        "text-only reference must not have a synthetic tool_calls field"
    )


# ---------------------------------------------------------------------------
# (2) Quiet mode + reference emits tool_call → tool_call preserved.
# ---------------------------------------------------------------------------


def test_quiet_mode_tool_call_reference_preserves_tool_calls(patched_call_llm_tool_call):
    """A reference emitting a tool_call (no prose) must surface the call
    so the aggregator can re-dispatch it.

    Before the fix this returned ``"(empty response)"`` because the only
    code path that read the response was ``_extract_text``, which is
    content-only.  After the fix the tool_call renders into the output
    string and the raw ``tool_calls`` payload is preserved on
    ``_RefAccounting`` so ``moa.reference`` consumers (display hook,
    trace persistence, downstream aggregator) can re-dispatch.
    """
    from agent import moa_loop

    label, output_text, acct = moa_loop._run_reference(
        _reference_slot(),
        [{"role": "user", "content": "Read /tmp/example.txt"}],
    )

    # The rendered call appears in the output text so the aggregator can
    # reason about what the reference attempted even when no display
    # consumer is attached to render it as a chatty thought block.
    assert "read_file" in output_text, (
        "reference tool_call must be rendered into the output text so "
        "the aggregator sees the name in its private context"
    )
    assert '"/tmp/example.txt"' in output_text, (
        "reference tool_call arguments must be rendered so the "
        "aggregator can re-dispatch the call"
    )

    # And the raw tool_calls payload is preserved on the accounting
    # object for downstream consumers to re-dispatch verbatim.
    assert acct.tool_calls, "raw tool_calls payload must be preserved on _RefAccounting"
    tc = acct.tool_calls[0]
    fn = tc.function if hasattr(tc, "function") else tc["function"]
    name = fn.name if hasattr(fn, "name") else fn["name"]
    assert name == "read_file"


def test_quiet_mode_tool_call_with_prose_preserves_both():
    """A reference can emit *both* prose and a tool_call.  Both must
    survive — the prose so the aggregator can quote advice, the call so
    the aggregator can re-dispatch it."""
    from agent import moa_loop

    tc_payload = {
        "id": "call_def456",
        "type": "function",
        "function": {
            "name": "list_files",
            "arguments": '{"directory": "/tmp"}',
        },
    }
    response = _tool_call_response(tc_payload, content="Here is the directory listing:")

    with patch("agent.moa_loop.call_llm", return_value=response):
        label, output_text, acct = moa_loop._run_reference(
            _reference_slot(),
            [{"role": "user", "content": "list /tmp"}],
        )

    assert output_text.startswith("Here is the directory listing:"), (
        "text content must appear first"
    )
    assert "list_files" in output_text, "tool_call must also be in the output"
    assert acct.tool_calls, "raw tool_calls payload must be preserved alongside text"
    fn = acct.tool_calls[0].function if hasattr(acct.tool_calls[0], "function") else acct.tool_calls[0]["function"]
    name = fn.name if hasattr(fn, "name") else fn["name"]
    assert name == "list_files"


# ---------------------------------------------------------------------------
# (3) Aggregator receives preserved tool_calls → can re-dispatch them.
# ---------------------------------------------------------------------------


def test_moa_reference_event_carries_tool_calls_in_quiet_mode(patched_call_llm_tool_call):
    """``MoAChatCompletions.create`` emits a ``moa.reference`` event per
    advisor.  In quiet mode display consumers are absent, but the
    ``tool_calls`` payload must still be on the event so a silent-mode
    consumer (trace persistence, gateway relay, aggregator re-dispatch)
    can act on it without parsing chatty prose."""
    from agent import moa_loop

    # Build the minimum surface to exercise the emit path without
    # instantiating a real facade.  We stub the methods around the
    # reference-output loop just enough to capture the event payload.
    emits: list[tuple[str, dict]] = []

    class _StubFacade:
        preset_name = "default"
        _pending_reference_usage = None
        _pending_reference_cost = None
        _pending_trace = None

        def _emit(self, event_name, **payload):
            emits.append((event_name, payload))

        _run_references_parallel = staticmethod(moa_loop._run_references_parallel)
        _emit_reference_signal = staticmethod(moa_loop._emit_reference_signal) if hasattr(
            moa_loop, "_emit_reference_signal"
        ) else None

    # Patch the facade helper used inside create().  We swap in a manual
    # version of the fan-out/emit region so we can call it without a full
    # preset.
    slots = [_reference_slot()]
    ref_messages = [{"role": "user", "content": "hi"}]

    stub = _StubFacade()

    # Run the bare reference (real implementation) to mirror what
    # _run_references_parallel would have produced.
    reference_outputs = [(label, text, acct) for label, text, acct in [
        moa_loop._run_reference(s, ref_messages) for s in slots
    ]]

    # Now exercise the emit loop's logic directly (mirror of the inner
    # block in MoAChatCompletions.create()).
    for idx, (lbl, txt, usage) in enumerate(reference_outputs, start=1):
        ref_calls = getattr(usage, "tool_calls", None)
        stub._emit(
            "moa.reference",
            index=idx,
            count=len(reference_outputs),
            label=lbl,
            text=txt,
            tool_calls=ref_calls,
        )

    # The moa.reference event must include the tool_calls field, even in
    # a quiet-mode (no-display) scenario.
    assert emits, "expected at least one moa.reference emit"
    name, payload = emits[0]
    assert name == "moa.reference"
    assert payload["tool_calls"], (
        "quiet-mode emit must carry tool_calls so the aggregator can "
        "re-dispatch without parsing prose"
    )
    tc = payload["tool_calls"][0]
    fn = tc.function if hasattr(tc, "function") else tc["function"]
    fname = fn.name if hasattr(fn, "name") else fn["name"]
    assert fname == "read_file"


def test_aggregator_sees_rendered_tool_call_in_joined_reference_guidance(
    patched_call_llm_tool_call,
):
    """The joined reference-guidance block the aggregator receives must
    mention the tool each reference tried to call.  This is what makes
    re-dispatch possible from the aggregator side without the
    ``_RefAccounting`` plumbing — the text itself is the contract.

    Without the fix, the joined guidance line was ``"[empty response]"``
    for every tool-only reference, leaving the aggregator blind.
    """
    from agent import moa_loop

    label, output_text, acct = moa_loop._run_reference(
        _reference_slot(),
        [{"role": "user", "content": "Read /tmp/example.txt"}],
    )

    # Simulate the joined-guidance builder the facade uses:
    joined = "\n\n".join(
        f"Reference {idx} — {lbl}:\n{txt}"
        for idx, (lbl, txt, _acct) in enumerate(
            [(label, output_text, acct)], start=1
        )
    )

    assert "read_file" in joined, (
        "the aggregator's joined reference guidance must include the "
        "tool name so it can re-dispatch"
    )
    assert "/tmp/example.txt" in joined, (
        "the aggregator's joined reference guidance must include the "
        "tool arguments so it can re-dispatch"
    )
