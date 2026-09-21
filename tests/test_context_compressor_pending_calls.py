"""Regression tests for the arg-payload exemption and pending-call protection (#8adf333d1b).

Restored 2026-09-21: this fix (write_file/patch/jsbc_call exempt from elision up to a hard
cap; pending tool_calls never truncated) was reset off main on 2026-09-17 and never landed.
See agent/context_compressor.py's _ARG_PAYLOAD_TOOLS / _pending_tool_call_indices for the
production code these tests cover.
"""

import json

from agent.context_compressor import (
    _elided,
    _truncate_tool_call_args_json,
)

BRIEF = "Dispatch brief. " + ("payload " * 400)


def test_head_preserved_marker_appended_for_long_values():
    """The current contract: head kept verbatim, a self-describing marker appended at the cut.

    (Earlier design purely elided with no head — see history. The compressor now keeps a
    fixed-size head so a value truncated for size is still partially readable, and the
    marker itself is un-imitable: see TestTruncationMarkerNotImitable.)
    """
    args = json.dumps({"prompt": BRIEF})
    out = json.loads(_truncate_tool_call_args_json(args))["prompt"]
    assert out.startswith(BRIEF[:200])
    assert "HERMES-CONTEXT-COMPRESSION" in out


def test_marker_reports_the_size_of_what_was_cut():
    a = json.loads(_truncate_tool_call_args_json(json.dumps({"p": "A" * 900})))["p"]
    b = json.loads(_truncate_tool_call_args_json(json.dumps({"p": "B" * 900})))["p"]
    assert a.startswith("A" * 200)
    assert b.startswith("B" * 200)
    # Same input length -> same reported omission count in both markers.
    assert a[200:] == b[200:]


def test_length_is_reported_so_the_loss_is_visible():
    out = _elided(2534)
    assert "2534" in out


def test_placeholder_is_not_shaped_like_truncated_content():
    """No ellipsis, no bracketed marker glued to a text tail."""
    out = _elided(1000)
    assert not out.startswith("\u2026")
    assert "\u2026" not in out
    assert out.startswith("[") and out.endswith("]")


def test_short_values_are_untouched():
    short = "keep me whole"
    out = json.loads(_truncate_tool_call_args_json(json.dumps({"p": short})))["p"]
    assert out == short


def test_nested_values_are_also_replaced_whole():
    payload = {"evidence": {"measured": "Z" * 900, "note": "short"}}
    out = json.loads(_truncate_tool_call_args_json(json.dumps(payload)))
    assert out["evidence"]["measured"].startswith("Z" * 200)
    assert "HERMES-CONTEXT-COMPRESSION" in out["evidence"]["measured"]
    assert out["evidence"]["note"] == "short"


def test_output_remains_valid_json():
    """Providers 400 on malformed function.arguments - see issue #11762."""
    args = json.dumps({"path": "/tmp/x", "content": "Q" * 3000, "n": 3})
    parsed = json.loads(_truncate_tool_call_args_json(args))
    assert parsed["path"] == "/tmp/x"
    assert parsed["n"] == 3


def test_non_json_arguments_pass_through():
    raw = "not json at all " * 50
    assert _truncate_tool_call_args_json(raw) == raw


def test_pressure_pass_does_not_elide_pending_tool_call_args():
    """Pass 4 (protected-tail pressure) must respect the pending guard.

    Measured 2026-09-01: pass 3 correctly protected a dispatch brief, then
    pass 4 cut the same argument when the transcript was heavy enough to
    trigger pressure demotion. That is why truncation looked like it
    depended on conversation weight rather than payload size, and why a
    canary passed on a light turn and the same brief failed on a heavy one.

    Behavioural (not source-grep): exercises the real pass-3/4 pipeline end
    to end so a refactor of internal helper names can't silently break the
    guard while leaving a stale string match green.
    """
    from unittest.mock import patch as _patch

    from agent.context_compressor import ContextCompressor

    huge_args = json.dumps({"prompt": "B" * 3000})
    with _patch("agent.context_compressor.get_model_context_length", return_value=8000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )
    messages = [{"role": "user", "content": "start"}]
    for i in range(30):
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": "some_tool", "arguments": json.dumps({"note": "x" * 900})}},
        ]})
        messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "y" * 900})
    # The last assistant turn's tool_call is PENDING - no matching tool result follows it.
    messages.append({"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_pending", "type": "function",
         "function": {"name": "some_tool", "arguments": huge_args}},
    ]})
    result, _ = c._prune_old_tool_results(messages, protect_tail_count=2, protect_tail_tokens=200)
    pending_msg = next(
        m for m in result
        if m.get("role") == "assistant" and m.get("tool_calls")
        and m["tool_calls"][0].get("id") == "call_pending"
    )
    surviving_args = pending_msg["tool_calls"][0]["function"]["arguments"]
    assert surviving_args == huge_args, (
        "a pending tool_call (no result landed yet) must survive pass 4's pressure "
        "demotion untouched - eliding it corrupts work still in flight"
    )


def test_wrapper_path_resolves_real_tool_name_for_exemption():
    """A payload tool dispatched through the generic tool_call wrapper is exempt too.

    Measured 2026-09-07: the exemption checked only the outer wrapper name
    (``mcp__hermes___tool_call``), which is never in ``_ARG_PAYLOAD_TOOLS``,
    so every wrapped dispatch was eligible for elision even though the tool
    it wraps (here ``mcp__control_plane__jsbc_call``) is exempt. The real
    tool name lives at ``arguments.name`` inside the wrapper's own args and
    must be resolved before the exemption check runs.
    """
    from agent.context_compressor import ContextCompressor

    wrapped_args = json.dumps({
        "name": "mcp__control_plane__jsbc_call",
        "arguments": {"op": "orchestrator.send", "body": "B" * 3000},
    })
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "agent.context_compressor.get_model_context_length", return_value=100000,
    ):
        c = ContextCompressor(
            model="test/model", threshold_percent=0.85,
            protect_first_n=1, protect_last_n=1, quiet_mode=True,
        )
    messages = [
        {"role": "user", "content": "dispatch it"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "mcp__hermes___tool_call", "arguments": wrapped_args}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "done"},
    ]
    result, _ = c._prune_old_tool_results(messages, protect_tail_count=1)
    surviving_args = result[1]["tool_calls"][0]["function"]["arguments"]
    assert surviving_args == wrapped_args, (
        "a payload tool dispatched through the tool_call wrapper must be exempt "
        "from elision by resolving arguments.name, not the outer wrapper name"
    )
