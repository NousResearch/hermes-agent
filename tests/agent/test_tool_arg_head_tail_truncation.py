"""Compaction of tool-call arguments must keep both ends of long string leaves.

Regression for the head-only shrink in ``_truncate_tool_call_args_json``: it rewrote
(never removed) every non-tail assistant message, so a ``write_file`` / ``execute_code``
payload came back as its first 200 chars plus ``...[truncated]`` — the body, and with it
any file ending, was gone from the model's own visible history.
"""

import json

from unittest.mock import patch

from agent.context_compressor import ContextCompressor, _truncate_tool_call_args_json


def _source(n: int = 20) -> str:
    """A realistic ~1.1 KB write_file body (imports at the top, returns at the bottom)."""
    return "\n".join(f"def handler_{i}(request):\n    return process(request, {i})\n" for i in range(n))


def test_long_leaf_keeps_both_ends_and_a_counted_marker():
    source = _source()
    args = json.dumps({"path": "C:/project/app.py", "content": source, "cross_profile": True})
    assert len(args) > 500  # over the caller's trigger, so the shrink pass runs

    content = json.loads(_truncate_tool_call_args_json(args))["content"]

    assert content.startswith(source[:200])
    assert content.endswith(source[-200:])  # the tail is the file's ending — must survive
    assert f"...[truncated {len(source) - 400:,} chars]..." in content
    assert len(content) >= 400  # head-only left 214 of 1139 chars


def test_leaves_that_fit_head_plus_tail_are_untouched():
    # Single leaf, blob under the 500-char trigger: byte-identical.
    small = json.dumps({"path": "a.py", "content": "x" * 300, "note": "hi"})
    assert len(small) < 500
    assert _truncate_tool_call_args_json(small) == small

    # Blob over the trigger but every leaf fits head+tail: nothing to reclaim, so no rewrite
    # (re-serializing would only reflow whitespace and can grow the blob).
    wide = json.dumps({"a": "y" * 250, "b": "z" * 250})
    assert len(wide) > 500
    assert _truncate_tool_call_args_json(wide) == wide


def test_json_stays_valid_and_non_string_leaves_survive():
    payload = json.dumps({
        "retries": 3,
        "enabled": True,
        "timeout": None,
        "items": [1, 2, 3],
        "nested": {"deep": {"blob": "q" * 900}},
        "list_of_str": ["m" * 700, "n"],
    })

    parsed = json.loads(_truncate_tool_call_args_json(payload))  # must not raise

    assert parsed["retries"] == 3
    assert parsed["enabled"] is True
    assert parsed["timeout"] is None
    assert parsed["items"] == [1, 2, 3]
    assert parsed["nested"]["deep"]["blob"].startswith("q" * 200)
    assert parsed["nested"]["deep"]["blob"].endswith("q" * 200)
    assert parsed["list_of_str"][0].endswith("m" * 200)
    assert parsed["list_of_str"][1] == "n"


def test_pass3_rewrites_in_place_with_head_and_tail_intact():
    """End-to-end through the ordinary compaction path (``_prune_old_tool_results``)."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )
    source = _source(80)
    args = json.dumps({"path": "app.py", "content": source})
    messages = [
        {"role": "user", "content": "write it"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "write_file", "arguments": args}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"bytes_written": 1300}'},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]

    out, _ = compressor._prune_old_tool_results(messages, protect_tail_count=2)

    # Rewritten in place, not dropped — that is why the mangled version stayed visible.
    assert len(out) == len(messages)
    assert out[1]["role"] == "assistant"
    shrunken = json.loads(out[1]["tool_calls"][0]["function"]["arguments"])
    assert shrunken["content"].startswith(source[:200])
    assert shrunken["content"].endswith(source[-200:])
    assert f"...[truncated {len(source) - 400:,} chars]..." in shrunken["content"]
    assert shrunken["path"] == "app.py"
