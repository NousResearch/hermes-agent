"""Regression for #121548: model-visible elisions are counted, not bare copyable prose."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.compression_marker import _COMPRESSION_MARKER_PREFIX, _COMPRESSION_MARKER_RE, elide_text
from agent.context_compressor import (
    ContextCompressor, _compact_fallback_turn, _sum_clarify, _truncate_tool_call_args_json,
)
from agent.skill_preprocessing import run_inline_shell


def test_elision_keeps_counts_budget_and_original_tail():
    source = "a" * 4100 + "END" * 700
    for head, tail, cap in ((700, 0, 700), (1400, 0, 1400), (4000, 1500, None), (199, 0, 199)):
        rendered = elide_text(source, head, tail, max_chars=cap)
        assert rendered.startswith(source[:1])
        assert _COMPRESSION_MARKER_PREFIX in rendered
        assert "...[truncated]" not in rendered
        assert _COMPRESSION_MARKER_RE.search(rendered)
        if tail:
            assert rendered.endswith(source[-tail:])
        if cap:
            assert len(rendered) <= cap
        marker_start = rendered.index(_COMPRESSION_MARKER_PREFIX)
        marker_end = rendered.index("⟫", marker_start) + 1
        omitted = len(source) - marker_start - (len(rendered) - marker_end)
        assert f"{omitted:,} of {len(source):,}" in rendered
    assert elide_text("short", 700, max_chars=700) == "short"


def test_model_visible_renderers_share_non_original_marker():
    source = "x" * 12000
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test/model", quiet_mode=True)
    rendered = (
        _compact_fallback_turn(source),
        _sum_clarify("clarify", {}, json.dumps({"user_response": source}), len(source), 1),
        compressor._serialize_for_summary([{"role": "user", "content": source}]),
        compressor._latest_user_task_snapshot([{"role": "user", "content": source}]),
        json.loads(_truncate_tool_call_args_json(json.dumps({"content": source})))["content"],
    )
    with patch("agent.skill_preprocessing.subprocess.run", return_value=SimpleNamespace(stdout=source, stderr="", returncode=0)):
        rendered += (run_inline_shell("printf text", None, 1),)
    for text in rendered:
        assert _COMPRESSION_MARKER_RE.search(text)
        assert "...[truncated]" not in text
