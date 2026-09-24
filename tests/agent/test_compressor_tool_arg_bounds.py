"""End-to-end proof that the summarizer tool-arg bounds actually reach the truncation site.

`_render_tool_call_for_summary` truncates tool-call arguments to `_TOOL_ARGS_HEAD` chars once
they exceed `_TOOL_ARGS_MAX`. These bounds are deployment-overridable via the
`compression.tool_arg_head_chars` / `compression.tool_arg_min_chars` config options, so the
defaults here must match the historical constants and an override must take effect.
"""

from __future__ import annotations

import pytest

from agent.context_compressor import ContextCompressor


def _compressor(**kwargs) -> ContextCompressor:
    return ContextCompressor(model="gpt-4o", quiet_mode=True, **kwargs)


def _tool_call(arguments: str) -> dict:
    return {"function": {"name": "read_file", "arguments": arguments}}


def test_defaults_match_historical_constants():
    c = _compressor()
    assert c._TOOL_ARGS_MAX == 1500
    assert c._TOOL_ARGS_HEAD == 1200


def test_default_truncation_kicks_in_above_1500_chars():
    c = _compressor()
    rendered = c._render_tool_call_for_summary(_tool_call("x" * 3000))
    # 1200 chars of head + "..." + the wrapper
    assert rendered.startswith("  read_file(")
    assert rendered.endswith("...)")
    assert "x" * 1200 in rendered
    assert "x" * 1201 not in rendered


def test_below_threshold_is_untouched():
    c = _compressor()
    short = "y" * 40
    rendered = c._render_tool_call_for_summary(_tool_call(short))
    assert short in rendered
    assert "..." not in rendered


@pytest.mark.parametrize(
    ("head", "floor"),
    [(200, 300), (0, 0), (3500, 4000), (50, 60)],
)
def test_override_reaches_the_truncation_site(head, floor):
    """A configured (head, floor) pair must be what the renderer actually applies."""
    c = _compressor(tool_arg_head_chars=head, tool_arg_min_chars=floor)
    assert c._TOOL_ARGS_HEAD == head
    assert c._TOOL_ARGS_MAX == floor

    payload = "z" * (floor + 500)
    rendered = c._render_tool_call_for_summary(_tool_call(payload))
    if floor >= len(payload):
        # Threshold above the payload → no truncation at all.
        assert "..." not in rendered
        assert payload in rendered
    else:
        assert rendered.endswith("...)")
        assert payload[:head] in rendered
        assert payload[: head + 1] not in rendered
