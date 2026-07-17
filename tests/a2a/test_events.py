"""Tool-progress metadata sent to the peer is bounded and operator-controllable.

An A2A server delegates work for a remote peer; echoing unbounded tool arguments
and results (file contents, shell output, secrets) into status metadata is both a
size and a disclosure hazard. ``a2a.tool_io`` controls the exposure:
``preview`` (default, bounded), ``none`` (names only), ``full`` (unbounded).
"""

from __future__ import annotations

from plugins.platforms.a2a.events import (
    _RESULT_PREVIEW_LIMIT,
    _tool_call_metadata,
    _tool_result_metadata,
)


def test_preview_mode_bounds_large_args():
    md = _tool_call_metadata("read_file", {"blob": "x" * 5000})
    assert md["hermes/kind"] == "tool-call"
    assert md["hermes/tool"] == "read_file"
    assert isinstance(md["hermes/args"], str)
    assert len(md["hermes/args"]) <= _RESULT_PREVIEW_LIMIT + 1  # + ellipsis


def test_preview_mode_keeps_small_structured_args():
    md = _tool_call_metadata("read_file", {"path": "a.py"})
    assert md["hermes/args"] == {"path": "a.py"}


def test_preview_mode_bounds_large_results():
    md = _tool_result_metadata("terminal", "y" * 5000)
    assert md["hermes/kind"] == "tool-result"
    assert isinstance(md["hermes/result"], str)
    assert len(md["hermes/result"]) <= _RESULT_PREVIEW_LIMIT + 1


def test_none_mode_omits_args_and_results():
    call = _tool_call_metadata("terminal", {"cmd": "cat secrets.env"}, "none")
    result = _tool_result_metadata("terminal", "API_KEY=sk-very-secret", "none")
    assert call["hermes/tool"] == "terminal" and "hermes/args" not in call
    assert result["hermes/tool"] == "terminal" and "hermes/result" not in result


def test_full_mode_preserves_unbounded_structure():
    md = _tool_call_metadata("read_file", {"path": "a.py", "blob": "x" * 5000}, "full")
    assert md["hermes/args"] == {"path": "a.py", "blob": "x" * 5000}
