"""bounded_result keeps the terminal's cwd when it truncates an oversize tool result.

The TUI's OSC 7 cwd report reads the top-level ``cwd`` a terminal result carries when the
command changed directory. A long-output command that also ``cd``-ed must still carry it
after the viewer payload limit replaces the result with a truncated envelope.
"""
import json

import pytest

from gateway.session_tool_events import MAX_TOOL_EVENT_CHARS, bounded_result

CWD = "/Users/josh/dev/sandbox"


def _long_terminal_result(**extra):
    """A terminal result whose JSON text is well over the limit (quotes double when escaped)."""
    result = {"output": '"' * 40000, "exit_code": 0, "error": None, **extra}
    text = json.dumps(result)
    assert len(text) > MAX_TOOL_EVENT_CHARS
    return result, text


@pytest.mark.parametrize("is_error, body_key", [(False, "output"), (True, "error")])
@pytest.mark.parametrize("as_text", [True, False], ids=["json-text", "dict"])
def test_oversize_result_keeps_its_top_level_cwd(is_error, body_key, as_text):
    result, text = _long_terminal_result(cwd=CWD)

    bounded = json.loads(bounded_result(text if as_text else result, is_error))

    assert set(bounded) == {body_key, "truncated", "cwd"}
    assert bounded["cwd"] == CWD
    assert bounded["truncated"] is True
    assert bounded[body_key] == text[: MAX_TOOL_EVENT_CHARS // 2]


def test_cwd_at_the_length_cap_is_kept():
    result, _ = _long_terminal_result(cwd="/" + "d" * 4095)

    assert json.loads(bounded_result(result, False))["cwd"] == "/" + "d" * 4095


def test_oversize_result_without_cwd_gets_no_cwd_key():
    _, text = _long_terminal_result()

    bounded = json.loads(bounded_result(text, False))

    assert set(bounded) == {"output", "truncated"}


def test_oversize_non_json_text_gets_no_cwd_key():
    bounded = json.loads(bounded_result("x" * 70000, False))

    assert set(bounded) == {"output", "truncated"}
    assert bounded["output"] == "x" * (MAX_TOOL_EVENT_CHARS // 2)


@pytest.mark.parametrize(
    "cwd",
    [7, None, "", ["/tmp"], {"path": "/tmp"}, "/" + "d" * 4096],
    ids=["int", "none", "empty", "list", "dict", "over-4096"],
)
def test_oversize_result_with_unusable_cwd_gets_no_cwd_key(cwd):
    result, _ = _long_terminal_result(cwd=cwd)

    bounded = json.loads(bounded_result(result, False))

    assert "cwd" not in bounded


def test_nested_cwd_is_not_lifted():
    result, _ = _long_terminal_result(meta={"cwd": CWD})

    assert "cwd" not in json.loads(bounded_result(result, False))


def test_oversize_json_array_text_gets_no_cwd_key():
    text = json.dumps([{"cwd": CWD}] + ["x" * 70000])

    assert "cwd" not in json.loads(bounded_result(text, False))


def test_under_limit_results_pass_through_unchanged():
    text = json.dumps({"output": "ok", "exit_code": 0, "cwd": CWD})
    assert bounded_result(text, False) is text

    parsed = {"output": "ü", "exit_code": 0, "cwd": CWD}
    assert bounded_result(parsed, True) == json.dumps(parsed, ensure_ascii=True, default=str)
