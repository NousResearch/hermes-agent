"""Streaming scrubber coverage for DeepSeek native DSML markup (#119261, #54283).

The per-delta path needs its own DSML handling: ``strip_think_blocks`` clean-up runs on the
finished string, so a stream consumer that renders deltas would still show the markup. These
tests pin the three shapes seen in the wild — a closed block, orphan closers from a degenerate
tail, and an opener cut before its ``>`` — plus the no-over-strip guarantee for prose.
"""

from __future__ import annotations

import pytest

from agent.think_scrubber import StreamingThinkScrubber

FW = "\uFF5C"  # fullwidth vertical bar, the token DeepSeek actually emits
OP = f"<{FW}DSML{FW}"
CL = f"</{FW}DSML{FW}"


def _run(*deltas: str) -> str:
    scrubber = StreamingThinkScrubber()
    out = "".join([scrubber.feed(delta) for delta in deltas]) + scrubber.flush()
    return out


def _assert_clean(text: str) -> None:
    for marker in ("DSML", "invoke", "parameter", "tool_call", "function_call"):
        assert marker not in text, f"{marker!r} leaked: {text!r}"


@pytest.mark.parametrize("pipe", [FW, "|"])
def test_closed_tool_calls_block_single_delta(pipe: str) -> None:
    op, cl = f"<{pipe}DSML{pipe}", f"</{pipe}DSML{pipe}"
    text = (
        f"{op}tool_calls>{op}invoke name=\"terminal\">{op}parameter name=\"cmd\">ls"
        f"{cl}parameter>{cl}invoke>{cl}tool_calls>"
    )
    assert _run("这是结果。\n", text, "\n\n完毕。") == "这是结果。\n\n\n完毕。"


def test_closed_block_split_across_deltas() -> None:
    deltas = [
        "开始。\n",
        f"{OP}tool_calls>",
        f'{OP}invoke name="read_file">',
        f'{OP}parameter name="path">x.txt',
        f"{CL}parameter>",
        f"{CL}invoke>",
        f"{CL}tool_calls>",
        "完成。",
    ]
    out = _run(*deltas)
    _assert_clean(out)
    assert "开始。" in out and "完成。" in out


def test_orphan_closers_from_degenerate_tail() -> None:
    assert _run("正文。\n", f"{CL}parameter>", f"{CL}invoke>") == "正文。\n"


def test_orphan_closer_mid_prose() -> None:
    assert _run("前文", f"{CL}parameter>", f"{CL}invoke>", "后文继续。") == "前文后文继续。"


def test_opener_cut_before_gt() -> None:
    """Bedrock deepseek.v3.2 cuts the text channel right after the opener's tag name."""
    assert _run("文字", f"{OP}function_calls") == "文字"


def test_unterminated_invoke_discards_remainder() -> None:
    out = _run("我来读文件。", f'{OP}invoke name="read_file">', f'{OP}parameter name="path">x.txt')
    _assert_clean(out)
    assert out == "我来读文件。"


def test_opener_held_across_delta_boundary() -> None:
    """A DSML opener split mid-name must not be emitted as prose before it resolves."""
    out = _run("文字", f"<{FW}DSM", f"L{FW}invo", "ke>", f"{CL}invoke>")
    _assert_clean(out)
    assert out == "文字"


def test_prose_mentioning_tag_name_survives() -> None:
    """No block-boundary gate for DSML, but plain prose about XML must still come through."""
    text = "XML 里 <invoke> 是标签写法，正文继续。"
    assert _run(text) == text


def test_dsml_opener_mid_line_still_stripped() -> None:
    """Unlike ``<think>``, a DSML opener mid-line still starts a block: it is never prose."""
    assert _run("前文", f"{OP}tool_calls>", "ls", f"{CL}tool_calls>", "后文") == "前文后文"


def test_reset_clears_dsml_block_state() -> None:
    scrubber = StreamingThinkScrubber()
    scrubber.feed(f"X{OP}tool_calls>")
    scrubber.reset()
    assert scrubber.feed("新的一轮正文。") == "新的一轮正文。"
