from agent.trajectory import convert_scratchpad_to_think, has_incomplete_scratchpad


def test_convert_scratchpad_to_think_rewrites_tags():
    content = "<REASONING_SCRATCHPAD>think</REASONING_SCRATCHPAD> done"
    assert convert_scratchpad_to_think(content) == "<think>think</think> done"


def test_has_incomplete_scratchpad_detects_real_unclosed_tag():
    content = "Answering...\n<REASONING_SCRATCHPAD>still thinking"
    assert has_incomplete_scratchpad(content) is True


def test_has_incomplete_scratchpad_ignores_fenced_code_block_mentions():
    content = """Here is the grep output:

```text
<REASONING_SCRATCHPAD>
```
"""
    assert has_incomplete_scratchpad(content) is False


def test_has_incomplete_scratchpad_ignores_blockquote_mentions():
    content = "> literal token <REASONING_SCRATCHPAD>\n\nFinal answer."
    assert has_incomplete_scratchpad(content) is False


def test_has_incomplete_scratchpad_ignores_inline_code_mentions():
    content = "The user literally typed `<REASONING_SCRATCHPAD>` in the prompt."
    assert has_incomplete_scratchpad(content) is False


def test_has_incomplete_scratchpad_still_flags_real_tag_after_quote():
    content = "> quoted literal <REASONING_SCRATCHPAD>\n\n<REASONING_SCRATCHPAD>real"
    assert has_incomplete_scratchpad(content) is True
