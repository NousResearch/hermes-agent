"""Tests for ``hermes_cli.code_fences`` — Python fence parser."""

from hermes_cli.code_fences import parse_code_fences


def test_simple_backtick_fence():
    source = "```python\nprint('hello')\n```\n"
    fences = parse_code_fences(source)
    assert len(fences) == 1
    f = fences[0]
    assert f["closed"] is True
    assert f["fence_char"] == "`"
    assert f["fence_length"] == 3
    assert f["language"] == "python"
    assert f["raw_content"] == "print('hello')"


def test_no_language_defaults_to_text():
    source = "```\nsome code\n```\n"
    fences = parse_code_fences(source)
    assert fences[0]["language"] == "text"


def test_longer_fence():
    source = "````\ncode\n````\n"
    fences = parse_code_fences(source)
    assert fences[0]["fence_length"] == 4


def test_closer_shorter_than_opener():
    source = "`````\ncode\n```\n"
    fences = parse_code_fences(source)
    assert fences[0]["closed"] is False


def test_triple_tilde():
    source = "~~~js\nconst x = 1;\n~~~\n"
    fences = parse_code_fences(source)
    assert fences[0]["fence_char"] == "~"
    assert fences[0]["language"] == "js"
    assert fences[0]["raw_content"] == "const x = 1;"


def test_mismatched_character_does_not_close():
    source = "~~~\ncode\n```\n"
    fences = parse_code_fences(source)
    assert fences[0]["closed"] is False


def test_unclosed_fence():
    source = "```python\npartial code"
    fences = parse_code_fences(source)
    assert fences[0]["closed"] is False
    assert fences[0]["raw_content"] == "partial code"


def test_multiple_fences_latter_unclosed():
    source = "```py\na\n```\nsome text\n```js\nb\n"
    fences = parse_code_fences(source)
    assert len(fences) == 2
    assert fences[0]["closed"] is True
    assert fences[0]["raw_content"] == "a"
    assert fences[1]["closed"] is False


def test_language_first_token():
    source = "```typescript linenums\nx = 1\n```\n"
    fences = parse_code_fences(source)
    assert fences[0]["language"] == "typescript"


def test_language_lowercased():
    source = "```PYTHON\ncode\n```\n"
    fences = parse_code_fences(source)
    assert fences[0]["language"] == "python"


def test_preserves_tabs():
    source = "```py\n\tcode\n```\n"
    fences = parse_code_fences(source)
    assert "\t" in fences[0]["raw_content"]


def test_preserves_trailing_spaces():
    source = "```py\ncode  \n```\n"
    fences = parse_code_fences(source)
    assert fences[0]["raw_content"] == "code  "


def test_diff_detected():
    source = "```\n--- old\n+++ new\n```\n"
    fences = parse_code_fences(source)
    assert fences[0]["language"] == "diff"
