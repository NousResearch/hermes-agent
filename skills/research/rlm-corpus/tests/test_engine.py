"""Tests for engine-layer pure helpers: parsing, truncation, citations.

Kernel / LLM-integration tests are intentionally out of scope for this suite;
they require a running jupyter kernel and a live LLM endpoint.
"""
from __future__ import annotations

from rlm_engine import (
    extract_code_block,
    extract_final,
    format_answer_with_references,
    _truncate,
)


# ---------------------------------------------------------------------------
# Protocol parsing
# ---------------------------------------------------------------------------


def test_extract_repl_code_block():
    text = "Sure, let me try.\n\n```repl\nprint('hi')\n```\n\nthoughts..."
    assert extract_code_block(text) == "print('hi')"


def test_extract_code_block_none_when_absent():
    assert extract_code_block("no code here") is None


def test_extract_final_simple():
    kind, payload = extract_final("before FINAL(hello) after")
    assert kind == "answer"
    assert payload == "hello"


def test_extract_final_with_nested_parens():
    kind, payload = extract_final(
        "work done.\nFINAL(The answer is f(x) = 2x (see [paper.pdf]).)"
    )
    assert kind == "answer"
    assert payload == "The answer is f(x) = 2x (see [paper.pdf])."


def test_extract_final_var():
    kind, payload = extract_final("FINAL_VAR(my_answer)")
    assert kind == "var"
    assert payload == "my_answer"


def test_extract_final_none_when_absent():
    assert extract_final("no final marker in this text") is None


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


def test_truncate_preserves_head_and_tail():
    s = "A" * 500 + "B" * 500 + "C" * 500
    out = _truncate(s, 200)
    assert len(out) < len(s)
    assert out.startswith("A")
    assert out.endswith("C")
    assert "truncated" in out


def test_truncate_no_op_when_short():
    assert _truncate("short", 1000) == "short"


# ---------------------------------------------------------------------------
# Citation formatting
# ---------------------------------------------------------------------------


def test_format_answer_with_references_resolves_filenames():
    corpus = {
        "alpha.md": {
            "metadata": {
                "title": "Alpha",
                "authors": ["A. One"],
                "year": 2023,
            },
        },
        "beta.md": {
            "metadata": {
                "title": "Beta",
                "authors": ["B. Two", "C. Three"],
                "year": 2024,
            },
        },
    }
    answer = "Alpha says X [alpha.md] but Beta pushes back [beta.md]."
    out = format_answer_with_references(answer, corpus)
    assert answer in out
    assert "**References**" in out
    assert "[alpha.md] **Alpha**" in out
    assert "B. Two, C. Three" in out


def test_format_answer_without_citations_is_untouched():
    answer = "no citations here at all"
    corpus = {}
    assert format_answer_with_references(answer, corpus) == answer


def test_format_answer_ignores_unknown_citations():
    corpus = {"known.md": {"metadata": {"title": "Known", "authors": [], "year": 2020}}}
    answer = "see [unknown.md] and [known.md]"
    out = format_answer_with_references(answer, corpus)
    assert "[known.md]" in out
    assert "[unknown.md] **" not in out  # no reference entry for unknown
