"""Regression tests for the gateway fallback secret-redaction patterns.

Issue #81073: ``_GATEWAY_SECRET_PATTERNS`` anchored with ``\\b``, which under
Python 3's default Unicode semantics treats CJK/fullwidth letters as word
characters. A token glued to a CJK character (``xx中sk-...``) is therefore NOT
at a ``\\b`` boundary, so the fallback pattern pass silently left it
unredacted. The anchors are now ASCII word lookarounds
(``(?<![A-Za-z0-9_])`` / ``(?![A-Za-z0-9_])``) so Unicode-glued tokens are
caught the same as ASCII-neighbored ones.
"""

import pytest

from gateway.run import _GATEWAY_SECRET_PATTERNS

# Each entry is (text with token, expected-absence substring).
_FAKE_TOKENS = [
    ("sk-" + "a" * 20, "sk-"),
    ("ghp_" + "b" * 20, "ghp_"),
    ("xapp-1-" + "c" * 20, "xapp-"),
    ("xoxb-" + "d" * 20, "xoxb-"),
    ("hf_" + "e" * 20, "hf_"),
    ("glpat-" + "f" * 20, "glpat-"),
    ("Bearer " + "g" * 20, "Bearer g"),
]


def _pattern_pass(text: str) -> str:
    """Run exactly the fallback pass ``_redact_gateway_user_facing_secrets``
    applies after the Tirith-grade redactor."""
    redacted = text
    for pattern in _GATEWAY_SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]",
            redacted,
        )
    return redacted


@pytest.mark.parametrize("token,prefix", _FAKE_TOKENS)
def test_cjk_glued_tokens_are_redacted(token, prefix):
    """A token glued directly to a CJK character must be redacted (#81073)."""
    redacted = _pattern_pass("xx中" + token)
    assert "[REDACTED]" in redacted
    assert prefix not in redacted, f"CJK-glued token leaked: {redacted!r}"


@pytest.mark.parametrize("token,prefix", _FAKE_TOKENS)
def test_fullwidth_digit_glued_tokens_are_redacted(token, prefix):
    """A token glued to a fullwidth digit (Unicode ``\\w``) must be redacted."""
    redacted = _pattern_pass("xx１" + token)
    assert "[REDACTED]" in redacted
    assert prefix not in redacted, f"fullwidth-glued token leaked: {redacted!r}"


@pytest.mark.parametrize("token,prefix", _FAKE_TOKENS)
def test_ascii_neighbor_tokens_still_redacted(token, prefix):
    """ASCII-neighbored tokens keep redacting (no regression)."""
    redacted = _pattern_pass("xx " + token)
    assert "[REDACTED]" in redacted
    assert prefix not in redacted, f"ASCII-neighbor token leaked: {redacted!r}"


@pytest.mark.parametrize("token,prefix", _FAKE_TOKENS)
def test_embedded_in_ascii_word_not_matched(token, prefix):
    """A token embedded inside an ASCII identifier must NOT match — same
    boundary semantics as ``\\b``."""
    redacted = _pattern_pass("xx" + token + "yy")
    assert "[REDACTED]" not in redacted, (
        f"embedded token over-matched: {redacted!r}"
    )


def test_bearer_prefix_is_preserved():
    """The Bearer prefix survives redaction (group(1) preserved)."""
    redacted = _pattern_pass("Authorization: Bearer " + "z" * 20)
    assert "Bearer " in redacted
    assert "Bearer zzz" not in redacted


def test_prose_without_tokens_unchanged():
    """Ordinary prose must pass through untouched."""
    text = "Hello, world. Nothing secret here."
    assert _pattern_pass(text) == text
