"""Hermetic gap tests — hit structure (4 tests).

Implements the hit structure sub-section of the hermetic_test_gap_analysis.md:

* test_ep_hit_06_relevance_score_negative_input_clamped_to_zero
* test_ep_hit_07_relevance_score_above_one_clamped_to_one
* test_ep_hit_08_snippet_zero_length
* test_ep_hit_09_title_unicode_multibyte_truncated_at_byte_boundary_safe
"""

from __future__ import annotations

import json

from tests.test_executive_v2.canary_b1.evidence_pack import (
    SNIPPET_MAX_LEN,
    TITLE_MAX_LEN,
    _make_hit_v2,
    SOURCE_TTL_DAYS,
)


OBS = "2026-07-08T20:00:00+00:00"
UPD = "2026-07-08T20:00:00+00:00"


def _hit(relevance: float, snippet: str = "snip", title: str = "title"):
    return _make_hit_v2(
        source="gbrain",
        hit_id="b1-hit",
        title=title,
        relevance_score=relevance,
        snippet=snippet,
        source_uri="gbrain://b1-hit",
        source_updated_at=UPD,
        retrieval_mode="metadata_only",
        observed_at=OBS,
        ttl_days=SOURCE_TTL_DAYS["gbrain"],
    )


# ─────────────────────────────────────────────────────────────────────
# 2.5.1 — adversarial input: relevance clamped to [0,1]
# ─────────────────────────────────────────────────────────────────────


def test_ep_hit_06_relevance_score_negative_input_clamped_to_zero():
    h = _hit(relevance=-0.5)
    assert h.relevance_score == 0.0


def test_ep_hit_07_relevance_score_above_one_clamped_to_one():
    h = _hit(relevance=1.5)
    assert h.relevance_score == 1.0


# ─────────────────────────────────────────────────────────────────────
# 2.5.3 — edge case: snippet of length 0
# ─────────────────────────────────────────────────────────────────────


def test_ep_hit_08_snippet_zero_length():
    h = _hit(relevance=0.5, snippet="")
    assert h.snippet == ""
    # SNIPPET_MAX_LEN is an upper bound only
    assert SNIPPET_MAX_LEN >= 0


# ─────────────────────────────────────────────────────────────────────
# 2.5.4 — unicode boundary: title truncated safely
# ─────────────────────────────────────────────────────────────────────


def test_ep_hit_09_title_unicode_multibyte_truncated_at_byte_boundary_safe():
    """Title of multi-byte emoji repeated is truncated to TITLE_MAX_LEN chars
    (NOT bytes) — and the resulting dataclass is JSON-serializable.
    """
    title_in = "🚀" * 1000  # ~4096 bytes in UTF-8
    h = _hit(relevance=0.5, title=title_in)
    # Char count (Python str length) — slice uses [:TITLE_MAX_LEN]
    assert len(h.title) <= TITLE_MAX_LEN, (
        f"title len {len(h.title)} > TITLE_MAX_LEN {TITLE_MAX_LEN}"
    )

    # JSON-safe serialization (via dataclasses.asdict + default=str)
    from dataclasses import asdict
    d = asdict(h)
    s = json.dumps(d, default=str, ensure_ascii=False)
    parsed = json.loads(s)
    assert "title" in parsed
    assert len(parsed["title"]) <= TITLE_MAX_LEN