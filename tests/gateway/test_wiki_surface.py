"""WIKI-SLACK P6-A — EntityPage → Slack Block Kit renderer tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.wiki_surface import (
    BLOCK_CEILING,
    SECTION_TEXT_LIMIT,
    UncitedSentenceError,
    render_wiki_blocks,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "entity_page_acme.json"


def _load_fixture() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _all_section_text(blocks: list[dict]) -> str:
    return "\n".join(
        b["text"]["text"] for b in blocks if b.get("type") == "section"
    )


def test_every_sentence_renders_untruncated():
    page = _load_fixture()
    blocks = render_wiki_blocks(page)
    rendered = _all_section_text(blocks)
    # Each narrative sentence's prose (minus the [cite:*] marker) appears verbatim.
    assert "Acme Corp currently runs its data platform on Neon" in rendered
    assert "The most recent touchpoint was a kickoff call on 2026-06-01" in rendered
    assert "open commitment to deliver a pilot integration by Q3 2026" in rendered


def test_each_citation_renders_as_clickthrough_receipt():
    page = _load_fixture()
    blocks = render_wiki_blocks(page)
    rendered = _all_section_text(blocks)
    for i, c in enumerate(page["citations"], start=1):
        # narrative carries the numbered receipt, sources list links the IRI.
        assert f"[{i}]" in rendered
        assert c["source_iri"] in rendered


def test_uncited_sentence_raises():
    page = _load_fixture()
    page["markdown"] += " Acme also raised a Series B last year."  # no [cite:*]
    with pytest.raises(UncitedSentenceError):
        render_wiki_blocks(page)


def test_no_section_exceeds_slack_limit():
    # A long page with many cited sentences splits across sections, each within
    # Slack's char limit; nothing is dropped (sources also chunked).
    sentences = [
        f"Fact number {n} about the entity is recorded [cite:c{n}]." for n in range(200)
    ]
    page = {
        "iri": "urn:atlas:bigentity",
        "title": "Big Entity",
        "markdown": " ".join(sentences),
        "citations": [
            {"chunk_id": f"c{n}", "source_iri": f"urn:src:{n}"} for n in range(200)
        ],
    }
    blocks = render_wiki_blocks(page)
    for b in blocks:
        if b.get("type") == "section":
            assert len(b["text"]["text"]) <= SECTION_TEXT_LIMIT
    assert len(blocks) <= BLOCK_CEILING


def test_block_ceiling_truncates_with_signpost():
    # Each sentence ~ one full section → enough sentences to exceed 50 blocks.
    pad = "x" * 2800
    sentences = [f"Fact {n} {pad} [cite:c{n}]." for n in range(60)]
    page = {
        "iri": "urn:atlas:huge",
        "title": "Huge",
        "markdown": " ".join(sentences),
        "citations": [
            {"chunk_id": f"c{n}", "source_iri": f"urn:src:{n}"} for n in range(60)
        ],
    }
    blocks = render_wiki_blocks(page)
    assert len(blocks) <= BLOCK_CEILING
    # Overflow is signposted, not silently dropped.
    ctx = [b for b in blocks if b.get("type") == "context"]
    assert any("truncated" in e["text"] for b in ctx for e in b.get("elements", []))


def test_empty_citations_still_renders_narrative_if_cited_inline():
    # A page whose only sentence carries a marker but with no citations list:
    # the guard passes (marker present); no Sources section is emitted.
    page = {
        "iri": "urn:atlas:x",
        "title": "X",
        "markdown": "X is a thing [cite:c1].",
        "citations": [],
    }
    blocks = render_wiki_blocks(page)
    assert any(b.get("type") == "section" for b in blocks)
    assert all("*Sources*" not in (b.get("text", {}).get("text", "")) for b in blocks)


def test_renders_what_changed_section():
    page = _load_fixture()
    page["changes"] = [
        {"predicate": "runsOn", "object": "Postgres", "kind": "added"},
        {"predicate": "runsOn", "object": "Linear", "kind": "retired"},
    ]
    blocks = render_wiki_blocks(page)
    text = "\n".join(
        b["text"]["text"] for b in blocks if b.get("type") == "section"
    )
    assert "What changed since you last looked" in text
    assert "Postgres" in text and "Linear" in text
