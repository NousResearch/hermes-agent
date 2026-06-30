"""WIKI-SLACK P6-A — render an Atlas EntityPage into Slack Block Kit (pure logic).

The ``/wiki <entity>`` persona payoff surface: turn the cited entity "story"
page (army-of-one ``GET /v1/wiki/{iri}``) into Slack Block Kit, UN-TRUNCATED
(handles Slack's 3000-char section limit + 50-block message ceiling), with every
``[cite:*]`` receipt rendered as a numbered click-through source line.

Anti-Goodhart guard: a narrative sentence with NO ``[cite:*]`` marker makes
``render_wiki_blocks`` raise ``UncitedSentenceError`` — uncited prose can never
reach Slack (defensive belt; the upstream synthesizer ProvenanceError already
makes an uncited page structurally impossible).

EntityPage shape (mirrors the WikiPageResponse the route returns)::

    {"iri": str, "title": str, "markdown": str,
     "citations": [{"chunk_id": str, "source_iri": str}, ...]}

DEFERRED: the WIKI-SLACK plan also specified a "what changed since last time"
delta section. The shipped WIKI-PAGE EntityPage does not yet carry per-user
last-view state, so the delta is a follow-up (needs a last-view store +
``build_entity_page(since=...)``). The un-truncated + cited-receipt +
anti-Goodhart invariants — the core value — are implemented here.
"""

from __future__ import annotations

import re
from typing import Any

# Slack Block Kit limits.
SECTION_TEXT_LIMIT = 2900  # < 3000 hard cap, leaving margin for mrkdwn glyphs
BLOCK_CEILING = 50

_CITE_RE = re.compile(r"\[cite:([^\]]+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9]")


class UncitedSentenceError(ValueError):
    """A narrative sentence carried no [cite:*] receipt — refuse to render."""

    def __init__(self, sentence: str) -> None:
        super().__init__(f"uncited sentence cannot reach Slack: {sentence!r}")
        self.sentence = sentence


def _split_sentences(markdown: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(markdown.strip()) if s.strip()]


def _assert_every_sentence_cited(markdown: str) -> None:
    for sentence in _split_sentences(markdown):
        if _WORD_RE.search(sentence) and not _CITE_RE.search(sentence):
            raise UncitedSentenceError(sentence)


def _number_citations(citations: list[dict[str, Any]]) -> dict[str, int]:
    """chunk_id -> 1-based receipt number, in citation order."""
    return {c["chunk_id"]: i + 1 for i, c in enumerate(citations)}


def _markdown_with_receipts(markdown: str, numbers: dict[str, int]) -> str:
    """Replace each [cite:chunk_id] with a bracketed receipt number [n]."""

    def _sub(m: re.Match[str]) -> str:
        cid = m.group(1)
        n = numbers.get(cid)
        return f" [{n}]" if n is not None else ""

    return _CITE_RE.sub(_sub, markdown)


def _section(text: str) -> dict[str, Any]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _chunk_text_into_sections(text: str) -> list[dict[str, Any]]:
    """Split prose into <=SECTION_TEXT_LIMIT section blocks WITHOUT dropping text.

    Splits on sentence boundaries; a single sentence longer than the limit is
    hard-wrapped so nothing is silently truncated.
    """
    sections: list[dict[str, Any]] = []
    buf = ""
    for sentence in _split_sentences(text):
        piece = sentence if not buf else f"{buf} {sentence}"
        if len(piece) <= SECTION_TEXT_LIMIT:
            buf = piece
            continue
        if buf:
            sections.append(_section(buf))
            buf = ""
        # Sentence alone may exceed the limit — hard-wrap it.
        while len(sentence) > SECTION_TEXT_LIMIT:
            sections.append(_section(sentence[:SECTION_TEXT_LIMIT]))
            sentence = sentence[SECTION_TEXT_LIMIT:]
        buf = sentence
    if buf:
        sections.append(_section(buf))
    return sections


def render_wiki_blocks(page: dict[str, Any]) -> list[dict[str, Any]]:
    """Render an EntityPage dict into a list of Slack Block Kit blocks.

    Raises ``UncitedSentenceError`` if any narrative sentence lacks a receipt.
    Output is capped at ``BLOCK_CEILING`` blocks; overflow is replaced by a
    context line pointing to the full page rather than silently dropped.
    """
    title = str(page.get("title") or page.get("iri") or "entity")
    iri = str(page.get("iri") or "")
    markdown = str(page.get("markdown") or "")
    citations: list[dict[str, Any]] = list(page.get("citations") or [])

    _assert_every_sentence_cited(markdown)

    numbers = _number_citations(citations)
    body = _markdown_with_receipts(markdown, numbers)

    blocks: list[dict[str, Any]] = [
        {"type": "header", "text": {"type": "plain_text", "text": title[:150]}},
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"wiki · current facts only · `{iri}`"}
            ],
        },
    ]
    blocks.extend(_chunk_text_into_sections(body))

    if citations:
        blocks.append({"type": "divider"})
        lines = [
            f"*[{numbers[c['chunk_id']]}]* <{c['source_iri']}|{c['source_iri']}>"
            for c in citations
        ]
        # Chunk the source list into <=limit section blocks (a long page can have
        # more sources than fit one 3000-char block).
        buf = "*Sources*"
        for line in lines:
            candidate = f"{buf}\n{line}"
            if len(candidate) > SECTION_TEXT_LIMIT:
                blocks.append(_section(buf))
                buf = line
            else:
                buf = candidate
        if buf:
            blocks.append(_section(buf))

    # 50-block ceiling: never silently truncate — replace the overflow tail with
    # a pointer to the full page.
    if len(blocks) > BLOCK_CEILING:
        kept = blocks[: BLOCK_CEILING - 1]
        kept.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"…page truncated for Slack — open the full wiki for "
                            f"`{iri}` in Atlas."
                        ),
                    }
                ],
            }
        )
        blocks = kept

    return blocks


__all__ = ["UncitedSentenceError", "render_wiki_blocks", "SECTION_TEXT_LIMIT", "BLOCK_CEILING"]
