"""Splitting a document into retrievable pieces, without losing where they came from.

Retrieval is only as trustworthy as its citations. A chunk that cannot say which file, and
which lines of it, it came from is an assertion the reader has to take on faith — which is
the single most expensive failure mode an enterprise knowledge system has. So provenance is
carried on the chunk itself rather than reconstructed later, and every field needed to
produce a citation is populated at split time.

The splitter is structural, not semantic: it breaks on blank lines and never mid-paragraph,
and it tracks Markdown heading depth so each chunk knows the section it sits in. That is
deliberate. A smarter splitter would need a model, a model would need a provider, and the
dependency surface of this layer is the standard library.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterator, Optional

#: Target size in characters. Roughly 250-400 tokens of English prose — large enough to
#: carry an argument, small enough that several fit in a tool result without crowding out
#: the conversation that asked for them.
DEFAULT_MAX_CHARS = 1600

#: Trailing characters repeated into the next chunk, so a statement split across a boundary
#: is still findable from either side.
DEFAULT_OVERLAP_CHARS = 200

#: Below this, a trailing fragment is appended to the previous chunk instead of standing
#: alone. A 40-character chunk matches queries it cannot answer.
MIN_CHUNK_CHARS = 120

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of one document."""

    chunk_id: str
    source_id: str
    doc_path: str
    ordinal: int
    text: str
    start_line: int
    end_line: int
    #: Enclosing Markdown headings, outermost first. Empty for formats without headings.
    heading_trail: tuple[str, ...] = ()
    doc_title: str = ""

    @property
    def citation(self) -> str:
        """How this chunk identifies itself to a reader — and to a model.

        Deliberately not a URL. The path is what a person can open and what an auditor can
        check; a link would be a promise about a web server NOVA does not run.
        """
        where = f"{self.doc_path}:{self.start_line}"
        if self.start_line != self.end_line:
            where += f"-{self.end_line}"
        section = " › ".join(self.heading_trail)
        return f"{where} ({section})" if section else where

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "doc_path": self.doc_path,
            "ordinal": self.ordinal,
            "text": self.text,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "heading_trail": list(self.heading_trail),
            "doc_title": self.doc_title,
            "citation": self.citation,
        }


@dataclass
class _Block:
    """A paragraph-sized run of lines, with the heading trail in force above it."""

    text: str
    start_line: int
    end_line: int
    heading_trail: tuple[str, ...]


def chunk_document(
    text: str,
    *,
    source_id: str,
    doc_path: str,
    doc_title: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Split *text* into chunks, each knowing exactly where in the document it came from."""
    if max_chars < MIN_CHUNK_CHARS:
        raise ValueError(f"max_chars must be at least {MIN_CHUNK_CHARS}, got {max_chars}")
    overlap_chars = max(0, min(overlap_chars, max_chars // 2))

    blocks = list(_blocks(text))
    if not blocks:
        return []

    chunks: list[Chunk] = []
    pending: list[_Block] = []
    pending_len = 0

    def flush() -> None:
        nonlocal pending, pending_len
        if not pending:
            return
        body = "\n\n".join(block.text for block in pending)
        chunks.append(
            _make_chunk(
                body,
                source_id=source_id,
                doc_path=doc_path,
                doc_title=doc_title,
                ordinal=len(chunks),
                start_line=pending[0].start_line,
                end_line=pending[-1].end_line,
                heading_trail=pending[0].heading_trail,
            )
        )
        pending, pending_len = [], 0

    for block in blocks:
        pieces = _split_oversized(block, max_chars) if len(block.text) > max_chars else [block]
        for piece in pieces:
            addition = len(piece.text) + (2 if pending else 0)
            if pending and pending_len + addition > max_chars:
                flush()
            pending.append(piece)
            pending_len += len(piece.text) + (2 if len(pending) > 1 else 0)
    flush()

    # A trailing scrap says nothing on its own; fold it back into its predecessor.
    if len(chunks) > 1 and len(chunks[-1].text) < MIN_CHUNK_CHARS:
        tail = chunks.pop()
        previous = chunks.pop()
        chunks.append(
            _make_chunk(
                previous.text + "\n\n" + tail.text,
                source_id=source_id,
                doc_path=doc_path,
                doc_title=doc_title,
                ordinal=previous.ordinal,
                start_line=previous.start_line,
                end_line=tail.end_line,
                heading_trail=previous.heading_trail,
            )
        )

    return _with_overlap(
        chunks,
        overlap_chars=overlap_chars,
        source_id=source_id,
        doc_path=doc_path,
        doc_title=doc_title,
    )


def _blocks(text: str) -> Iterator[_Block]:
    """Paragraph-sized blocks, carrying the heading trail in force at that point.

    A heading line is emitted as part of the block that follows it rather than on its own,
    so a chunk that begins at a section boundary begins with the section's title — which is
    both better retrieval and a better citation.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    trail: list[tuple[int, str]] = []
    buffer: list[str] = []
    start = 1

    for number, line in enumerate(lines, start=1):
        heading = _HEADING.match(line)
        if heading:
            if buffer:
                yield _Block("\n".join(buffer).strip(), start, number - 1, _trail(trail))
                buffer = []
            depth = len(heading.group(1))
            while trail and trail[-1][0] >= depth:
                trail.pop()
            trail.append((depth, heading.group(2)))
            buffer = [line]
            start = number
            continue
        if not line.strip():
            if buffer and "".join(buffer).strip():
                yield _Block("\n".join(buffer).strip(), start, number - 1, _trail(trail))
            buffer = []
            start = number + 1
            continue
        if not buffer:
            start = number
        buffer.append(line)

    if buffer and "".join(buffer).strip():
        yield _Block("\n".join(buffer).strip(), start, len(lines), _trail(trail))


def _trail(stack: list[tuple[int, str]]) -> tuple[str, ...]:
    return tuple(title for _, title in stack)


def _split_oversized(block: _Block, max_chars: int) -> list[_Block]:
    """Break a single block that is larger than a whole chunk.

    Prefers sentence ends, then whitespace, then a hard cut. A minified JSON blob or a
    base64 payload has none of the first two — it is still split rather than dropped,
    because refusing to index a document because of one long line is worse than a seam in
    the middle of a token nobody will search for.
    """
    pieces: list[_Block] = []
    remaining = block.text
    line = block.start_line
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
        cut = cut + 1 if cut > max_chars // 2 else window.rfind(" ")
        if cut <= max_chars // 4:
            cut = max_chars
        head, remaining = remaining[:cut].strip(), remaining[cut:].strip()
        end = line + head.count("\n")
        pieces.append(_Block(head, line, end, block.heading_trail))
        line = end
    if remaining:
        pieces.append(_Block(remaining, line, block.end_line, block.heading_trail))
    return pieces


def _with_overlap(
    chunks: list[Chunk],
    *,
    overlap_chars: int,
    source_id: str,
    doc_path: str,
    doc_title: str,
) -> list[Chunk]:
    """Prepend each chunk's predecessor's tail, so a split statement is findable from both sides.

    Line numbers are *not* widened to cover the borrowed text. The citation must point at
    where the chunk's own content lives; claiming the overlap's lines would make every
    citation drift a few lines earlier than the truth.
    """
    if overlap_chars <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        tail = previous.text[-overlap_chars:].lstrip()
        out.append(
            _make_chunk(
                f"{tail}\n\n{current.text}" if tail else current.text,
                source_id=source_id,
                doc_path=doc_path,
                doc_title=doc_title,
                ordinal=current.ordinal,
                start_line=current.start_line,
                end_line=current.end_line,
                heading_trail=current.heading_trail,
            )
        )
    return out


def _make_chunk(
    body: str,
    *,
    source_id: str,
    doc_path: str,
    doc_title: str,
    ordinal: int,
    start_line: int,
    end_line: int,
    heading_trail: tuple[str, ...],
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id(source_id, doc_path, ordinal, body),
        source_id=source_id,
        doc_path=doc_path,
        ordinal=ordinal,
        text=body.strip(),
        start_line=start_line,
        end_line=max(start_line, end_line),
        heading_trail=heading_trail,
        doc_title=doc_title,
    )


def chunk_id(source_id: str, doc_path: str, ordinal: int, text: str) -> str:
    """A stable id for one chunk.

    Content-addressed rather than sequential, so re-ingesting an unchanged document
    produces the same ids and an incremental update can tell what actually changed. The
    position is included because the same paragraph appearing twice in one document is two
    chunks, not one.
    """
    material = f"{source_id}\0{doc_path}\0{ordinal}\0{text}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:32]


def document_digest(text: str) -> str:
    """Content hash of a whole document, used to skip unchanged files on re-ingest."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
