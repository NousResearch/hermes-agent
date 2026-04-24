"""Unit tests for the ingestion pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion import (
    SCHEMA_VERSION,
    _looks_like_heading,
    _read_url_list,
    extract_references,
    ingest_crawl,
    ingest_directory,
    ingest_file,
    ingest_urls,
    parse_sections,
)
from web_fetch import FetchedPage


FIXTURES = Path(__file__).parent / "fixtures" / "tiny-corpus"


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_markdown_sections_detected():
    text = "# Title\n\nintro\n\n## Section A\n\nbody A\n\n## Section B\n\nbody B\n"
    sections = parse_sections(text, "md")
    headings = [s.heading for s in sections]
    assert headings == ["Title", "Section A", "Section B"]


def test_latex_sections_detected():
    text = "\\section{Intro}\nbody\n\\subsection{Healing Length}\nmore\n\\section{Results}\nend"
    sections = parse_sections(text, "tex")
    assert [s.heading for s in sections] == ["Intro", "Healing Length", "Results"]
    assert [s.level for s in sections] == [1, 2, 1]


def test_section_parser_falls_back_to_body_on_unstructured_text():
    sections = parse_sections("just a bunch of plain text with no headings.", "txt")
    assert len(sections) == 1
    assert sections[0].heading == "Body"


def test_pdf_heading_heuristic():
    assert _looks_like_heading("2.1 Methods")
    assert _looks_like_heading("Introduction")
    assert _looks_like_heading("RESULTS")
    assert not _looks_like_heading(
        "This is a long sentence of body text, not a heading at all."
    )
    assert not _looks_like_heading("short sentence.")


def test_extract_references_blank_line_separated():
    text = (
        "# doc\n## References\n\nA (2001).\n\nB (2002). Follow up.\n\nC (2003)."
    )
    sections = parse_sections(text, "md")
    refs = extract_references(sections, text)
    assert len(refs) == 3
    assert refs[0].raw.startswith("A (")


def test_extract_references_line_separated_fallback():
    text = (
        "# doc\n## References\nA (2001). Paper A.\nB (2002). Paper B.\nC (2003). Paper C.\n"
    )
    sections = parse_sections(text, "md")
    refs = extract_references(sections, text)
    # Blank-line split yields one blob, year-line fallback gives 3.
    assert len(refs) == 3


# ---------------------------------------------------------------------------
# End-to-end on fixtures
# ---------------------------------------------------------------------------


def test_ingest_single_markdown(tmp_path):
    out = tmp_path / "cache"
    result = ingest_file(FIXTURES / "alpha.md", out, FIXTURES)
    assert result["status"] == "ingested"
    doc = json.loads((out / "alpha.md.json").read_text())
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["metadata"]["title"].startswith("Alpha:")
    assert doc["metadata"]["year"] == 2023
    assert doc["stats"]["char_count"] > 0
    # References were parsed
    assert any("Volovik" in r["raw"] for r in doc["references"])


def test_ingest_directory_and_idempotency(tmp_path):
    out = tmp_path / "cache"
    first = ingest_directory(FIXTURES, out, workers=1)
    assert first["manifest"]["counts"]["errors"] == 0
    assert first["manifest"]["counts"]["ingested"] >= 3

    second = ingest_directory(FIXTURES, out, workers=1)
    assert second["manifest"]["counts"]["skipped"] == first["manifest"]["counts"]["ingested"]
    assert second["manifest"]["counts"]["ingested"] == 0


def test_malformed_pdf_does_not_abort_batch(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "good.md").write_text("# Good\n\nbody\n")
    (src / "broken.pdf").write_bytes(b"not a real pdf")
    out = tmp_path / "cache"
    r = ingest_directory(src, out, workers=1)
    assert r["manifest"]["counts"]["ingested"] == 1
    assert r["manifest"]["counts"]["errors"] == 1
    assert (out / "_ingest_errors.json").exists()


def test_dry_run_lists_without_writing(tmp_path):
    out = tmp_path / "cache"
    r = ingest_directory(FIXTURES, out, dry_run=True)
    assert r["dry_run"] is True
    assert r["count"] >= 3
    assert not any(out.glob("*.json")) if out.exists() else True


# ---------------------------------------------------------------------------
# URL / crawl ingestion (no network — injected stub fetcher)
# ---------------------------------------------------------------------------


class StubFetcher:
    name = "stub"

    def __init__(self, pages):
        self._pages = {p.url: p for p in pages}
        self._crawl_results = list(pages)

    def fetch_markdown(self, url, *, timeout=60):
        if url not in self._pages:
            raise RuntimeError(f"no stub page for {url}")
        return self._pages[url]

    def crawl(self, start_url, **kwargs):
        return list(self._crawl_results)


def test_ingest_urls_writes_cache_entries(tmp_path):
    cache = tmp_path / "cache"
    pages = [
        FetchedPage(
            url="https://example.com/a",
            markdown="# Alpha\n\n## Intro\n\nbody about Bogoliubov\n",
            title="Alpha",
        ),
        FetchedPage(
            url="https://example.com/b",
            markdown="# Beta\n\nsome text",
            title="Beta",
        ),
    ]
    fetcher = StubFetcher(pages)
    r = ingest_urls(
        urls=["https://example.com/a", "https://example.com/b"],
        cache_dir=cache,
        fetcher=fetcher,
    )
    assert r["manifest"]["counts"]["ingested"] == 2
    assert r["manifest"]["counts"]["errors"] == 0

    files = list(cache.glob("*.json"))
    assert len(files) >= 2

    for r in r["results"]:
        doc = json.loads(Path(r["cache_file"]).read_text())
        assert doc["schema_version"] == SCHEMA_VERSION
        assert doc["metadata"]["source_type"] == "url"
        assert doc["metadata"]["source_url"].startswith("https://example.com/")


def test_ingest_urls_idempotent(tmp_path):
    cache = tmp_path / "cache"
    pages = [FetchedPage(url="https://example.com/a", markdown="# A\n\nbody")]
    fetcher = StubFetcher(pages)

    first = ingest_urls(urls=["https://example.com/a"], cache_dir=cache, fetcher=fetcher)
    second = ingest_urls(urls=["https://example.com/a"], cache_dir=cache, fetcher=fetcher)
    assert first["manifest"]["counts"]["ingested"] == 1
    assert second["manifest"]["counts"]["skipped"] == 1
    assert second["manifest"]["counts"]["ingested"] == 0


def test_ingest_urls_records_fetch_errors(tmp_path):
    cache = tmp_path / "cache"
    fetcher = StubFetcher([FetchedPage(url="https://example.com/good", markdown="# ok")])
    r = ingest_urls(
        urls=["https://example.com/good", "https://example.com/missing"],
        cache_dir=cache,
        fetcher=fetcher,
    )
    counts = r["manifest"]["counts"]
    assert counts["ingested"] == 1
    assert counts["errors"] == 1
    assert (cache / "_ingest_errors.json").exists()


def test_ingest_crawl_processes_multi_page_result(tmp_path):
    cache = tmp_path / "cache"
    pages = [
        FetchedPage(url="https://site/p1", markdown="# P1\n\ntext", title="P1"),
        FetchedPage(url="https://site/p2", markdown="# P2\n\ntext", title="P2"),
        FetchedPage(url="https://site/p3", markdown="# P3\n\ntext", title="P3"),
    ]
    fetcher = StubFetcher(pages)
    r = ingest_crawl(
        start_url="https://site/",
        cache_dir=cache,
        fetcher=fetcher,
        max_depth=3,
        limit=10,
    )
    assert r["manifest"]["counts"]["ingested"] == 3


def test_read_url_list_ignores_blanks_and_comments(tmp_path):
    p = tmp_path / "urls.txt"
    p.write_text(
        "# top comment\n"
        "\n"
        "https://example.com/one\n"
        "   https://example.com/two   \n"
        "# another comment\n"
        "https://example.com/three\n"
    )
    urls = _read_url_list(p)
    assert urls == [
        "https://example.com/one",
        "https://example.com/two",
        "https://example.com/three",
    ]
