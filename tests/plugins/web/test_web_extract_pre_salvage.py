"""web_extract must not silently drop <pre> code blocks lost by vendor markdown conversion.

Scraping vendors (Firecrawl cloud / Nous Tool Gateway) produce clean markdown that empties
<pre> content on WordPress copy-block plugins (Elementor rawHTML). The raw HTML still carries
the content — the Firecrawl provider now always fetches it alongside markdown and appends a
"Salvaged <pre> block(s)" section for anything the markdown lost.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from plugins.web.firecrawl import provider as firecrawl_provider

RUNBLOCK = (
    "vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 4 --gpu-memory-utilization 0.95 "
    "--enable-chunked-prefill --enable-prefix-caching"
)
DESIGN = "# DESIGN DOC\nVOID RAIDERS - Space Invaders style game spec, paddle and blaster."
HTML = (
    "<!DOCTYPE html><html><body>"
    f"<h1>Article</h1><p>prose</p><pre>{RUNBLOCK}</pre><pre>{RUNBLOCK}</pre>"
    f"<pre>{DESIGN}</pre>"
    "</body></html>"
)
EMPTY_FENCE_MD = "# Article\nSome prose.\n\n```none\n\n```\nMore prose."


def _scrape(formats, format_, payload):
    seen = {}

    def scrape(**kw):
        seen.update(kw)
        return payload

    client = SimpleNamespace(scrape=scrape)
    with patch.object(firecrawl_provider, "_get_firecrawl_client", return_value=client), \
            patch.object(firecrawl_provider, "check_website_access", return_value=None), \
            patch.object(firecrawl_provider, "is_safe_url", return_value=True):
        entry = asyncio.run(firecrawl_provider._scrape_one(
            "https://t.test/x", formats, format_))
    return entry, seen


def test_pre_block_texts_strips_tags_and_dedupsable():
    blocks = firecrawl_provider._pre_block_texts(HTML)
    assert len(blocks) == 3
    assert blocks[0] == RUNBLOCK
    assert blocks[0] == blocks[1]  # copy-buffer markup duplicates the visible block


def test_salvage_appends_dropped_blocks_without_dupes():
    out = firecrawl_provider._salvage_dropped_pre_blocks(HTML, EMPTY_FENCE_MD)
    assert "Salvaged <pre> block(s)" in out
    assert out.count(RUNBLOCK) == 1  # duplicated <pre> salvaged once
    assert "VOID RAIDERS" in out


def test_salvage_skips_blocks_present_in_markdown():
    md = EMPTY_FENCE_MD + "\n" + RUNBLOCK
    out = firecrawl_provider._salvage_dropped_pre_blocks(HTML, md)
    assert RUNBLOCK not in out
    assert "VOID RAIDERS" in out


def test_salvage_edge_cases_noop():
    assert firecrawl_provider._salvage_dropped_pre_blocks("", "md") == ""
    assert firecrawl_provider._salvage_dropped_pre_blocks(HTML, "") == ""
    assert firecrawl_provider._salvage_dropped_pre_blocks("<pre>unclosed", "x") == ""
    assert firecrawl_provider._salvage_dropped_pre_blocks("<p>no pre</p>", "plain md") == ""


def test_scrape_one_salvages_into_content_and_always_fetches_html():
    payload = {"markdown": EMPTY_FENCE_MD, "html": HTML, "metadata": {"status_code": 200}}
    entry, seen = _scrape(["markdown", "html"], "markdown", payload)
    assert seen["formats"] == ["markdown", "html"]
    assert "tensor-parallel-size 4" in entry["content"]
    assert "Salvaged <pre> block(s)" in entry["content"]
    assert entry["raw_content"] == entry["content"]


def test_scrape_one_markdown_intact_gets_no_salvage():
    payload = {"markdown": EMPTY_FENCE_MD + "\n" + RUNBLOCK, "html": HTML,
               "metadata": {"status_code": 200}}
    entry, _ = _scrape(["markdown", "html"], "markdown", payload)
    assert entry["content"].count(RUNBLOCK) == 1


def test_scrape_one_html_format_returns_html_untouched():
    payload = {"markdown": EMPTY_FENCE_MD, "html": HTML, "metadata": {"status_code": 200}}
    entry, _ = _scrape(["html"], "html", payload)
    assert entry["content"] == HTML
    assert "Salvaged" not in entry["content"]


def test_extract_requests_both_formats_for_markdown():
    """provider.extract() must request raw HTML whenever the caller did not ask for html explicitly."""
    provider = firecrawl_provider.FirecrawlWebSearchProvider()
    seen = []

    async def fake_scrape(url, formats, format_):
        seen.append(formats)
        return {"url": url, "title": "", "content": "", "error": None}

    with patch.object(firecrawl_provider, "_use_keyless_ring", return_value=False), \
            patch.object(firecrawl_provider, "_scrape_one", side_effect=fake_scrape):
        asyncio.run(provider.extract(["https://t.test/x"], format="markdown"))
        asyncio.run(provider.extract(["https://t.test/x"], format="html"))
    assert seen[0] == ["markdown", "html"]
    assert seen[1] == ["html"]


def test_web_extract_schema_exposes_format_param():
    from tools.web_tools import WEB_EXTRACT_SCHEMA

    fmt = WEB_EXTRACT_SCHEMA["parameters"]["properties"]["format"]
    assert fmt["enum"] == ["markdown", "html"]
