"""Blog source grounding — extract, search, inject, and verify primary-source links.

Every AI/PM stream post should cite at least one primary source with a
verifiable link. This module:

  1. Extracts named papers, benchmarks, and datasets from the draft body.
  2. Searches the arXiv API for matching papers.
  3. Injects markdown links into the body for matched papers.
  4. Verifies all URLs in the body return HTTP 200 (HEAD request).
  5. Orchestrates the above via ground_post(draft).

Builder stream is exempt from the zero-link gate (see blog_gate.py).
"""
from __future__ import annotations

import re
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from typing import Optional

# arXiv API endpoint.
_ARXIV_API = "http://export.arxiv.org/api/query"

# Timeout for arXiv API and link verification (seconds).
_TIMEOUT = 15

# Patterns for named-paper extraction.
# Matches: "GPT-4", "BERT", "LLaMA 2", "Mixtral 8x7B", "Chinchilla", "Gemini Ultra"
# Also matches: "the Chinchilla paper", "the GPT-4 Technical Report"
_PAPER_PATTERNS = [
    # Named model + optional number: GPT-4, LLaMA 2, Mixtral 8x7B
    r"\b((?:GPT|BERT|LLaMA|Llama|Mixtral|Chinchilla|Gemma|Gemini|Claude|Mistral|Qwen|Falcon|T5|PaLM|Dolly|Vicuna|Alpaca)[\s-]?\d?\d?)\b",
    # "X paper" / "X et al." / "X Technical Report"
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:paper|et\s+al\.?|Technical\s+Report)\b",
    # Benchmark/dataset names: MMLU, GSM8K, HumanEval, BIG-Bench, GLUE, SWE-bench
    r"\b(MMLU|GSM8K|HumanEval|BIG-?Bench|GLUE|SWE-?[Bb]ench|MATH|ARC|HellaSwag|TruthfulQA|Winogrande|C-Eval)\b",
]

# Compiled patterns.
_COMPILED_PATTERNS = [re.compile(p) for p in _PAPER_PATTERNS]

# Markdown link regex: [text](url)
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)]+)\)")
# Bare URL regex.
_BARE_URL_RE = re.compile(r"(?<![\(\w])https?://[^\s)]+")


def _extract_paper_references(body_md: str) -> list[str]:
    """Extract named papers, benchmarks, and datasets from the body.

    Returns a deduplicated list of reference strings (paper/benchmark names).
    Does NOT extract things that are already linked — we only want to find
    names that could benefit from a primary-source link.
    """
    # Find positions of existing markdown links so we can skip text inside them.
    existing_link_spans = []
    for m in _MD_LINK_RE.finditer(body_md):
        existing_link_spans.append((m.start(), m.end()))

    def _in_existing_link(pos: int) -> bool:
        for start, end in existing_link_spans:
            if start <= pos < end:
                return True
        return False

    refs: list[str] = []
    seen: set[str] = set()

    for pattern in _COMPILED_PATTERNS:
        for m in pattern.finditer(body_md):
            if _in_existing_link(m.start()):
                continue
            ref = m.group(1).strip()
            # Normalise: collapse internal whitespace.
            ref = re.sub(r"\s+", " ", ref)
            if len(ref) < 2:
                continue
            key = ref.lower()
            if key not in seen:
                seen.add(key)
                refs.append(ref)

    return refs


def _search_arxiv(title: str, max_results: int = 3) -> Optional[str]:
    """Search the arXiv API for a paper matching the title.

    Returns the arXiv abstract URL (https://arxiv.org/abs/XXXX.XXXXX) of the
    top result, or None if no match found or the API is unreachable.
    """
    query = urllib.parse.quote(title)
    url = f"{_ARXIV_API}?search_query=ti:{query}&max_results={max_results}&sortBy=relevance"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SahilBlog/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            xml_data = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return None
    except Exception:
        return None

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return None

    # arXiv Atom feed: <entry><id>http://arxiv.org/abs/XXXX.XXXXX</id>
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall(".//atom:entry", ns):
        entry_id = entry.find("atom:id", ns)
        if entry_id is not None and entry_id.text:
            raw_id = entry_id.text.strip()
            # Normalize to https://arxiv.org/abs/XXXX.XXXXX
            if raw_id.startswith("http://arxiv.org/abs/"):
                return raw_id.replace("http://", "https://", 1)
            return raw_id
    return None


def _inject_links(body_md: str, links: dict[str, str]) -> str:
    """Inject markdown links for matched reference names.

    links: {reference_name: url}

    For each reference, finds the first unlinked occurrence in the body and
    wraps it in a markdown link: [[Name](url)]. Only the first occurrence is
    linked (avoid over-linking). Existing markdown links are never touched.

    Returns the modified body_md.
    """
    if not links:
        return body_md

    # Track existing link spans to avoid modifying text inside them.
    existing_link_spans = []
    for m in _MD_LINK_RE.finditer(body_md):
        existing_link_spans.append((m.start(), m.end()))

    def _in_existing_link(pos: int) -> bool:
        for start, end in existing_link_spans:
            if start <= pos < end:
                return True
        return False

    # Process links in reverse order of position to avoid offset issues.
    replacements: list[tuple[int, int, str]] = []

    for ref_name, url in links.items():
        # Escape for regex.
        escaped = re.escape(ref_name)
        # Find first occurrence not already inside a link.
        for m in re.finditer(escaped, body_md):
            if not _in_existing_link(m.start()):
                replacement = f"[{ref_name}]({url})"
                replacements.append((m.start(), m.end(), replacement))
                break  # Only link the first occurrence

    # Sort by position descending so offsets don't shift.
    replacements.sort(key=lambda r: r[0], reverse=True)

    result = body_md
    for start, end, replacement in replacements:
        result = result[:start] + replacement + result[end:]

    return result


def _verify_links(body_md: str, timeout: int = _TIMEOUT) -> list[str]:
    """Verify all URLs in the body return HTTP 200 (HEAD request).

    Returns a list of dead URLs (strings). Empty list = all links alive.
    Handles bare URLs and markdown link URLs.
    """
    urls: set[str] = set()

    # Markdown links.
    for m in _MD_LINK_RE.finditer(body_md):
        urls.add(m.group(2))

    # Bare URLs not already in markdown links.
    for m in _BARE_URL_RE.finditer(body_md):
        urls.add(m.group(0))

    dead: list[str] = []
    for url in urls:
        if not _check_url(url, timeout):
            dead.append(url)

    return dead


def _check_url(url: str, timeout: int = _TIMEOUT) -> bool:
    """Send a HEAD request and return True if HTTP 200-299."""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "SahilBlog/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, Exception):
        return False


def ground_post(draft: dict, stream: str = "ai") -> dict:
    """Orchestrate source grounding for a single blog post.

    Pipeline:
      1. Extract paper/benchmark references from the body.
      2. Search arXiv for each reference.
      3. Inject markdown links for matched papers.
      4. Verify all URLs in the body (including pre-existing links).
      5. Return a result dict with:
         - "body_md": updated body with injected links
         - "grounds": count of primary sources linked
         - "dead_links": list of URLs returning non-200
         - "references_found": list of reference names extracted

    For builder stream, skip arXiv search (no grounding needed) but still
    verify any existing links.

    Returns the result dict. Never raises — degrades gracefully.
    """
    body = draft.get("body_md", "") or ""

    refs = _extract_paper_references(body)

    links: dict[str, str] = {}
    if stream != "builder":
        for ref in refs[:5]:  # Cap at 5 to avoid API rate limits.
            url = _search_arxiv(ref)
            if url:
                links[ref] = url

    if links:
        body = _inject_links(body, links)

    # Verify all links (both existing and newly injected).
    dead = _verify_links(body)

    return {
        "body_md": body,
        "grounds": len(links),
        "dead_links": dead,
        "references_found": refs,
    }