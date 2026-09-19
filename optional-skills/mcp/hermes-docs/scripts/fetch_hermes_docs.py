"""Fetch Hermes Agent documentation pages and build DocPage objects.

This script scrapes the Hermes documentation from the official website
(hermes-agent.nousresearch.com/docs) or from a local checkout of the repo.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx

# Import the DocPage from the server module — works when run via `python -m scripts.fetch_hermes_docs`
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from server.hermes_docs_server import DocPage
except ImportError:
    # Define it inline so the script works standalone
    from dataclasses import dataclass

    @dataclass
    class DocPage:
        slug: str
        title: str
        section: str
        content: str
        word_count: int = 0


DOCS_BASE_URL = "https://hermes-agent.nousresearch.com/docs"
SITEMAP_URL = f"{DOCS_BASE_URL}/sitemap.xml"

# Known doc categories (manually curated from the repo structure)
KNOWN_SECTIONS = {
    "getting-started": "Getting Started",
    "user-guide": "User Guide",
    "features": "Features",
    "guides": "Guides",
    "reference": "Reference",
    "developer-guide": "Developer Guide",
    "integrations": "Integrations",
}


def slug_to_section(slug: str) -> str:
    """Infer the doc section from a URL slug."""
    parts = slug.split("/")
    if parts and parts[0] in KNOWN_SECTIONS:
        return KNOWN_SECTIONS[parts[0]]
    if parts:
        return parts[0].replace("-", " ").title()
    return "General"


def slug_to_title(slug: str) -> str:
    """Derive a human-readable title from a slug path."""
    name = slug.split("/")[-1]
    name = name.replace(".md", "").replace(".mdx", "")
    name = name.replace("-", " ").replace("_", " ")
    # Title case with exceptions
    exceptions = {"a", "an", "the", "and", "or", "for", "of", "in", "to", "with"}
    words = name.split()
    titled = []
    for i, w in enumerate(words):
        if i == 0 or w.lower() not in exceptions:
            titled.append(w.capitalize() if w.islower() else w)
        else:
            titled.append(w.lower())
    return " ".join(titled)


def extract_page_from_html(html: str, slug: str) -> str | None:
    """Extract the main content from a Hermes docs HTML page."""
    # Try to find the article/main content area
    # Docusaurus uses <article> or <main> or div.markdown
    patterns = [
        r'<article[^>]*>([\s\S]*?)</article>',
        r'<main[^>]*class="[^"]*doc[^"]*"[^>]*>([\s\S]*?)</main>',
        r'<div[^>]*class="[^"]*markdown[^"]*"[^>]*>([\s\S]*?)</div>\s*</',
        r'<div[^>]*class="[^"]*theme-doc-markdown[^"]*"[^>]*>([\s\S]*?)</div>',
    ]
    for pattern in patterns:
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            return clean_html_to_text(m.group(1))

    # Fallback: just strip tags and return body text
    body = re.search(r'<body[^>]*>([\s\S]*?)</body>', html, re.IGNORECASE)
    if body:
        return clean_html_to_text(body.group(1))
    return None


def clean_html_to_text(html: str) -> str:
    """Convert HTML to clean plain text."""
    # Remove script and style tags
    html = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', html)
    html = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', html)
    # Remove HTML tags, preserve structure with newlines
    html = re.sub(r'<br\s*/?>', '\n', html)
    html = re.sub(r'</(p|div|h[1-6]|li|blockquote|pre|tr|th|td)>', '\n', html)
    html = re.sub(r'<[^>]+>', '', html)
    # Decode common HTML entities
    html = html.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    html = html.replace('&quot;', '"').replace('&#39;', "'").replace('&#x27;', "'")
    # Decode numeric entities
    html = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), html)
    # Collapse whitespace
    html = re.sub(r'\n\s*\n', '\n\n', html)
    html = re.sub(r'[ \t]+', ' ', html)
    return html.strip()


def discover_urls_from_sitemap(client: httpx.Client) -> list[str]:
    """Discover documentation URLs from the sitemap."""
    try:
        r = client.get(SITEMAP_URL, timeout=10)
        r.raise_for_status()
        # Parse sitemap XML
        urls = re.findall(r'<loc>([^<]+)</loc>', r.text)
        # Filter to docs pages only
        doc_urls = [u for u in urls if "/docs/" in u and not u.endswith("/tags/")]
        return doc_urls
    except Exception:
        print("Sitemap fetch failed, falling back to known pages")
        return []


def discover_urls_from_sidebar(client: httpx.Client) -> list[str]:
    """Scrape the docs sidebar for all page links."""
    try:
        r = client.get(DOCS_BASE_URL, timeout=10)
        r.raise_for_status()
        # Find links in the sidebar navigation
        links = re.findall(r'href="(/docs/[^"]+)"', r.text)
        seen: set[str] = set()
        urls: list[str] = []
        for link in links:
            full = f"https://hermes-agent.nousresearch.com{link}"
            if full not in seen:
                seen.add(full)
                urls.append(full)
        return urls
    except Exception:
        return []


def fetch_all_docs() -> list[DocPage]:
    """Fetch all Hermes documentation pages."""
    pages: list[DocPage] = []

    with httpx.Client(follow_redirects=True, timeout=15) as client:
        # Discover URLs
        urls = discover_urls_from_sitemap(client)
        if not urls:
            print("Sitemap empty, using sidebar discovery")
            urls = discover_urls_from_sidebar(client)
        if not urls:
            print("Sidebar discovery also failed — using known slugs")
            # Fallback to major known doc pages
            urls = [
                f"{DOCS_BASE_URL}/{slug}"
                for slug in [
                    "getting-started/installation",
                    "getting-started/quickstart",
                    "getting-started/configuration",
                    "user-guide/cli",
                    "user-guide/configuration",
                    "user-guide/desktop",
                    "user-guide/docker",
                    "user-guide/tui",
                    "user-guide/which-file-does-what",
                    "user-guide/bot-mode",
                    "user-guide/configuring-models",
                    "user-guide/git-worktrees",
                    "user-guide/checkpoints-and-rollback",
                    "user-guide/windows-native",
                    "user-guide/windows-wsl-quickstart",
                    "user-guide/import-from-other-agents",
                    "user-guide/features/mcp",
                    "user-guide/features/skills",
                    "user-guide/features/profiles",
                    "user-guide/features/sessions",
                    "user-guide/features/tools",
                    "user-guide/features/memory",
                    "user-guide/features/egress",
                    "user-guide/features/cron",
                    "user-guide/features/telegram",
                    "guides/use-mcp-with-hermes",
                    "reference/optional-skills-catalog",
                    "reference/faq",
                ]
            ]

        print(f"Discovered {len(urls)} potential doc pages")

        for url in urls:
            try:
                # Extract slug from URL
                slug = url.replace(DOCS_BASE_URL, "").strip("/")
                if not slug or slug.endswith(".xml"):
                    continue

                r = client.get(url, timeout=10)
                if r.status_code != 200:
                    continue

                content = extract_page_from_html(r.text, slug)
                if not content or len(content) < 50:
                    continue

                title = slug_to_title(slug)
                section = slug_to_section(slug)

                pages.append(DocPage(slug=slug, title=title, section=section, content=content))
                print(f"  ✓ {slug} ({len(content)} chars)")

            except Exception as e:
                print(f"  ✗ {url}: {e}")

    print(f"\nIndexed {len(pages)} documentation pages")
    return pages


def fetch_from_repo(repo_path: str | Path) -> list[DocPage]:
    """Build index from a local Hermes repo checkout instead of scraping."""
    repo = Path(repo_path)
    docs_dir = repo / "website" / "docs"

    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    pages: list[DocPage] = []

    for md_file in sorted(docs_dir.rglob("*.md*")):
        # Skip _category_.json and other non-doc files
        if md_file.name.startswith("_") or md_file.name == "index.mdx":
            continue

        rel_path = md_file.relative_to(docs_dir)
        slug = str(rel_path.with_suffix("")).replace("\\", "/")

        # Determine section from first directory component
        section_parts = rel_path.parts
        section = slug_to_section(section_parts[0]) if section_parts else "General"

        content = md_file.read_text(encoding="utf-8")
        # Strip frontmatter if present
        content = re.sub(r'^---\n[\s\S]*?\n---\n', '', content)
        # Remove imports and JSX
        content = re.sub(r'import\s+.*?from\s+[\'"].*?[\'"]\n', '', content)
        content = re.sub(r'<\w+[^>]*/>', '', content)

        title = slug_to_title(slug)

        pages.append(
            DocPage(slug=slug, title=title, section=section, content=content.strip())
        )
        print(f"  ✓ {slug} ({len(content)} chars)")

    print(f"\nLoaded {len(pages)} pages from local repo")
    return pages


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--repo":
        path = sys.argv[2] if len(sys.argv) > 2 else "."
        pages = fetch_from_repo(path)
    else:
        pages = fetch_all_docs()

    # Print summary
    sections: dict[str, int] = {}
    for p in pages:
        sections[p.section] = sections.get(p.section, 0) + 1

    print("\nSection summary:")
    for section, count in sorted(sections.items()):
        print(f"  {section}: {count} pages")
    print(f"\nTotal: {len(pages)} pages")
