---
name: defuddle
description: "Use when you need to extract clean article content and metadata from a web page URL or raw HTML. Defuddle removes clutter (sidebars, nav, ads, footers) and returns clean Markdown or HTML with rich metadata (title, author, published date, schema.org data). Complements walled-web-research (access) and headless-web-research (data) for the content-cleanup step."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [defuddle, content-extraction, readability, markdown, web-scraping, html-cleanup, article-parser]
    related_skills: [walled-web-research, headless-web-research, web-content-extraction, obsidian]
---

# Defuddle — Web Content Extraction

## Overview

Defuddle is a CLI and library by Steph Ango (kepano) that extracts the main content from any web page and returns clean HTML or Markdown with rich metadata. It was created for the [Obsidian Web Clipper](https://github.com/obsidianmd/obsidian-clipper) and is designed as a modern replacement for Mozilla Readability with better metadata extraction, consistent footnote/math/code handling, and mobile-style-based clutter detection.

In the Hermes research skill stack, Defuddle fills the **content-cleanup layer** — the step between *getting the HTML* (via curl, r.jina.ai, Wayback, browser_snapshot) and *using the content* (summarization, note-taking, research reports). It replaces fragile manual CSS-selector approaches with a purpose-built, actively maintained tool.

| Property | Value |
|----------|-------|
| Package | `defuddle` on npm |
| Latest | 0.19.2 (July 2026, actively maintained) |
| License | MIT |
| Dependency | `commander` only (1 dep, 2.6 MB unpacked) |
| Runtime | Node.js (npx or global install) |
| Repo | https://github.com/kepano/defuddle |
| Homepage | https://defuddle.md |

## When to Use

- You have a URL and need the article content as clean Markdown for summarization, note-taking, or research.
- You have raw HTML (from curl, r.jina.ai, Wayback, or browser_snapshot) and need to strip clutter (sidebars, nav, ads, comments, footers).
- You need page metadata: title, author, description, published date, domain, favicon, main image, language, schema.org data.
- You want Obsidian-compatible output with YAML frontmatter for direct note creation.
- A page returns 403 to default User-Agent and you need content extraction with a custom UA string.
- You need to extract a single property (e.g. just the title or author) without downloading the full article body.

## Don't Use For

- Client-side rendered SPAs where the HTML shell has no content (use `browser_snapshot` or `r.jina.ai` first to get rendered HTML, then pipe to Defuddle).
- Sites behind Cloudflare challenges or login walls (use `walled-web-research` access ladder first).
- Structured data extraction from JSON APIs (use `headless-web-research` JSON-LD patterns).
- Bulk crawling or sitemap enumeration (Defuddle processes one page at a time).

## Skill Stack Position

```
┌─────────────────────────────────────────────────────┐
│              ACCESS LAYER (get the HTML)             │
│  curl · r.jina.ai · Wayback · browser_snapshot       │
│  walled-web-research · headless-web-research         │
└──────────────────────┬──────────────────────────────┘
                       │ raw HTML
                       ▼
┌─────────────────────────────────────────────────────┐
│         CLEANUP LAYER (this skill)                   │
│  Defuddle: strip clutter, extract content + meta     │
│  → clean Markdown / HTML / JSON / frontmatter        │
└──────────────────────┬──────────────────────────────┘
                       │ clean content
                       ▼
┌─────────────────────────────────────────────────────┐
│              USE LAYER (consume content)             │
│  Summarize · save to Obsidian · research report      │
│  obsidian · automated-briefings · arxiv              │
└─────────────────────────────────────────────────────┘
```

## Prerequisites

Node.js and npx must be available on the host:

```bash
node --version    # needs v18+
npx --version
```

No global install required — `npx -y defuddle@latest` downloads on first use and caches. For repeated use, install globally:

```bash
npm install -g defuddle
```

## Core Commands

### 1. URL directly to Markdown (simplest)

```bash
npx -y defuddle@latest parse <url> --markdown
```

### 2. Pipe HTML from curl (agent workflow — most flexible)

```bash
curl -sL <url> | npx -y defuddle@latest parse --markdown
```

This is the preferred pattern for Hermes agents because it separates fetching (where you control headers, proxies, retries) from parsing.

### 3. Pipe from access tools (walled sites)

```bash
# Wayback Machine → Defuddle cleanup
curl -sL "https://web.archive.org/web/2026/https://example.com/article" | npx -y defuddle@latest parse --markdown
```

Do NOT pipe r.jina.ai output into Defuddle — r.jina.ai returns Markdown, not HTML, and Defuddle expects HTML. When using r.jina.ai, you already have clean content.

### 4. JSON output with full metadata

```bash
npx -y defuddle@latest parse <url> --json
```

Returns: `content`, `title`, `description`, `author`, `domain`, `favicon`, `image`, `language`, `published`, `site`, `wordCount`, `metaTags`, `schemaOrgData`, `parseTime`.

### 5. Markdown with YAML frontmatter (Obsidian-compatible)

```bash
npx -y defuddle@latest parse <url> --markdown --frontmatter
```

Output:
```yaml
---
title: "Article Title"
author: "Author Name"
site: "Site Name"
published: 2025-10-20T00:00:00+00:00
source: "https://example.com/article"
domain: "example.com"
language: "en"
description: "Article description"
word_count: 143
---

Article content in Markdown...
```

This integrates directly with the `obsidian` skill for note creation.

### 6. Extract a single property

```bash
npx -y defuddle@latest parse <url> --property title
npx -y defuddle@latest parse <url> --property author
npx -y defuddle@latest parse <url> --property description
npx -y defuddle@latest parse <url> --property domain
```

### 7. Save output to file

```bash
npx -y defuddle@latest parse <url> --markdown --output /tmp/article.md
```

### 8. Custom User-Agent (403/FORBIDDEN fix)

```bash
npx -y defuddle@latest parse <url> --markdown \
  --user-agent "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
```

### 9. Language preference

```bash
npx -y defuddle@latest parse <url> --markdown --lang fr
```

### 10. Parse a local HTML file

```bash
npx -y defuddle@latest parse page.html --markdown
npx -y defuddle@latest parse page.html --json
```

## CLI Reference

```
Usage: defuddle parse [options] [source]

Parse HTML content from a file, URL, or stdin

Arguments:
  source                     HTML file path, URL, or "-" to read from stdin

Options:
  -o, --output <file>        Output file path (default: stdout)
  -m, --markdown             Convert content to markdown format
  --md                       Alias for --markdown
  -j, --json                 Output as JSON with metadata and content
  -f, --frontmatter          Prepend YAML frontmatter (title, author, source,
                             etc.) to the output
  -p, --property <name>      Extract a specific property (e.g., title,
                             description, domain)
  --debug                    Enable debug mode
  -l, --lang <code>          Preferred language (BCP 47, e.g. en, fr, ja)
  -u, --user-agent <string>  Custom User-Agent header for HTTP requests (helps
                             with 403/FORBIDDEN responses)
  -h, --help                 display help for command
```

## JSON Response Schema

| Property | Type | Description |
|----------|------|-------------|
| `author` | string | Author of the article |
| `content` | string | Cleaned up extracted content (HTML or Markdown) |
| `description` | string | Description or summary of the article |
| `domain` | string | Domain name of the website |
| `favicon` | string | URL of the website's favicon |
| `image` | string | URL of the article's main image |
| `language` | string | Language in BCP 47 format (e.g. `en`, `en-US`) |
| `metaTags` | object | Meta tags |
| `parseTime` | number | Time taken to parse in milliseconds |
| `published` | string | Publication date of the article |
| `site` | string | Name of the website |
| `schemaOrgData` | object | Raw schema.org data extracted from the page |
| `title` | string | Title of the article |
| `wordCount` | number | Total number of words in the extracted content |
| `debug` | object | Debug info (when `--debug` is used) |

## Integration Recipes

### Recipe A: URL → Obsidian note (one-liner)

```bash
npx -y defuddle@latest parse "https://example.com/article" --markdown --frontmatter > /tmp/note.md
```

Then use the `obsidian` skill to save it to the vault.

### Recipe B: Walled site → clean content

```bash
# Step 1: get HTML via Wayback Machine (from walled-web-research skill)
curl -sL "https://web.archive.org/web/2026/https://walled-site.com/article" > /tmp/raw.html

# Step 2: clean with Defuddle
npx -y defuddle@latest parse /tmp/raw.html --markdown
```

Note: do NOT pipe r.jina.ai output into Defuddle. r.jina.ai already returns clean Markdown, not HTML — Defuddle expects HTML input and will return empty on Markdown. Use Defuddle with raw HTML sources (curl, Wayback, browser snapshots). When using r.jina.ai, you already have clean content and don't need Defuddle.

### Recipe C: Research workflow — fetch + extract + summarize

```bash
# Fetch and clean in one pipe
curl -sL "https://example.com/article" | npx -y defuddle@latest parse --markdown --output /tmp/article.md

# Read the cleaned content for summarization
cat /tmp/article.md
```

### Recipe D: Bulk metadata extraction (single property)

When you only need the title or author from multiple pages, extract just that property:

```bash
npx -y defuddle@latest parse "https://example.com/page1" --property title
npx -y defuddle@latest parse "https://example.com/page2" --property title
```

### Recipe E: Browser snapshot → Defuddle (SPA content)

For client-side rendered sites where curl gets only a JS shell:

1. Navigate with `browser_navigate` to the target URL.
2. Extract the rendered HTML with `browser_console`:
   ```javascript
   document.documentElement.outerHTML
   ```
3. Save to a file and pipe to Defuddle:
   ```bash
   npx -y defuddle@latest parse /tmp/rendered.html --markdown
   ```

## What Defuddle Standardizes

### Headings
- First H1/H2 matching the title is removed (no duplicate).
- All H1s converted to H2s.
- Anchor links in headings removed (plain headings).

### Code blocks
- Line numbers and syntax highlighting stripped.
- Language preserved as `data-lang` attribute and class.

### Footnotes
- Inline references standardized to `<sup>` format.
- Footnote sections converted to ordered lists with backref links.
- In Markdown: `[^1]` syntax with footnote definitions.

### Math
- MathJax and KaTeX converted to standard MathML.
- Full bundle adds LaTeX conversion via `temml` and `mathml-to-latex`.

### Callouts
- GitHub markdown alerts, Obsidian callouts, Bootstrap alerts standardized.
- In Markdown: Obsidian-style `> [!info]` callout syntax.

## Common Pitfalls

1. **SPAs have no content for Defuddle to parse.** Defuddle works on static HTML. If the page is client-side rendered (React, Vue, Wix), `curl` will get a JS shell with no article text. Use `browser_snapshot` or `r.jina.ai` to get the rendered HTML first, then pipe to Defuddle. See Recipe E above.

2. **403/FORBIDDEN from sites blocking default User-Agent.** Use `--user-agent "Mozilla/5.0 ..."` with a real browser UA string. Sites like Wikipedia and many news outlets block bot-like User-Agents.

3. **Forgetting `-y` with npx.** Without `npx -y`, the CLI prompts for confirmation to install the package, which hangs in non-interactive terminal sessions. Always use `npx -y defuddle@latest`.

4. **Using `defuddle@latest` for reproducibility.** `@latest` always gets the newest version. For reproducible results in scripts, pin a version: `npx -y defuddle@0.19.2`. The package is actively developed with 44 versions in 5 months.

5. **stdin mode requires `-` or no argument.** When piping HTML via stdin, you can either pass `-` explicitly or omit the source argument entirely. Both work: `cat page.html | npx defuddle parse` and `cat page.html | npx defuddle parse -`.

6. **`--frontmatter` only works with `--markdown`.** Frontmatter is a YAML block prepended to Markdown output. It has no effect with `--json` or default HTML output.

7. **`--json` and `--markdown` are mutually exclusive in practice.** `--json` returns the content field as HTML. If you want JSON with Markdown content, use the library API with `separateMarkdown: true` option, not the CLI.

8. **Past security advisory (fixed).** GHSA-5mq8-78gm-pjmq was an XSS via unescaped string interpolation, fixed in v0.9.0. Current versions (0.19.x) are clean. Always use v0.9.0+.

9. **npm may trigger security scan warnings.** Hermes terminal security scanning flags `npx defuddle@latest` due to the historical OSV advisory. The advisory is fixed in current versions. The flag is informational, not a blocker.

10. **Global install vs npx caching.** `npx -y` downloads and caches the package on first run (~3s). Subsequent runs are instant. For high-frequency use, `npm install -g defuddle` avoids the npx overhead entirely.

## Verification Checklist

- [ ] `node --version` returns v18+ and `npx --version` works
- [ ] `npx -y defuddle@latest parse https://stephango.com/saw --markdown` returns clean article text
- [ ] `npx -y defuddle@latest parse https://stephango.com/saw --json` returns JSON with `title`, `content`, `author` fields
- [ ] `npx -y defuddle@latest parse https://stephango.com/saw --markdown --frontmatter` prepends YAML block
- [ ] `curl -sL https://stephango.com/saw | npx -y defuddle@latest parse --markdown` produces same output as direct URL mode
- [ ] `npx -y defuddle@latest parse https://stephango.com/saw --property title` returns only the title string

## Source

- Repository: https://github.com/kepano/defuddle
- npm: https://www.npmjs.com/package/defuddle
- Homepage: https://defuddle.md
- Author: Steph Ango (@kepano), CEO of Obsidian
- License: MIT