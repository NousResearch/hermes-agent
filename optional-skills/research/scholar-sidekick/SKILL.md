---
name: scholar-sidekick
description: Resolve scholarly identifiers (DOI, PMID, PMCID, ISBN, arXiv, ISSN, ADS bibcode, WHO IRIS URL) into formatted citations (10,000+ CSL styles) and bibliography exports (BibTeX, RIS, EndNote, CSV…), and check retraction, open-access, and citation-fabrication status. Calls a documented REST API over plain HTTP — no install, no API key needed for the free tier.
version: 1.0.0
author: Scholar Sidekick
license: MIT
metadata:
  hermes:
    tags: [citations, bibliography, doi, pmid, arxiv, csl, bibtex, ris, retraction, open-access, citation-verification, research]
    related_skills: [arxiv, research-paper-writing]
---

# Scholar Sidekick — Citations, Retraction & Open-Access

Turn a scholarly identifier into a formatted citation, a bibliography file, or an
integrity check (retraction / open-access / fabrication), via a documented REST API.
**No API key and no install required** — plain HTTPS calls over `curl`. An optional
RapidAPI key only raises rate limits.

## When to Use
- The user has an identifier (DOI, PMID, PMCID, ISBN, arXiv, ISSN, ADS bibcode, WHO IRIS URL) and wants metadata, a formatted citation, or a bibliography file.
- "Cite this in APA/Vancouver/Chicago…", "give me a BibTeX/RIS file", "export these refs".
- "Has this been retracted?", "is this open access?", "is this citation real / did you make it up?"
- Do NOT use to *search* for papers by topic — that's discovery (see the `arxiv` skill). This assumes you already have an identifier.

## Surfaces — call the API, never scrape the UI
The site is built for agents. The contract lives at:
- https://scholar-sidekick.com/llms.txt (index of agent surfaces)
- https://scholar-sidekick.com/AGENTS.md (REST + MCP guide)
- https://scholar-sidekick.com/openapi/openapi.yml (OpenAPI 3.1)

Always call the JSON REST API below. Do not drive the website form.

## Authentication & limits
Calls to `scholar-sidekick.com/api/*` work **anonymously — there is no first-party API
key** — at a rate-limited free tier (~40 format / 10 export requests per window), which
is plenty for normal, human-driven agent use. For higher limits, Scholar Sidekick is
offered on RapidAPI: subscribe at
https://rapidapi.com/scholar-sidekick-scholar-sidekick-api/api/scholar-sidekick and call
it through the RapidAPI gateway with your `X-RapidAPI-Key`. Use the anonymous
`scholar-sidekick.com` endpoints by default; move to RapidAPI only for volume.

## Quick Reference
Base URL: `https://scholar-sidekick.com`

| Need | Endpoint | Body |
|------|----------|------|
| Format a citation | `POST /api/format` | `{text, style, output}` |
| Export a bibliography file | `POST /api/export` | `{text, format}` |
| Retraction / correction / EoC check | `POST /api/retraction-check` | `{id}` |
| Open-access status + best legal URL | `POST /api/oa-check` | `{id}` |
| Verify a claimed citation (fabrication) | `POST /api/verify` | `{claimed: {title, doi}}` |
| Service health | `GET /api/health` | — |

## Procedure

### Format a citation
```bash
curl -sS -X POST "https://scholar-sidekick.com/api/format" \
  -H "Content-Type: application/json" \
  -d '{"text": "10.1038/nphys1170", "style": "vancouver", "output": "text"}'
```
- `text`: one identifier, or several newline-separated for a batch. Pass verbatim — `PMID:`, `arXiv:`, ISBN hyphens, and `https://doi.org/…` are all tolerated.
- `style`: `vancouver` (default), `ama`, `apa`, `ieee`, `cse`, or any CSL style ID (`chicago-author-date`, `harvard-cite-them-right`, `modern-language-association`, `nature`, `bmj`, `the-lancet`, …).
- `output`: `text` or `json`.
Response: `{ "ok": true, "items": [{ "formatted": "…" }], "text": "…" }`.

### Export a bibliography file
```bash
curl -sS -X POST "https://scholar-sidekick.com/api/export" \
  -H "Content-Type: application/json" \
  -d '{"text": "10.1038/nphys1170\nPMID:30049270", "format": "bibtex"}' \
  -o refs.bib
```
- `format`: `bibtex`, `ris`, `csl-json`, `endnote-xml`, `refworks`, `nbib`, `rdf`, `csv`, `txt`.

### Check retraction
```bash
curl -sS -X POST "https://scholar-sidekick.com/api/retraction-check" \
  -H "Content-Type: application/json" \
  -d '{"id": "10.1016/S0140-6736(97)11096-0"}'
```
Returns `{ ok, doi, result: { isRetracted, hasCorrections, hasConcern, notices[], title } }`
(Crossref + Retraction Watch). One identifier per call — field is **`id`**.

### Check open access
```bash
curl -sS -X POST "https://scholar-sidekick.com/api/oa-check" \
  -H "Content-Type: application/json" \
  -d '{"id": "10.1371/journal.pone.0173664"}'
```
Returns `{ ok, doi, result: { isOa, oaStatus, bestLocation: {url, license, version}, locations[] } }`
(Unpaywall). One identifier per call — field is **`id`**.

### Verify a claimed citation (catch fabrication)
```bash
curl -sS -X POST "https://scholar-sidekick.com/api/verify" \
  -H "Content-Type: application/json" \
  -d '{"claimed": {"title": "The title exactly as cited", "doi": "10.xxxx/xxxxx"}}'
```
Citation fields go inside a **`claimed`** object: `title` (required) plus one identifier
(`doi`, `pmid`, …) and optional `authors` / `year` / `container`. Returns
`{ ok, verdict, confidence, matched }`, verdict ∈ `matched` / `mismatch` / `not_found` /
`parsing_error`. `mismatch` = the identifier resolves but the title doesn't — the dominant
AI-fabrication pattern (real DOI + invented title; Topaz et al., Lancet 2026). Use this for
"is this citation real?", not a plain format/resolve.

## Pitfalls
- Never scrape the web UI — the JSON API is faster and stable.
- Pass identifiers verbatim; don't strip prefixes.
- Body fields differ per endpoint: `format`/`export` use `text`; `retraction-check`/`oa-check` use `id` (one identifier per call); `verify` wraps fields in `claimed`. Don't mix them up.
- ISBNs have no DOI, so retraction/OA return a "no DOI" result for books.
- Don't fabricate a fallback: if a call fails or returns `ok:false`, report that — never invent a citation, retraction status, OA verdict, or a "matched" verdict.

## Verification
- `curl -sS https://scholar-sidekick.com/api/health` returns `{ "ok": true, … }`.
- A good `/api/format` response has `items[].formatted` non-empty.

## Optional: MCP server (power users)
Scholar Sidekick is also an MCP server (tools: `resolveIdentifier`, `formatCitation`,
`exportCitation`, `checkRetraction`, `checkOpenAccess`, `verifyCitation`). That path
requires installing the server and a RapidAPI key, so the REST calls above are the
zero-setup default:
```bash
npx -y scholar-sidekick-mcp@latest   # needs RAPIDAPI_KEY in env
```
