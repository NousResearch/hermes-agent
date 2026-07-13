---
name: scholar-sidekick
description: Cite, export, and verify papers from a DOI, PMID, or ISBN.
version: 1.0.0
author: Mark Lavercombe (mlava)
license: MIT
metadata:
  hermes:
    tags: [citations, bibliography, doi, pmid, arxiv, csl, bibtex, ris, retraction, open-access, citation-verification, research]
    related_skills: [arxiv, research-paper-writing]
---

# Scholar Sidekick Skill

Turns a scholarly identifier (DOI, PMID, PMCID, ISBN, arXiv, ISSN, ADS bibcode, WHO IRIS
URL) into a formatted citation, a bibliography file, or an integrity check — retraction,
open access, or fabrication. It does **not** search for papers by topic: it assumes you
already have an identifier. For discovery, use the `arxiv` skill instead.

## When to Use
- The user has an identifier and wants metadata, a formatted citation, or a bibliography file.
- "Cite this in APA/Vancouver/Chicago…", "give me a BibTeX/RIS file", "export these refs".
- "Has this been retracted?", "is this open access?", "is this citation real / did you make it up?"

## Prerequisites
None. The API is public and needs no install and no key — call it anonymously and it works.

Anonymous callers get a rate-limited free tier (~40 format / 10 export requests per
window), which is ample for human-driven agent use. Two optional upgrades, neither
required by this skill:

- **First-party key** — sign in at https://scholar-sidekick.com/account, issue an `ssk_`
  key, and send `Authorization: Bearer ssk_…` for a higher allowance.
- **RapidAPI gateway** — for volume, subscribe at
  https://rapidapi.com/scholar-sidekick-scholar-sidekick-api/api/scholar-sidekick and
  send `X-RapidAPI-Key` to the gateway host instead.

The same six operations are also exposed as a hosted MCP server at
`https://scholar-sidekick.com/api/mcp` (`resolveIdentifier`, `formatCitation`,
`exportCitation`, `checkRetraction`, `checkOpenAccess`, `verifyCitation`), which also
accepts anonymous calls. Use it only if you want tool-native access; the REST calls below
need no setup at all.

## How to Run
Invoke every operation as a JSON POST through the `terminal` tool. Base URL
`https://scholar-sidekick.com`; all endpoints take and return JSON:

```bash
curl -sS -X POST "https://scholar-sidekick.com/api/format" \
  -H "Content-Type: application/json" \
  -d '{"text": "10.1038/nphys1170", "style": "vancouver", "output": "text"}'
```

Call the JSON API, never the website form. The agent-facing contract is published at
https://scholar-sidekick.com/llms.txt (index of surfaces),
https://scholar-sidekick.com/AGENTS.md (REST + MCP guide), and
https://scholar-sidekick.com/openapi/openapi.yml (OpenAPI 3.1).

## Quick Reference

| Need | Endpoint | Body |
|------|----------|------|
| Format a citation | `POST /api/format` | `{text, style, output}` |
| Export a bibliography file | `POST /api/export` | `{text, format}` |
| Retraction / correction / EoC check | `POST /api/retraction-check` | `{id}` |
| Open-access status + best legal URL | `POST /api/oa-check` | `{id}` |
| Verify a claimed citation (fabrication) | `POST /api/verify` | `{claimed: {title, doi}}` |
| Service health | `GET /api/health` | — |

## Procedure

### 1. Format a citation
```bash
curl -sS -X POST "https://scholar-sidekick.com/api/format" \
  -H "Content-Type: application/json" \
  -d '{"text": "10.1038/nphys1170", "style": "vancouver", "output": "text"}'
```
- `text`: one identifier, or several newline-separated for a batch. Pass verbatim — `PMID:`, `arXiv:`, ISBN hyphens, and `https://doi.org/…` are all tolerated.
- `style`: `vancouver` (default), `ama`, `apa`, `ieee`, `cse`, or any CSL style ID (`chicago-author-date`, `harvard-cite-them-right`, `modern-language-association`, `nature`, `bmj`, `the-lancet`, …).
- `output`: `text` or `json`.

Response: `{ "ok": true, "items": [{ "formatted": "…" }], "text": "…" }`.

### 2. Export a bibliography file
```bash
curl -sS -X POST "https://scholar-sidekick.com/api/export" \
  -H "Content-Type: application/json" \
  -d '{"text": "10.1038/nphys1170\nPMID:30049270", "format": "bibtex"}' \
  -o refs.bib
```
- `format`: `bibtex`, `ris`, `csl-json`, `endnote-xml`, `refworks`, `nbib`, `rdf`, `csv`, `txt`.

### 3. Check retraction
```bash
curl -sS -X POST "https://scholar-sidekick.com/api/retraction-check" \
  -H "Content-Type: application/json" \
  -d '{"id": "10.1016/S0140-6736(97)11096-0"}'
```
Returns `{ ok, doi, result: { isRetracted, hasCorrections, hasConcern, notices[], title } }`
(Crossref + Retraction Watch). One identifier per call — the field is **`id`**.

### 4. Check open access
```bash
curl -sS -X POST "https://scholar-sidekick.com/api/oa-check" \
  -H "Content-Type: application/json" \
  -d '{"id": "10.1371/journal.pone.0173664"}'
```
Returns `{ ok, doi, result: { isOa, oaStatus, bestLocation: {url, license, version}, locations[] } }`
(Unpaywall). One identifier per call — the field is **`id`**.

### 5. Verify a claimed citation (catch fabrication)
```bash
curl -sS -X POST "https://scholar-sidekick.com/api/verify" \
  -H "Content-Type: application/json" \
  -d '{"claimed": {"title": "The title exactly as cited", "doi": "10.xxxx/xxxxx"}}'
```
Citation fields go inside a **`claimed`** object: `title` (required) plus an identifier
(`doi`, `pmid`, … — recommended) and optional `authors` / `year` / `container`. Returns
`{ ok, verdict, confidence, matched }`, verdict ∈ `matched` / `mismatch` / `ambiguous` /
`not_found`. `mismatch` = the identifier resolves but the title doesn't — the dominant
AI-fabrication pattern (real DOI + invented title; Topaz et al., Lancet 2026). `ambiguous`
= the identifier resolves to one paper but the claimed title matches a different real paper
(a wrong-identifier citation error). Use this for "is this citation real?", not a plain
format or resolve — those never catch a real DOI carrying an invented title.

## Pitfalls
- Never scrape the web UI — the JSON API is faster and stable.
- Pass identifiers verbatim; don't strip prefixes.
- Body fields differ per endpoint: `format`/`export` use `text`; `retraction-check`/`oa-check` use `id` (one identifier per call); `verify` wraps fields in `claimed`. Don't mix them up.
- ISBNs have no DOI, so retraction and open-access checks return a "no DOI" result for books.
- Don't fabricate a fallback: if a call fails or returns `ok:false`, report that — never invent a citation, retraction status, OA verdict, or a "matched" verdict.

## Verification
```bash
curl -sS https://scholar-sidekick.com/api/health
```
Returns `{ "ok": true, … }`. A healthy `/api/format` response has `items[].formatted` non-empty.
