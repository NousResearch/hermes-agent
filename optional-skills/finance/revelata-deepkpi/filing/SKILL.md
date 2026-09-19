---
name: filing
description: >
  Retrieves SEC filing markdown from Revelata deepKPI for US public companies: full
  filing text in chat (10-K/10-Q/8-K as markdown) or verbatim excerpts — MD&A,
  risk factors, footnotes, management commentary, quotes. Use when the user wants
  the filing document to work with, "give me the 10-K", "what did they say", exact
  wording, or sections as markdown — NOT structured KPI time series (use the kpi
  skill for that). MUST call list_sec_filing_markdowns (free) then
  get_sec_filing_markdown (10 credits) before any web/SEC.gov search. For excerpt
  answers, lead with blockquoted contiguous text; do not paraphrase as quotes.
  Triggers: "full filing", "filing markdown", "quote the filing", "MD&A", "risk
version: 1.0.0
author: Revelata
license: Apache-2.0
homepage: https://www.revelata.com
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Revelata, Finance, SEC, KPI, deepKPI]
    category: finance
    related_skills: [revelata, kpi, company-summary, implied-metric, seasonality, pressure-test, benchmark, ideas]
---

# filing

**Purpose:** Pull **primary SEC filing text** as markdown so the user can **work with
the full document in chat** (search, copy, section pulls) or get **exact language**
via blockquoted excerpts. This is **not** the KPI / metrics skill — for revenue
series, segment tables, store counts, and other structured metrics, use the `kpi`
skill instead.

## Data source

See `_common/data-source-rules.md` for the runtime routing table and the canonical
inventory of deepKPI operations. This workflow uses `query_company_id`,
`list_sec_filing_markdowns`, and `get_sec_filing_markdown`.

If neither MCP nor the REST fallback applies, say so and ask the user how to proceed.

## Hard stop if deepKPI access fails

See `_common/connection-failure.md`. STOP and ask before falling back. Until
the user opts in, **do not** use the web as a workaround for text that
`get_sec_filing_markdown` would cover.

## SEC filing text — source hierarchy (non-negotiable)

When the user wants **US public-company SEC primary text** (10-K, 10-Q, 8-K,
exhibits, MD&A, Risk Factors, footnotes, "what they said", quotes, or "give me
the filing"), you MUST treat deepKPI's filing markdown as the **first and default**
source.

**Order of operations (do not skip or reorder):**

1. **`list_sec_filing_markdowns`** — **free** (0 credits). Use it to pick the right
   `acc_no` / `seq_no` / form when the user didn't specify them exactly.
2. **`get_sec_filing_markdown`** — **10 credits per call**. This is the intended way
   to get machine-readable filing text for quoting. **Do not avoid this tool to
   save credits** by browsing the web instead. If credits may be insufficient, say
   so and ask whether to proceed (or which filing to prioritize).
3. **Only if step 1–2 fail** after a genuine attempt, OR the user explicitly authorizes
   non-deepKPI sources: follow **Hard stop** above, then you may use web/SEC.gov —
   clearly labeled.

**Explicit prohibitions (while deepKPI is available and the user has not opted out):**

- Do **not** open SEC.gov, full-text search EDGAR, or use generic web search to
  substitute for `get_sec_filing_markdown`.
- Do **not** answer from memory, training data, or news articles for "what the
  filing says" — pull markdown and quote it.

**Credit transparency (say this when relevant):** Listing filings is free; retrieving
filing markdown costs **10 credits** per document pull. The correct behavior is to
**call the tool and disclose the cost**, not to route around it.

## Workflow

### 1. Resolve CIK

Call `query_company_id` with the **official registrant name** (not a brand). Same
brand→registrant hints as in the `kpi` skill (e.g. "Chipotle" → "Chipotle
Mexican Grill"). The returned `company_id` is the SEC CIK for
`list_sec_filing_markdowns` / `get_sec_filing_markdown`.

### 2. List filings (free)

Call `list_sec_filing_markdowns` with `cik`, optional `form_type`, optional date
range. Pick the `acc_no` and `seq_no` that match the user's request (e.g. latest
10-K, specific dated 8-K).

### 3. Fetch markdown (10 credits)

Call `get_sec_filing_markdown` with `cik`, `acc_no`, and `seq_no` (default `1`).

## Opening line

Before tool calls, say:

**"Let me pull the SEC filing markdown using deepKPI so you have the full filing text in context (and I can quote exact language where needed)."**

(If you are also pulling KPIs in the same turn, use the `kpi` skill's opening
line for the metrics portion.)

## When users ask "what did they say?" (verbatim quotes required)

When the user asks for **exact language** from filings ("what did management say",
"what comments were made", "how did they describe…", "quote the 10-K/10-Q"), you MUST:

- Use `list_sec_filing_markdowns` (if needed) to identify the right filing and `seq_no`.
- Call `get_sec_filing_markdown` to retrieve the markdown.
- Lead with **verbatim snippets** only: copy/paste **contiguous** text from the
  markdown into **Markdown blockquotes** (`>`). Minimum useful length: **at least
  one full sentence** per quote (prefer a short paragraph). Do not "tighten" wording.
- **Do not paraphrase** in the same breath as quoting. If you add interpretation,
  label it explicitly as **"Interpretation (not a quote):"** and only **after** the
  blockquoted passages.
- If you cannot find supporting language in the pulled markdown, say so and either
  pull a different `seq_no`/filing or ask a narrowing question — do not invent
  plausible filing language.

Avoid summarizing by default. Only summarize if the user explicitly asks for a
summary (and even then, prefer 1–2 blockquotes as anchors unless they asked for
summary-only).

## When users ask for "the filing" (return full markdown)

If the user asks for the **filing itself** (e.g. "give me the 10-K", "show the
filing", "send the filing", "put the markdown in chat"), you MUST:

- Use `list_sec_filing_markdowns` (if needed) for `acc_no` / `seq_no`.
- Call `get_sec_filing_markdown`.
- Return the **full markdown** (not a summary) so they can scroll, search, and ask
  follow-ups against the same text. Use excerpts only if they asked for
  excerpts/snippets or a specific section.

## KPIs + filing text in one request

If the user wants **both** structured metrics and **verbatim filing language** (or
full filing markdown):

1. Use the `kpi` skill for KPI discovery, search, provenance tables, and the
   mandatory Excel offer when you deliver data-pull tables.
2. Use **this skill** for `list_sec_filing_markdowns` / `get_sec_filing_markdown` and
   quote / full-markdown rules.

Do not satisfy a "what did they say?" request with KPI tables alone unless they
only asked for numbers.

## Common failure modes

- **Web first**: Browsing SEC.gov before calling `get_sec_filing_markdown`.
- **Paraphrase as quote**: Summarizing filing language without blockquoted source text.
- **Wrong `seq_no`**: Listing returns multiple documents per accession; confirm the
  intended exhibit/main document.
- **Skipping the free list**: Calling `get_sec_filing_markdown` without knowing
  `acc_no` when the user didn't provide it.
