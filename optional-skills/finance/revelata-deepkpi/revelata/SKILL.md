---
name: revelata
description: >
  Broad orchestrator for Revelata deepKPI research on US public companies — use for
  complex multi-step queries that cross multiple workflows (KPIs + filing text + segments,
  pressure-test plus follow-up data pulls, peer benchmarks plus seasonality, etc.).
  Routes to narrow skills (kpi, company-summary, filing, seasonality, implied-metric,
  pressure-test, benchmark, ideas) when a single workflow fits. Pulls SEC metrics
  (10-K/10-Q/8-K): segments, unit KPIs (stores, comps, ARPU, users), statements, cash
  flow; filing markdown and verbatim quotes; implied Q4/per-unit, seasonality, Excel.
  For "pull data", historicals, "find the KPI", seasonality, .xlsx, filing excerpts,
  pressure-test analyst PDFs, peer benchmarks, idea generation, or company summaries
  across multiple needs.
version: 1.0.0
author: Revelata
license: Apache-2.0
homepage: https://www.revelata.com
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Revelata, Finance, SEC, KPI, deepKPI]
    category: finance
    related_skills: [kpi, company-summary, filing, implied-metric, seasonality, pressure-test, benchmark, ideas]
  openclaw:
    emoji: "📊"
    requires:
      env:
        - DEEPKPI_API_KEY
      bins:
        - curl
    primaryEnv: DEEPKPI_API_KEY
---

# Revelata deepKPI orchestrator

Query structured KPIs, SEC filing text, and company summaries for US public
companies, powered by [Revelata](https://www.revelata.com).

This is the **broad orchestrator** for the Revelata bundle. Use it when a request
crosses multiple workflows. For a single, well-scoped task, route to the narrow
skill below.

## Data access

See `_common/data-source-rules.md` — single source of truth for how to call
deepKPI from MCP-capable runtimes (Claude, ChatGPT) vs non-MCP runtimes
(OpenClaw). Every skill below depends on this.

## Hard stop on connection failures

See `_common/connection-failure.md`. If deepKPI access fails, STOP and ask
the user before falling back to other sources. This rule applies to **every**
skill in the bundle.

## Sub-skill routing

**Read the relevant narrow skill before doing any work.** Multiple skills may
apply to a single request — chain through all that fit.

| User need | Skill |
|-----------|-------|
| No specific company or thesis — "what should I invest in?", "what stock should I buy?", **new ideas**, **interesting companies**, boredom / open exploration | `ideas` — interactive funnel + `company_summary_search` screen; then `kpi` for deep pulls |
| Pull historical KPIs / financials from deepKPI | `kpi` |
| **What a company does**, **segments / geographies**, or **thematic** "who does X?" lists | `company-summary` — named company: `get_company_summary` / `get_company_segments`; discovery: `company_summary_search` then hydrate as needed |
| Pull **full filing markdown** into chat, **verbatim SEC text**, quotes, MD&A / risk-factor language, or "what did they say" | `filing` — **before any web/SEC.gov**: `list_sec_filing_markdowns` (free) → `get_sec_filing_markdown` (10 credits). Return full markdown when they want the document; blockquoted excerpts for quote-style asks; no paraphrase as quotes |
| Derive missing Q4 numbers, a segment remainder, or per-unit economics (ASP, ARPU, AUV, take rate) | `implied-metric` |
| Split annual forecasts into quarterly estimates / seasonality patterns | `seasonality` |
| Produce an Excel workbook (.xlsx) from deepKPI data | Whichever data skill applies (`kpi`, `seasonality`, etc.); then follow `_common/revelata-excel-styling-rules.md` |
| Pressure-test a sell-side / analyst report against SEC filing data (HTML report, Chart.js, provenance links) | `pressure-test` |
| **Peers / comps**, **benchmark** a company, **most similar** public companies, or operational comp sets | `benchmark` — always read the `kpi` skill first for KPI mechanics |

**Default entry points by intent:**

- For **metrics and modeling feeds**, start with `kpi` (it chains to `implied-metric`,
  `seasonality`, and `_common/revelata-excel-styling-rules.md` as needed).
- For **business description, segment structure, or thematic company lists**, start
  with `company-summary`.
- For **filing markdown (full or excerpts) and verbatim quotes**, start with `filing`.

If a request mixes these, run **every** narrow skill that applies.

## Sub-skill summary

**`kpi`** — Primary workflow for **structured KPIs** (company ID, `list_kpis` /
`search_kpis`, gap handling including Q4 derivation), provenance rules, in-chat
tables, and the mandatory post-pull Excel offer.

**`company-summary`** — `get_company_summary` (what they do) and
`get_company_segments` (how they break out segments / geos);
`company_summary_search` for **thematic** "who does X?" discovery. Not for KPI
time series.

**`filing`** — **SEC filing markdown** in chat: full filing or excerpts,
verbatim quotes, MD&A / risk language. Tools: `list_sec_filing_markdowns` →
`get_sec_filing_markdown`, source hierarchy (tool-first, not web), blockquote
discipline for quote-style asks. Distinct purpose from KPI time series.

**`implied-metric`** — Compute metrics that deepKPI doesn't report directly from
data that IS reported: `Q4 = FY − (Q1+Q2+Q3)`, missing segment = total − known
segments, per-unit economics (revenue per store, ARPU, ASP), take rates,
penetration rates, geographic mix percentages. Standalone so custom user-built
skills can invoke it.

**`seasonality`** — Compute each quarter's typical share of the full fiscal year
from 2–3 years of actuals, then apply those ratios to split an annual projection
into quarterly estimates.

**`pressure-test`** — End-to-end workflow for uploaded analyst PDFs: extract
4–6 claims, pull granular deepKPI metrics per claim, build paired "supports" /
"complicates" evidence with mandatory provenance hyperlinks, and emit a
Revelata-branded interactive HTML report (follow that skill’s html-template and
chart-patterns references). After the draft HTML exists, run that skill’s
mandatory **draft HTML QA** rubric (double-check), then revise and re-check as
needed before presenting the file. **Always** read `kpi` for KPI mechanics; use
`filing` when the test needs verbatim filing passages; use `company-summary` for
business description / segment structure / thematic lists when useful; use
`implied-metric` when filling Q4 or segment gaps.

**`benchmark`** — Curated **operational** benchmark sets (not trading multiples):
`company_summary_search` for candidates, KPI fingerprint + alignment, diff-driven
segment sub-benchmarks, chat 1-pager, optional HTML (follow that skill’s
html-template reference), Excel via
`_common/revelata-excel-styling-rules.md`. **Always** read `kpi` first.

**`ideas`** — Onramp when the user has **no ticker and no thesis**: rapid
interview (multiple-choice by default), `company_summary_search` for a 10–15
name screen with thesis-matched KPIs, then deep pulls on 1–2 chosen names per
`kpi`. Use `list_kpis` / `search_kpis` for diagnostic KPI concepts. **No**
default Excel offer. Not for users who already named a company or a specific
metric — use `kpi` instead.
