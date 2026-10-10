---
name: company-summary
description: >
  Answer "what does this company do?" and segment / geography questions using deepKPI:
  get_company_summary (narrative summary) and get_company_segments (segment & operating
  structure). For discovery — "which companies do X", "who operates in Y" — use
  company_summary_search across summaries, then pull full summaries for names
  that need detail. NOT for numeric KPI time series — use the kpi skill for that.
  Triggers: what does X do, describe the business, segments, geographies,
version: 1.0.0
author: Revelata
license: Apache-2.0
homepage: https://www.revelata.com
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Revelata, Finance, SEC, KPI, deepKPI]
    category: finance
    related_skills: [revelata, kpi, filing, implied-metric, seasonality, pressure-test, benchmark, ideas]
---

# company-summary

**Purpose:** Ground answers in deepKPI's **10-K-derived company summary** and **segment
breakdown** endpoints. Use **summary search** when the user is looking for **names**
that match a **theme** or **activity**.

For **historical metrics**, provenance tables, and Excel-style pulls, use the `kpi`
skill instead.

## Data source

See `_common/data-source-rules.md` for the runtime routing table and the canonical
inventory of deepKPI operations. This workflow uses `query_company_id`,
`get_company_summary`, `get_company_segments`, and `company_summary_search`.

`query_company_id` resolves **name / brand → `company_id`** (SEC CIK for US issuers).
Use it whenever the user names a company and you do not already have `company_id`.

## Hard stop if deepKPI access fails

See `_common/connection-failure.md`. STOP and ask before falling back.

## Credits (budget consciously)

| Call | Cost |
|------|------|
| **`get_company_summary`** | **3 credits** on success (one company, latest 10-K narrative). |
| **`get_company_segments`** | **3 credits** on success (one company, structured segments / geos from latest 10-K). |
| **`company_summary_search`** | **1 credit per company returned** (`top_k_companies`, max **15**). Empty results: **0**. |

## Workflow A — Named company (default)

Use this when the user is asking about **one or more specific issuers** by name or
ticker.

### What does the company do? (business, model, strategy)

1. **`query_company_id`** if you need **`company_id`**.
2. **`get_company_summary`** — return the **narrative** from the response; that is the
   canonical answer to "what do they do?", "describe the business", "how do they
   make money at a high level?", etc.

Do **not** substitute web summaries or training-data descriptions when this endpoint
is available.

### Segments, geographies, how they break out the business

1. **`query_company_id`** if needed.
2. **`get_company_segments`** — use for **segment structure**, **operating geographies**,
   **reportable segments**, and similar **organizational** questions (what lines /
   regions they disclose), as returned by the API.

For **segment revenue / margins over time**, follow the `kpi` skill (`list_kpis` /
`search_kpis`) after you have the segment names you care about.

### Both narrative and structure

Call **`get_company_summary`** and **`get_company_segments`** when the question needs
**prose plus** how they **slice** the business (same `company_id`).

## Workflow B — Thematic discovery (no single target company)

Use **`company_summary_search`** when the user wants a **set of companies** defined by
**what they do** or **where / how they operate**, e.g.:

- "Companies that **make** …", "**operate in** …", "**exposed to** …"
- "Who **reports** …", "**US-listed** firms in …" (as natural language — tune the `query`)

Steps:

1. **`company_summary_search`** with a clear natural-language **`query`** and
   **`top_k_companies`** (often **10**; max **15**). **1 credit per row returned.**
2. Present the ranked list (names / `company_id` / ticker as returned).
3. If the user (or the task) needs **full narrative** for one or more hits — not just
   the fact they matched — call **`get_company_summary`** on those `company_id`s
   (**3 credits** each). Use **`get_company_segments`** on selected names if they ask
   how those companies **break out** operations.

If results are weak, **rephrase `query`** before giving up.

## Out of scope — benchmarking vs this skill

**Not covered here:** "**Companies most similar to [Target]**", formal **peer sets**
for valuation comps, or **benchmark** workflows keyed off a **specific** company as
the anchor. Use the `benchmark` skill for that. This skill's
**`company_summary_search`** is for **theme / activity** discovery ("who does X?"),
not a drop-in replacement for target-based similarity benchmarking.

## Opening line

Pick one tone that matches the ask, e.g.:

- Named company: **"I'll pull their summary and segment breakdown if needed from deepKPI."**
- Thematic list: **"I'll search deepKPI's company summaries for matches, then pull full summaries where we need detail."**

## Common failure modes

- **Answering "what do they do?" from memory** instead of **`get_company_summary`**.
- **Guessing segment names** instead of **`get_company_segments`** (or KPI search) when
  the user asked how the business is **broken out**.
- **Using `company_summary_search`** when the user named **one company** and only need
  summary/segments — use **Workflow A**.
- **Expecting full segment financial history** from **`get_company_segments` alone** —
  use the `kpi` skill for time series.
