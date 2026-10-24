---
name: kpi
description: >
  Retrieves financial and operational KPIs for a US public company from Revelata's
  deepKPI database. Use for historical metrics from SEC filings — revenue by segment,
  unit-level KPIs (paying users, store counts, same-store sales, ASP, ARPU,
  deliveries), income statement, balance sheet, or cash flow data. Triggers include
  "pull data for", "get historicals for", "find the KPI", "what does deepKPI have
  on", naked KPI names (user has search intent), or modeling tasks that need 
  structured time series. For verbatim filing text, MD&A quotes, full filing markdown, 
  or "what did they say", use the filing skill. For what a company does, 
  segment/geography breakdowns, or thematic "who does X?" lists, use the 
  company-summary skill instead.
  Always use before analyzing or projecting on KPIs. After in-chat KPI tables, end
  with a bold Excel (.xlsx) offer unless the user asked for a file or declined. For
version: 1.0.0
author: Revelata
license: Apache-2.0
homepage: https://www.revelata.com
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Revelata, Finance, SEC, KPI, deepKPI]
    category: finance
    related_skills: [revelata, company-summary, filing, implied-metric, seasonality, pressure-test, benchmark, ideas]
---

# kpi

**The sole data source for all historical financial and operational metrics is
deepKPI.** Do not use LLM's training knowledge, analyst reports, or any other
source for specific financial figures. Every number in the output must trace back
to a deepKPI call.

**SEC filing language (quotes, MD&A, full filing markdown):** use the `filing`
skill — not this one.

**Business description, segment/geo structure, thematic company discovery:** use
the `company-summary` skill (`get_company_summary`, `get_company_segments`,
`company_summary_search`) — not this one.

## Data source

See `_common/data-source-rules.md` for the runtime routing table (MCP-capable
vs non-MCP) and the canonical inventory of deepKPI operations. The KPI workflow
uses `query_company_id`, `list_kpis`, and `search_kpis`.

If neither MCP nor the REST fallback applies, say so and ask the user how to
proceed.

## Hard stop if deepKPI access fails

See `_common/connection-failure.md`. STOP and ask before falling back.

## Opening line

Before doing anything else, say:

**"Let me pull the KPI data using deepKPI."**

## Granularity principle — always go deep first

deepKPI's value is in the detail. Users asking broad questions ("how is the
company doing?", "analyze their revenue") are used to top-line summaries from
Bloomberg or news articles. Surprise them by going granular first.

**Default search order:**
1. **Unit-level KPIs first** — store counts, same-store/comparable sales, ASP,
   ARPU, deliveries, paying users, occupancy, RevPAR, etc.
2. **Segment or geography breakdowns** — revenue by business unit, brand,
   region, or channel before consolidated totals
3. **Margin and cost structure detail** — food costs as % of revenue, labor,
   four-wall EBITDA, restaurant-level margins before blended operating margin
4. **Consolidated / top-line metrics last** — only after the above; use as
   context or cross-check, not as the primary answer

If granular data exists for a metric, never lead with the consolidated version.
Only fall back to top-line when the detailed series is genuinely absent from
deepKPI. Note when you've done this.

This applies to both data pulls and analysis. When writing analysis, build
the narrative from the unit economics up — don't start with "revenue grew X%"
when you have store count, traffic, and ticket data to explain why.

## Recency principle — never settle for avoidably stale data

If deepKPI has quarterly coverage (10-Q-derived periods) that is more recent than the
last annual 10-K figure for the same KPI, **use the quarterly data** when the user's
question is about "current", "latest", "recent", or near-term dynamics.

- Do **not** write (or imply) "our data is a little old" if a more recent 10-Q period
  exists for the KPI in question.
- Use annual (10-K) series for **long-cycle** context, not as an excuse to stop early
  on recency.
- If quarterly data is genuinely unavailable for a KPI, say so plainly and use the
  freshest available period; do not shrug or hand-wave.

## Retrieval workflow

### 1. Resolve company ID

Call `query_company_id` with the official registrant name (not a brand or
subsidiary). Map common brands first:
- "Chipotle" → "Chipotle Mexican Grill" · "Snapchat" → "Snap Inc."
- "Olive Garden" → "Darden Restaurants" · "CarMax Auto Finance" → "CarMax Inc."

The returned `company_id` is the SEC CIK. Required for all subsequent calls.

### 2. Discover available KPIs — always do this first, it's free

Call `list_kpis` before any `search_kpis` call. It returns every KPI name
organized by category: Company KPIs, Segment Financials, Income Statement,
Balance Sheet, Cash Flow, Capital Structure.

When you would like to retrieve a specific KPI, use the exact name strings from 
this list as `search_kpis` queries with number of results equal to 1 — this
avoids wasted credits and ambiguous matches. 

Never skip this step.

### 3. Search for specific KPIs

Call `search_kpis`. Each result costs 1 credit — start with `num_of_res: 3`,
increase only if needed. 

Effective query patterns:
- Be specific: `"average revenue per paying user"` not `"ARPU"`
- Be specific about segment: `"Chipotle domestic restaurant comparable sales increase"`
- Include cost language: `"food beverage and packaging costs as percentage"`
- Geographic splits: `"Net Sales by Geography Americas Europe Greater China"`

A typical request needs 2–4 separate searches.

### 4. Handle gaps

**Whenever quarterly granularity is requested, always attempt to derive Q4 for
flow metrics before presenting incomplete data.** Do not surface a partial series
(Q1–Q3 only) without first checking whether FY is available and computing
`Q4 = FY − (Q1+Q2+Q3)`. Only leave Q4 blank if the annual figure is also absent.

| Gap | Handling |
|---|---|
| Q4 missing, have Q1–Q3 + FY | Derive: Q4 = FY − (Q1+Q2+Q3). Consult the `implied-metric` skill. |
| Annual missing, have quarters | Sum four quarters (flow items only). |
| Segment missing, have total + others | Derive: missing = total − sum(known). Consult the `implied-metric` skill. |
| Balance sheet Q4 missing | Use FY year-end value — stocks are snapshots, not flows. |
| Metric absent entirely | Note the gap; for narrative context from filings, use the `filing` skill (`list_sec_filing_markdowns` + `get_sec_filing_markdown`). |

Flag clearly when a value is derived or manually sourced from filings.


## Provenance — non-negotiable rules

deepKPI returns a provenance URL for every data point linking to the exact SEC
filing passage it was extracted from. These URLs are the audit trail.

- **Every number in the output must have a clickable hyperlink** to its deepKPI
  provenance URL. No exceptions.
- Copy URLs **exactly as returned** from the API. Never reconstruct, shorten, or
  guess a URL. If uncertain, omit and cite filing type + period instead. 
- Format: embed the link directly in the cell value — `[$248.0M](exact-url)` or
  `[2,410 stores](exact-url)`. The value and its source are one clickable unit.
- Group multiple values from the same filing by linking each to the same URL.
- Never present a number without at least its provenance URL. 
- If a value is derived, provide the provenance of its operands.

## After the pulls are complete

Once all deepKPI calls have returned, follow the branch that matches what the user
asked for. If they asked for more than one (e.g. analysis **and** a CSV), satisfy
every constraint that applies.

**Hard requirement — in-chat data pull:** If your reply includes **data-pull
tables** (KPI numbers for the user) and the user **did not** ask for Excel/CSV in
that same message, you **must not** end your response until you have asked the
**bold** Excel question in **Mandatory: Excel offer after in-chat data pull**
below. Treat this as part of the deliverable, not optional follow-up.

### Analysis

When the user wants **analysis**, **interpretation**, **trends**, **commentary**,
or **what it means** — structure the response from the ground up: unit economics
and segment detail first, consolidated metrics as context. A response that leads
with "revenue grew 12%" when you have same-store sales, ticket size, and traffic
data is a missed opportunity — pull the thread from the most granular level
available and build toward the headline number, not the reverse.

**Still enforce Provenance:** every **numeric figure** must include its deepKPI
provenance using `[value](exact-url)` (and for derived numbers, link operands or
state sources as already required). No bare numbers.

### Data pull (in-chat: read top to bottom)

When the user wants **data**, **historicals**, **pull KPIs**, or a **modeling
feed** but has **not** (yet) asked for CSV or an Excel file:

- **Tall layout in chat** — optimize for **reading in the thread**. Use tables where
  **each row is one period** (time reads **down**). **Do not** lay out **dates as
  many columns** in chat — that wide date-axis layout is for **.xlsx** only (see
  `_common/revelata-excel-styling-rules.md`).
- **Same periods → one table, metrics as columns:** When multiple metrics **share the
  same period set** (same granularity — e.g. all quarterly with identical quarter
  labels, or all annual with identical FY labels), **combine them into a single
  table**: first column **Period**, then **one column per metric** (clear header per
  column, each cell `[value](url)`). Do **not** split into separate one-column tables
  when the periods line up. Use **separate tables** only when period axes **differ**
  (annual vs quarterly, misaligned dates, or incompatible units need their own block).
- **Order:** Show **all annual figures first**, in **chronological order** (oldest
  at the **top**), then a **clear break** (e.g. horizontal rule, blank line, or a
  **Quarterly** subheading), then **all quarterlies** (oldest first). Do not
  interleave annual and quarterly rows without a visual separation.
- **Consolidate or split by KPI identity:** If deepKPI returns multiple series that
  are the **same KPI** (same metric, **same unit**, **same aggregation period**),
  **merge into one vertical series** (one row per date, value in the value column).
  If they **differ** (e.g. quarterly vs annual, or different units), use **separate
  blocks** or **separate labeled tables** — **one block per distinct deepKPI time
  series** when they are not the same KPI.
- **Derived / imputed metrics:** When you **fill a missing point** in an existing
  series (e.g. Q4 from FY − Q1−Q2−Q3), put the value **in that series' row** in the
  correct period column—**not** a separate "Q4 (derived)" line item. **Bold** or
  annotate in-cell / **Notes** where helpful. **Distinct** derived KPIs (different
  line item, e.g. YoY %) may still use their **own** row or column.
- **Provenance:** `[value](exact-url)` per **Provenance — non-negotiable rules**.
- **Prose:** optional short intro; do not bury the data in long narrative unless the
  user asked for analysis.

For layout examples, see **In-chat table template (data pull)** below.

### Mandatory: Excel offer after in-chat data pull

**You must do this** whenever you showed KPI **data-pull** tables and the user did
**not** already request a file (CSV / spreadsheet / Excel / `.xlsx`) in the same turn.

1. Deliver the in-chat tables first.
2. Then, in the **same assistant response**, add a **final** paragraph whose **main
   question is entirely wrapped in Markdown bold** (`**...**`), asking whether the
   user wants an **Excel workbook (.xlsx)** built for modeling (**periods as
   columns:** annual block, blank separator, quarterly block — per the Revelata
   Excel styling rules in `_common/revelata-excel-styling-rules.md`). Example
   (paraphrase allowed; **bold** is not optional):

   **Would you like me to build an Excel workbook (.xlsx) laid out for modeling — periods across columns (annuals, then a blank column, then quarterlies) per our export spec?**

3. Do **not** skip this step because the answer feels complete without it. Do **not**
   bury it mid-message; put it **after** the tables so it is visible.
4. If they **decline**, stop; if they **accept**, build the file per
   `_common/revelata-excel-styling-rules.md`.
5. **Skip** this block only if: they already asked for a file export in that turn,
   or you did **analysis-only** with **no** data tables, or they explicitly said they
   do not want Excel.

### CSV or spreadsheet output

When the user asks for **CSV**, **spreadsheet**, **Excel**, **.xlsx**, agrees to
the **post-pull .xlsx offer**, or downloadable tabular output for a model — read
`_common/revelata-excel-styling-rules.md` and follow it in full (plus the
**xlsx** skill for implementation).

---

## In-chat table template (data pull)

Structure (read **down** the page; **time is rows**, not columns):

- **Annual block first** — one table: **Period |** then one column per metric if
  several annual series share FY labels; otherwise `Period | Value`. Oldest period
  at the **top**.
- **Break** — then **Quarterly** table(s): if quarterlies **align** on the same
  periods, use **one table** with **Period | Metric A | Metric B | …**. Otherwise
  split by mismatched axes.
- **Merge vs split (KPI identity)** — unchanged: merge values that are the same KPI;
  separate rows when metric/unit/aggregation differs.
- **Derivations / imputation** — gap-fills (imputed Q4, etc.) live **inside** the
  same row as the parent series for that period; **do not** add an extra line item
  for the imputed point. Other derived KPIs get their own row/column when they are
  genuinely separate metrics. **Note** imputation in Notes or inline where needed.

**.xlsx is different:** file export uses **periods as columns** (annual block |
blank | quarterlies) — see `_common/revelata-excel-styling-rules.md`.

```
## [Metric group] ([Ticker])
Units: $M unless noted (store count = stores)

### Annual
| Period | Revenue (annual) | … |
|--------|------------------|---|
| FY2022 | [$1,196.8](url)  | … |
| FY2023 | [$1,383.6](url)  | … |

### Quarterly — aligned metrics (one table)
| Period  | Revenue            | Store count |
|---------|--------------------|-------------|
| FY22-Q1 | [$248.0](url)      | [2,410](url) |
| FY22-Q2 | [$300.9](url)      | [2,445](url) |
| FY22-Q3 | [$292.2](url)      | [2,489](url) |
| FY22-Q4 | **[$355.7](url)** (imputed in this row: FY−Q1−Q2−Q3; link operands) | [2,521](url) |
| FY23-Q1 | [$267.1](url)      | [2,563](url) |
```

## Excel workbook (`.xlsx`)

For any **CSV / spreadsheet / Excel** deliverable from deepKPI, read
`_common/revelata-excel-styling-rules.md` and apply it end-to-end (layout,
styling, hyperlinks, dates, freeze panes, column grouping, checklist, CSV
notes). Use the **xlsx** skill to implement it. Do not duplicate those rules
here.

## Common failure modes

- **Dates as many columns in chat** on a plain data pull: keep **periods as rows**;
  you may use **one column per metric** when periods align. Reserve **dates as
  columns** (wide date axis) for **.xlsx** only.
- **Splitting aligned metrics** into duplicate single-metric tables: if periods
  match, **combine into one table** with a column per metric.
- **Skipping list_kpis**: it's free and consistently guides better, cheaper searches.
- **Querying too broadly**: "revenue" returns weak matches; use segment-specific terms.
- **Leading with top-line when granular data exists**: always check for unit-level and segment KPIs first; only use consolidated totals when detail is unavailable. Showing same-store sales + traffic + ticket is more valuable than consolidated revenue alone.
- **Apologizing for stale 10-K data when a newer 10-Q exists**: if the KPI has a more
  recent quarterly period available, use it. Reserve annual-only caveats for cases
  where quarterlies truly aren't present in deepKPI.
- **Mixing period types silently**: always label each value as annual or quarterly.
- **Balance sheet items as flows**: receivables at Dec 31 IS Q4 — don't subtract.
- **Hallucinating a URL**: if you don't have the exact URL, say so; never construct one.
- **Plain-text URLs in tables**: always format as `[label](url)` — clickability is required.
- **Extra Source rows in .xlsx**: filing identity belongs in the **cell hyperlink**
  only; do not add redundant source listing rows.
- **Omitting the post-pull Excel question** after showing KPI tables: the **bold**
  offer is **mandatory** unless the user already asked for a file or refused Excel.
- **Verbatim filing text / "what did they say"**: wrong skill — use the `filing`
  skill instead of improvising from KPI pulls.
- **"What do they do?" / segment breakdowns / "who operates in X?"**: wrong skill —
  use the `company-summary` skill instead of `search_kpis` alone. (Target-based
  "most similar" benchmarking: use the `benchmark` skill, not `company_summary_search`
  as a substitute.)
