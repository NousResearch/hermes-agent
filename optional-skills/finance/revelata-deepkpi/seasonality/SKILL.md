---
name: seasonality
description: >
  Computes seasonal quarterly ratios from historical actuals and applies them to
  split annual projections into quarterly estimates. Trigger for: quarterly
  breakdown of annual forecasts, in-progress fiscal year projections, any request
  involving quarterly splits or seasonal patterns. Always retrieve underlying data
  via the kpi skill first. For `.xlsx`/CSV workbook exports, follow
version: 1.0.0
author: Revelata
license: Apache-2.0
homepage: https://www.revelata.com
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Revelata, Finance, SEC, KPI, deepKPI]
    category: finance
    related_skills: [revelata, kpi, company-summary, filing, implied-metric, pressure-test, benchmark, ideas]
---

# seasonality

Computes each quarter's typical share of the full fiscal year, then applies those
ratios to project quarterly values from annual forecasts.

**All historical inputs must come from deepKPI** (via the `kpi` skill). Call
`list_kpis` first to discover exact KPI names for both the quarterly and annual
series before searching. Never use hardcoded or assumed figures.

**Do not** apply to balance sheet items (cash, receivables, inventory) — these
are point-in-time stocks; use the fiscal year-end value as Q4 directly.

## Data source

See `_common/data-source-rules.md`. Deep KPI data flows through the standard
`query_company_id` / `list_kpis` / `search_kpis` workflow used by the `kpi` skill.

## Hard stop if deepKPI access fails

See `_common/connection-failure.md`. STOP and ask before falling back.

## Core formula

```
Q_ratio(year, q)  = Q_value(year, q) / FY_value(year)
seasonal_ratio(q) = simple average of Q_ratio across most recent 3 years
projected_Q_value = projected_FY × seasonal_ratio(q)
```

Four ratios must sum to ~100%. If materially off, normalize before applying.

## Steps

1. **Retrieve** annual + quarterly actuals from deepKPI (minimum 2 years; 3 is
   standard). Capture the provenance URL for every value returned.
2. **Always derive Q4** — companies don't file a Q4 10-Q so Q4 will never be
   directly in deepKPI. Always compute `Q4 = FY − (Q1+Q2+Q3)` for every year
   in the series. Use the `implied-metric` skill. Put Q4 **in the quarterly
   revenue (metric) row**, Q4 column—**not** a separate "Q4 (derived)" row. Flag in
   **Notes**.
3. **Compute per-year ratios** — divide each quarter by its fiscal year total.
4. **Average** across the 3 most recent complete years.
5. **Flag anomalies** — if one year is distorted (product launch, acquisition,
   COVID), note it in the Notes column and consider excluding from the average.
6. **Apply** ratios to each projected annual figure.

## In-progress fiscal year

When Q1–Q2 are actuals and you need to project Q3–Q4 only:

```
remainder     = projected_FY − Q1_actual − Q2_actual
Q3_reweighted = Q3_ratio / (Q3_ratio + Q4_ratio)
Q4_reweighted = Q4_ratio / (Q3_ratio + Q4_ratio)
projected_Q3  = remainder × Q3_reweighted
projected_Q4  = remainder × Q4_reweighted
```

The same re-weighting logic applies for any subset of remaining quarters.

## Output format (in-chat tables)

Follow the `kpi` and `implied-metric` skills for how tables read in chat.

**From the `kpi` skill (data pull):**

- **Provenance:** Every numeric figure uses a clickable deepKPI link —
  `[value](exact-url)` — per the `kpi` skill's **Provenance — non-negotiable rules**
  (copy URLs exactly as returned; never reconstruct or guess). For **derived**
  values, link the **operands** or state sources as that skill requires.
- **KPI identity:** Merge into one series when deepKPI returns the **same KPI**
  (same metric, unit, aggregation); use **separate** rows or tables when they
  differ.
- **Order:** **Annual figures first**, chronological (**oldest at top**), then a
  **clear break** (rule, blank line, or subheading), then **quarterlies** (oldest
  first). Do not interleave annual and quarterly without separation.
- **Tall vs wide:** For ordinary historical feeds, prefer **tall** tables — **each
  row is one period**, first column **Period**, then **one column per metric** when
  periods align — per the `kpi` skill's **In-chat table template (data pull)**. **Do
  not** use a wide date axis "many period columns" layout for `.xlsx`-style modeling
  in chat; that layout is for files (`_common/revelata-excel-styling-rules.md`).

**Actuals block (aligned quarters + Q4 across years)** — use the `implied-metric`
skill's **Multi-period table** pattern (wide grid is allowed **here** for this
derivation layout):

- **Time flows left to right** — column headers are period labels (`FY22-Q1` …
  `FY22-Q4`), earliest at left.
- **One row per time series** — each operand row maps 1:1 to one deepKPI series; do
  not fragment into separate rows per quarter (e.g. not "Revenue - Q1" / "Revenue -
  Q2").
- **Imputed Q4 in-stream** — the quarterly **metric row** carries Q1–Q3 from
  deepKPI and **Q4 as imputed** (formula or linked value) in the Q4 columns—**no**
  extra line item for "Q4 (derived)". **Annual** row stays separate when it is a
  different series. **Bold** imputed cells only if it helps; **Notes** documents
  imputation.
- **Links embed in cell values** — `[$248.0M](exact-deepkpi-url)`; no separate Source
  column.
- **Notes** is the **bottom row** of the block; populate only in columns where
  something non-standard occurred (derivation, anomaly, re-weighting, etc.).

Show **all work**: operands (FY, Q1–Q3) visible with links; Q4 appears **in the
quarterly row**. Ratios block still shows per-year math. Per the `implied-metric` skill.

**Ratios** and **Projections** blocks can use compact tables; still hyperlink any
cell that is a direct deepKPI-reported value.

If the reply includes these **data** tables and the user did **not** ask for CSV /
Excel / `.xlsx` in the same turn, end with the **bold** workbook offer per the
`kpi` skill (**Mandatory: Excel offer after in-chat data pull**).

Three blocks in sequence: **Actuals → Ratios → Projections**.

```
## Seasonality — [Metric] ([Ticker])
Units: $M unless noted

ACTUALS
                   | FY22-Q1       | FY22-Q2       | FY22-Q3       | FY22-Q4    | FY23-Q1       | FY23-Q2       | FY23-Q3       | FY23-Q4    | FY24-Q1       | FY24-Q2       | FY24-Q3       | FY24-Q4    |
-------------------|---------------|---------------|---------------|------------|---------------|---------------|---------------|------------|---------------|---------------|---------------|------------|
Revenue (quarterly)| [$248](url)   | [$301](url)   | [$292](url)   | **$355** (FY−Q1−Q2−Q3) | [$267](url)   | [$325](url)   | [$297](url)   | **$495** (same) | [$248](url)   | [$301](url)   | [$292](url)   | **$715** (same) |
Revenue (annual)   |               |               |               |[$1,196](url)|              |               |               |[$1,384](url)|              |               |               |[$1,556](url)|
Notes              |               |               |               | imputed Q4 |               |               |               | imputed Q4 |               |               |               | imputed Q4 |

SEASONAL RATIOS
     | FY2022 | FY2023 | FY2024 | 3-yr avg |
-----|--------|--------|--------|----------|
Q1   | 20.7%  | 19.3%  | 15.9%  | 18.6%    |
Q2   | 25.1%  | 23.5%  | 19.3%  | 22.6%    |
Q3   | 24.4%  | 21.5%  | 18.8%  | 21.6%    |
Q4   | 29.7%  | 35.8%  | 45.9%  | 37.1%    |
Sum  | 100%   | 100%   | 100%   | 100%     |
Notes| Q4 from imputed quarter / FY | (same) | (same) | |

PROJECTIONS (FY20XXE = $X)
                  | FY2XE-Q1 | FY2XE-Q2 | FY2XE-Q3 | FY2XE-Q4 |
------------------|----------|----------|----------|----------|
Projected ($M)    |          |          |          |          |
Notes             |          |          |          |          |
```

The Notes row must be populated for: derived Q4 values, excluded anomaly years,
re-weighted in-progress quarter projections, stock items used as Q4 directly, or
any year where the ratio was estimated rather than computed from deepKPI data.

## Exporting to Excel or CSV

Use `_common/revelata-excel-styling-rules.md` for all spreadsheet layout,
styling, numeric date headers, freeze panes, column grouping, hyperlinks on
**value cells** (green, no underline — **no** separate Sources sheet/rows for
filings), number formats, and the pre-delivery checklist. Implement with the
**xlsx** skill.

**Seasonality model addenda (on top of that spec):**

- **Three ALL CAPS section blocks** in document order: **ACTUALS**, **SEASONAL
  RATIOS**, **PROJECTIONS**. Row semantics match the markdown template earlier in
  this skill; **period columns** follow `_common/revelata-excel-styling-rules.md`
  (**annual block**, blank separator, **quarterly block** — not quarterlies-left as
  in legacy examples).
- **Every ratio and every projected quarter** is a live `=` formula. Do not
  hardcode seasonal ratios or projection numbers. Examples:
  - **Q4 actual (in the quarterly metric row, Q4 column):**
    `=FY_cell - Q1_cell - Q2_cell - Q3_cell` — **not** a separate "Q4 derived" row
  - **Per-year Q ratio:** `=Q_cell / FY_cell`
  - **Average ratio:** `=AVERAGE(ratio_year1, ratio_year2, …)`
  - **Projected quarter:** `=projected_FY_cell * avg_ratio_cell`
  - **In-progress re-weighting:** `=Q3_ratio_cell / (Q3_ratio_cell + Q4_ratio_cell)`
- The model must **recalculate** when FY projection assumptions change.
- **CSV:** Cannot store live formulas or real hyperlinks. Add `provenance_url` and
  `formula` (raw strings) as needed; warn the user and recommend `.xlsx`.

Run the pre-delivery checklist in `_common/revelata-excel-styling-rules.md` before
sharing the file; add seasonality-specific checks: all ratios/projections are
formulas; Notes row populated per this skill's rules where applicable.

## Caveats to flag in Notes column

- **Imputed Q4**: flag in Notes every year where Q4 = FY − Q1 − Q2 − Q3 (not directly reported), even though the value sits **in the quarterly row**
- **Anomaly year excluded**: identify the year, the distortion, and that it was excluded
- **Shifting pattern**: if the quarterly mix is trending over time, note the direction
- **Non-December fiscal year**: match Q1–Q4 labels to the company's fiscal calendar
  (e.g., AAPL ends September, NVDA ends January)
- **In-progress re-weighting**: note which quarters are actuals vs. projected
- **Geographic/segment splits**: same technique applies; note the series used
