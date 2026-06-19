# Offline Filing Data Layout

This reference describes how investment-assistant offline filing data is
organized on disk. Use it when building a compact intake index or deciding
which existing filing-analysis materials to open.

## Root

Default root:

```text
data/investment_assistant/
```

Symbol directories:

```text
data/investment_assistant/symbols/<SYMBOL>/
```

Examples:

```text
data/investment_assistant/symbols/US.NVDA/
data/investment_assistant/symbols/US.MRVL/
data/investment_assistant/symbols/US.SNDK/
```

Symbols use the market-prefixed form, such as `US.MRVL`. Some providers may
store provider-native tickers inside JSON payloads; use the directory name as
the canonical workflow symbol.

## Read Order

For research intake, read files in this order:

1. `manifest.json`
2. `fmp_company_profile.json` when present
3. `sec_companyfacts.json`
4. `filing_metadata.json`
5. `filing_summary.meta.json`
6. `filing_summary.md` only when the symbol is selected for a filing-analysis
   read

Do not read raw filing HTML or extracted section markdown during intake. Those
belong to filing-summary generation or deeper research.

## Files

### `manifest.json`

Purpose: layer inventory and freshness gate.

Useful fields:

- `artifact_type`
- `symbol`
- `market`
- `source_status`
- `generated_at`
- `updated_at`
- `layers`
- `warnings`

Each `layers.<layer>` item can include:

- `status`: `fresh`, `partial`, `stale`, `unavailable`, `skipped`, etc.
- `source`
- `path`
- `meta_path`
- `asof`
- `updated_at`
- `warnings`
- `error`

Use manifest status to decide whether a missing file is expected, stale, or a
data-quality problem.

### `fmp_company_profile.json`

Purpose: compact business and security profile.

Useful fields commonly include:

- company name
- description
- sector and industry
- exchange
- market cap
- beta
- active trading flag
- ETF/fund/ADR flags
- country and currency
- website

Use this for light theme-fit checks. Do not use provider profile text as proof
of exact segment revenue or current AI revenue.

### `sec_companyfacts.json`

Purpose: exact numeric fundamentals from SEC/companyfacts or edgartools.

Useful fields:

- `fundamentals`
- `numeric_evidence`
- `metric_provenance`
- `warnings`

Common `fundamentals` keys:

- `ttm_revenue`
- `annual_revenue`
- `ttm_net_income`
- `annual_net_income`
- `gross_profit`
- `operating_income`
- `net_margin`
- `roe`
- `total_assets`
- `total_liabilities`
- `shareholders_equity`
- `debt_to_assets`

Rules:

- Exact financial numbers must come from this file or another structured
  numeric artifact.
- Check `numeric_evidence.llm_generated`; it should be `false` for authoritative
  metrics.
- Use `metric_provenance` when a number matters to the decision.
- Missing numbers remain missing.

### `filing_metadata.json`

Purpose: filing availability, forms, dates, accessions, and URLs.

Useful fields:

- `filings.latest_10k`
- `filings.latest_10q`
- `filings.latest_8k`

Each filing entry may include:

- `form`
- `filing_date`
- `period_of_report`
- `accession_number`
- `primary_document`
- `filing_url`
- `homepage_url`
- `text_url`
- `size`

Use filing dates to decide whether the filing-analysis material is fresh enough
for deep research.

### `filing_summary.meta.json`

Purpose: status and warnings for the offline filing summary.

Useful fields:

- `artifact_type`
- `symbol`
- `status`
- `generated_at`
- `source_files`
- `warnings`
- usage/runtime metadata when present

Read this before opening `filing_summary.md`. A `partial` summary can still be
useful, but warnings about skipped or truncated sections should be surfaced in
the intake artifact.

### `filing_summary.md`

Purpose: offline LLM-generated narrative summary of selected filing sections.

Open this only for symbols selected in:

- `must_read_filing_analysis`
- `optional_read_filing_analysis` when budget allows
- `promote_from_watchlist_for_reading`

Expected sections:

- Source Files
- Business Overview
- Recent Operating Discussion
- Demand Signals
- Margin / Cost Signals
- AI / Data Center Relevance
- Key Risks
- Changes vs Prior Filing
- Open Questions
- Data Quality Notes

Rules:

- Use it for qualitative management commentary and risk interpretation.
- Do not extract exact financial numbers from it as authoritative.
- Preserve source labels such as `[latest_10q / Item 2]` when citing.

### `filing_sections.json` and `filing_sections/**.md`

Purpose: raw extracted filing sections used to generate summaries.

Examples:

```text
filing_sections/latest_10k/part_i_item_1.md
filing_sections/latest_10k/part_i_item_1a.md
filing_sections/latest_10k/part_ii_item_7.md
filing_sections/latest_10q/part_i_item_2.md
filing_sections/latest_8k/item_202.md
```

Do not read these in research intake. They are for filing-summary generation or
manual deep dives after a symbol has already been selected.

### `raw_filings/*.html`

Purpose: raw SEC filing HTML.

Do not read in research intake.

### Optional FMP Files

Depending on data availability, a symbol directory may include:

- `fmp_peer_group.json`
- `fmp_normalized_metrics.json`
- `fmp_analyst_expectations.json`
- `fmp_earnings_transcripts.json`
- `fmp_ownership_signal.json`
- `fmp_insider_signal.json`
- `fmp_etf_exposure.json`

Treat these as optional. If status is `unavailable` or warnings show quota/API
errors, record the data gap instead of inferring a negative or positive signal.

## Intake Decisions

Use light files to decide read actions:

- Open `filing_summary.md` when business mix, cycle position, segment exposure,
  customer risk, capex, backlog, or margin durability is central to the layer
  decision.
- Keep `profile_metrics_only` when the symbol is an ETF, duplicate share class,
  lower-priority peer, or sufficiently represented by structured metrics for
  this pass.
- Promote a watchlist candidate only when it covers a missing layer/subdomain or
  appears materially stronger than a current deep candidate based on light
  evidence.
- Defer when manifest or filing-summary status is missing/stale and no refresh
  path is part of the current run.

## Source Discipline

- Cite file names in the output when they drive an intake decision.
- Use `sec_companyfacts.json` for exact metrics.
- Use `fmp_company_profile.json` for business descriptions and metadata.
- Use `filing_metadata.json` and `filing_summary.meta.json` for freshness.
- Use `filing_summary.md` only after the intake decision says to open it.
