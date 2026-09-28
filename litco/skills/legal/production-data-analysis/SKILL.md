---
name: production-data-analysis
description: Analyze a bulk data production and write the memo.
version: 0.2.0
author: LitCo, Hermes Agent
platforms: [linux, macos]
metadata:
  hermes:
    tags: [legal, ediscovery, production, data-analysis, duckdb, damages]
    related_skills: [litkit-corpus-pull, legal-cite-check, discovery-letter-brief, litkit-deliverable]
---

# Production Data Analysis Skill

Handle an opposing party's data production end to end: retrieval, byte verification, decryption, full-population profiling, charts, and an illustrated analysis memo for the case team. For the produced documents themselves (imaged email and files in LitKit), load `references/litkit-access-procedure.md`.

## When to Use

- A production arrives as bulk data (month-partitioned CSV or CSV.gz, Spark part files, Parquet) rather than imaged documents.
- The lawyer asks to download, unzip with a password, and analyze.
- The producing party discloses its collection protocol (custodian list, search-term table, cover letter assigning Bates ranges to requests) and the team needs verifiable specifics on what it could and could not have collected.

## Prerequisites

- The host's data venv (DuckDB, pandas, matplotlib, python-docx) and `7z` for encrypted archives.
- The block volume for large productions.
- The `litkit` toolset for cross-checks against produced documents.

## Procedure

### 1. Retrieve and verify the bytes

- Record the true file size before downloading; verification is meaningless without it.
- For a signed-URL host (Box and similar), use the browser only to mint the direct URL, then download with `curl --retry 10 --retry-all-errors -C - -o file.zip "$URL"` as a background process after confirming the server answers range requests (`206`, `accept-ranges: bytes`).
- A link behind a firm's single sign-on (SharePoint, OneDrive) needs the requester's own access; ask them to attach the file or share a direct link rather than driving a login wall.
- Verify the byte count before touching the archive.

### 2. Decrypt and extract

- `7z x -y -p'PASSWORD' archive.zip -o<outdir>`; `7z l -slt -p'PASSWORD'` shows each member's encryption and method.
- Record the cipher and how the password arrived, with a hygiene flag if it traveled over an ordinary channel. Never write the password into a deliverable.

### 3. Profile with DuckDB: full population, no sampling

- Query partitions in place: `read_csv('month=*/*.csv.gz', hive_partitioning=true, header=true, nullstr=[''])`.
- Never alias a column `rows`; use `n_rows`.
- Order: schema and field-semantics checks; totals, distinct keys, min and max dates, amount sums; per-partition aggregates; the distributions the case theory cares about.
- Validate monetary-field meanings before naming a metric. Distinguish a producer-authored dictionary from an analyst's interpretation; cross-tab each amount field and count equal, greater, and smaller values across related fields. Without documentation, use the literal field name.
- Save a measure definition beside each cited cut: entity key, scope, unit, denominator, window, and whether values are observed, estimated, or forecast.
- `COPY (SELECT ...) TO 'analysis/cuts/<cut>.csv' (HEADER)` every cut the memo cites and save its SQL as `analysis/sql/<cut>.sql`; the memo builder reads the CSVs. Never transcribe query output by hand.

### 4. Charts

- One matplotlib script emits every exhibit as PNG at 200 dpi, each titled "Exhibit N — <finding>". Escape dollar signs (`\$`) so mathtext does not eat them.
- Inspect each exhibit with `vision_analyze` before embedding.

### 5. Memo and the fact-check gate

- Follow the matter's house style (template in the matter's files or matter memory); build from a script and regenerate on every revision; stamp the version in the caption.
- Reconcile metric definitions before comparing numbers: revenue versus volume, list price versus transaction amount versus the competitive benchmark. A discount label does not establish a price reduction.
- Reproduce an opponent's KPI on its own population before interpreting a gap.
- Damages envelopes start from observed exposure and a labeled price-difference scenario, not an assumed overcharge; deduplicate overlapping components; leave the benchmark to the expert.
- Describe dispersion as dispersion, exposure counts as potential exposure, and treatment timing from dated rollout evidence.
- Rerun every number quoted in the memo through a fresh verification query before delivery, and never state a figure a query did not return. A failed lookup is reported with its blocker.
- Register is predictive, not adversarial: say what the data shows and what it cannot show; label expert-dependent inferences.
- Flag coverage gaps against the pleading (a class period that starts before the produced window) as meet-and-confer items.
- Search-protocol assertions ("no string contains X") are re-derived from the parsed term table with abbreviations and variants, and the per-string custodian count is stated.
- Cross-check against produced documents through LitKit where the memo relies on them, with Bates cites from documents actually read.
- Deliver with `litkit-deliverable` (class `memo`); figures lifted later into a filing go through `legal-cite-check`.

## Pitfalls

- Bind comparative figures to explicit entity keys, not list positions, so reordering never swaps values.
- A correct calculation over mixed products or a forecast period does not establish a product-specific historical fact.
- An announced exit is not a completed closure; one firm's forecast is not an industry result.

## Verification

- Every number in the memo traces to a saved cut and its SQL.
- The byte count and archive listing are recorded in the provenance section.
