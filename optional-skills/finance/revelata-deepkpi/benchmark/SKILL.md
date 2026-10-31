---
name: benchmark
description: >
  Operational peer / comp discovery for US public companies via deepKPI semantic search — not
  valuation multiples. Build KPI fingerprint from SEC data; company_summary_search (same-vertical
  first); align KPIs; diff gaps → segment sub-benchmarks. Outputs: tight chat 1-pager, optional
  Revelata HTML, optional Excel (.xlsx). Use when comparing to peers, finding comps or most-similar
  companies, benchmarking, comp sets, valuation peers, thematic "what's like X?", or segment
  pure-plays. Exclude unrelated verticals (e.g., gym vs car wash) except clear segment-to-segment,
version: 1.0.0
author: Revelata
license: Apache-2.0
homepage: https://www.revelata.com
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Revelata, Finance, SEC, KPI, deepKPI, Comps, Peers]
    category: finance
    related_skills: [revelata, kpi, company-summary, filing, implied-metric, seasonality, pressure-test, ideas]
---

# benchmark

Find comparable public companies ("benchmarks") for a target company or segment using deepKPI.
The output is not a mechanical list of industry peers — it's a curated set where every benchmark
adds a specific analytical lens, and where the gaps in coverage are as informative as the matches.

**Do not pair companies that do wildly different work** — different end customers, different
underlying need, different operating model — just because semantic search surfaced them together
(e.g. a gym chain and a car-wash operator). That is not a useful whole-company benchmark. The
only exception is when a **specific segment** of each business plausibly targets the **same
customers** with products that serve the **same job-to-be-done**; in that case, treat the match as
**segment-to-segment** (Step 3 sub-benchmarks, clearly labeled), not as "these two conglomerates
are comps."

**Vertical first, then wider.** Build the core benchmark set from companies in the **same
vertical** as the target's primary business (same industry, same customer job, same operating
shape) before you add names from **outside** that vertical. Broader or cross-vertical searches are
for filling gaps after same-vertical peers are in place, for a user-requested "outside the sector"
lens, or for Step 3 segment sub-benchmarks — not as the default first pass.

The workflow is explicitly iterative. Segment sub-benchmarks are not pre-assumed based on what
segments exist; they are discovered by diffing the target's KPI fingerprint against what the
whole-company benchmarks can actually explain. If something in the target's operations has no
counterpart in the whole-company set, that's the signal to go deeper.

## Data source

See `_common/data-source-rules.md` for the runtime routing table and the canonical
inventory of deepKPI operations.

## Hard stop if deepKPI access fails

See `_common/connection-failure.md`. STOP and ask before falling back.

## Dependencies

Before starting, consult:
- The `kpi` skill — retrieval workflow, provenance rules, scaling artifact
  handling, KPI search strategy. Required for all data pulls.
- `_common/revelata-excel-styling-rules.md` — only if user requests Excel output.
- The `implied-metric` skill — when filling Q4 or segment gaps in candidate
  alignment.

Every number displayed must have a `[value](provenance-url)` link. No exceptions.

## Output mode — ask first

**Before pulling any data**, ask the user which output format they want:

> "Would you like an **interactive HTML report** (opens in browser, clickable provenance links, Chart.js **time-series** charts) or **in-chat markdown** output? Or both?"

- If **HTML**: generate the full HTML report (dark Revelata branding, Chart.js line charts for KPI **trends over time**, fingerprint grid). Also produce the chat 1-pager as a compact summary.
- If **in-chat**: produce only the chat 1-pager with full markdown provenance links (see rules below). Offer the HTML at the end.
- If **both**: produce chat 1-pager first, then generate the HTML.

Offer Excel at the end regardless of which mode is chosen, unless the user already requested it.

## Outputs

| Format | When | Description |
|--------|------|-------------|
| Chat 1-pager | Always | Fingerprint pill rows + master table + KPI alignment + ≤5 notes — **all values hyperlinked** |
| HTML report | If user chose HTML or both | Dark Revelata-branded report with fingerprint grid, benchmark cards, and Chart.js **KPI trends over time** (line charts: target vs benchmarks on a shared period axis) |
| Excel workbook | User requests .xlsx | KPI alignment table, periods as columns per `_common/revelata-excel-styling-rules.md` |

### In-chat provenance link rules (non-negotiable)

Every numeric value shown in chat **must** be a markdown hyperlink to its source filing passage. No bare numbers.

Format: `[+4.9%](https://www.revelata.com/docviewer?pvid=11240)` — the pvid comes from the deepKPI response.

This applies everywhere in the chat output: fingerprint bullets, the KPI alignment table, notes, and the diff insight. If a value has no pvid (e.g. an estimate or derived metric), label it explicitly as `~$X est.` or `implied` with no link. Never omit the link for a directly sourced value.

Example of correct in-chat KPI alignment table row:
```
| TXRH | [+4.9%](https://www.revelata.com/docviewer?pvid=11240) | [$8.7M](https://www.revelata.com/docviewer?pvid=10061) | [714](https://www.revelata.com/docviewer?pvid=11537) |
```

---

## Workflow

The four steps below run in sequence. Steps 1–3 are analytical; Step 4 is output generation.
Run all deepKPI calls in parallel wherever possible to save time.

---

### Step 1: Build the target's KPI-enriched fingerprint

This is the foundation everything else builds on. Pull it from deepKPI — do not use general
knowledge for specific metrics.

**Calls:** `query_company_id` → `get_company_summary` → `get_company_segments` → `list_kpis`
→ `search_kpis` (for each segment's signature metrics)

**What to extract:**

**A. Narrative** — 2-3 sentences: business model, revenue drivers, customer type, go-to-market,
geography. Ground it in numbers where possible (unit count, revenue scale, market share if stated).

**B. Segment map** — For each reportable segment:
- Name and revenue share %
- One-sentence description of the economic model (how it makes money, who pays)
- 3-5 signature KPIs with actual values from deepKPI, each with provenance link

The segment KPIs are what make this fingerprint useful for diffing. For an airline: ASMs, load
factor, CASM, TRASM. For a refinery: capacity bpd, utilization %, gross margin per barrel.
For a restaurant segment: unit count, comp sales %, AUV. Pull the real numbers.

**C. KPI signature** — The 3-5 metrics that matter most for the consolidated company. These
become the columns in the KPI alignment table in the output.

**Output of Step 1:** An internal working document (not shown to user) with:
```
Target: [TICKER] — [Company] (CIK: NNNNNN, FY: DATE)
Primary vertical (for comp search ordering): [e.g. full-service restaurant, regional bank, SaaS HCM]
Narrative: ...
Segments:
  [Segment A] (XX% rev): [description] | KPIs: metric1=[val](url), metric2=[val](url), ...
  [Segment B] (XX% rev): [description] | KPIs: metric1=[val](url), ...
KPI signature: metric1, metric2, metric3 [, metric4, metric5]
```

---

### Step 2: Find candidates and align their fingerprints

**A. Search**

Run `company_summary_search` with 3-5 queries. Each query should describe a *business model*,
not just a sector name. The goal is semantic richness — describe what the company does, for whom,
how it earns money, and what drives its unit economics.

**Order matters — same vertical first.** At least **the first two queries** must stay tightly
inside the target's **primary vertical** (from Step 1: summary, segments, and the "Primary
vertical" line). Pull and KPI-align those results before you run looser or cross-vertical queries.
Use additional queries to go **outside** the vertical only **after** you have a credible
same-vertical candidate set, or when the user explicitly wants analogies from other sectors, or
when Step 3 needs a segment-specific search.

Examples of the contrast:
- Weak: `"airline"` → Strong: `"hub-and-spoke network carrier international domestic routes premium cabin loyalty co-brand revenue per ASM"`
- Weak: `"restaurant chain"` → Strong: `"multi-brand casual full-service dining operator franchised same-store sales average unit volume"`
- Weak: `"software company"` → Strong: `"B2B SaaS subscription recurring revenue net revenue retention enterprise mid-market"`

Use `top_k_companies: 10-12` per query. Deduplicate across queries. Target 6-12 candidates.

**B. KPI alignment**

For each candidate: call `list_kpis` (free), then `search_kpis` for the target's KPI signature
metrics (from Step 1C). This produces aligned rows — the target's metrics sitting next to each
candidate's equivalent metrics.

**Recency rule (do this from the start):** when the user's question is about "current" / "latest"
comparability, prefer the **most recent quarterly periods** available (10-Q-derived) for each KPI
over stopping at the last 10-K annual figure. Use annual series for long-cycle context, but do not
ship a benchmark set that effectively says "data is old" if newer 10-Q periods exist for the KPI.

The alignment reveals two things:
1. How close each candidate is on the dimensions that matter (similar range = structurally comparable)
2. Where candidates have no data or wildly different values (poor fit on that dimension)

Coverage coding:
- **Green**: metric present and plausible
- **Yellow**: metric present but value seems off (scaling artifact, possible cross-contamination)
- **Red**: no data

**C. Tier the candidates**

Based on the aligned KPI fingerprints (not just the description similarity), and only after
passing the **structural fit gate** below:

- **Tier 1**: Closest structural match — same business model and same customer/need, similar values on 3+ signature KPIs
- **Tier 2**: Useful comparators — same overall job-to-be-done; good match on some KPIs, different on others; adds a specific lens
- **Tier 3**: Same **kind** of business as the target, different scale, mix, or geography — still
  a coherent comp (e.g. regional vs national airline). **Not** unrelated verticals "for color."

**D. Structural fit gate (drop before the output table)**

Before anything appears in the chat 1-pager or HTML as a whole-company benchmark, ask: *Would
an investor reasonably describe this company as doing the **same job** for the **same type of
buyer** as the target?* If **no**, exclude it from Tier 1–3 for whole-company analysis — even if
a few metrics line up by coincidence. If there is a **real but narrow** overlap (same
wallet, same purchase occasion, same enterprise buyer for a specific product line), keep the
candidate **only** as a **segment** sub-benchmark in Step 3 with an explicit "Maps to: Segment …"
story, not as a peer for the consolidated company.

---

### Step 3: Diff — find what the whole-company benchmarks don't explain

This is the step that surfaces the most analytically valuable insights.

**The question:** After aligning KPI fingerprints in Step 2, which parts of the target's
operations have *no good coverage* in the whole-company benchmark set?

A "coverage gap" exists when:
1. The target has a reportable segment whose signature KPIs appear nowhere in the Tier 1/2 benchmark set, OR
2. The target reports KPIs that are structurally foreign to all benchmark candidates
   (e.g., `refinery throughput bpd` when all candidates are airlines), OR
3. A major segment has a fundamentally different economic model than what any whole-company
   benchmark represents

**For each gap identified:**

a. Build a **segment-specific fingerprint** using the segment's own KPIs from Step 1B
   ("this segment, if it were a standalone public company, would look like:")

b. Run a fresh `company_summary_search` describing that segment as a standalone business.
   Sub-benchmarks must still satisfy **same customer + same need** for that segment; do not
   pull unrelated verticals "for coverage."
   Use the segment's KPIs as the anchor — e.g., for a refinery segment:
   `"independent petroleum refiner crude oil throughput barrels per day capacity utilization refining margin per barrel crack spread"`.
   Not: `"refinery"`.

c. Pull KPI alignment for sub-benchmark candidates using the *segment's* KPI signature
   (not the parent company's consolidated metrics)

d. Tag all matches as `Maps to: Segment: [name]` in the master table

**Important framing:** The sub-benchmark tells you what that segment *would* look like as a
standalone profit-maximizing business. When the segment operates under a different economic logic
inside its parent (cost-center, captive supplier, hedge), say so explicitly — the gap between
the sub-benchmark's economics and the segment's implied economics *is the analytical insight*.

**Example of the diff in action:**
- Target: Delta (airline + refinery)
- Step 2 finds: UAL, AAL, LUV, ALK — all cover the airline KPIs (load factor, CASM, ASMs)
- Diff: `refinery throughput bpd`, `utilization %`, `gross margin/bbl` appear in Delta's segment
  map but in *none* of the airline benchmarks
- Step 3 triggers: build refinery segment fingerprint → search → find PBF, CVI, DINO as
  sub-benchmarks → align on throughput/utilization/margin
- Insight that emerges: Monroe's 200K bpd capacity is closest to CVI (207K bpd combined);
  PBF shares East Coast PADD 1 geography; but Monroe runs as a hedge, not a profit center —
  the merchant refiner margins ($8–16/bbl) represent the *opportunity cost* Delta pays to own it

The segment sub-benchmark logic is not limited to named segments. If the target has any
operational dimension that reads as structurally foreign to the whole-company comps —
a financial arm, a logistics network, a retail footprint embedded in a manufacturer —
ask whether a standalone pure-play exists in the S&P 1500 that illuminates it.

---

### Step 4: Build outputs

#### Chat 1-pager

Tight. No prose sidebars. Everything fits on one screen. In the **benchmark table** and **KPI
alignment** table, put **same-vertical** names first (clearest structural peers in the primary
vertical), then cross-vertical or segment-mapped comps, with `Maps to` spelling out any segment
story.

```
# Benchmarks for [Company] ([TICKER])

[TARGET FINGERPRINT — 2-3 sentences. Every sourced metric hyperlinked:
e.g. [+2.0% blended comps](https://www.revelata.com/docviewer?pvid=578),
[$12.1B revenue](https://www.revelata.com/docviewer?pvid=11442)]
Segments: [Segment A] XX% rev | [Segment B] XX% rev

## Benchmark table

| Ticker | Maps to | Similar on | Different on | Scale vs [TARGET] |
|--------|---------|------------|--------------|-------------------|

## KPI alignment — latest available period(s)

| Company | [metric1] | [metric2] | [metric3] | [segment metric] | Period |
|---------|-----------|-----------|-----------|-----------------|--------|
| [TARGET] | [val](pvid-url) | [val](pvid-url) | [val](pvid-url) | [val](pvid-url) | FY... or FY..-Q.. |
| BENCH1   | [val](pvid-url) | [val](pvid-url) | — | — | FY... or FY..-Q.. |

Column headers are plain text. Only the value cells carry hyperlinks.
Estimates/derived values: `~$X est.` with no link.

## Notes
- [≤5 bullets — only material caveats: scaling flags, cross-contamination, structural differences,
  coverage gaps, economic logic mismatches (cost-center vs. profit-maximizer)]

## Sources
[Ticker (CIK) — Green/Yellow/Red — FY end — any corrections]
```

If a section would be empty, omit it. If the KPI alignment table would exceed 10 rows, drop Tier 3
from it (keep Tier 3 in the benchmark table only).

#### HTML report

Read `references/html-template.md`. Key structure:

- **Hero fingerprint grid** (`table.fp-grid`): KPIs × benchmark columns only — **no prose inside the grid**
  (no "Why it matters", thesis blurbs, or `fp-context`-style rows). Put narrative in benchmark cards
  and diff insight. **`.legend-strip`** (color key) sits **below** the grid, not in the page header.
- **Benchmark cards** (one per company): order **same-vertical** peers before cross-vertical or
  segment-mapped names. Tier-colored top border, similar/different columns,
  key KPI values hyperlinked. Segment sub-benchmarks use purple border + "Segment:" badge.
- **KPI trends over time (charts + alignment table)**: This section is where deepKPI's edge shows up—**filing-resolved time series** most vendors do not reconstruct at this depth. For the **2–3 most important** signature KPIs, pull **multi-period** values for the target **and every benchmark** (prefer quarterly where coverage is solid; otherwise annual—follow the `kpi` skill for period selection, scaling, and provenance). Render **Chart.js line charts** (not single-period grouped bars): **x-axis = time** (ordered FY/FQ labels), **one line per company** so the reader compares **trajectory, inflection, and pace** across competitors on the **same** metric. Put the full **KPI alignment** table (with hyperlinked values) **below** the charts as today. If a KPI truly has fewer than three comparable periods across the set, you may show a sparse line chart **or** a one-line caption pointing readers to the table—do not imply a cross-sectional bar is equivalent to the time-series view.
- **Chronology QA for every time-series chart (mandatory)**: Ensure the plotted points are in **chronological order** for every company series and the x-axis labels are in **ascending time**. **Do not rely on input order.** Sort by a real time key (prefer **period end date**; otherwise parse FY/FQ labels into a comparable key) before passing points to Chart.js. A zig-zagging line is almost always an ordering bug.
- **Y-axis QA for every time-series chart (mandatory)**: **Double-check** each chart's **Y scale** before shipping HTML. The visible **min/max domain must include every non-null plotted point** across **all** series (target + benchmarks). **Do not** set fixed `min`/`max`, `suggestedMin`/`suggestedMax`, or `beginAtZero` unless those bounds are **recomputed from the union** of all series values—wrong bounds **clip lines off-canvas** so they look missing. Prefer automatic scaling plus **padding** (e.g. Chart.js `scales.y.grace` as a %) so tight clusters stay visible. For **log** scales, omit non-positive points or avoid log if that would drop the series; confirm lines still render. **Mentally verify each tab**: any ticker with data must show a **visible** polyline or points—not a legend entry with no trace.
- **Diff insight box** (cyan glow, same as synthesis in pressure test): 2 paragraphs on what
  the benchmark set reveals that whole-company analysis would miss.
- **Notes** and **Sources** footer.

#### Excel workbook

Follow `_common/revelata-excel-styling-rules.md`.
One sheet for whole-company benchmarks, one sheet per segment sub-benchmark group.
Annual periods as columns; green cells for target values; clickable provenance links.

---

## Failure modes

- **Assuming segment sub-benchmarks without doing the diff**: Don't pre-decide "this company
  has a real estate arm so I'll find REITs." Let the KPI gap analysis surface it. The diff step
  earns its keep precisely because the gaps are often surprising.
- **Narrative-only fingerprint**: If Step 1 doesn't pull actual KPI values, the diff in Step 3
  has nothing to work with. The fingerprint must have numbers.
- **Prose inside the HTML hero grid**: Do not add "Why it matters", thesis blurbs, or other
  commentary rows inside `table.fp-grid` — the grid is KPI labels and numbers only. Use benchmark
  cards and diff insight for narrative (`references/html-template.md`).
- **Skipping KPI alignment for candidates**: Checking description similarity is not enough.
  Pull the actual metrics — two companies can describe themselves similarly but operate very
  differently when you look at unit economics.
- **Treating all segments equally**: Not every segment warrants a sub-benchmark search.
  Focus on segments where: (a) the parent discloses separate KPIs, and (b) the KPIs are foreign
  to the whole-company benchmark set.
- **Time series plotted out of order**: If any chart line zig-zags "backward," the points are not time-sorted. Sort each series by **period end date** (or FY/FQ key) before charting, and make sure the shared x-axis label list is in ascending order too.
- **Y-axis clips time series in HTML**: Hard-coded or guessed Y bounds, or `beginAtZero` used without checking the data band, often **hides entire lines** outside the drawable range. Always derive bounds from **all** series (or use defaults + `grace`) and re-check before delivery.
- **Too much prose in chat**: The 1-pager is a reference artifact. Cut until it hurts.
- **Skipping list_kpis**: Free call that prevents wasted credits on metrics that don't exist.
- **Unrelated verticals as comps**: A gym and a car wash are not whole-company benchmarks for
  each other unless you have a defensible segment-level story (same customers, same need). If you
  cannot write that one sentence without hand-waving, drop the name.
- **Cross-vertical before in-vertical**: Do not foreground or prioritize outside-the-vertical
  names until same-vertical peers are identified and KPI-aligned. If the first search pass drifts
  cross-vertical, reset with tighter in-vertical queries before presenting the table.
