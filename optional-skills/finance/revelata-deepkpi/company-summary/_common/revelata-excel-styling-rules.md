# Revelata Excel styling rules

Canonical Excel (.xlsx) layout and styling for deepKPI financial exports: wide
layout, annual then quarterly columns, Calibri, pale-green fill on deepKPI input
cells, dark-blue hyperlinked numbers, numeric date headers, time series values
as Excel numbers (not text), freeze panes, column grouping, no redundant Source
rows; same metric at quarterly and annual frequency (same definition/unit) on
one row across period columns, not two rows; derived cells must be live
formulas, never hardcoded computed values.

Apply whenever building a spreadsheet from deepKPI data. Pair with the **xlsx**
skill for implementation.

**In-chat** KPI tables stay **tall** (periods as rows). **Excel** workbooks are
**wide** (periods across columns) for modeling. This file is the **single source
of truth** for deepKPI `.xlsx` structure and visuals.

Always invoke the **xlsx skill** for spreadsheet output — it governs formula
discipline, tooling, and general conventions. The sections below add **financial /
deepKPI-specific** requirements on top.

## DeepKPI Excel export spec

Use whenever the user wants a **file** (or accepted a post-pull `.xlsx` or `.csv` 
offer).

### Sheet defaults (apply to the whole workbook)

- **Font:** **Calibri** for the sheet. **Hyperlinked value cells** (provenance links)
  use **dark blue** font (`#0563C1` / `FF0563C1`) and the **number format** in
  **Hyperlinks** below so they read like real links — see that section. Other body
  (non-linked) styling follows **Color conventions**.
- **Gridlines:** **Off** — the worksheet should present with **no visible cell
  grid** (Excel: gridlines hidden for the sheet).
- **Empty data cells** (no value for that period): show a **dash** (`-` or
  accounting-style em dash), not a blank cell — consistent for the data region
  (period columns from Col E onward, and any comparable body area).

### Time series values — store as numbers (not text)

- Any **numeric** value written from deepKPI **time series** (reported actuals,
  imputed amounts, ratios, rates, and other quantitative points) must be stored as
  **Excel number values** (numeric cell type), **not** as text — so sorting, charts,
  `SUM`, and arithmetic work without coercion.
- **Do not** write series values as strings that only *look* like numbers (e.g.
  quote-prefixed cells, padded text, or literals like `"1,234.5"` as the stored
  value). Use a **real numeric** `cell.value` (or equivalent) and apply **number
  format** (currency, percent, thousands, decimals) for display.
- **Percent / basis points:** store the **numeric** magnitude the sheet should use
  (per your convention, e.g. decimal fraction for percent format), still as a
  number — not a text label such as `"15%"` unless the underlying source is
  genuinely non-numeric.
- **Hyperlinked value cells:** the **cell value** remains numeric when the metric is
  numeric; the provenance hyperlink is layered on the cell — do not replace the
  number with a text-only representation. Display them with **dark blue** font and
  the **hyperlink number format** in **Hyperlinks** (percent-style for rates; for
  counts/currency use the sheet accounting format **with the same dark blue**).

### Title cell **C1** (not A1)

Columns **A** and **B** are **hidden** (audit trail). A title in **A1** would not
**show** — put the workbook title in **C1** (first visible column). Leave **A1:B1**
empty.

- **C1** is a short **title / description** of the workbook: **bold**, **font
  size 12**, **Calibri**, **normal sentence case** (not ALL CAPS). Include the
  **full official company name** as registered (e.g. from `query_company_id` /
  SEC registrant), plus what the sheet contains (e.g. historical KPIs / deepKPI
  extract). Example shape: `Chipotle Mexican Grill, Inc. — quarterly and annual
  KPI historicals (deepKPI)`. For long titles, merge **C1** rightward across the
  label area (e.g. **C1:D1**) or as needed; do not place the title in hidden
  columns.
- **Row 2 must be empty** — leave the **entire row blank** (no titles, no merged
  filler). This separates the **C1** title from the data block.
- **Row 3** is the first row used for **period headers** (dates in Col E onward) and
  any ALL CAPS section labels that sit on the same row as those headers if your
  layout uses that pattern; **row 4** is the **first** KPI **data** row unless your
  template needs an extra header row (if so, keep **one** blank row still **only**
  between the title row and where dates begin).

### Period column headers (store dates as numbers)

- Do **not** type period labels as plain text in header cells. **Store real Excel
  date serials** in the **date header row** (**row 3** in the default layout: **C1**
  title → blank row 2 → **headers row 3**), in Col E onward, then apply **number format**:
  - **Quarterly columns:** `mmm-yy` (displays **Mar-22**, **Dec-23**, etc.).
  - **Annual columns:** use a consistent year-end (or fiscal year-end) **date
    value** in the cell and format as **`yyyy`** (or equivalent) so it reads clearly;
    do not use text like `FY2022` unless it is still a formatted date number.
- That same header row remains **bold** and **underlined** (font), per layout rules.

### Freeze panes

- Set **freeze panes** so when the user scrolls:
  - The **period header row** (date labels in Col E onward) **stays visible** below
    the title and blank spacer; and
  - **Columns A through D** **stay visible** horizontally.
- **Default row layout:** Row **1** = **C1** title (leave **A1:B1** empty) · Row **2** =
  empty · Row **3** = period headers · Row **4** = first data row → set freeze to
  **`E4`** (top-left of the scrollable grid: first data cell under the date headers,
  right of Col D).
- If you add rows above the date header for extra titles only, still keep **exactly
  one fully blank row** between the last title row and the date header row, then
  freeze at the cell **under** the header row and **right of** column D.

### Layout — revenue build model convention

Format all output spreadsheets as a **revenue build model**:

- **One row per KPI or distinct time series** — use **one row with many
  date columns** when multiple deepKPI series are the **same KPI** (same metric
  definition and **same unit**). **Quarterly and annual variants of the same
  metric** (e.g. `Sales - Olive Garden - 3 Month` and `Sales - Olive Garden -
  Annual`) belong on the **same row**: put **annual** values under the **annual
  column block** and **quarterly** values under the **quarterly** block — **do
  not** put them on two separate rows. Only **split into separate rows** when the
  series truly differ (different definition, unit, segment, or a different metric
  — not merely a different aggregation period). Do not fragment one series into
  "Revenue - Q1" / "Revenue - Q2" rows. **Each value lands in the column that
  matches its reporting period;** Col A may list multiple deepKPI source strings
  if you merged quarterly + annual into one row (or keep one canonical display
  label in Col C).
- **Imputation / gap-fill (same series row):** When you **impute** a missing point
  in an existing stream (e.g. **Q4 = FY − Q1 − Q2 − Q3** because there is no Q4
  10-Q), put the value **only** in **that series' row**, in the **missing period
  column(s)**—as a **live `=` formula** (black font, no green input styling). **Do
  not** add a separate line item such as "Q4 (derived)", "Q4 Revenue", or "Imputed
  Q4" beneath the quarterly row. **Especially in Excel**, the quarterly KPI stays
  **one row**; Q4 cells reference FY and Q1–Q3. **Annual figures for that same
  metric** (same definition and unit, only FY vs quarterly frequency) sit on **this
  same row** in the annual columns — not on a second row. Document imputation in
  **Notes** or an agreed footnote row, not as an extra KPI row.
- **Distinct derived KPIs** (YoY %, margins, segment remainder **as its own metric**,
  build subtotals): **operand** rows above where needed; the **derived KPI** gets
  its **own** row—because it is a **different** line item, not a hole in one stream.
  Those cells are still **live formulas** (see **Derived values — formulas only**).
- **Time periods flow left to right** within each block. Row labels in column A.
- **Three-column label system**:
  - Col A (hidden): deepKPI metric name string — serves as audit trail
  - Col B: Display label (what users see)
  - Col C: Units tag, italic (e.g., `(in thousands)`)
  - Col D onward: time-period data values
- **KPI names (Col C — display labels) — one pattern for the whole sheet:** Before
  filling rows, choose a **single** convention for how **metric + qualifier /
  segment** reads. **Prefer** separating the qualifier with an **em dash or hyphen**
  (`Revenue — Americas`, `Dues - Black Card`) — **avoid** parenthetical modifiers
  (`Metric (segment)`) when a dash-separated label reads naturally. Apply the
  pattern **everywhere** — do not mix styles for parallel constructs (e.g. not
  *Black Card Penetration* on one row and *Dues — Black Card* on another).
- **Abbreviations in Col C:** Prefer **full words**; use abbreviations **sparingly**
  and only when they are **standard in the industry** (e.g. **ARPU**, **ASP**,
  **EPS**). **Do not** use opaque or company-specific codes (e.g. **BOC**, internal
  segment codes) unless the user explicitly asked for that exact label. If a KPI 
  has a specific abbreviation already baked in that you don't recognize, e.g. CCHBC
  then use it as it appears.
- Col A may still hold the raw deepKPI string; Col C should read as a **consistent
  series of human-readable labels**.
- **Annual and quarterly column blocks:** **Annual columns first** (FY2017,
  FY2018 … or 4-digit year labels), left → right, oldest first. Then **one blank
  column**. Then **quarterly columns** (Mar-22, Jun-22, Sep-22, Dec-22 …), left →
  right, oldest first. Do **not** put quarterlies before annuals.
- **Column outline / grouping (expand–collapse):** Create **two** separate Excel
  **column groups** (outline) so the user can **collapse or expand** the **annual**
  block vs the **quarterly** block: (1) group **only** the contiguous **annual**
  period columns (including the date header row for those columns); (2) group
  **only** the contiguous **quarterly** period columns. Leave the **blank separator
  column ungrouped** (between the two groups). Columns **A–D** are **not** grouped.
  Match Excel **Data → Group** behavior (or your library's equivalent), e.g.
  openpyxl: set `outlineLevel = 1` on each `column_dimensions` entry spanning the
  annual columns, then again for the quarterly span (**sibling** groups, not one
  nested group). Leave groups **expanded** by default so numbers are visible unless
  the xlsx skill says otherwise.
- **Date header row** (Col E onward): **numeric Excel dates** with display formats
  above (**Mar-22** style for quarterlies); **Bold** and **underlined** font.
  (Calibri.)
- **ALL CAPS section header rows** (e.g. `REVENUE BUILD`): **Bold** and **font
  underlined** (underline the text in Col C for that row; background unchanged
  unless the xlsx skill says otherwise).
- **Color conventions** (Revelata KPI inputs — overriding generic xlsx fill rules
  for this file type):
  - **deepKPI-sourced input values** ("hardcoded" from API): **fill / background**
    **`#E2EFDA`** (Excel ARGB `FFE2EFDA`). **Font color:** use **`#0563C1`**
    (`FF0563C1`) — **dark blue** — on cells that have an **Excel hyperlink** to
    provenance (so links look like links). Use **`#00B050`** (`FF00B050`) only for
    **non-hyperlinked** sourced numerics if any (rare); default is **dark blue** on
    all linked value cells.
  - **Derived / calculated formula cells**: **Black** font (default or explicit); no
    green tint unless the xlsx skill differs.
  - **Col A source-label text**: **Blue** `#0070C0` (`FF0070C0`) where still specified below.
  - **Red** `#FF0000` (`FFFF0000`): estimated or external values (use sparingly)
- **% change rows**: italic label + italic data, formatted as accounting-style %:
  `_(* #,##0.0%_);_(* \(#,##0.0%\);_(* "-"?_);_(@_)`
- **Number format** for all data values (accounting with dash for zero):
  `_(* #,##0_);_(* \(#,##0\);_(* "-"_);_(@_)`
- **Subtotals**: bold; last item before total gets a thin bottom border.
- **Grand totals**: bold, double top border.
- **Sub-items**: indent label in Col C with two leading spaces.

```
Col A (hidden)            | Col B | Col C                  | Col D          | FY2022       | FY2023       |   | Mar-22       | Jun-22       | Sep-22       | Dec-22       | … |
--------------------------|-------|------------------------|----------------|--------------|--------------|---|--------------|--------------|--------------|--------------|---|
                          |       | REVENUE BUILD          |                |              |              |   |              |              |              |              |   |  ← ALL CAPS bold + underlined
revenue; qtr+FY ($)       |  33   | Revenue                | (in thousands) | [1,196,826] inp| [1,383,600] inp|   | [248,017] inp| [300,941] inp| [292,246] inp| [355,622] drv| … | ← one row: FY block + quarterly block; Q4 imputed = FY−Q1−Q2−Q3 in Dec column, black—not a separate "Q4" row; not a second "annual only" row
                          |  34   |   % YoY                |                |   +15.6%     |   +12.5%     |   |     n/a      |   +21.1%     |   +15.2%     |              | … | ← italic; distinct KPI → own row OK
```

*(Legend: **row 2 blank** between **C1** title and headers; date header row —
**numeric** dates, bold + underlined; empty inputs — `-`; **C1** = bold 12pt title
with full company name; **A1:B1** empty.)*

### Derived values — formulas only

**Every derived cell** must be a live Excel **`=`
formula** referencing inputs. **Imputed quarters** (e.g. Q4 = FY − Q1 − Q2 − Q3) live
in the **same row** as that quarterly series, in the gap column—**not** on a
separate "Q4 derived" row. **Annual and quarterly pulls of the same metric** stay on
that **same row** (annual columns vs quarterly columns), not a second row for "Annual"
only. **Distinct** derived KPIs (YoY %, ratios, remainder
lines, projections) use **their own** rows because they are different series.

**Never** type, paste, or "hardcode" the computed number into a derived cell — even if
it matches what you calculated in chat or in your head. The workbook must
**recalculate** when inputs change; a static numeric literal in a derived cell is a
failure. Only **deepKPI-sourced inputs** (values taken straight from the API) may be
literal values in cells (still with hyperlinks per this skill).

### Hyperlinks — provenance only (no extra Source rows)

Every deepKPI value must carry an **actual Excel hyperlink** on the **value cell** —
that is the filing reference. **Do not** add separate **Source** rows, **Source**
columns, or other redundant blocks that only restate the filing name; the link is
sufficient.

- **Display:** Use the **metric value** (or short label) as the link text, not
  "10-K FY2024" unless that is the value shown.
- **Style (linked value cells):** **Font color** **dark blue** **`#0563C1`**
  (Excel ARGB `FF0563C1`). **Do not** rely on underline to signal links — Excel's
  per-character underline on hyperlinked numbers is unreliable across clients and
  libraries.
- **Number format** on **hyperlinked** numeric cells (rates / percents / ratios stored
  as decimal fractions, e.g. `0.105` → 10.5%):

  `_(#,##0.0%_);_((#,##0.0%);_(-_);_(@_)`

  For **counts, currency, or other non-percent** linked values, keep the sheet's
  standard accounting format from **Number format** above (`_(* #,##0_);_(* \(#,##0\);…`)
  but **still** apply **dark blue** (`FF0563C1`) on the font so the cell reads as a
  link.
- **Calibri** on link cells.

Use `cell.hyperlink` + numeric `cell.value`, or `=HYPERLINK("url", …)` with a numeric
cell — apply **font color** and **number format** on the cell as above (not default
hyperlink styling).

Never store a provenance URL as **plain text only** in the data grid.

**CSV:** Cannot store hyperlinks. Add a `provenance_url` column with raw URL
strings, and warn the user that links are not clickable — recommend .xlsx.

### Pre-delivery checklist (Excel output)

Before presenting the file link, verify every item. Do not deliver the file until all boxes pass.

- [ ] No spurious row fragmentation (no "metric - Q1" rows); merged same-KPI
      series = one row; **quarterly + annual for the same metric (same definition /
      unit)** = **one row** (annual block + quarterly block), not two rows; **imputed
      Q4 (etc.) in that row's column**, not a separate "Q4 derived" line item;
      distinct KPIs = separate rows
- [ ] **Annual columns are the left block**, blank separator, **quarterly columns
      on the right** — not the reverse
- [ ] **Column groups** defined: **annual** block and **quarterly** block each
      collapsible via outline (separator + cols A–D **outside** groups)
- [ ] **Calibri** as the workbook body font
- [ ] **Gridlines hidden** (no visible cell grid on the sheet)
- [ ] **C1** (not A1): **bold**, **12pt**, sentence case, **full official company name** +
      data description; **A1:B1** empty (hidden columns)
- [ ] **Row 2** entirely **blank** between title and date headers
- [ ] **Date headers**: **Excel date serials** in cells, formatted **`mmm-yy`**
      (etc.), **not** typed text — **bold** + **underlined** (default: **row 3**)
- [ ] **Freeze panes** at **`E4`** by default (or equivalent: below header row,
      right of **D**)
- [ ] **No** extra **Source** rows/columns for filings — only **hyperlinks** on value
      cells
- [ ] **Hyperlinks** on value cells: **dark blue** font `FF0563C1`; number format
      `_(#,##0.0%_);_((#,##0.0%);_(-_);_(@_)` for percent-class metrics; accounting
      format + same blue for non-percent linked values; **do not** depend on underline
- [ ] **Empty** data cells show a **dash** (not blank) in the data region
- [ ] **deepKPI input** cells: fill **`#E2EFDA`**; **hyperlinked** inputs **dark blue**
      `FF0563C1` per **Hyperlinks**
- [ ] **Derived** cells: **black** font (no green input styling); formulas live
- [ ] Every derived value is a live `=` formula referencing input cells — no hardcoded numbers
- [ ] **Time series numerics** stored as **Excel numbers**, not text strings
- [ ] Number format `_(* #,##0_);_(* \(#,##0\);_(* "-"_);_(@_)` applied to data cells as applicable
- [ ] Every deepKPI-sourced cell has an actual Excel hyperlink (not a plain-text URL string)
- [ ] **ALL CAPS** section headers: **bold** and **underlined**
- [ ] **Col C** display labels: **one** naming pattern sheet-wide; **prefer** em dash
      or hyphen over **parentheses** for qualifiers; abbreviations only when
      **industry-standard** (e.g. ARPU), not opaque codes (e.g. BOC)
- [ ] Col A (deepKPI metric name) and Col B (index) are hidden

## Common failure modes (Excel)

- **Extra Source rows**: filing identity lives in the **cell hyperlink** only.
- **Green font on hyperlinked values** — use **dark blue** `FF0563C1` and the **percent
  format** (or accounting + blue for non-percent) so links look like links **without**
  relying on underline.
- **Text-only date headers** — use **numeric** Excel dates + `mmm-yy` / year format.
- **Text-stored time series values** — numeric KPI points must be **Excel numbers**,
  not strings; use number format for display, not text that mimics a number.
- **Quarterly columns before annual** — **annuals left**, blank, **quarterlies right**.
- **Hardcoded derived numbers** — if a cell is computed from other cells, it must
  be a **formula**, not a pasted constant (even if "correct").
- **Parenthetical KPI labels** (`Name (modifier)`) or **opaque abbreviations** in
  Col C when **dash-separated** names or **spelled-out** words would be clearer.
- **Extra row for imputed Q4** — imputation belongs **in the quarterly series row**,
  not a second row for the same stream.
- **Two rows for quarterly vs annual of the same metric** — if definition and unit
  match and only aggregation differs (e.g. `… - 3 Month` vs `… - Annual`), use **one
  row** with FY values in the annual columns and quarterlies in the quarterly columns.
