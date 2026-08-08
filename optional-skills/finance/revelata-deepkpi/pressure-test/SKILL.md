---
name: pressure-test
description: >
  Pressure-test any analyst report on a US public company. Upload a PDF of a sell-side
  report and say "Pressure test this" — the skill extracts the analyst's key arguments,
  pulls granular SEC filing data via deepKPI to build supporting and counter-evidence
  for each claim, then produces an interactive HTML report with Revelata dark branding,
  Chart.js visualizations (blue palette), source attribution (amber = analyst, cyan = SEC),
  and clickable provenance hyperlinks to the exact filing passages. Use whenever the user
  mentions "pressure test", "pressure-test", "critically analyze", "triangulate", "fact-check",
  or "challenge" in the context of an analyst report, research note, or investment thesis.
  Also triggers when the user uploads a PDF that looks like a sell-side report and asks
version: 1.0.0
author: Revelata
license: Apache-2.0
homepage: https://www.revelata.com
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Revelata, Finance, SEC, KPI, deepKPI, HTML, Analyst]
    category: finance
    related_skills: [revelata, kpi, company-summary, filing, implied-metric, seasonality, benchmark, ideas]
---

# pressure-test

Take an analyst report and rigorously test every argument against SEC filing data.
The output is an interactive HTML report that a portfolio manager can open in a browser
and use to form an independent view — every number is clickable back to its source.

## Data source

See `_common/data-source-rules.md` for the runtime routing table and the canonical
inventory of deepKPI operations.

## Hard stop if deepKPI access fails

See `_common/connection-failure.md`. STOP and ask before falling back.

## When this skill triggers

- User says "pressure test this" (or similar) with an uploaded analyst report
- User uploads a sell-side PDF and asks for critical analysis, counterpoints, or fact-checking
- User asks to "triangulate" or "challenge" an investment thesis

## Chat pacing and verbosity

Keep the **conversation** thin; put the depth in the **HTML report**.

- After reading the PDF, state the **4–6 main arguments** you will pressure-test
  (short titles or one line each — this is the roadmap for the user).
- Treat each request as self-contained: **do not reference, compare to, or borrow from any other analyst reports** you may have analyzed earlier in the same chat/session unless the user explicitly asks you to do so. Assume you are pressure-testing **only the report(s) uploaded with the current request**.
- Also state you'll pull **market reaction context** (price on report date vs today),
  then move straight into data pulls.
- Do **not** narrate every tool call, intermediate table, or reasoning step in chat.
  Avoid dumping raw deepKPI responses into the thread.
- While pulling data, use a **brief progress line per argument**, e.g.
  `Pulling SEC data for 2/5: [short argument title]…` then continue silently until
  the next argument or until the HTML is ready.
- **Automatically build and save the HTML report** when pulls and drafting are
  complete. Do **not** ask whether to generate the file unless the user explicitly
  asked for a non-HTML deliverable only.
- **Before presenting the HTML**, run the **pressure-test double-check** pass
  (see below). In chat, a single line is enough, e.g. `Running pressure-test QA on the draft HTML…`
- After the artifact is **double-checked** and any **Critical** issues are
  addressed (or explicitly accepted by you as the authoring agent): **present**
  the file (`present_files` or equivalent) and add **1–2 sentences** on the
  sharpest takeaway — no full recap.

## High-level workflow

1. **Read the report** — extract the analyst's 4-6 key arguments/claims
2. **Identify the company** — resolve the ticker/name to a CIK via deepKPI
3. **Pull SEC data** — for each argument, search deepKPI for the most granular
   metrics that can confirm or complicate the claim
4. **Build evidence pairs** — for each argument, write a "supports the thesis"
   card and a "complicates the thesis" card, grounded in specific numbers
5. **Write the synthesis** — 2 paragraphs that weave the evidence into a nuanced
   take the analyst didn't give you
6. **Generate the HTML** — an interactive, dark-themed report with Chart.js charts,
   source attribution, and provenance links (**default deliverable — produce it without asking**)
7. **Double-check the draft HTML** — mandatory QA pass before the user sees the file;
   structured review only (see **Step 7**)
8. **Revise and iterate** — you decide which findings to fix; re-run the double-check
   after material edits until **Overall: Pass** or remaining issues are only **Suggested** / **Optional**
9. **Deliver** — present the final HTML to the user (**short wrap-up only**)

Each of these steps is detailed below. Follow **Chat pacing and verbosity** for what to say in-thread vs. what belongs only in the HTML.

## Dependencies

This skill relies on the `kpi` skill for **KPI** retrieval and on the `filing`
skill when you need **verbatim SEC filing passages** or **full filing markdown**
(not just structured metrics). Consult the `kpi` skill for metrics workflow,
provenance, and gaps; consult the `filing` skill when claims must be checked
against exact filing language. Use the `company-summary` skill for **business
description**, **segment structure**, or **thematic** lists when helpful (not
required for every run). Use the `implied-metric` skill when filling Q4 or
segment gaps. This skill handles the analytical framework and HTML output.

For the mandatory QA rubric on the finished HTML, read **`references/double-check.md`**
(internal to this skill — not a separate user-facing skill).

## Scope

This skill works for **US public companies** with SEC filings indexed in deepKPI.
If the report covers a non-US company or a private company, tell the user that
deepKPI doesn't have coverage and offer to do a qualitative analysis instead
(without the data-backed evidence cards and provenance links).

---

## Step 1: Read the analyst report

Read the uploaded PDF. Extract:

- **Company name and ticker**
- **Report metadata**: author/firm, date, rating action (upgrade/downgrade/initiation),
  price target if applicable. Store the firm name internally for source pills and claim
  banners, but **do not surface the firm name in the footer or synthesis**.
- **4-6 key arguments** the analyst makes. These are the claims you'll pressure-test.
  Look for: revenue/growth commentary, margin trends, segment performance, balance
  sheet / leverage, competitive positioning, specific catalysts or risks.

Write each argument as a 1-2 sentence summary. You'll use these as the section
headers in the final report. **In chat**, share this list once as the identified
main arguments; do not pre-write full evidence cards in the thread.

## Step 1.5: Market reaction context (price move since publication)

Get the stock **close price** on:

- **Report date** (use the report's publication date; if only a date is given, use that day's close)
- **Current date** (most recent close)

Compute percent move: \(\\Delta\\% = (P_\\text{now}/P_\\text{report} - 1)\\times 100\\).

### Reporting rules (keep concise)

- Add a small **\"Price since report\"** box near the top of the HTML with the two closes and the \(\\Delta\\%\\).
- Keep the commentary **blunt and non-narrative**: factual finance language only.
- If price data cannot be retrieved reliably in the current environment, state that in the HTML box and continue with the SEC pressure test.

## Step 2: Resolve the company in deepKPI

Use the `kpi` skill workflow:

1. Call `query_company_id` with the company's official registrant name (not brand)
2. Call `list_kpis` (free) to see every available metric
3. Scan the KPI list to plan your searches — map each analyst argument to the
   most granular KPIs available

## Step 6: Generate the HTML report

Read `references/html-template.md` in this folder for the complete HTML/CSS/JS template.

The report structure includes:

- Header link label: **\"Analysis Pressure Test (GitHub)\"**
- Title: \"[Company] ([Ticker]): Pressure-Testing the [Firm] [Action]\"

## Step 7: Double-check the draft HTML (mandatory)

After the draft HTML file is written (saved path known), you **must** run the
**pressure-test double-check** before `present_files` / showing it to the user.

1. **Read** `references/double-check.md` (internal QA reference for this skill).
2. **Execute the procedure** in that document against the draft HTML:
   - If your host can invoke a **separate reviewer agent**, give it that doc plus
     the HTML path (and the analyst PDF path when available) and collect its output.
   - Otherwise, **perform the same pass yourself** in a distinct step: read the HTML
     end-to-end, build the claim graph, run rules **R1–R12**, and emit the **exact
     output format** specified there (structured review — **do not rewrite** the
     HTML inside the double-check output).
3. Pass the **source PDF path** into the double-check whenever you still have it;
   otherwise R1 and R8 must appear under **`Not checked`** in the review.

**You** (the authoring agent) **own revisions**: use the review's severity-ranked
findings to patch the HTML. Typical loop:

- Fix all **Critical** before delivery. **Fail** overall → treat as blocking until
  addressed or the user explicitly waives QA (only if they say so).
- **Needs Revision** → fix Critical, then re-run Step 7 on the updated file.
- **Pass** with only **Suggested** / **Optional** → you may still polish, then deliver.

Re-running Step 7 after substantive edits is **recommended** so regressions do not
ship.

---

## Reference files

| File | When to read |
|------|-------------|
| `references/html-template.md` | Always — contains the full HTML/CSS/JS template |
| `references/chart-patterns.md` | When building charts — palette, dash patterns, examples |
| `references/double-check.md` | **Always after draft HTML exists** — mandatory internal QA rubric and output format |
