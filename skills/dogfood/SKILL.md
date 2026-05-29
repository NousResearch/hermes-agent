---
name: dogfood
description: "Exploratory QA of web apps: find bugs, evidence, reports."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [qa, testing, browser, web, dogfood]
    related_skills: []
---

# Dogfood: Systematic Web Application QA Testing

## Overview

This skill guides you through systematic exploratory QA testing of web applications using the browser toolset. You will navigate the application, interact with elements, capture evidence of issues, and produce a structured bug report.

## Prerequisites

- Browser toolset must be available (`browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_vision`, `browser_console`, `browser_scroll`, `browser_back`, `browser_press`)
- A target URL and testing scope from the user

## Inputs

The user provides:
1. **Target URL** — the entry point for testing
2. **Scope** — what areas/features to focus on (or "full site" for comprehensive testing)
3. **Output directory** (optional) — where to save screenshots and the report (default: `./dogfood-output`)

## Workflow

Follow this 6-phase systematic workflow:

### Phase 1: Plan

1. Create the output directory structure:
   ```
   {output_dir}/
   ├── screenshots/       # Evidence screenshots
   └── report.md          # Final report (generated in Phase 5)
   ```
2. Identify the testing scope based on user input.
3. Build a rough sitemap by planning which pages and features to test:
   - Landing/home page
   - Navigation links (header, footer, sidebar)
   - Key user flows (sign up, login, search, checkout, etc.)
   - Forms and interactive elements
   - Edge cases (empty states, error pages, 404s)

### Phase 2: Explore

For each page or feature in your plan:

1. **Navigate** to the page:
   ```
   browser_navigate(url="https://example.com/page")
   ```

2. **Take a snapshot** to understand the DOM structure:
   ```
   browser_snapshot()
   ```

3. **Check the console** for JavaScript errors and failed network requests:
   ```
   browser_console(clear=true)
   ```
   Do this after every navigation and after every significant interaction. Silent JS errors are high-value findings. Capture two distinct kinds of evidence from the console output and keep them separate when recording an issue:
   - **Console evidence** — uncaught exceptions, unhandled promise rejections, warnings.
   - **Network evidence** — failed requests surfaced in the console (4xx/5xx responses, CORS failures, mixed-content blocks). Note the request URL and status code.
   Copy the exact text verbatim; you will paste it into the issue's evidence fields.

4. **Take an annotated screenshot** to visually assess the page and identify interactive elements:
   ```
   browser_vision(question="Describe the page layout, identify any visual issues, broken elements, or accessibility concerns", annotate=true)
   ```
   The `annotate=true` flag overlays numbered `[N]` labels on interactive elements. Each `[N]` maps to ref `@eN` for subsequent browser commands.

5. **Test interactive elements** systematically:
   - Click buttons and links: `browser_click(ref="@eN")`
   - Fill forms: `browser_type(ref="@eN", text="test input")`
   - Test keyboard navigation: `browser_press(key="Tab")`, `browser_press(key="Enter")`
   - Scroll through content: `browser_scroll(direction="down")`
   - Test form validation with invalid inputs
   - Test empty submissions

6. **After each interaction**, check for:
   - Console errors: `browser_console()`
   - Visual changes: `browser_vision(question="What changed after the interaction?")`
   - Expected vs actual behavior

### Phase 3: Collect Evidence

For every issue found:

1. **Take a screenshot** showing the issue:
   ```
   browser_vision(question="Capture and describe the issue visible on this page", annotate=false)
   ```
   Save the `screenshot_path` from the response — you will reference it in the report.

2. **Record the details** — capture every field the report template needs so it can be filled without guessing later:
   - URL where the issue occurs
   - Steps to reproduce (numbered)
   - Expected behavior
   - Actual behavior
   - **Console evidence** — verbatim console errors, or "None observed"
   - **Network evidence** — failed requests with URL and status code, or "None observed"
   - Screenshot path (from the `screenshot_path` in the `browser_vision` response)

3. **Classify the issue** using the issue taxonomy (see `references/issue-taxonomy.md`):
   - **Severity:** Critical / High / Medium / Low
   - **Category:** Functional / Visual / Accessibility / Console / UX / Content
   - **Reproduction confidence:** Confirmed (reproduced 2+ times) / Likely (seen once) / Intermittent (could not reproduce reliably). To reach "Confirmed", re-run the reproduction steps at least once.
   - **Suggested owner:** Frontend / Backend/API / Design / Content / Infra — the area most likely to own the root cause.
   - **Next action:** the concrete fix or investigation you recommend.

### Phase 4: Categorize

1. Review all collected issues.
2. De-duplicate — merge issues that are the same bug manifesting in different places.
3. Assign final severity and category to each issue.
4. Sort by severity (Critical first, then High, Medium, Low).
5. Count issues **both** by severity and by category for the breakdown section. The two counts must each sum to the same total.
6. Distill the three-line executive summary: a one-sentence overall **verdict**, the single most important **headline** finding, and the recommended **next action**.

### Phase 5: Report

Generate the final report using the template at `templates/dogfood-report-template.md`.

This report is meant to be readable in messaging platforms, so **use no Markdown tables** — the template expresses everything as labeled fields and bullet lists. The report must include:

1. **Three-line executive summary** — Verdict, Headline, Next action (one sentence each).
2. **Issue breakdown** — counts by severity and by category, as bullet lists (no tables). Both lists sum to the same total.
3. **Per-issue blocks** (no tables, sorted Critical → Low), each with:
   - Issue number and title
   - Severity and category
   - URL where observed
   - Reproduction confidence, suggested owner, and next action
   - Description, numbered steps to reproduce, expected vs actual behavior
   - **Evidence — Screenshot** (`MEDIA:<screenshot_path>` for inline images)
   - **Evidence — Console** (verbatim, or "None observed")
   - **Evidence — Network** (failed requests with status codes, or "None observed")
4. **All issues at a glance** — one bullet per issue.
5. **Coverage matrix** — one bullet per page/feature marked Tested / Partial / Not tested, with flows exercised or reason skipped, plus any blockers.
6. **Artifact inventory** — the report path and every screenshot with a note on what it shows.
7. **Final smoke-check checklist** — the verification gate completed in Phase 6.

Save the report to `{output_dir}/report.md`.

### Phase 6: Verify Completeness

Before finalizing, **explicitly verify the report against the smoke-check checklist** at the bottom of the template. Do not deliver the report until every box can be checked:

- Executive summary is exactly three lines (Verdict / Headline / Next action).
- The severity breakdown and the category breakdown each sum to the same total, and that total matches the number of per-issue blocks.
- Every issue has severity, category, reproduction confidence, suggested owner, and next action.
- Every issue has at least one populated evidence field (screenshot, console, or network).
- Every `MEDIA:` screenshot path points to a file that actually exists in `{output_dir}/screenshots/`.
- The coverage matrix lists both tested and not-tested areas.
- The report contains no Markdown tables.

If any check fails, return to the relevant phase, gather the missing evidence or fix the count, and re-verify. Only then deliver the report.

## Tools Reference

| Tool | Purpose |
|------|---------|
| `browser_navigate` | Go to a URL |
| `browser_snapshot` | Get DOM text snapshot (accessibility tree) |
| `browser_click` | Click an element by ref (`@eN`) or text |
| `browser_type` | Type into an input field |
| `browser_scroll` | Scroll up/down on the page |
| `browser_back` | Go back in browser history |
| `browser_press` | Press a keyboard key |
| `browser_vision` | Screenshot + AI analysis; use `annotate=true` for element labels |
| `browser_console` | Get JS console output and errors |

## Tips

- **Always check `browser_console()` after navigating and after significant interactions.** Silent JS errors are among the most valuable findings.
- **Use `annotate=true` with `browser_vision`** when you need to reason about interactive element positions or when the snapshot refs are unclear.
- **Test with both valid and invalid inputs** — form validation bugs are common.
- **Scroll through long pages** — content below the fold may have rendering issues.
- **Test navigation flows** — click through multi-step processes end-to-end.
- **Check responsive behavior** by noting any layout issues visible in screenshots.
- **Don't forget edge cases**: empty states, very long text, special characters, rapid clicking.
- When reporting screenshots to the user, include `MEDIA:<screenshot_path>` so they can see the evidence inline.
