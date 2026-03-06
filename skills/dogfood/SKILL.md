---
description: Perform agent-driven exploratory QA (Dogfooding) on web applications to find visual, functional, and accessibility bugs.
---

# Dogfood Skill: Agent-Driven Exploratory QA

This skill enables the agent to systematically explore, test, and find issues in web applications, acting as an automated QA tester. It is heavily inspired by Vercel's agent-browser dogfooding workflow (Issue #315).

## Prerequisites

To perform this skill, you must have access to:
1. `browser_*` tools (e.g. `browser_navigate`, `browser_snapshot`, `browser_click`) OR a dedicated `agent-browser` execution flow to interact with the target site.
2. `vision_analyze` tool or equivalent if visual validation is required for layout bugs.
3. Access to write files locally (to save screenshots and generate the final markdown report).

## Goal

To rigorously test a specified web application URL, identify bugs (visual layout errors, functional breaks, console errors, accessibility issues), categorize them by severity, and produce a structured Markdown QA Report.

## Workflow

When asked to "dogfood", "QA test", or "explore" a URL, follow these exact 5 phases:

### Phase 1: Planning & Setup
1. Identify the target URL.
2. If authentication is requested, prepare the login flow.
3. Determine the scope (e.g., "just the landing page" vs. "the entire checkout process").

### Phase 2: Systematic Exploration
Explore the page(s) systematically:
1. **Visual Scan:** Take initial screenshots of different viewport states. Look for overlapping text, broken CSS, or unaligned elements.
2. **Interactive Elements:** Click buttons, open dropdowns, and toggle menus. Verify that the UI responds.
3. **Form Testing:** If present, attempt to submit empty forms to trigger validation errors. Try edge cases (e.g. extremely long strings, invalid emails).
4. **Navigation:** Check header links, footer links, and breadcrumbs. Ensure they lead to valid destinations instead of 404s.

### Phase 3: Evidence Collection
For *every* issue you find:
1. Take a screenshot describing the issue state.
2. Note the precise reproduction steps.
3. Record the exact behavior expected versus what actually occurred.

### Phase 4: Categorization and Severity
Consult the `references/issue-taxonomy.md` file to properly map each issue. Every issue must be assigned:
- A **Category** (Visual, Functional, UX, Accessibility, Console, Content).
- A **Severity** (Critical, High, Medium, Low). 

### Phase 5: Report Generation
1. Read the `templates/dogfood-report-template.md`.
2. Compile all findings into a structured markdown report.
3. Save the report to disk (e.g., `dogfood_report_{date}.md`), embedding local file paths to the screenshots you captured as evidence.

## Constraints & Rules
- **Be Systematic:** Do not randomly click. Have a plan for traversing the page state.
- **Provide Actionable Evidence:** An issue without steps to reproduce or a screenshot is worthless. Always document *how* a human developer can recreate the bug.
- **Do not read source code:** Function as a black-box tester. Your perspective should be that of the end-user.
- **Capture Console Errors if possible:** If your browser tooling supports it, note any JavaScript exceptions or failed network requests (4xx/5xx).
