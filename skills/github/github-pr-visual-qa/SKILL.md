---
name: github-pr-visual-qa
description: "Use when verifying the visual/UI state of a web app for a specific PR or deploy — finding the preview/live URL from PR metadata, screenshotting changed pages, checking console/network errors, and comparing rendered UI against what the diff changed, reporting only evidence-backed findings."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [qa, visual, browser, pull-requests, deploy, preview, regression]
    related_skills: [dogfood, github-pr-workflow, github-auth]
---

# PR / Deploy Visual QA

## Overview

After a PR opens or a deploy lands, the question is narrow and specific: *does the
changed UI actually render correctly on the deployed app?* This skill answers it with
a **diff-driven, evidence-only** workflow:

1. Find the preview/live URL from the PR's own metadata (don't guess it).
2. Open the pages the diff touched and capture screenshots + console + network.
3. Compare what rendered against what the diff *says* should have changed.
4. Report findings where every visual claim is backed by a capture.

It deliberately scopes to *this change* — unlike the broader exploratory sweep in the
[[dogfood]] skill. Use the two together: this skill for targeted post-PR regression,
[[dogfood]] for whole-app exploration. It reuses [[github-pr-workflow]] conventions for
auth and owner/repo extraction, and [[dogfood]]'s `references/issue-taxonomy.md` for
severity.

## When to Use

- "QA this PR", "does the preview look right?", "verify the deploy", "screenshot the
  changes", "check the staging URL for #1234".
- After CI is green and a preview/deploy is available, before merge/sign-off.

**Don't use for:**
- Whole-site exploratory QA with no specific change in mind → use [[dogfood]].
- Pure functional/unit testing with no rendered UI → run the test suite instead.
- Performance regressions specifically → that's a benchmarking concern, not this.

## The Iron Rule: never assert visual state without evidence

LLMs hallucinate UI. This skill exists to prevent that. Bind every visual claim to a
capture taken *this run*:

- **No screenshot / DOM snapshot → no claim.** Say "not verified", never "looks fine".
- **The URL must load before you describe it.** A guessed URL pattern that 404s, or a
  stale cached screenshot, is worse than saying nothing.
- **"Expected" comes from the diff, not from imagination.** If the diff changes a label
  to "Save changes", that string is the expectation — quote it.
- **Distinguish "I saw X" from "X should be there."** The report's Expected-vs-Observed
  table forces this separation.
- **Be explicit about what you could not reach** (auth wall, feature flag, missing seed
  data) instead of silently skipping it.

## Workflow

### Phase 1 — Identify the change surface

Find out *what* changed so you know what to look at. From the PR branch:

```bash
# Files touched by the PR (vs the base branch)
git diff --name-only main...HEAD

# The actual changes — read these to extract concrete expectations:
#   new/changed copy strings, new routes, renamed components, toggled states.
git diff main...HEAD -- '*.tsx' '*.jsx' '*.ts' '*.js' '*.vue' '*.html' '*.css'
```

Map changed files → user-facing pages/routes. A change to `src/pages/settings/Billing.tsx`
means: go look at `/settings/billing`. Write down 3–8 **concrete, checkable expectations**
(e.g. *"the Billing tab is now visible in Settings"*, *"primary button label reads
'Save changes'"*). These become the rows of the report's Expected-vs-Observed table.

### Phase 2 — Locate the preview/live URL (don't guess)

Run the helper — it reads the PR's deployment statuses, commit statuses, check runs, and
bot comments, and prints tagged candidate URLs:

```bash
bash skills/github/github-pr-visual-qa/scripts/find-preview-url.sh           # current branch
bash skills/github/github-pr-visual-qa/scripts/find-preview-url.sh 1234       # by PR number
bash skills/github/github-pr-visual-qa/scripts/find-preview-url.sh --sha <SHA>
```

It uses `gh` when authenticated, else falls back to `GITHUB_TOKEN` + `curl` (same
detection as [[github-pr-workflow]]). Pick the best candidate by source trust:

| Source | Meaning | Trust |
|--------|---------|-------|
| `deployment` | GitHub Deployment `environment_url` | highest — the platform's own record |
| `status` / `check-run` | Provider posted a `target_url` (Vercel, Netlify, …) | high |
| `pr-comment` | URL in a bot comment | medium — confirm it's the deploy, not docs |

If nothing is found: **ask the user for the URL** or use a known staging/live URL. Never
invent a URL pattern (`pr-1234.preview.app.com`) and report on it as if confirmed.

**Then confirm the URL actually serves the app before trusting it:**

```bash
curl -sS -o /dev/null -w "%{http_code} %{url_effective}\n" -L "<candidate_url>"
```

A non-2xx/3xx code, an SSO/login wall, or a "deployment building" page means you are NOT
looking at the change yet — note it as BLOCKED rather than proceeding.

### Phase 3 — Capture evidence on changed pages

Use the browser toolset (full mechanics in [[dogfood]]). For each changed page:

```text
browser_navigate(url="<verified_url>/settings/billing")
browser_console(clear=true)          # baseline; capture again after interactions
browser_vision(question="Does the Billing tab render? Read the primary button label verbatim.", annotate=true)
browser_snapshot()                   # DOM/accessibility tree — ground-truth for copy/state
```

Capture, for every changed page:
- A **screenshot** (save the returned `screenshot_path`).
- **Console output** after load and after each interaction — uncaught errors are top findings.
- **Network failures** — note any 4xx/5xx for the page's own requests (visible via the
  browser's network panel / failed-request reporting).
- The **verbatim rendered copy/state** for each expectation from Phase 1 (read it off the
  snapshot or vision answer — do not paraphrase from memory).

If the browser toolset isn't available, fall back to a headless screenshot + HTTP check
and say so in the report — partial evidence labeled as partial, not dressed up as full QA.

### Phase 4 — Compare expected vs. observed

For each Phase 1 expectation, fill one row: Expectation (from diff) · Observed (from
capture) · Match ✅/❌/➖ · Evidence path. A ❌ where the diff intended the change is a
regression; a ➖ means you couldn't verify — state the blocker, don't assume pass.

### Phase 5 — Report

Fill `templates/visual-qa-report-template.md`. Required: the verdict line, the
Expected-vs-Observed table, console/network captures, issues (severity per
[[dogfood]]'s `references/issue-taxonomy.md`), and the honesty notes listing anything
unreachable or inferred. When showing screenshots to the user, inline them with
`MEDIA:<screenshot_path>`.

Verdict scale: **PASS** (all expectations met, no new errors) · **PASS-WITH-NITS** (cosmetic
only) · **FAIL** (a changed area is broken/regressed) · **BLOCKED** (couldn't reach the
change — say why).

## Common Pitfalls

1. **Guessing the preview URL.** Run the finder; if it's empty, ask. A confidently-reported
   404 is the worst outcome.
2. **Reporting on a stale or "building" deploy.** Confirm HTTP 200 *and* that content
   rendered for *this* SHA before describing anything.
3. **Describing UI from the diff instead of the screen.** The diff is the *expectation*;
   the screenshot is the *observation*. Never collapse the two.
4. **Skipping the console.** Silent JS errors on a changed page are high-severity and
   easy to miss without an explicit `browser_console()` call.
5. **Silent skips.** An auth wall or feature flag that hides the change must appear in the
   honesty notes as BLOCKED/not-verified, not be omitted.
6. **Scope creep into full-site QA.** Stay on the changed surface; route broad sweeps to
   [[dogfood]].
7. **Paraphrasing copy.** Quote rendered strings verbatim from the snapshot — a near-match
   ("Save" vs "Save changes") is exactly the kind of regression this skill should catch.

## Verification Checklist

- [ ] Identified changed files and wrote concrete, checkable expectations from the diff
- [ ] Found the preview/live URL from PR metadata (or got it from the user) — not guessed
- [ ] Confirmed the URL returns 2xx and renders the app for this SHA
- [ ] Captured screenshot + console + network for every changed page
- [ ] Filled the Expected-vs-Observed table; every ✅ has an evidence path
- [ ] Every visual claim is backed by a capture; unreachable areas marked not-verified
- [ ] Report saved from the template with a verdict and honesty notes
