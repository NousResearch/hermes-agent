# Post-Deploy Visual QA — PR #{pr_number}

**Target URL:** {verified_url}
**URL source:** {url_source}  <!-- deployment status / commit status / check-run / bot comment / user-provided -->
**Commit:** {sha}
**Date:** {date}
**Scope:** changes in this PR ({files_changed_count} files)

> Every visual claim below is backed by a screenshot or console/network capture taken
> during this run. Where evidence is missing, the row says **"not verified"** — it does
> not assert a passing state.

---

## Verdict

**Ship-readiness:** {PASS | PASS-WITH-NITS | FAIL | BLOCKED}
**One-line:** {one_sentence}

| Check | Result | Evidence |
|-------|--------|----------|
| Preview URL loads (HTTP 200, content rendered) | {pass/fail/not verified} | MEDIA:{screenshot} |
| No console errors on changed pages | {pass/fail/not verified} | {count} errors — see below |
| No failed network requests (4xx/5xx) | {pass/fail/not verified} | {count} failures |
| Changed UI copy/states render as expected | {pass/fail/not verified} | per-expectation table |

---

## Expected vs. Observed (diff-driven)

<!-- One row per concrete, checkable expectation derived from the PR diff. -->
<!-- "Expected" must come from the code/copy in the diff — NOT from assumption. -->

| # | Expectation (from diff) | Observed | Match? | Evidence |
|---|-------------------------|----------|--------|----------|
| 1 | {e.g. button label reads "Save changes"} | {what the screenshot/DOM actually showed} | ✅/❌/➖ | MEDIA:{path} |
| 2 | {e.g. /settings renders new "Billing" tab} | {observed} | ✅/❌/➖ | MEDIA:{path} |

Legend: ✅ matches · ❌ mismatch (regression) · ➖ could not verify (state the blocker)

---

## Console & Network

**Console errors:**
```
{console_output_or "none captured"}
```

**Failed network requests:**
```
{method url -> status, or "none captured"}
```

---

## Issues Found

<!-- Repeat per issue. Severity per dogfood references/issue-taxonomy.md. -->

### Issue #{n}: {title}
| Field | Value |
|-------|-------|
| Severity | {Critical/High/Medium/Low} |
| Category | {Functional/Visual/Console/Network/Content/UX} |
| Page | {url_path} |
| Introduced by this PR? | {yes/likely/pre-existing/unknown} |

**Steps to reproduce:** {steps}
**Expected:** {expected}
**Actual:** {actual}
**Evidence:** MEDIA:{screenshot}

---

## Coverage & Honesty Notes

- **Pages checked:** {list}
- **Changed areas NOT reachable** (and why): {list — auth wall, feature flag, no seed data, etc.}
- **Anything inferred without direct evidence:** {state it explicitly, or "none — all claims have evidence"}
