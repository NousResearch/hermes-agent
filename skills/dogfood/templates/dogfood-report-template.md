# Dogfood QA Report

**Target:** {target_url}
**Date:** {date}
**Scope:** {scope_description}
**Tester:** Hermes Agent (automated exploratory QA)

> This report is written for messaging platforms: no Markdown tables, short
> scannable lines, and evidence inline. Keep it that way when filling it in.

---

## Executive Summary

<!--
Exactly three lines. Keep each to one sentence so it reads cleanly in a chat
client. Fill all three even when there are zero issues.
  Line 1 — Verdict: overall health of the app for the tested scope.
  Line 2 — Headline: the single most important finding (or "no blocking issues").
  Line 3 — Next action: the one thing the team should do next.
-->

- **Verdict:** {one_line_overall_health_verdict}
- **Headline:** {one_line_most_important_finding}
- **Next action:** {one_line_recommended_next_step}

---

## Issue Breakdown

**By severity:**
- 🔴 Critical: {critical_count}
- 🟠 High: {high_count}
- 🟡 Medium: {medium_count}
- 🔵 Low: {low_count}
- Total: {total_count}

**By category:**
- Functional: {functional_count}
- Visual: {visual_count}
- Accessibility: {accessibility_count}
- Console: {console_count}
- UX: {ux_count}
- Content: {content_count}

<!-- The two breakdowns must each sum to the same Total. -->

---

## Issues

<!--
Repeat the block below once per issue, sorted Critical → High → Medium → Low.
No tables — use the labeled fields so each issue stays readable in a chat
thread. Every block must have severity, category, reproduction confidence,
suggested owner, next action, and at least one populated evidence field.
-->

### Issue #{issue_number}: {issue_title}

- **Severity:** {severity}
- **Category:** {category}
- **URL:** {url_where_found}
- **Reproduction confidence:** {confidence}
  <!-- Confirmed (reproduced 2+ times) / Likely (reproduced once) / Intermittent (could not reproduce reliably). See references/issue-taxonomy.md. -->
- **Suggested owner:** {owner_area}
  <!-- e.g. Frontend, Backend/API, Design, Content, Infra. See references/issue-taxonomy.md. -->
- **Next action:** {recommended_fix_or_investigation}

**Description:**
{detailed_description_of_the_issue}

**Steps to Reproduce:**
1. {step_1}
2. {step_2}
3. {step_3}

**Expected Behavior:**
{what_should_happen}

**Actual Behavior:**
{what_actually_happens}

**Evidence — Screenshot:**
MEDIA:{screenshot_path}

**Evidence — Console:**
```
{console_error_output_or_"None observed"}
```

**Evidence — Network:**
```
{failed_requests_with_status_codes_or_"None observed"}
```

---

<!-- End of per-issue block -->

## All Issues at a Glance

<!-- One bullet per issue (no table), same order as above. -->
- #{n} [{severity} / {category}] {title} — {url}

---

## Coverage Matrix

<!--
One bullet per page/feature area (this replaces a coverage table). Mark each as
Tested / Partial / Not tested, then note the flows exercised or the reason it
was skipped. List both what was covered and what was not.
-->

- {area_or_page}: {Tested / Partial / Not tested} — {flows_exercised_or_reason_skipped}

**Blockers encountered:**
- {anything_that_prevented_testing_an_area_or_"None"}

---

## Artifact Inventory

<!-- Account for every file produced so reviewers can find the evidence. -->

- Report: {output_dir}/report.md
- Screenshots ({screenshot_count} total) in {output_dir}/screenshots/:
  - {screenshot_filename} — {what_it_shows}
- Other artifacts: {other_files_or_"None"}

---

## Final Smoke-Check Checklist

<!-- Verify every box before sending the report. Do not finalize with any box unchecked. -->

- [ ] Executive summary is exactly three lines (Verdict / Headline / Next action)
- [ ] Severity breakdown and category breakdown each sum to the same Total
- [ ] The Total matches the number of per-issue blocks
- [ ] Every issue has severity, category, reproduction confidence, suggested owner, and next action
- [ ] Every issue has at least one populated evidence field (screenshot, console, or network)
- [ ] Console was checked after each navigation and significant interaction
- [ ] Every `MEDIA:` screenshot path points to a file that exists in the artifacts directory
- [ ] Coverage matrix lists both tested and not-tested areas
- [ ] Report contains no Markdown tables (messaging-platform friendly)

---

## Notes

{any_additional_observations_or_recommendations}
