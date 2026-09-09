<!--
Copy to  <drive>:\logs\<TOPIC>_<YYYY-MM-DD>.md  at the end of any session that
changed tracked files or this ledger (see README.md "Agent conduct").

The first four lines below are a REQUIRED machine-readable header — the
completeness check in scripts/collect-logs.{ps1,sh} (via
scripts/lib/report_completeness.py) matches every RUN- id in the ledger to a
report, and REPORT-MANIFEST.md is the index it uses. After writing this file,
add a row to logs/ledger/reports/REPORT-MANIFEST.md.
-->

# Session report — <one-line topic> (`RUN-YYYY-MM-DD-NNN`)

Run: RUN-YYYY-MM-DD-NNN
Covers: CHG-YYYY-MM-DD-NNN, ERR-YYYY-MM-DD-NNN, DECISION-YYYY-MM-DD-NNN   <!-- or "—" -->
Source-Of-Truth: in-bundle   <!-- or an absolute path / URL if this file is a redacted mirror of a fuller report kept elsewhere; or "ledger-only" if the run legitimately produced no report -->

**Agent:** <who>, in `<repo path>`
**Date:** YYYY-MM-DD, local (<utc offset>)
**Version cut:** `NF-vX.Y.Z` (<MAJOR|MINOR|PATCH>) — <pushed | committed local-only | not committed>
**Ledger:** <the CHG-/ERR-/DECISION- ids this run created or closed, one clause each>

---

## What this pass was

<2–4 sentences: the task as given, and whether it landed cleanly.>

## <numbered section per distinct change>

<what changed, why, the affected paths, and how it was verified.>

## Verification

- <git state: branch, ahead/behind origin, working tree clean?>
- <tests run and their result — name the suites, give the counts>
- <anything deliberately NOT run, and why>

## For the owner / North Forge GPT

1. <open items, follow-ups, decisions still owed>
