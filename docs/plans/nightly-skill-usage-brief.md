# Nightly skill usage brief

## Goal

Add a small local helper that turns Hermes skill usage telemetry into a reviewable maintenance brief without reading or printing skill bodies.

## Why

Joe wants Hermes to compound via reusable skills, but skill sprawl becomes noise if stale, unused, or unpinned procedures accumulate. A deterministic brief lets a cron job surface skill-maintenance actions only when there is something to review.

## Scope

- Create `scripts/skill_usage_brief.py` using only the Python standard library.
- Read `skills/.usage.json` and discover installed `SKILL.md` files under a configurable skills directory.
- Report active, unpinned skills that are stale or never used; optionally include missing usage telemetry for deeper audits.
- Support Markdown and JSON output.
- Support exact `[SILENT]` output when there are no attention items.
- Do not read or print skill bodies beyond frontmatter metadata needed for name/description/provenance.

## Non-goals

- No automatic archiving, pruning, deletion, or skill mutation.
- No network calls.
- No direct integration into the curator scheduler.

## Verification

- Unit tests cover stale/unused/missing-usage classification, pinned/archive suppression, JSON output, and exact `[SILENT]` behavior.
- Smoke compile the script and test file.
