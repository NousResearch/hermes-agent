# Nightly Side Project Triage Spec

## Goal

Provide a small offline helper that turns Joe's hand-maintained side-project idea notes into a ranked review queue. The helper must not fetch external services or contact anyone; it only reads a Markdown file and prints Markdown or JSON for Joe to review.

## Input contract

- Markdown headings define sections.
- Top-level bullets define candidate ideas.
- Optional leading priority tag: `[high]`, `[medium]`, or `[low]`.
- Optional metadata can appear as `key: value` fragments separated by semicolons, pipes, or em dashes.
- Supported metadata: `effort`, `leverage`, `reuse`, `next`, `status`, `audience`.
- Guidance/safety/template sections are ignored.

## Output contract

- Markdown output includes:
  - total idea count
  - ranked top picks
  - parking lot for blocked/low scoring items
  - missing next-action checklist
- JSON output includes normalized entries and scores for automation.

## Scoring contract

Favor Joe's operating principles: high leverage, low effort, and reuse of existing tools. Penalize blocked items and high-effort ideas. The exact score is an implementation detail, but ordering must be deterministic.

## Verification

- Unit tests cover parsing, scoring/order, guidance-section skipping, missing-next-action reporting, and CLI JSON smoke.
