# Cyber VC Analyst Productization Roadmap

This roadmap packages the existing `cyber-vc-analyst` skill into a practical
Hermes Slack workflow while establishing the shape of a broader private cyber
investment workflow repository.

## Phase 1: Stabilize The Skill Contract

- Keep the Hermes-side capability as a skill, not a new core tool.
- Preserve two primary modes:
  - company analysis
  - theme analysis
- Standardize evidence separation:
  - private vault context
  - Return on Security market intelligence
  - public evidence
- Keep company memo write-back under `4.Resources/Companies`.
- Keep theme memo write-back under `3.Areas/cyber futures frontier`.

## Phase 2: Productize The Slack Workflow

- Prefer `!cyber-vc-analyst ...` in Slack threads.
- Support concise thread summaries with optional vault write-back.
- Add normalized prompt shapes for:
  - company
  - theme
  - compare
  - triage
- Treat long-form memos as durable artifacts and Slack as the operating surface.

## Phase 3: Establish The Private Repo Shape

Recommended top-level directories:

- `docs/`
- `prompts/`
- `schemas/`
- `fixtures/`
- `release-notes/`
- `skills/`

Use the private repo as the system of record for prompts, schemas, fixtures,
and release discipline. Sync the runtime-facing skill into Hermes when needed.

## Phase 4: Add Regression Discipline

- Validate required frontmatter and section order.
- Validate allowed enum values and score ranges.
- Validate evidence-class separation.
- Keep fixtures for:
  - company analysis with ROS MCP
  - theme analysis with ROS MCP
  - degraded market-intelligence path

## Phase 5: Expand The Workflow Surface

Add private workflow modules for:

- company analysis
- thematic analysis
- company comparison
- diligence-gap analysis
- market maps
- meeting-note to company brief conversion
- watchlist updates

## Current Repo Artifacts

This repository now carries starter versions of:

- Slack workflow docs
- private-repo blueprint docs
- company, theme, compare, and triage schemas
- starter fixtures across the main Slack-facing modes
- release notes scaffolding

That is enough to make the current Hermes skill operationally clearer while
leaving the private repo as the natural home for deeper workflow ownership.
