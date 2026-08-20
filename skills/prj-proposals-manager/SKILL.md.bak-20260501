---
name: proposal-management
description: Manage proposal lifecycle from intake to delivery across coordinating agents or roles. Use when the user asks to create, update, track, accept, or close a proposal, or when handling PRD confirmation, technical review, acceptance, or revision workflows. Works with any agent platform (Cursor, Hermes, OpenClaw, etc.)
---

# Proposal Management

## Overview

A platform-agnostic skill for managing proposal lifecycles across multi-role workflows (e.g. coordinator / PM / dev). Covers intake, clarification, PRD confirmation, technical review, development handoff, acceptance, and delivery.

## Configuration

Current environment configuration (for Hermes):

| Variable | Value | Description |
|----------|-------|-------------|
| `PROPOSALS_ROOT` | `~/.hermes/proposals` | Directory holding proposal index and files |
| `TEMPLATES_DIR` | `~/.hermes/proposals/templates` | Subdirectory for templates |
| `PM_OUTPUT_DIR` | `~/.hermes/proposals/workspace-pm/proposals` | Where PM stores PRD documents |
| `DEV_OUTPUT_DIR` | `~/.hermes/proposals/workspace-dev/proposals` | Where dev stores project artifacts |
| `TEST_OUTPUT_DIR` | `~/.hermes/proposals/workspace-test/proposals` | Where Test Expert stores test cases and results |
| `RESEARCH_OUTPUT_DIR` | `~/.hermes/proposals/workspace-research/proposals` | Where Research Analyst stores iteration research reports |
| `PROPOSAL_DOCS_INDEX` | `~/.hermes/proposals/proposal-docs-index.md` | Index of all PRD and technical solution documents |

These values are hardcoded for the current Hermes environment. Do not ask — use them directly.

## Proposal ID Format

`P-YYYYMMDD-XXX` — zero-padded sequential number per day.

To determine the next ID, read `proposal-index.md` and find the highest `XXX` for today's date.

## Proposal States

Use these exact names across all roles — no custom variants:

```
intake → clarifying → prd_pending_confirmation → approved_for_dev → in_dev → in_acceptance → accepted → delivered
                                                                                ↓
                                                                          needs_revision → in_dev
```

## Project Docs Directory Convention

Every project under `${DEV_OUTPUT_DIR}/<project-slug>/` must have a `docs/` subdirectory containing all proposal-related documentation and a local index. This keeps historical context self-contained within each project.

```
${DEV_OUTPUT_DIR}/<project-slug>/
├── docs/
│   ├── index.md              # Local document index (version history)
│   ├── proposal.md           # Original proposal intake document
│   ├── prd.v1.md            # PRD v1 (versioned)
│   ├── prd.v2.md            # PRD v2 (if revised)
│   ├── technical-solution.v1.md  # Technical solution v1 (versioned)
│   └── technical-solution.v2.md  # Technical solution v2 (if revised)
└── (project source files)
```

### docs/index.md Format

```markdown
# P-YYYYMMDD-XXX: <Title> — Documents

## Proposal

| Version | File | Updated |
|---------|------|---------|

## PRD

| Version | File | Updated | Notes |
|---------|------|---------|-------|

## Technical Solution

| Version | File | Updated | Notes |
|---------|------|---------|-------|
```

**Rules:**
- `docs/proposal.md` is the single source of truth for the original proposal (never versioned)
- `prd.md` is a symlink or copy of the current PRD version for convenience
- `technical-solution.md` is a symlink or copy of the current technical solution for convenience
- Versioned files use `.vN.md` suffix (v1, v2, ...)
- When a new version is created, update `docs/index.md` with the new entry
- The index is the single source of truth for version history within the project

## Proposal Docs Index

Maintain a single source of truth at `${PROPOSAL_DOCS_INDEX}` (`~/.hermes/proposals/proposal-docs-index.md`) that maps every proposal to its current and historical documents.

### proposal-docs-index.md Format

```markdown
# Proposal Documents Index

## P-YYYYMMDD-XXX: <Title>

| Document | Path | Version | Updated |
|----------|------|---------|---------|
| Proposal | `workspace-dev/proposals/<slug>/docs/proposal.md` | - | YYYY-MM-DD |
| PRD | `workspace-dev/proposals/<slug>/docs/prd.v1.md` | v1.0 | YYYY-MM-DD |
| Technical Solution | `workspace-dev/proposals/<slug>/docs/technical-solution.v1.md` | v1.0 | YYYY-MM-DD |

---

## P-YYYYMMDD-YYY: <Title>
...
```

**Index maintenance rules:**
- Create the index entry when the proposal is registered
- Update the version and date whenever a document is revised
- Do NOT copy full documents into the index — the index only tracks paths and versions
- The index provides a quick overview of all proposals and their document versions without needing to open each project

### Historical PRD References

When a proposal evolves (e.g., scope change, revision), historical PRD versions should be preserved:

- Store old PRD versions with version suffix: `prd.v1.md`, `prd.v2.md`
- The index tracks the current version and previous versions
- When requesting PRD confirmation, include reference to the previous version if this is a revision cycle

## Workflow

```
- [ ] Step 1a/1b: Intake – register proposal (from existing codebase OR new)
- [ ] Step 2: Clarify – up to 3 rounds
- [ ] Step 3: Route to PM if needed
- [ ] Step 4: PRD confirmation gate
- [ ] Step 5: Technical expectations gate (up to 3 rounds)
- [ ] Step 6: Output technical solution
- [ ] Step 7: Hand off to dev
- [ ] Step 8: Acceptance review
- [ ] Step 9: Deliver or revise
```

### Step 1a: Register from Existing Codebase

When the request is to clone an existing GitHub repo and register it as a proposal (rather than building from scratch):

1. **Clone the repo** to `${DEV_OUTPUT_DIR}/<slug>/`
   ```bash
   git clone https://<token>@github.com/<owner>/<repo>.git ${DEV_OUTPUT_DIR}/<slug>/
   ```
   Use token `ghp_XXXXX` (YeLuo45)

2. **Create project docs** under `${DEV_OUTPUT_DIR}/<slug>/docs/`:
   - `index.md` — local document index
   - Any existing README.md should be in Chinese; if English, replace content

3. **Update `proposal-index.md`** — add entry with status `delivered` (already built), `in_dev` (if still developing), or `accepted` (if already accepted)

4. **Update proposals-manager website** via GitHub API:
   ```python
   # GET current file + SHA
   GET https://api.github.com/repos/YeLuo45/proposals-manager/contents/data/proposals.json
   
   # PUT new content with SHA (triggers GitHub Actions rebuild)
   PUT https://api.github.com/repos/YeLuo45/proposals-manager/contents/data/proposals.json
   body: { "message": "feat: add P-YYYYMMDD-XXX <name>", "content": <base64>, "sha": <sha> }
   ```

5. **Sync to `proposals-document` repo** — commit updated project docs to `project-docs/<slug>/` in the proposals-document repo

6. **Fix GitHub repo description** to Chinese via PATCH API:
   ```python
   PATCH https://api.github.com/repos/YeLuo45/<repo>
   body: { "description": "<Chinese description>", "homepage": "<pages-url>" }
   ```

### Step 1b: Register New Proposal (from scratch)

1. Read `${PROPOSALS_ROOT}/proposal-index.md` to determine the next ID
2. Copy `${TEMPLATES_DIR}/request-intake-template.md` to `${PROPOSALS_ROOT}/P-YYYYMMDD-XXX.md`
3. Fill in Basic Information and Original Request
4. Add an entry in `proposal-index.md` under Active Proposals with status `intake`
5. Add an entry in `${PROPOSAL_DOCS_INDEX}` for this proposal
6. Create `${DEV_OUTPUT_DIR}/<slug>/docs/index.md` with the initial index structure

### Step 2: Clarify Requirements

- Ask the requester up to 3 rounds of clarifying questions focused on: goal, scope, constraints, acceptance criteria
- Record each round in the proposal file under Clarification
- After 3 rounds or when clear, record Final Assumptions
- Update status to `clarifying`

### Step 3: Route to PM

If the request is an idea or rough draft, hand off to the PM role for PRD generation.

- PM saves PRD to `${PM_OUTPUT_DIR}/YYYY-MM-DD-<slug>-prd.md`
- PM also copies the PRD as `${DEV_OUTPUT_DIR}/<slug>/docs/prd.v1.md`
- Update `PRD Path` in `proposal-index.md` once PM delivers
- Update both `${PROPOSAL_DOCS_INDEX}` and `${DEV_OUTPUT_DIR}/<slug>/docs/index.md` with the new PRD path and version

### Step 4: PRD Confirmation Gate

When PM returns the PRD:

1. Present the PRD to the requester and ask for confirmation
2. Start a confirmation timeout (recommended: 5 minutes)
3. Record the timeout reference in `PRD Confirmation Countdown ID`

**If confirmed**: set `PRD Confirmation` to `confirmed`, cancel the timeout

**If timeout**: set `PRD Confirmation` to `timeout-approved`, record in `Timeout Resolution`

#### Timeout Implementation by Platform

| Platform | Method |
|----------|--------|
| Hermes | Use `cron` with `schedule` as ISO timestamp (e.g. `schedule: "2026-04-16T12:43:00+08:00"`), or track manually with timestamps in proposal file |
| OpenClaw | `cron` with `schedule.kind="at"`, `atMs=<now+300000>`, `payload.kind="systemEvent"` |
| Cursor | Use the countdown-manager skill if available, or track manually with timestamps |
| Other | Record a deadline timestamp and check on next interaction |

**Hermes cron timeout example:**
```python
cron(action='create', schedule='2026-04-16T12:43:00+08:00', prompt='Check proposal P-YYYYMMDD-XXX PRD confirmation timeout', name='prd-timeout-P-YYYYMMDD-XXX')
```

### Step 5: Technical Expectations Gate

Before outputting a technical solution:

1. Ask the requester about: stack, performance, cost, deployment, maintainability, dependency constraints
2. Up to 3 rounds of questions
3. Start a confirmation timeout (same mechanism as Step 4)
4. Record in `Technical Expectations Countdown ID`

**If confirmed**: set `Technical Expectations` to `confirmed`, write constraints to `Technical Assumptions Summary`

**If timeout**: set `Technical Expectations` to `timeout-approved`, proceed with current assumptions, record in `Timeout Resolution`

### Step 6: Technical Solution

- Output the technical solution at `${PROPOSALS_ROOT}/P-YYYYMMDD-XXX-tech-solution.md`
- Also copy to `${DEV_OUTPUT_DIR}/<slug>/docs/technical-solution.v1.md`
- Update status to `approved_for_dev`
- Update both `${PROPOSAL_DOCS_INDEX}` and `${DEV_OUTPUT_DIR}/<slug>/docs/index.md` with the new technical solution path and version

### Step 7: Hand Off to Dev

- Update status to `in_dev`
- Dev creates `${DEV_OUTPUT_DIR}/<slug>/docs/` directory if not exists (should already exist from Step 1)
- Dev saves project output to `${DEV_OUTPUT_DIR}/<slug>/`
- The `docs/` directory must contain `index.md`, `proposal.md`, `prd.vN.md` (if exists), and `technical-solution.vN.md`
- Update `Project Path` in `proposal-index.md`

### Step 8: Acceptance Review

When dev reports completion, verify all of the following:

**Requirements consistency:**
- Matches requester-confirmed requirements
- Aligns with PRD
- No scope creep or shortcuts

**Functional verification (hands-on, not screenshots only):**
- Core functionality works end-to-end
- No errors in console/logs (warnings OK)
- Existing functionality not broken
- Build succeeds (e.g. `npm run build`, `cargo build`, `go build`)

**Delivery completeness:**
- File paths provided
- Startup/access instructions provided
- Verification results or screenshots provided

**Quality:**
- No obvious gaps
- No UI/logic conflicts
- Known limitations documented

Update status to `in_acceptance` during review.

### Step 9: Deliver or Revise

**If accepted**: update status to `accepted` or `delivered`, report to requester

**If not accepted**: update status to `needs_revision`, output structured revision notes:

```markdown
## Revision Notes

- **Issue**: <description>
- **Impact**: <what is affected>
- **Expected fix**: <how to fix and how to verify>
```

Record revision notes in `proposal-index.md` Notes field.

## Dev Delivery Quality Checks

Three hard indicators to verify before accepting:

1. **Build exit code**: must be 0
2. **Output directory not empty**: list core files to confirm
3. **Core source/service files exist**: verify key files are present

If dev claims completion without providing evidence, run the checks yourself.

### Takeover Triggers

The coordinator should take over from dev when:
- Dev fails delivery twice consecutively
- Dev session interrupted by API/quota errors
- Dev session abnormally short (< 30s) yet claims completion
- Fix is simple and clearly identified

### Recording Fixes

When the coordinator directly fixes issues, record in:
1. Project memory file (e.g. `MEMORY.md`) under relevant section
2. Daily log (e.g. `memory/YYYY-MM-DD.md`)
3. Proposal's `Notes` or `Main Fixes Applied` field

## Index Entry Template

When adding to `proposal-index.md`:

```markdown
### P-YYYYMMDD-XXX: <Title>

- `Proposal ID`: `P-YYYYMMDD-XXX`
- `Title`: <title>
- `Owner`: <coordinator>
- `Current Status`: `intake`
- `PRD Path`: (to be filled by PM)
- `Technical Solution`: (to be filled)
- `Project Path`: (to be filled by dev)
- `Acceptance`: -
- `PRD Confirmation`: pending
- `PRD Confirmation Countdown ID`: -
- `Technical Expectations`: pending
- `Technical Expectations Countdown ID`: -
- `Last Update`: YYYY-MM-DD
- `Notes`:
```

## Templates

This skill expects three templates in `${TEMPLATES_DIR}/`:

| Template | Purpose |
|----------|---------|
| `request-intake-template.md` | Initial proposal registration with clarification fields and confirmation gates |
| `proposal-status-template.md` | Status tracking with linked assets, confirmation gates, and revision notes |
| `acceptance-checklist-template.md` | Structured acceptance review with functional/quality/delivery checks |

If templates do not exist at the expected path, create them based on the index entry template above and the acceptance review checklist in Step 8.
