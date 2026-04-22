# Agentic Cron Orchestration Kit

Turn Hermes into a recurring operator instead of a session-bound helper.

## Outcome
Target ship proof for this week: verify that in under 30 minutes you can:
1. install or point at Hermes
2. copy these starter files
3. create four recurring jobs
4. keep one project moving automatically with daily and weekly review loops

Until the clean-room timed run is recorded, treat the under-30-minute setup as a target claim to validate, not a shipped guarantee.

## Who this is for
- technical founders
- AI operators
- solo builders
- anyone tired of manually re-prompting agents to keep work moving

## Canonical starter workflow
This kit ships one opinionated workflow:
- **Monday kickoff** — pick and lock the week’s priority
- **Daily CEO review** — identify blocker, next move, and execution order
- **Evening doc sync** — update notes/checklists so state is durable
- **Friday ship review** — force a launch decision and package artifacts

## Folder layout
- `prompts/` — self-contained job prompts
- `templates/` — note and checklist templates
- `scripts/` — local preflight script for the canonical setup path

## Fast start
### 1. Run local preflight
```bash
bash scripts/preflight.sh
```

### 2. Create or update your project notes
Copy the templates from `templates/` into your project notes location or Obsidian vault.

Recommended notes:
- Weekly Factory note
- Current week pipeline note
- per-project CEO note
- ship checklist

### 3. Create the four starter jobs
Use the prompts in `prompts/` as the job instructions.

Important: the prompt files are workflow templates, not zero-context magic. Before a fresh run, inject the exact paths for:
- your factory note
- your current-week pipeline note
- your active CEO note
- your ship checklist
- the repo/workspace the loop should inspect

The clean-room proof run on 2026-04-17 verified that one recurring evening-doc-sync workflow could be scheduled and run from fresh notes context in **1.74 minutes**, but only after those exact note/workspace paths were supplied explicitly.

Recommended schedules:
- weekly kickoff: `0 9 * * 1`
- daily CEO review: `0 9 * * 2-5`
- evening doc sync: `0 18 * * 1-5`
- Friday ship review: `0 15 * * 5`

### 4. Verify one loop
Run one job immediately and confirm it:
- reads the right notes
- updates the notes directly
- outputs a concise operator update
- leaves the next critical move obvious

## Suggested cron creation examples
These are example job shapes. Adjust paths and deliver target.

### Weekly kickoff
- **name:** Weekly MVP Kickoff
- **schedule:** `0 9 * * 1`
- **deliver:** `origin`
- **prompt source:** `prompts/weekly-kickoff.md`

### Daily CEO review
- **name:** Daily CEO Review
- **schedule:** `0 9 * * 2-5`
- **deliver:** `origin`
- **prompt source:** `prompts/daily-ceo-review.md`

### Evening doc sync
- **name:** Evening Documentation Sync
- **schedule:** `0 18 * * 1-5`
- **deliver:** `local`
- **prompt source:** `prompts/evening-doc-sync.md`

### Friday ship review
- **name:** Friday Ship Review
- **schedule:** `0 15 * * 5`
- **deliver:** `origin`
- **prompt source:** `prompts/friday-ship-review.md`

## Definition of done for setup
Setup is done when:
- Hermes is reachable
- the four starter prompts exist
- notes/checklists exist in durable storage
- at least one job has been run manually
- one recurring job is scheduled successfully

## Scope guardrails
This kit is intentionally not:
- a dashboard
- a multi-tenant control plane
- a universal workflow framework
- a replacement for product judgment

It is a starter system for keeping one operator and one agent moving continuously.
