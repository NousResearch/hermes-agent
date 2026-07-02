# Hermes Interactive Pilot Plan

## Goal

Build Claudio's interactive Hermes harness from a fresh repo, using `yallaplay-wiki/` as the central knowledge bin and keeping Slack/scheduler out of scope until the CLI experience is proven.

## Success Criteria

- Hermes can answer interactive questions using the correct domain skill.
- Durable knowledge lookup starts from `yallaplay-wiki/`.
- Read-only tool wrappers work for the first migrated capabilities.
- Safety gates are explicit before any teammate-visible or production write.
- The pilot can be abandoned without changing the existing Claudio bot/scheduler repo.

## Phase 0 - Repo Baseline

- Fresh git repo with no inherited app code.
- `yallaplay-wiki/` added as a submodule.
- `.hermes.md` defines project routing and hard safety rules.
- `SOUL.md` defines Claudio's identity for the lab profile.
- Domain skills exist as skeletons.

## Phase 1 - Interactive Knowledge

- Implement `tools/wiki_search.py`.
- Add templates for findings, definitions, methodology, queries, and schema notes.
- Test prompts that require prior knowledge before any live tool call.

## Phase 2 - Read-Only Tools

Port or wrap the safest existing interactive capabilities first:

1. Snowflake read-only query workflow.
2. Embrace read-only observability: Metrics API, per-user dashboard sessions, and native MCP profile config for aggregate drilldowns.
3. App Insights read-only backend observability: Log Analytics KQL over the shared dev/prod workspace with explicit prod and PC-test-client guards.
4. Safest LiveOps read wrappers: backend config fetch/history/snapshots before live account or support surfaces.
5. Repo/C# source search.
6. Bedrock model listing and provider health checks.
7. Meeting/local collaboration read workflows.

Avoid writes in this phase.

## Phase 3 - Comparison Run

Use known questions from current Claudio workflows and compare:

- Correct skill routing.
- Tool-call reliability.
- Final answer quality.
- Safety behavior.
- Latency/cost/model fit.
- Knowledge capture quality.

## Phase 4 - Promotion Decision

Only after the interactive CLI pilot is reliable, decide whether to add:

- A production interactive Hermes profile.
- Slack adapter.
- Scheduler/job adapter.
- More write-capable gated tools.

## Non-Goals

- Replacing the existing Slack bot now.
- Replacing the existing scheduler now.
- Migrating all historical code or sessions.
- Creating one Hermes profile per persona.
