# Note Capture Implementation Audit

This audit tracks the current implementation status of the canonical
note-capture and staged projection work against the requested objective.

## Objective

Put the canonical note-capture and projected-storage model in place across repo
and runtime. Keep the work on a separate branch. Require PR review and user
approval. Run full testing. Ensure the projection rules are reviewed and
approved before runtime rollout.

## Current Branch Status

- Current branch: `feature/canonical-note-projection`
- Evidence source: `git branch --show-current`
- Status: complete

## Repo Implementation Status

### Canonical capture logic

- File: `hermes_cli/vault_para_triage.py`
- Status: implemented
- Evidence:
  - canonical capture root helpers exist
  - capture events are written under `.hermes/note-capture/events/`
  - per-store staged projections are written under
    `.hermes/note-capture/staging/<store>/<entry_id>/...`
  - projection status summary is written under
    `.hermes/note-capture/status/latest.json`
  - current code still routes through vault-scoped target strings rather than
    the broader target-registry model now defined in the contract

### Skill and user-facing docs

- File: `optional-skills/productivity/vault-para-triage/SKILL.md`
- Status: updated
- Evidence:
  - describes canonical event logging
  - describes staged downstream projections
  - describes source archive and feedback correction semantics

### Generated website docs

- File:
  `website/docs/user-guide/skills/optional/productivity/productivity-vault-para-triage.md`
- Status: regenerated
- Evidence:
  - generated page matches updated skill description and projection rules

### Reviewable projection rule spec

- File: `docs/note-capture-projection-contract.md`
- Status: written
- Evidence:
  - defines canonical storage
  - defines target classes:
    `obsidian_vault`, `filesystem`
  - defines target lifecycle states:
    `active`, `deferred`, `pending_migration`, `unavailable`
  - defines projected path derivation rule
  - defines trust boundaries
  - explicitly calls out the daily-note exception
  - documents the reorganisation rule

### Target-space review and backlog mapping

- File: `docs/note-capture-target-space-review.md`
- Status: written
- Evidence:
  - maps current known capture and ingress flows to current vault, non-vault,
    and future-target lifecycle states
  - identifies local/runtime materialization flows not yet mapped to reviewed
    live targets
  - identifies backlog candidates such as household-bill capture into a
    household-finance target

### Runtime rollout spec

- Files:
  - `docs/note-capture-runtime-rollout.md`
  - `docs/note-capture-runtime-proposed-patches.md`
  - `docs/note-capture-runtime-event-schema.md`
- Status: written
- Evidence:
  - names exact runtime files to edit
  - provides exact proposed wording for those runtime changes
  - defines corrected runtime event semantics for canonical capture output

## Runtime Implementation Status

- Status: applied
- Evidence:
  - live runtime skill text now describes canonical capture events plus staged
    downstream projections
  - live runtime memory files now describe vault and non-vault destinations as
    downstream projected targets rather than the primary write surface
  - live cron docs now mark daily-note handling as a special-case runtime path
  - live `jobs.json` prompts now distinguish generic note capture from the
    plugin-backed daily-note workflow
- Runtime files updated:
  - `~/HermesData/runtime/hermes-core/skills/note-taking/vault-routing/SKILL.md`
  - `~/HermesData/runtime/hermes-core/memories/MEMORY.md`
  - `~/HermesData/runtime/hermes-core/profiles/orchestrator/memories/MEMORY.md`
  - `~/HermesData/runtime/hermes-core/second-brain-infra/hermes/cron-jobs.md`
  - `~/HermesData/runtime/hermes-core/cron/jobs.json`

## Testing Status

### Completed checks

- `scripts/run_tests.sh tests/hermes_cli/test_vault_para_triage.py tests/plugins/test_vault_triage_feedback_plugin.py`
- `source venv/bin/activate && python website/scripts/generate-skill-docs.py`
- `source venv/bin/activate && python -m py_compile hermes_cli/vault_para_triage.py plugins/vault_triage_feedback/__init__.py optional-skills/productivity/vault-para-triage/scripts/vault_para_triage.py`
- `source venv/bin/activate && python optional-skills/productivity/vault-para-triage/scripts/vault_para_triage.py --help`

### Proven by those checks

- capture and feedback code paths are working for the covered tests
- store path-prefix projection mapping is covered by tests
- projection summary contract and enabled-store reporting are covered by tests
- generated skill docs still build successfully after the source changes
- the new/updated Python entrypoints compile cleanly
- the helper CLI exposes the updated capture-and-projection wording

### Not yet proven

- live runtime behavior after runtime edits
- broader integration behavior across external sync jobs
- PR review gate, because no PR has been opened yet

## Approval Gates

### Still required from user

1. Review the branch diff
2. Approve the change set for PR submission and merge when ready

### Still required after approval

1. Prepare PR for review
2. Obtain review and explicit user approval before merge

## Completion Status

Implementation work is complete. Review and merge gates remain open.

Remaining required work:

- PR review gate has not yet happened
- merge approval has not yet happened
