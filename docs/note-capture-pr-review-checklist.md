# Note Capture PR Review Checklist

Use this checklist when the canonical note-capture and staged projection work
is ready for PR review.

## Scope Check

- Confirm the branch is `feature/canonical-note-projection`
- Confirm the PR includes repo-side canonical capture logic
- Confirm runtime edits are only included after projection-rule approval
- Confirm daily-note behavior is either unchanged or explicitly documented as
  a temporary exception

## Contract Check

- Review `docs/note-capture-projection-contract.md`
- Review `docs/note-capture-target-space-review.md`
- Confirm the approved target classes are:
  `obsidian_vault`, `filesystem`
- Confirm the approved target lifecycle states are:
  `active`, `deferred`, `pending_migration`, `unavailable`
- Confirm the projected path derivation rule is:
  `<path_prefix>/<resolved-target-path>/<filename>`
- Confirm the trust boundary rule is:
  Hermes writes canonical capture data and staged projections; external sync
  writes live vault, iCloud, non-vault user-space targets, and other
  downstream stores
- Confirm the daily-note exception is explicit and not accidental
- Confirm the reorganisation rule preserves canonical target identities even if
  live paths change

## Repo Code Check

- Review `hermes_cli/vault_para_triage.py`
- Verify capture events are written under `.hermes/note-capture/events/`
- Verify staged projections are written under
  `.hermes/note-capture/staging/<store>/<entry_id>/...`
- Verify source notes are archived under
  `.hermes/note-capture/source-archive/<entry_id>/...`
- Verify feedback corrections rewrite staged projections rather than pretending
  to mutate downstream live stores directly

## Docs Check

- Review `optional-skills/productivity/vault-para-triage/SKILL.md`
- Review `website/docs/user-guide/skills/optional/productivity/productivity-vault-para-triage.md`
- Review `docs/note-capture-runtime-rollout.md`
- Review `docs/note-capture-runtime-proposed-patches.md`
- Review `docs/note-capture-implementation-audit.md`
- Review `docs/note-capture-target-space-review.md`

## Runtime Check

After runtime edits are applied:

- Review `~/HermesData/runtime/hermes-core/skills/note-taking/vault-routing/SKILL.md`
- Review `~/HermesData/runtime/hermes-core/memories/MEMORY.md`
- Review `~/HermesData/runtime/hermes-core/profiles/orchestrator/memories/MEMORY.md`
- Review `~/HermesData/runtime/hermes-core/second-brain-infra/hermes/cron-jobs.md`
- Review `~/HermesData/runtime/hermes-core/cron/jobs.json`
- Confirm those files no longer teach direct vault writes for generic captured
  notes

## Test Evidence

- Confirm focused tests passed:
  `tests/hermes_cli/test_vault_para_triage.py`
- Confirm focused tests passed:
  `tests/plugins/test_vault_triage_feedback_plugin.py`
- Confirm docs generation succeeded:
  `source venv/bin/activate && python website/scripts/generate-skill-docs.py`
- Confirm Python compile smoke checks succeeded:
  `source venv/bin/activate && python -m py_compile ...`
- Confirm helper CLI smoke check succeeded:
  `source venv/bin/activate && python optional-skills/productivity/vault-para-triage/scripts/vault_para_triage.py --help`

## Approval Gates

- Reviewer approves the repo-side contract and implementation
- User approves the projection rules
- User approves the runtime wording changes
- Runtime verification passes after rollout
- Only then is the change ready to merge
