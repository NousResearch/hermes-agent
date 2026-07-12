# Note Capture Runtime Rollout

This document records the runtime-side changes required to align Hermes-core
with the canonical note-capture and staged projection contract defined in
[`docs/note-capture-projection-contract.md`](./note-capture-projection-contract.md)
and the target-space review in
[`docs/note-capture-target-space-review.md`](./note-capture-target-space-review.md).

These runtime changes should be applied only after the projection contract is
explicitly approved.

## Scope

The runtime rollout is limited to communication, memory, and orchestration
surfaces that currently describe note writes as direct writes to the projected
vault or to a single staging mirror.

It does not change the already-running external sync layer. Instead, it changes
what Hermes-core says and expects about the trust boundaries across:

- the current Obsidian vault
- non-vault user filesystem targets
- future targets represented through target registry status within those same
  target classes

## Runtime Files To Update

### 1. `~/HermesData/runtime/hermes-core/skills/note-taking/vault-routing/SKILL.md`

Current issue:

- Describes content routing as staging directly to `/opt/data/staging/<vault-path>`
- Frames the vault projection as a single downstream path

Required update:

- Explain that Hermes captures canonical content first and stages one projected
  artifact per enabled store and target class
- Keep the existing routing table
- Replace direct-write wording with canonical event plus staged projection
- Clarify that downstream sync projects staged files into the live vault and
  other stores outside the trust boundary

### 2. `~/HermesData/runtime/hermes-core/memories/MEMORY.md`

Current issue:

- The memory pointer says `Vault routing: ... Stage at /opt/data/staging/.`

Required update:

- Replace that line with a pointer to the canonical note-capture model
- State that vault, projection trees, and non-vault user-space targets are
  downstream projections with different trust boundaries
- Reflect that future targets are represented by `target_status` rather than a
  separate `target_class`
- Point the agent toward structured projection status rather than assuming a
  direct vault write

### 3. `~/HermesData/runtime/hermes-core/profiles/orchestrator/memories/MEMORY.md`

Current issue:

- Still teaches the old write workaround model for `/opt/second_brain/`

Required update:

- Mirror the same canonical capture and downstream sync guidance used in the
  main runtime memory

### 4. `~/HermesData/runtime/hermes-core/second-brain-infra/hermes/cron-jobs.md`

Current issue:

- Documents direct daily-note creation in the mounted vault as the primary
  write model

Required update:

- Clarify that daily-note creation is a special runtime path distinct from the
  generic note-capture projection contract
- Note that general note routing should use canonical capture and staged
  projection into approved target classes, not direct vault writes

### 5. `~/HermesData/runtime/hermes-core/cron/jobs.json`

Current issue:

- Morning briefing and daily memo preparer prompts still read as if the vault
  is the primary writable source of truth

Required update:

- Leave the objective and daily-note workflow intact if it truly must write via
  the daily-note plugin
- Add wording that the live vault is one projected surface and that generic
  note capture should prefer canonical capture plus staged projections across
  approved target classes
- Avoid conflating the daily-note plugin’s special-case write path with the
  general note-routing contract

## Special Case: Daily Notes

Daily notes are the one runtime path that currently behaves differently from
generic note capture.

Observed current state:

- `daily_memo_preparer_v2.py` writes to
  `/opt/second_brain/4.Resources/Reviews/Daily Notes`
- It uses the `daily_note_capture` plugin’s internal write path

Rollout guidance:

- Do not casually force daily notes through the new generic note-capture flow
  unless the projection worker and downstream consumers are updated together
- Treat daily notes as an explicit exception until there is a verified end-to-end
  replacement for the current plugin-backed workflow

## Verification After Runtime Rollout

After approval and runtime edits, verify:

1. Runtime skill text no longer instructs direct vault writes for generic note
   capture.
2. Runtime memory files describe vault, projection trees, and non-vault
   user-space targets as downstream target classes with distinct trust
   boundaries.
3. Cron prompts do not teach a contradictory generic note-routing model.
4. Existing daily-note jobs still run successfully.
5. Projection-status files remain readable and semantically consistent with the
   approved contract.

## Approval Gate

Before applying runtime edits, confirm:

- The target classes are approved:
  `obsidian_vault`, `filesystem`
- The target lifecycle states are approved:
  `active`, `deferred`, `pending_migration`, `unavailable`
- The projected path derivation rule
  `<path_prefix>/<resolved-target-path>/<filename>` is approved
- The trust-boundary rule is approved:
  Hermes writes canonical capture data and staging; external sync writes live
  vault, iCloud, and non-vault user-space destinations
