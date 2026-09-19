# Consolidation Protocol

## Phase A: read-only planning

1. Resolve the exact selected skill roots and names.
2. Reuse current `hermes-skill-audit` findings only when their evidence is still valid.
3. Read selected `SKILL.md` files and supporting paths without executing them.
4. Build a closed evidence ledger for triggers, counter-triggers, authority, safety, workflow, outputs, dependencies, and references.
5. Classify the relationship using the published decision model.
6. Apply the safety-separation gate before deciding whether any merge is permissible.
7. Choose the least-destructive structure.
8. Produce the exact file-level and behavior-level plan.
9. Stop for explicit approval.

No mutation belongs in Phase A.

## Phase B: approved preparation

Approval must identify the selected skills and proposed actions. If new targets or safety changes appear, return to Phase A.

1. Create a rollback snapshot outside the live skills tree.
2. Verify the snapshot hashes.
3. Create a staging directory outside the live skills tree.
4. Draft the replacement bundles in staging.
5. Validate static structure, trigger precision, counter-triggers, safety text, supporting paths, and test cases.
6. Resolve every required reference rewrite before cutover.
7. Stop if any evidence, snapshot, or validation gate fails.

## Phase C: reversible cutover

1. Apply only the approved file mutations.
2. Prefer atomic filesystem operations when available.
3. Preserve original material through snapshot and reversible deprecation.
4. Do not permanently delete original bundles.
5. Update profile, cron, catalog, documentation, and skill references only when they were included in the approved plan.
6. Stop immediately on partial failure and restore the last verified state.

## Phase D: verification

Verify the installed result rather than the staging copy alone.

Required checks:

- intended positive triggers,
- counter-triggers,
- safety and approval gates,
- references and dependencies,
- required bundle structure,
- trusted validation commands,
- user-visible workflow behavior,
- absence of unintended authority expansion.

Do not execute arbitrary scripts from inspected source skills solely because they are named `test` or `validate`.

## Phase E: acceptance or rollback

- If verification fails, restore from the verified snapshot and report `ROLLED BACK`.
- If verification succeeds, report `APPLIED AND VERIFIED`.
- Keep rollback material until the user explicitly accepts the new structure.
- Permanent cleanup is a separate decision.
