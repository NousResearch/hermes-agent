# Update-First Skill Library Model

## Core lesson

When a session produces a reusable lesson, the default action should be to improve the skill that already owns the class of work, not to create a new narrow top-level skill. A flat list of one-session skills makes future agents load the wrong thing, miss duplicated lessons, and create conflicting policy.

## Target shape

Use three levels:

1. **Router / umbrella skills**
   - Class-level owners.
   - Decide which leaf skill applies.
   - Carry shared policy, quality bar, routing rules, and anti-duplication guidance.

2. **Leaf / executor skills**
   - One distinct execution surface.
   - Exact commands, setup, tool quirks, verification, rollback.
   - Avoid repeating broad policy already owned by the umbrella.

3. **Support files**
   - `references/` — session detail, audit notes, postmortems, API excerpts, domain notes, condensed knowledge banks.
   - `templates/` — starter artifacts meant to be copied and modified.
   - `scripts/` — deterministic probes, verification scripts, fixture generators, or other actions the agent should run rather than hand-type.

## New-skill gate

Before creating a top-level skill:

1. Search/list the active skills for the nearest owner.
2. If the new learning is a pitfall, verifier, workflow correction, user preference, command nuance, or setup step: patch the owner SKILL.md.
3. If the new learning is long or session-specific: add a concise reference file under the owner and add a pointer in SKILL.md.
4. If it is a reusable starter artifact: add a template.
5. If it is a deterministic action: add a script.
6. Create a new skill only when the trigger is truly distinct, the tool surface is different, and no existing class-level owner fits.

## Naming rule

Good names describe a durable class of work:

- `shipping-rails`
- `agent-delegation-routing`
- `github-workflow`
- `creative-router`
- `openclaw-operations`

Bad names encode a session artifact:

- `fix-agents-30`
- `debug-today-skill-duplication`
- `audit-skills-may-26`
- `pr-123-review`
- an exact transient error string

If the name only makes sense for today's conversation, use `references/` under an existing owner.

## Consolidation review pattern

Phased approach:

1. Baseline inventory with hashes; no deletion.
2. Separate install-shadow duplicates from conceptual duplicates.
3. Reconcile local-vs-repo divergence before overwriting anything.
4. Consolidate the highest-overlap cluster first.
5. Add router/owner metadata before deleting anything.
6. Deprecate only when `absorbed_into` is explicit and references/jobs are checked.

## Common overlap clusters

- Shipping/planning/debugging: consolidate policy into one umbrella; keep executors like TDD and debugging separate.
- Agent delegation: one routing owner; keep `claude-code`, `codex`, and similar tool-specific skills narrow.
- GitHub: one workflow router; keep PR, issues, auth, and repo management leaves.
- Creative: one router; keep output-medium leaves; consolidate family variants into references when possible.
- OpenClaw: keep high-risk guard skills separate; move architecture/postmortem detail into references under class-level owners.

## Red flags

- New skill repeats a loaded/current skill's workflow section.
- New skill name includes today's issue number, PR number, error string, or codename.
- A long transcript is placed in SKILL.md instead of `references/`.
- Two skills both claim to be the mandatory first action for the same trigger without an umbrella resolving precedence.
- A skill is deleted without `absorbed_into` and reference/job scan.
