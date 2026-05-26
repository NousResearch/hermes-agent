# Divergent Skill Reconciliation

Use when active user-local skills differ from repo/bundled source skills and the goal is to consolidate without losing local lessons.

## Core pattern

1. Treat the active runtime skill tree as production state. Do not overwrite it from repo by filename.
2. Create/verify backups before any destructive action.
3. Generate a machine-readable inventory with name, path, hash, source layer, and validation errors.
4. For each divergent skill, diff repo/source vs active local and classify every local hunk:
   - **Generic promote:** useful to Hermes users generally; move to repo SKILL.md or repo references.
   - **MJ-local preserve:** user/profile/host/provider/client-specific; keep local or in a local reference.
   - **Repo preserve:** repo has generic content missing locally; keep it during merge.
   - **Needs split:** mixed generic + local; extract generic shell, keep specifics local.
   - **Needs verification:** documentation claims an env var/config/feature exists; prove it in source before promoting.
   - **Stale/cleanup later:** removable only after reference scan and explicit approval.
5. Write a decision artifact before editing skills. The artifact should include paths, diff summary, classification, recommendation, stop conditions, and proof gates.
6. Promote the lowest-risk generic changes first. Avoid starting with the largest/highest-blast-radius umbrella.
7. After each batch, rerun inventory and owner-routing checks.

## Phase 2A example pattern

For system-critical divergent skills, a safe order is:

1. Clean generic promote (`hermes-agent-skill-authoring`-style update-first guidance).
2. Small docs hunks requiring source verification (`kanban-worker` env/config details).
3. Generic bridge/process docs with local details split out (`linear` + local rules file).
4. Large operational umbrella split into references (`hermes-agent`).

## Stop conditions

Stop before editing if:

- The proposed edit would remove a local-only lesson.
- A repo promotion would expose private/user-specific profile, provider, client, or host details.
- A reference path named by the skill does not exist.
- A claimed runtime env var/config key cannot be found in source.
- Exact duplicate removal or loader behavior changes become necessary.
