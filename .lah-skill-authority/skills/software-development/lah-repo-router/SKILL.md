---
name: lah-repo-router
description: "LAH Stack schema v4 executable role-aware routing authority."
category: software-development
---

# LAH Repo Router (schema v4)

This is the Git canonical source for the v4 router. The installed Hermes copy is the execution authority, and the Codex copy is derived by the installation contract. The single canonical machine-readable source is:

`lah-repo-router/references/repo_mappings.json`

Executable entrypoint:

```bash
bash scripts/dry-run-route.sh references/repo_mappings.json /tmp/lah-mission.txt
```

Installation and three-way drift checks are performed with `scripts/sync-installations.cjs` and `scripts/validate-installation-drift.cjs`. Installation-local files are never an independent source of routing truth.

The receipt resolves roles, not one guessed repository. Roles:

`IMPLEMENTATION`, `EXECUTION_RUNTIME`, `GOVERNANCE`, `MEMORY`, `CONTEXT`, `SKILL_KNOWLEDGE`, `BUSINESS_ASSET`.

Schema v4 keeps legacy fields for compatibility, but canonical fields are role fields, `write_intents`, `role_evidence`, `ontology_status`, `explicit_target`, and `conflicts`.

Fail closed:

- `AMBIGUOUS`: multiple plausible owners for one required role.
- `UNRESOLVED`: missing or materially contradictory current evidence.
- `BLOCKED`: owner known, requested write forbidden or outside scope.
- `RESOLVED`: every required role and write intent deterministic and policy-valid.

Historical mission prefixes and aliases are contextual signals only. Current implementation topology plus current repo-local ownership contracts decide ownership. Explicit targets are preserved and verified; they never override ownership. CodeGraph may add evidence but cannot convert an unresolved explicit conflict into `RESOLVED`.

`context_repos` never grant write permission. `write_allowed_repos` comes only from explicit role-specific `write_intents`. Different repositories for implementation, execution, governance, and memory are expected, not ambiguous.

Active operational memory authority is `cartelogic-v2`. Archived OpenClaw agent-memory is non-writable. `lah-stack-skills` owns reusable LAH skills, workflows, and shared agent knowledge.

After routing, apply the selected repository's local bootstrap contract. Routing does not replace CodeGraph freshness checks or repo-local safety rules.

Validation:

```bash
node scripts/test-v4-router.cjs
node scripts/validate-routing-drift.cjs
```
