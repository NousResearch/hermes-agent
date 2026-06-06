# Hermes Nous + Hermes Lab Migration Review Plan

Generated: 2026-05-24  
Scope: read-only audit pack for migrating legacy Hermes Nous and Hermes Lab into Hermes Agent.

## Non-Negotiable Rules

- Do not delete either legacy folder until the deletion readiness checklist is approved by the user.
- Do not migrate `.env`, token seeds, keys, credentials, or secret-bearing files into docs, Obsidian, prompts, context packs, or skills.
- Do not bulk-copy legacy folders into Hermes Agent. Every migrated item needs a target, lineage, and review status.
- Treat employee, owner, finance, client, and Lark-export material as permission-scoped knowledge, not general agent memory.
- Runtime behavior must be ported into Hermes Agent-native profiles, skills, tools, or config. Historical runtime files are not automatically active.

## Audit Inputs

| Source | Path | Files | Bytes |
|---|---|---:|---:|
| Hermes Nous | `/Users/rattanasak/Documents/Viber Project/Tech Tools/_archived-hermes-20260524/HermesNous.retired-20260524` | 21,165 | 1,503,175,959 |
| Hermes Lab | `/Users/rattanasak/Documents/Viber Project/Tech Tools/_archived-hermes-20260524/Hermes Labs.retired-20260524` | 6,011 | 110,006,690 |
| Total | both sources | 27,176 | 1,613,182,649 |

Full file-level inventory with SHA-256 hashes is in `migration-manifest.json`.

## Classification Summary

| Category | Files | Meaning | Default Decision |
|---|---:|---|---|
| dependency-or-generated | 25,961 | `node_modules`, virtualenvs, build outputs, generated package internals | ignore/archive; recreate from lockfiles only |
| runtime-candidate | 513 | scripts, source, config, modules, templates, skills | review and port selected behavior |
| knowledge-candidate | 449 | docs, knowledge, memory, lessons, playbooks, patterns, reports | review and import/promote |
| archive-review | 134 | uncategorized historical files | human review |
| secret-review | 100 | path/name indicates secrets, credentials, tokens, or auth-sensitive code | security inventory only |
| logs-or-cache | 19 | logs, cache, incident traces | archive only if evidence is needed |

`secret-review` includes false positives from dependency names such as `tokenizer.py`, but the rule remains conservative: nothing in this class is prompt-visible until Security Boundary Lead clears it.

## Workstreams

| Workstream | Lead Role | Main Inputs | Output |
|---|---|---|---|
| Inventory Integrity | Migration Orchestrator | `migration-manifest.json` | source coverage report, missing-source list |
| Security Boundary | Security Boundary Lead | secret-review, employee, owner, finance, Lark exports | redaction plan, private-scope map |
| Synerry Business Brain | Synerry Business Strategist | MD Assist, HermesNous knowledge, playbooks, pitching flow, TOR analyzer | Synerry operating model and pitch workflow packs |
| Knowledge Promotion | Knowledge Curator | knowledge, lessons, patterns, playbooks, reports | reviewed Obsidian notes and domain KB entries |
| Runtime Port | Runtime Skill Engineer | skills, scripts, modules, config, templates | Hermes Agent skills/tools/profiles only where useful |
| QA Acceptance | QA Verification Lead | migrated packs, code paths, context loader | tests, smoke checks, live scenario checklist |
| Sunset | Sunset Lead | approved manifest, archive state, rollback needs | deletion readiness approval |
| WOW Review Surface | WOW Operator Designer | migration status, risks, workflow maps | review dashboard/spec for fast human decisions |

## Review Phases

### Phase 0: Freeze and Preserve

- Confirm both legacy roots still exist.
- Confirm `migration-manifest.json` hash coverage matches source file counts.
- Do not run formatters, package installs, or migration scripts against legacy roots.
- Record old-folder sizes and top-level directories.

Exit gate: user confirms these are the final source folders.

### Phase 1: Security Triage

- Review all `secret-review` records.
- Split into:
  - actual secret/config requiring secure archive
  - dependency false positive
  - auth/security code worth porting conceptually
- Never move actual secret values into Hermes Agent docs.
- For `.env` files, only migrate key names and setup requirements if still needed.

Exit gate: `secret-review` count is explained and no secret content is included in generated knowledge.

### Phase 2: Knowledge Triage

- Review HermesNous `knowledge`, `lessons`, `patterns`, `playbooks`, `reports`, `sources`, `lark-catalog`, `review-queue`.
- Review Hermes Lab `memory/team`, root reports, handoffs, project specs, and rules.
- Classify each candidate as:
  - `import-to-obsidian`
  - `promote-to-domain`
  - `convert-to-playbook`
  - `convert-to-skill`
  - `archive-only`
  - `reject-stale`

Exit gate: every knowledge candidate has an owner, target, and review state.

### Phase 3: Synerry and Pitching Layer

- Use MD Assist as source of truth for Synerry/company/owner context.
- Do not re-ask questions already captured there.
- Build the following business workflows as first-class knowledge/workflow packs:
  - TOR Intake and Go/No-Go
  - Pitch War Room
  - Proposal Builder
  - Case Study Writer
  - Market Research Pack
  - Finance/Margin Check
  - Post-Pitch Lessons

Exit gate: a real Synerry pitch scenario can route through the workflow without relying on HermesNous runtime.

### Phase 4: Runtime Port

- Review legacy skills/scripts/modules.
- Port only behavior that remains valuable.
- Use Hermes Agent-native destinations:
  - `.hermes/skills`
  - `.hermes/profiles`
  - `tools/`
  - `agent/`
  - `docs/hermes-agent-standalone`
  - `~/ObsidianVault/HermesAgent`
- Do not activate legacy runtime files directly.

Exit gate: selected runtime behavior has tests or smoke verification.

### Phase 5: Acceptance Scenarios

Minimum scenarios before deletion:

1. Ask Hermes Agent for Synerry company/owner context and verify it uses migrated knowledge.
2. Run a TOR analysis scenario.
3. Run a Pitch War Room scenario.
4. Run a competitive/market research scenario for proposal support.
5. Ask for project status/context from a migrated project card.
6. Ask for an old incident/lesson and verify source lineage appears.
7. Run `scripts/run_tests.sh tests/knowledge_center/ -q`.

Exit gate: all scenarios pass and user approves deletion readiness.

## Current Findings That Need Attention

- Hermes Agent Knowledge Center code exists and current verification target is
  `scripts/run_tests.sh tests/knowledge_center/ -q`.
- `~/ObsidianVault/HermesAgent/projects` has 40 project notes with `domain:` frontmatter.
- The restored graph superseded the short-lived standalone `domains/` layout.
  New promoted domain knowledge should land under
  `~/ObsidianVault/HermesAgent/knowledge/domain/<domain>/`; legacy `domains/`
  remains supported only when it already exists.
- HermesNous has the highest-value knowledge materials: `patterns`, `lessons`, `playbooks`, `knowledge`, `lark-catalog`, `review-queue`.
- Hermes Lab has the highest-value runtime and governance materials: `modules`, `src`, `scripts`, `memory/team`, `config`, handoff/spec reports.

## Approval Gates

| Gate | Required Approval |
|---|---|
| A | Source folders and manifest are accepted as complete |
| B | Secret/security triage accepted |
| C | Knowledge target map accepted |
| D | Synerry workflow map accepted |
| E | Runtime port list accepted |
| F | Tests and live scenarios accepted |
| G | Deletion readiness checklist accepted |
