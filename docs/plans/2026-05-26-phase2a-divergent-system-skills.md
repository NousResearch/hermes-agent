# Phase 2A Divergent System Skills Reconciliation

Scope approved by MJ: start Phase 2A tracking for the highest-risk divergent skills. This file is a decision/proof artifact only. No skill deletion, archive, rename, or behavior-changing merge is approved by this document.

## Scope

Reviewed active user-local skill copies against repo source copies for:

1. `hermes-agent`
2. `hermes-agent-skill-authoring`
3. `linear`
4. `kanban-worker`

Source layers:

- Active runtime: `/Users/alfred/.hermes/skills/.../SKILL.md`
- Repo source: `/Users/alfred/.hermes/hermes-agent/skills/.../SKILL.md`

## Executive decision

Do **not** overwrite local copies with repo copies and do **not** delete local copies yet.

Next implementation should be a controlled promotion/preservation pass:

1. Promote broadly reusable local additions into repo skills or repo references.
2. Preserve MJ/host/profile-specific operational lessons in local skill copies or explicitly local references.
3. Restore/protect generic repo content that active local copies currently lack, especially cross-platform sections.
4. Re-run inventory and routing harness after every batch.

## Classification legend

- **Generic promote:** useful to Hermes users generally; candidate to move into repo source.
- **MJ-local preserve:** specific to MJ, this Mac, his profiles, his provider choices, or OpenClaw adjacency; keep local or in local reference.
- **Repo preserve:** repo has general content absent from local; do not lose it during reconciliation.
- **Needs split:** mixed generic + MJ-local; extract generic shell into repo, keep specifics local.
- **Stale/cleanup later:** likely removable only after proof/reference scan.

---

## 1. `hermes-agent`

- Active: `/Users/alfred/.hermes/skills/autonomous-ai-agents/hermes-agent/SKILL.md`
- Repo: `skills/autonomous-ai-agents/hermes-agent/SKILL.md`
- Diff size: 428 unified diff lines.

### Main differences

Active local adds many Hermes operational lessons:

- Skill consolidation / update-first library design pointer.
- Linear/local-first performance applicability reference.
- Self-evolution / GEPA / DSPy applicability reference.
- External Hermes critique / memsearch / semantic-recall video reference.
- External/cross-agent memory layer reference.
- Default profile alias display caveat.
- Per-profile gateway install recipe using `HERMES_HOME=`.
- Shape-C profile creation recipe.
- Compression / compaction model diagnostics.
- Fallback provider shape and verification rules.
- Anthropic Max OAuth policy change and Max-vs-API routing caveat.
- Provider-routing diagnostics before flipping profile providers.
- Active profile sticky trapdoor reference.
- Post-incident catchup and shipping reference.
- Telegram typing indicator diagnostic note.
- Background process notification spam diagnostic notes.
- macOS storage cleanup reference.
- Hermes profile wiring into external work tracking.
- Session cost/hygiene and `/compress` guidance.
- Agent runtime honesty / between-turn limitations.

Repo source has generic material that active local lacks or replaced:

- `platforms: [linux, macos, windows]` frontmatter.
- Windows-specific quirks section.
- Some contributor/runtime sections that should remain broadly useful.

### Hunk classification

- **Generic promote:**
  - Compression / compaction diagnostics.
  - Fallback providers config shape and verification.
  - Per-profile gateway install with `HERMES_HOME=`.
  - Provider-routing diagnostics before profile/provider flips.
  - Active-profile sticky trapdoor concept.
  - Post-incident catchup audit sequence.
  - Telegram typing indicator lifecycle.
  - Background-process notification spam diagnostics.
  - Session cost/hygiene guidance, if made provider-neutral and not Opus-price-specific.
  - Runtime honesty / between-turn limitation, if phrased generally.

- **MJ-local preserve:**
  - MJ-specific provider preferences and Anthropic Max billing/routing notes.
  - Shape-C profile creation specifically “like Bill Printer / Ruta”.
  - macOS storage cleanup note that names MJ’s host quirks.
  - Any OpenClaw adjacency, Blazer-specific profile, or Telegram-home-channel detail.
  - Cost numbers specific to current provider economics.

- **Repo preserve:**
  - `platforms` frontmatter.
  - Windows-specific quirks.
  - General contributor sections that support cross-platform users.

- **Needs split:**
  - Background notification spam: generic mechanism belongs repo; MJ preference for off/error-only belongs local.
  - Auth pool / Max OAuth: generic Anthropic policy belongs repo if current; MJ wallet routing belongs local.
  - Skills consolidation pointer: generic update-first concept belongs repo/reference; MJ approval/proof artifacts remain plan docs.

### Recommendation

Create a repo reference bundle under `skills/autonomous-ai-agents/hermes-agent/references/` for generic operational lessons, then keep local `hermes-agent` as a thin MJ overlay pointing at local-only references. Do not merge by blind copy.

### Proof gate before editing

- Fresh `skill_view hermes-agent` still shows the trigger pointers.
- Repo `hermes-agent` keeps `platforms` and Windows sections.
- Any moved reference path exists.
- Routing harness still maps Hermes skill-library lessons to `hermes-agent`.

---

## 2. `hermes-agent-skill-authoring`

- Active: `/Users/alfred/.hermes/skills/software-development/hermes-agent-skill-authoring/SKILL.md`
- Repo: `skills/software-development/hermes-agent-skill-authoring/SKILL.md`
- Diff size: 70 unified diff lines.

### Main differences

Active local adds the exact update-first behavior this consolidation effort needs:

- First workflow step becomes: prefer updating an existing owner skill before creating anything new.
- Adds “Update-First Library Shape” section:
  - router / umbrella skills
  - leaf / executor skills
  - support files under `references/`, `templates/`, `scripts/`
- Adds classification rules for new content:
  - pitfall/command/verifier/workflow correction/user preference -> patch owner `SKILL.md`
  - long transcript/audit/API notes/postmortem/evidence -> owner `references/`
  - starter files -> `templates/`
  - deterministic action -> `scripts/`
  - create new skill only with distinct trigger + distinct tools + no owner
- Adds pitfall: avoid one-session-one-skill names.
- Expands duplicate-skill pitfall and renumbers existing pitfalls.

### Hunk classification

- **Generic promote:** all active additions are broadly useful and should go into repo source.
- **Needs split:** the reference pointer `references/update-first-skill-library.md` must either be created in repo or changed to the current plan/reference path.
- **Repo preserve:** existing validator, formatting, and related-skill rules.
- **MJ-local preserve:** none obvious in this diff.

### Recommendation

Promote this active copy into repo source, after ensuring the referenced update-first reference exists. This is the cleanest and lowest-risk Phase 2A merge.

### Implementation proof (2026-05-26)

Implemented first because it was the lowest-risk generic promotion.

- Repo `SKILL.md` updated with the active update-first workflow and pitfalls.
- Repo references created:
  - `skills/software-development/hermes-agent-skill-authoring/references/update-first-skill-library.md`
  - `skills/software-development/hermes-agent-skill-authoring/references/divergent-skill-reconciliation.md`
  - `skills/software-development/hermes-agent-skill-authoring/references/session-end-memory-skill-maintenance.md`
- Stale hardcoded `/home/bb/hermes-agent` examples replaced with `<repo-root>` in both repo and active local copy.
- Inventory after authoring + kanban-worker correction: exact duplicates 78, divergent copies 13, active validation failures 0.
- Owner routing harness after promotion: 21/21 passed.

### Proof gate before editing

- Repo skill validates.
- Any referenced `references/update-first-skill-library.md` exists or pointer is corrected.
- `skills_owner_routing_check.py` still passes the skill-create-warning case.

---

## 3. `linear`

- Active: `/Users/alfred/.hermes/skills/productivity/linear/SKILL.md`
- Repo: `skills/productivity/linear/SKILL.md`
- Diff size: 31 unified diff lines.

### Main differences

Active local adds “Hermes / Kanban bridge pattern”:

- Linear as visible system-of-record.
- Hermes Kanban as execution queue.
- Projects by system/product, boards by execution boundary.
- Linear issue + Kanban card idempotency key.
- Comment Kanban task id back to Linear.
- Proof comments before Done.
- Prefer local `~/.hermes/bin/linear-agent` helper when available.
- References to `references/hermes-kanban-bridge.md` and `references/agent-wiring-pattern.md`.

### Hunk classification

- **Generic promote:** Linear/Kanban bridge model is broadly useful for Hermes agent work, if phrased as an optional Hermes workflow.
- **MJ-local preserve:** exact local helper path `~/.hermes/bin/linear-agent` is MJ/Hermes-host specific unless the helper is repo-supported. Keep as “if present” or local note.
- **Needs split:** project names/board names belong in local `LINEAR-AGENT-RULES.md`, not repo skill.
- **Repo preserve:** core Linear GraphQL/API docs.

### Recommendation

Promote a generic bridge section into repo `linear`, but move local project/board routing details into a local reference/rules file. Keep the local helper mention conditional.

### Proof gate before editing

- `linear` skill still has API basics and GraphQL examples.
- Reference files named by the bridge section exist.
- Linear tracking rules remain in `~/.hermes/LINEAR-AGENT-RULES.md` for local specifics.

---

## 4. `kanban-worker`

- Active: `/Users/alfred/.hermes/skills/devops/kanban-worker/SKILL.md`
- Repo: `skills/devops/kanban-worker/SKILL.md`
- Diff size: 25 unified diff lines.

### Main differences

Active local changes:

- Worktree branch default changes from placeholder `<branch>` to `${HERMES_KANBAN_BRANCH:-wt/$HERMES_KANBAN_TASK}`.
- Adds “Notification routing” section for `notification_sources` in `~/.hermes/config.yaml`:
  - `['*']` accepts all profiles.
  - explicit list restricts profiles.
  - omitted key keeps profile isolation.

### Hunk classification

- **Generic promote:** worktree branch default is a useful concrete default if these env vars are guaranteed by Kanban worker runtime.
- **Generic promote:** notification routing is useful if `notification_sources` is a supported Hermes config key.
- **Needs verification:** confirm env vars and config key are actually implemented before promoting to repo source.
- **MJ-local preserve:** none obvious if supported by product; otherwise keep local only.
- **Repo preserve:** existing scratch/dir/worktree semantics and “Do NOT” section.

### Recommendation

Run a code/config verification before editing repo source:

- grep for `HERMES_KANBAN_BRANCH` and `HERMES_KANBAN_TASK` in runtime.
- grep for `notification_sources` in gateway/config handling.

If both are implemented, promote both hunks. If not, keep local or mark stale.

### Implementation proof (2026-05-26)

Verification result:

- `HERMES_KANBAN_TASK` is real in current source (`hermes_cli/kanban_db.py`, `tools/kanban_tools.py`, docs/tests).
- `HERMES_KANBAN_BRANCH` is **not** implemented in current repo source; only found in an unrelated worktree/local docs.
- `notification_sources` is **not** implemented in current repo source; only found in local/worktree skill docs and this decision file.

Decision:

- Did **not** promote the local `HERMES_KANBAN_BRANCH` worktree default into repo source.
- Did **not** promote the local `notification_sources` section into repo source.
- Corrected the active local `kanban-worker` copy back to repo behavior by removing the two unverified/stale hunks.

Result:

- `kanban-worker` is no longer a divergent active/source skill.
- Inventory after correction: exact duplicates 78, divergent copies 13, active validation failures 0.
- Owner routing harness after correction: 21/21 passed.

### Proof gate before editing

- Runtime grep proves env vars/config key exist.
- Existing Kanban worker tests still pass if any docs tests exist.
- No change to task isolation semantics.

---

## Proposed Phase 2A implementation order

1. `hermes-agent-skill-authoring` — easiest generic promote, directly fixes future skill sprawl.
2. `kanban-worker` — verify two implementation assumptions, then promote or preserve.
3. `linear` — promote generic bridge; keep local org routing in `LINEAR-AGENT-RULES.md`.
4. `hermes-agent` — split into references carefully; highest value but biggest blast radius.

## Stop conditions

Stop and ask MJ before proceeding if:

- A proposed edit would remove any local-only lesson.
- A referenced file does not exist and requires creating new reference structure.
- A repo promotion would expose MJ-specific/private profile, provider, or OpenClaw details.
- Loader behavior changes or exact duplicate removal becomes necessary.

## Verification commands used for this decision pass

```bash
python3.13 scripts/skills_inventory_audit.py --repo-root /Users/alfred/.hermes/hermes-agent --out docs/plans/2026-05-26-skills-audit-data.json
python3.13 scripts/skills_owner_routing_check.py --owner-map docs/plans/skills-owner-map.yaml --cases docs/plans/skills-routing-cases.yaml --min-pass-rate 0.90
```

Current pre-edit status:

- Active skills: 114.
- Repo bundled skills: 87.
- Optional skills: 81.
- Active/source duplicate names: 91.
- Exact duplicate copies: 76.
- Divergent copies: 15.
- Active validation failures: 0.
- Owner routing harness: 21/21 passed.
