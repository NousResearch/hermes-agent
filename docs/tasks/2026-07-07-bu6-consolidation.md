# BU-6 — Hermes Consolidation Skills (brain-note, memory-consolidate)

Project: Hermes adoption (BuiltOnPurpose fork) — plan v4 Track B, unit BU-6 (feeds B2 + B2.5).
Base: origin/main at BU-5 merge (worktree refreshed pre-exec, same procedure as BU-5; counts
below assume the 11-skill post-BU-5 state — verify with `ls bop/skills/` before implementing).
Data-engineer pre-checks: N/A — no data pipelines read/written; deliverables are two skill
documents + installer/lint glue (covers Freshness, Temporal bias, Normalization, Sample size,
Pipeline integrity, Data lineage).

## Objective

Two new Hermes skills under `bop/skills/` (established format — read
`bop/skills/osr-intake/SKILL.md`). Source canon line for both:
`Ported 2026-07-07 from hermes-adoption-plan-v4 Track B (B2/B2.5 design) + MEMORY-CONTRACT v2.7 Hermes rows (BU-6).`

### Files to create/modify
- `bop/skills/brain-note/SKILL.md` (new)
- `bop/skills/memory-consolidate/SKILL.md` (new)
- `bop/install.sh` (modify: skill list grows to 13)
- `bop/tests/skills-lint.sh` (modify: cover 13 skills)
- this spec (tracked)

## Skill 1 — brain-note (B2: knowledge synthesis, lint-gated direct write)

- Scope: Hermes's OWN learnings only (facts it verified in its own runs — tool behaviors,
  estate facts, recurring patterns from its work). The claude-mem→vault distillation lane
  belongs to brain-distill — this skill NEVER duplicates it and says so.
- Write surfaces: `~/brain/wiki/entities/` and `~/brain/wiki/concepts/` ONLY, and ONLY when
  the wiki tier is enabled (the installed write-fence unlocks these two roots iff
  `~/.hermes/fence-wiki-enabled` exists or HERMES_FENCE_WIKI is truthy — B2 ops creates the
  marker). If a write is blocked or the marker is absent → fail closed, say
  "wiki tier not enabled (B2 pending)". `synthesis/` and `_meta/` are HUMAN-ONLY — never
  attempt them (fence blocks; skill states it).
- Pre-write lint gate (checklist IN the skill — no external tool exists): (1) frontmatter
  with title + `last_verified: YYYY-MM-DD` + `source:` provenance line naming the Hermes run
  /evidence; (2) dedup grep of entities/ + concepts/ for the topic — existing page → propose
  an EDIT of that page (surgical, preserve structure) rather than a new one; (3)
  links-not-copies: reference canonical sources (file paths, ledger ids), never copy numbers
  that live elsewhere; (4) one note per fact-cluster, kebab-case filename. Any check
  unsatisfiable → don't write; say which check failed.
- Untrusted-content + NPI rules per house standard (data-only; metadata-only).

## Skill 2 — memory-consolidate (B2.5: episodic → MEMORY.md + weekly digest)

- Input: Hermes's own episodic history (state.db via whatever session-history access the
  runtime provides the agent — if none is available in-run, consolidate from the current
  session context and say so; never invent history).
- Output 1: update `~/.hermes/MEMORY.md` — curated long-term memory, links-not-copies,
  STABLE facts only (identity, standing rules, durable estate facts). Explicitly forbidden:
  raw event streams, transcripts, one-off task chatter. Keep the file under ~200 lines;
  compress/supersede rather than append forever.
- Output 2 (weekly cadence, or on demand): one digest note
  `~/brain/raw/YYYY-MM-DD-hermes-week.md` — what Hermes worked on, durable outcomes,
  links to workspace reports — so /emerge, /trace, and memory-shepherd see Hermes activity
  (MEMORY-CONTRACT v2.7 accepted-gap bridge; state.db itself stays LOCAL-ONLY, never synced).
- Write surfaces: `~/.hermes/MEMORY.md` (fence exact-file allow) + `~/brain/raw/`
  (fence-allowed root). Nothing else.
- Designed to run under a future B3 cron (delegation-model friendly: deterministic steps,
  no judgment calls that require the primary model) — but fully usable on demand now.

## install.sh / tests

- install.sh: skill loop grows to 13. skills-lint.sh: SKILLS list 13; installed-lines 11→13;
  per-skill checks apply automatically.

## Out of scope

- B2 ops (creating the fence-wiki-enabled marker), B2.5/B3 cron arming (hermes cron config,
  harness-eval entry gate, supervised fires), MEMORY-CONTRACT edits (rows already exist,
  v2.7). Any change to hooks, config template, or the 11 existing skills.

## Done-condition / verification gate

- External /security scan: noted — brain-note writes into the shared brain vault (BOTH-tier
  wiki dirs) behind the fence marker + lint gate; memory-consolidate writes MEMORY.md (an
  agent-identity surface, self-modification adjacent — its STABLE-facts-only rule and the
  ledger of what may never enter MEMORY.md are the control). Orchestrator runs a security
  review pass on the changed bop/ files before closer dispatch; verdict rides the gate packet.
- `bash bop/tests/skills-lint.sh` exits 0 (13 skills).
- `bash bop/tests/hook-matrix.sh` unchanged (46 passed, 0 failed).
- `bash -n` clean; scratch install lands 13 skills; idempotent; `git status` clean vs list.
