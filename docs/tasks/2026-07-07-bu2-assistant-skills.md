# BU-2 — Hermes Assistant Skills (ledger-writer, capture-intake, transcript-followup)

Project: Hermes adoption (BuiltOnPurpose fork) — plan v4 Track A, unit BU-2 (feeds A2 cutover).
Data-engineer pre-checks: N/A — no data pipelines read/written; deliverables are three skill
documents + installer glue (covers Freshness, Temporal bias, Normalization, Sample size,
Pipeline integrity, Data lineage).

## Objective

Create three Hermes skills under `bop/skills/`, each a `SKILL.md` in the repo's established
Hermes skill format (frontmatter: name, description, version, platforms, metadata.hermes.tags —
model: `skills/dogfood/SKILL.md`), and extend `bop/install.sh` to deploy them. These PORT
existing, battle-tested Claude Code canon — the rules below are locked verbatim requirements,
not suggestions to rediscover. Each SKILL.md opens with a "Source canon" line:
`Ported 2026-07-07 from ~/.claude/skills/assistant/SKILL.md + ~/.claude/agents/assistant.md (BU-2).`

### Files to create/modify
- `bop/skills/ledger-writer/SKILL.md` (new)
- `bop/skills/capture-intake/SKILL.md` (new)
- `bop/skills/transcript-followup/SKILL.md` (new)
- `bop/install.sh` (modify: install skills)
- `bop/tests/skills-lint.sh` (new)
- this spec (tracked)

## Skill 1 — ledger-writer

The shared write-discipline skill the other two invoke. Locked rules (from agents/assistant.md):

- Ledger home `~/assistant/`: `ledger.md` (9-column table; status enum
  open/scheduled/waiting/done/parked), `log.md` (append-only receipts, format
  `## [date] <op> | ...`), `inbox/`, `archive/`. Read the ledger header row to get the live
  9-column order — never hardcode column names in the skill beyond citing the file as truth.
- Row ids: `A-####` sequential — next = highest existing across ledger AND archive;
  `CC-<10hex>` reserved for capture2cal bridge rows (this skill never mints those).
- `what` ≤15 words, metadata-only. NPI hard rule (GLBA): lending rows NEVER carry
  FICO/income/SSN/account numbers — metadata only, no exceptions.
- Single-writer conflict rule: never overwrite a row whose `last_touch` is newer than the
  skill's own last write — surface the conflict in the reply instead.
- `done` requires an evidence line (what proves it) written to `next_action` + a log receipt;
  `parked` requires a resume date (becomes `due`) OR a reason in `next_action` — never accept
  a bare done/park.
- Every add/done/park/import writes one `log.md` receipt line.
- Write surfaces: ONLY `~/assistant/**`. (The installed write-fence enforces this; the skill
  states it as its own rule too.)

## Skill 2 — capture-intake

Inputs: a Telegram message (text, forwarded content, or photo) OR files found in
`~/assistant/inbox/` (the existing manual-drop dir — the skill processes it when invoked).
Behavior (from canon):

- Parse item → ONE new ledger row per the ledger-writer rules (source=telegram or
  source=inbox), OR — if the item is durable cross-project knowledge rather than a task —
  a timestamped note into `~/brain/raw/` (kebab-case filename, one-line provenance header).
- Inbox originals move to `~/assistant/archive/inbox/`; images: extract the context into the
  row/note, then DELETE the image (standing Mike rule).
- One log receipt per item. Ambiguous item → ask in chat, don't guess.

## Skill 3 — transcript-followup

Input: a pasted call/voice transcript (Whisperflow etc.) in chat. Behavior (from canon):

- Extract commitments, promises, and leads → one ledger row each per ledger-writer rules
  (source=transcript, `what` ≤15 words).
- Lending-related items: metadata-only rows (GLBA) — zero NPI. The row text must be clean
  enough that `grep -iE 'ssn|fico|[0-9]{3}-[0-9]{2}-[0-9]{4}'` over the ledger stays quiet.
- End reply with the locked receipt: rows added, ids, one line each.

## install.sh extension

- Copy `bop/skills/<name>/` → `~/.hermes/skills/<name>/` (all three), idempotent like the
  hook install (plain cp, 644 on SKILL.md, print one `installed skill:` line each).
- Do NOT touch existing `~/.hermes/skills/` content beyond these three names.

## bop/tests/skills-lint.sh

Self-contained, no framework (match hook-matrix.sh style, `set -u`, PASS/FAIL counters):
1. For each of the three skills: SKILL.md exists; frontmatter block parses as YAML via
   `python3 -c "import yaml,sys; yaml.safe_load(...)"` fallback to a plain `---` sandwich
   check if PyYAML absent; `name:` matches its directory name; body contains the literal
   string "Source canon".
2. NPI self-audit: none of the three SKILL.md bodies contain a 9-digit SSN-shaped literal
   (`[0-9]{3}-[0-9]{2}-[0-9]{4}`) — examples in the docs must use `XXX-XX-XXXX` style.
3. install.sh dry-check: run with `HERMES_HOME=$(mktemp -d)` and assert the three skill dirs
   land there and re-running is a no-op (idempotent).
Exit non-zero on any FAIL; print `skills-lint: N passed, 0 failed`.

## Out of scope

- The A2 Claude-Code-side cutover (agent retirement, /assistant shim, CLAUDE.md rewires) —
  separate unit, separate lane.
- Calendar booking (A3), Gmail (A4), crons (B3), any change to hooks/config templates.
- No changes outside `bop/` + this spec file.

## Done-condition / verification gate

- External /security scan: noted — capture-intake and transcript-followup parse untrusted external
  input (Telegram messages/photos, pasted transcripts) and the unit encodes NPI/GLBA handling;
  the orchestrator runs Claude Code `/security-review` on the changed bop/ files before the
  closer dispatch, and its verdict rides the gate packet.
- `bash bop/tests/skills-lint.sh` exits 0.
- `bash bop/tests/hook-matrix.sh` still 44 passed, 0 failed (no regression).
- `git status` in the worktree shows only the files listed above.
- `bash bop/install.sh` against a scratch HERMES_HOME installs 3 skills + hooks cleanly.
