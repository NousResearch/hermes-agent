# North-Forge Changelog

Every change to tracked files or project configuration in this fork, newest first.
Format follows [Keep a Changelog](https://keepachangelog.com/). Version rules and the
`CHG-` id scheme are defined in [`README.md`](./README.md).

Heading format: `## [NF-vX.Y.Z] — YYYY-MM-DD — hermes@<sha> (N behind upstream/main)`

---

## [NF-v0.1.2] — 2026-09-06 — hermes@693641aa8b (0 behind upstream/main)

`ledger-schema v1 → v2`. Ledger-tooling only — no application code, no `.env`, no
attic-clone change. All entries in this block are `RUN-2026-09-06-001`. `NF-v0.2.0`
stays reserved for the first fork-identity commit. **Committed locally; not pushed —
handoff is the `logs/` zip.**

### Added

- **CHG-2026-09-06-015** — New **decision register**: `decisions/DECISION-LOG.md`
  (Open / Resolved / Register, same shape as `errors/ERROR-LOG.md`) and
  `templates/DECISION-ENTRY-TEMPLATE.md`. `DECISION-YYYY-MM-DD-NNN` ids, same
  immutability / supersession / cross-link rules as `ERR-`. Seeded with
  `DECISION-2026-09-06-001` (fork identity — migrated from the identity half of
  `ERR-2026-09-06-002`, links both ways) and `DECISION-2026-09-06-002` (attic-clone
  keep-or-delete — newly opened; never had an `ERR-` id). Paths:
  `logs/ledger/decisions/DECISION-LOG.md`, `logs/ledger/templates/DECISION-ENTRY-TEMPLATE.md`.
  Ref: — . Run: RUN-2026-09-06-001.
- **CHG-2026-09-06-016** — `AUDIT-TEMPLATE.md`: mandatory **section 1 "Since last
  handoff"** (Closed / New / Unchanged vs the previous audit's open items), written
  before the rest of the body; remaining sections renumbered 2–7. Paths:
  `logs/ledger/templates/AUDIT-TEMPLATE.md`. Ref: — . Run: RUN-2026-09-06-001.

### Changed

- **CHG-2026-09-06-017** — **Confidence tags** on findings and register entries:
  fixed vocabulary `Confirmed Fact` / `Field-Reasoned` / `Unverified`, defined in
  `README.md`, required on every `AUDIT-` finding, `ERR-` entry, and `DECISION-`
  entry. Added the field to `AUDIT-TEMPLATE.md`, `ERROR-ENTRY-TEMPLATE.md`, and
  `DECISION-ENTRY-TEMPLATE.md`. Paths: `logs/ledger/README.md`,
  `logs/ledger/templates/*`. Ref: — . Run: RUN-2026-09-06-001.
- **CHG-2026-09-06-018** — **Run IDs**: `RUN-YYYY-MM-DD-NNN`, one per agent
  invocation, assigned at session start; `Run:` field on every `CHG-` / `ERR-` /
  `DECISION-` entry and in the audit header. Traceability only — groups entries by
  session, does not replace their ids. Defined in `README.md`; field added to
  `CHANGE-ENTRY-TEMPLATE.md`, `ERROR-ENTRY-TEMPLATE.md`, `DECISION-ENTRY-TEMPLATE.md`,
  `AUDIT-TEMPLATE.md`. This session = `RUN-2026-09-06-001`. Paths:
  `logs/ledger/README.md`, `logs/ledger/templates/*`. Ref: — . Run: RUN-2026-09-06-001.
- **CHG-2026-09-06-019** — `README.md` + `INDEX.md` rolled to `ledger-schema v2`:
  directory diagram (+`decisions/`), naming table (+`DECISION-`, +`RUN-`), an
  `ERR-` vs `DECISION-` boundary table, new `Confidence tags` and `Run IDs`
  sections, workflow steps updated. `INDEX.md` split into **Open incidents
  (faults)** and **Open decisions (judgment calls)**; `ERR-2026-09-06-002` moved to
  Resolved (fault half fixed by `CHG-2026-09-06-014`, choice half → `DECISION-2026-09-06-001`).
  Paths: `logs/ledger/README.md`, `logs/ledger/INDEX.md`,
  `logs/ledger/errors/ERROR-LOG.md`. Ref: `ERR-2026-09-06-002`. Run: RUN-2026-09-06-001.

### Unchanged (called out)

- No `CHG-` bullet content was rewritten; `ERR-2026-09-06-001/003/004/005` blocks are
  untouched except a one-line legacy note in the `ERROR-LOG.md` preamble and a retro
  `Run:`/`Confidence` note on `ERR-2026-09-06-005`. No IDs reused or deleted.

---

## [NF-v0.1.1] — 2026-09-06 — hermes@693641aa8b (0 behind upstream/main)

Landed the `NF-v0.1.0` hardening set (it had been working-tree-only), cleared
Windows/pytest cruft that had re-accumulated in `north-forge-agent`, and **synced
the fork to `upstream/main`** — the ledger commit was rebased onto `693641aa8b`, so
the fork is no longer behind upstream. First commits on the fork; pushed to
`origin/main`. Fork *identity* (rebrand vs thin-downstream) is still deferred — the
"sync only" path was chosen. No application code changed by north-forge. See
`audits/AUDIT-2026-09-06-002-hardening-commit-and-hygiene.md`.

### Added

- **CHG-2026-09-06-010** — Committed the `AUDIT-2026-09-06-001` remediation set to
  local `main`: the `.gitignore` secrets/ledger/junk hunk, `.githooks/`
  (`secret-guard` + `pre-commit` + `pre-push`), and the whole `logs/ledger/` tree
  (`ledger-schema v1`, `AUDIT-2026-09-06-001`, templates). It was all uncommitted —
  a fresh clone had neither the secret guard nor the ledger. Not pushed (see
  `ERR-2026-09-06-002`). `pre-commit` `secret-guard` passed. Paths: `.gitignore`,
  `.githooks/`, `logs/ledger/`. Ref: `AUDIT-2026-09-06-002` F-01.

### Changed

- **CHG-2026-09-06-011** — `.gitignore`: added guards for working-tree artifacts
  that recur on this Windows checkout — `MagicMock/` (unittest.mock path leak);
  `*hermes-pytest-tmp*`, `*pytest-tmproot*`, `*pytest-of-*` (pytest tmp-path files
  written into cwd as literal `C:Users…` names — matched on the ASCII substring
  since the `:` survives as a name char under Git-Bash); `/logs.zip` (a zipped copy
  of `logs/`); and `/marguerite-and-penny-suno.txt` (an unrelated stray that a
  plain move failed to hold twice — canonical copy stays in
  `D:\north-forge-agent-attic\`). Paths: `.gitignore`. Ref: `ERR-2026-09-06-005`,
  `AUDIT-2026-09-06-002` F-02/F-03/F-04.

### Fixed

- **CHG-2026-09-06-012** — Removed recurred / stray junk from the `north-forge-agent`
  working tree: the `%SystemDrive%/ProgramData/…` cache tree (recurred after
  `CHG-2026-09-06-009`, exactly as `ERR-2026-09-06-004` foresaw; still git-ignored,
  so harmless), `MagicMock/`, four `C:Users…hermes-pytest-tmp…` pytest files,
  `logs.zip`, and the re-strayed `marguerite-and-penny-suno.txt`. Also cleared
  three stale `logs/full_suite_run*.log` runtime logs (~6.6 MB, git-ignored —
  housekeeping only). No tracked file deleted. Ref: `ERR-2026-09-06-004`
  (recurrence), `ERR-2026-09-06-005`.

### Changed

- **CHG-2026-09-06-014** — Synced the fork to `NousResearch/hermes-agent`
  `upstream/main`. `git rebase upstream/main` replayed the `CHG-2026-09-06-010`
  ledger commit (no conflicts — the 2 upstream commits `be58c276ee` /
  `693641aa8b` touch only `agent/`, `evals/`, `gateway/`, `tests/`, `website/`,
  disjoint from `.gitignore` / `.githooks/` / `logs/ledger/`). `main`:
  `820106d4a5 + [ledger]` → `693641aa8b + [ledger]`. Pushed to `origin/main` as a
  fast-forward (`ahead 3`: the 2 upstream commits + the ledger commit). Closes the
  version-drift half of `ERR-2026-09-06-002`; the fork-identity half stays OPEN
  (deferred by choice). Ref: `ERR-2026-09-06-002`.

### Housekeeping (no tracked-file change)

- **CHG-2026-09-06-013** — `git gc` on the three live repos on `D:\`
  (`north-forge-agent`, `hermes-webui`, `north-forge-hermes-edition`) to repack
  loose objects. The attic clone `nested-clone-2026-09-06` was left as-is (it is a
  delete-when-confirmed backup, not a working repo). Reclaimed sizes: see the
  `D:\logs` report.
- Git freshness verified: all three local branches are level with their `origin`
  (`north-forge-agent` `main`==`820106d4a5`, `hermes-webui` `master`==`e168b67e`,
  `north-forge-hermes-edition` `main`==`1c47b60`); `git pull --ff-only` on each was
  a no-op. `north-forge-agent` is still 2 behind `upstream/main` — `ERR-2026-09-06-002`.

### Known / carried forward

- `ERR-2026-09-06-001` (**OPEN**, HIGH) — live `ANTHROPIC_API_KEY`; now also noted
  at `D:\.env` (drive root, outside any repo). In-repo exposure structurally
  mitigated **and now committed**. Still pending the user's confirm-or-rotate.
- `ERR-2026-09-06-002` — version-drift half **closed** by `CHG-2026-09-06-014`
  (synced to `upstream/main` `693641aa8b`, `NF-v0.1.0`/`v0.1.1` pushed to `origin`).
  Identity half (rebrand vs thin-downstream, first identity commit, `NF-v0.2.0`) was
  deferred here. *→ Superseded 2026-09-06 (`NF-v0.1.2`, `CHG-2026-09-06-019`): the*
  *identity choice migrated to `DECISION-2026-09-06-001`; this `ERR-` is now RESOLVED.*
- `ERR-2026-09-06-005` (**RESOLVED**) — pytest/mock working-tree artifacts. Fixed
  by `CHG-2026-09-06-011` / `CHG-2026-09-06-012`.

---

## [NF-v0.1.0] — 2026-09-06 — hermes@820106d4a5 (2 behind upstream/main)

> **Committed 2026-09-06** on local `main` (`CHG-2026-09-06-010`, `NF-v0.1.1` block
> above); previously working-tree-only. Not pushed.

First ledger entry. Baseline repository cleanup and establishment of the project
ledger. No application code changed. All items below are in the **working tree,
uncommitted** — see the audit's "Open items" for the commit decision.

### Changed

- **CHG-2026-09-06-001** — Fast-forwarded local `main` `245e48008f` → `820106d4a5`
  (245 commits) to match `origin/main`. The local checkout was ~1 day stale. Clean
  fast-forward, no local commits existed. Motivated by `AUDIT-2026-09-06-001` F-03.
- **CHG-2026-09-06-004** — `.gitignore`: carved `logs/ledger/` out of the upstream
  `logs/` ignore rule (`logs/` → `logs/*` + `!logs/ledger/`) so this ledger is
  version-controlled while Hermes runtime logs in `logs/` stay ignored. Motivated by
  `AUDIT-2026-09-06-001` F-04.
- **CHG-2026-09-06-005** — `.gitignore`: added `/north-forge-agent/` guard so a repo
  cloned inside this checkout again cannot be swept into a commit. Motivated by
  `AUDIT-2026-09-06-001` F-01 / F-04.

### Added

- **CHG-2026-09-06-006** — Established `logs/ledger/` (`ledger-schema v1`):
  `README.md` (rules), `INDEX.md`, `CHANGELOG.md`, `audits/`, `errors/ERROR-LOG.md`,
  `templates/`. Seeded with `AUDIT-2026-09-06-001-repository-baseline.md`.
- **CHG-2026-09-06-008** — Added `.githooks/` — `secret-guard` engine plus
  `pre-commit` and `pre-push` hooks that refuse to commit or push any real
  `.env` / `.env.<anything>` / `.op.env` (only `*.example` / `*.sample` pass;
  `.envrc` allowed). `pre-push` scans the full tree at each pushed tip **and** the
  newly added commits. Activated with `core.hooksPath = .githooks` (run
  `sh .githooks/install` on a fresh clone). Bypass is `--no-verify` only.
  Paths: `.githooks/`. Ref: `ERR-2026-09-06-001`.

### Security

- **CHG-2026-09-06-007** — `.gitignore`: replaced the enumerated `.env` list with
  `.env` + `.env.*` + `.op.env` and `!*.example` / `!*.sample` negations. Closes a
  real gap — `.env.production`, `.env.staging`, and any other `.env.<name>` were
  **not** ignored before. Paths: `.gitignore`. Ref: `ERR-2026-09-06-003`.

### Fixed

- **CHG-2026-09-06-009** — Removed a misplaced `%SystemDrive%/ProgramData/…`
  Windows icon-cache tree from the repo root (written by a process with an
  unexpanded `%SystemDrive%` env var; not a git artefact) and added
  `.gitignore` guards: `/%SystemDrive%/`, `Thumbs.db`, `ehthumbs.db`,
  `[Dd]esktop.ini`, `$RECYCLE.BIN/`. Paths: `.gitignore`. Ref: `ERR-2026-09-06-004`.

### Housekeeping (outside the repo tree — no tracked-file change)

- **CHG-2026-09-06-002** — Moved the nested full clone `north-forge-agent/` (≈897 MB,
  its own `.git`) out of the checkout to
  `D:\north-forge-agent-attic\nested-clone-2026-09-06\`. It was a pristine second
  clone with zero unique commits; safe to delete once confirmed unneeded. Motivated
  by `AUDIT-2026-09-06-001` F-01.
- **CHG-2026-09-06-003** — Moved the unrelated stray file
  `marguerite-and-penny-suno.txt` out of the checkout to
  `D:\north-forge-agent-attic\`. Motivated by `AUDIT-2026-09-06-001` F-02.

### Known / carried forward

- `ERR-2026-09-06-001` (**OPEN**, HIGH) — live `ANTHROPIC_API_KEY` in `.env`.
  Exposure now structurally mitigated (CHG-007/008/009); still open pending the
  user's confirm-or-rotate decision on the key itself.
- `ERR-2026-09-06-002` (**OPEN**, MEDIUM) — fork is 2 behind upstream and carries no
  north-forge identity commit yet.
- `ERR-2026-09-06-003` (**RESOLVED**) — `.gitignore` `.env` coverage gap. Fixed by CHG-2026-09-06-007.
- `ERR-2026-09-06-004` (**RESOLVED**) — misplaced `%SystemDrive%` cache tree. Fixed by CHG-2026-09-06-009.

---

<!--
Template for the next block — copy from templates/CHANGE-ENTRY-TEMPLATE.md:

## [NF-vX.Y.Z] — Unreleased — hermes@<sha> (N behind upstream/main)

### Added / Changed / Fixed / Removed / Security
- **CHG-YYYY-MM-DD-NNN** — <what changed>. <why>. Paths: `<...>`. Ref: <ERR-/AUDIT- id>.
-->
