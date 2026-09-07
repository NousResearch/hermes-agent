# North-Forge Changelog

Every change to tracked files or project configuration in this fork, newest first.
Format follows [Keep a Changelog](https://keepachangelog.com/). Version rules and the
`CHG-` id scheme are defined in [`README.md`](./README.md).

Heading format: `## [NF-vX.Y.Z] — YYYY-MM-DD — hermes@<sha> (N behind upstream/main)`

---

## [NF-v0.4.0] — 2026-09-07 — hermes@61d30533f7 (13 behind upstream/main)

First pass to touch **application code** for identity (`RUN-2026-09-07-002`). The
rebrand had so far been documentation, tooling, and art only (`BRANDING.md` §2
kept the seeded persona and the CLI banner/help text as "intentionally
Hermes-compatible"). `AGENT E`'s first-launch witness run (`FIRST-LAUNCH-WITNESS_2026-09-07`)
confirmed the consequence: a first-time user who bootstraps with `north-forge.cmd`
gets an agent that says *"You are Hermes Agent, built by Nous Research"* and a
`hermes --version` / `--help` banner reading *"Hermes Agent"* — no North Forge
anywhere in the running product. This pass moves those specific surfaces into
North Forge's identity and re-files them in `BRANDING.md`; it also fixes the
cross-volume bootstrap slow path the same witness run measured (`ERR-2026-09-07-001`).
**MINOR** — a North Forge identity surface extended into the runtime; the engine,
repo layout, the `hermes` command name, `HERMES_*` env vars, and the
`hermes-agent` distribution name are all unchanged, and no behaviour changes
except what the agent calls itself. Not pushed pending review (higher-stakes than
a docs pass). HEAD `61d30533f7` is level with `origin/main` but `upstream/main`
has moved **13 commits** since `NF-v0.3.0` was cut this morning — a sync is a
separate task, not folded in here.

### Changed

- **CHG-2026-09-07-007** — Identity surfaces moved from `BRANDING.md` category 2
  (Hermes-compatible, do not touch) to category 1 (North-Forge-owned identity),
  and rebranded:
  - **Seeded / fallback persona identity line.** `hermes_cli/default_soul.py`
    `DEFAULT_SOUL_MD` (seeded into `HERMES_HOME/SOUL.md` on first run) and
    `agent/prompt_builder.py` `DEFAULT_AGENT_IDENTITY` (in-memory fallback when no
    SOUL.md is present, e.g. `skip_context_files` / subagent) — first sentence
    changed from *"You are Hermes Agent, built by Nous Research."* to *"You are
    North Forge, an adaptive AI agent (built on the Hermes Agent engine by Nous
    Research)."* Engine attribution kept (BRANDING.md honesty principle). The rest
    of the behaviour spec (reply-sizing rule, named prohibitions, earned-depth) is
    upstream's and is untouched; the two constants remain **byte-identical** (712
    chars) as `default_soul.py`'s own header comment requires. `_LEGACY_TEMPLATE_SOULS`
    / `_SCAFFOLD_*` in the same file were **not** touched — they must stay
    Hermes-verbatim to keep auto-upgrade detection working. Not added to
    `_LEGACY_TEMPLATE_SOULS`: the outgoing text, so homes seeded between the fork
    start and this commit keep the old persona until edited — deliberate (that
    list carries a "never silently overwrite a user's SOUL.md" guarantee; the
    handful of such homes are all disposable test installs). Flagged for review.
  - **CLI display name.** `hermes_cli/banner.py` `format_banner_version_label()`
    and `hermes_cli/_parser.py` top-level parser `description`: *"Hermes Agent
    v0.21.0 …"* → *"North Forge v0.21.0 …"*, *"Hermes Agent - AI assistant with
    tool-calling capabilities"* → *"North Forge - AI assistant …"*. The version
    number, the `· upstream <sha>` suffix, the `Install directory` / `Install
    method` / `Python` / `OpenAI SDK` lines, and **every `hermes …` example** in
    the help epilogue are unchanged (that is the command name — stays). Two
    degraded-path echoes of the same label updated to match:
    `hermes_cli/_startup_fast.py` (import-failure fallback) and `cli.py`
    (`HERMES_FAST_STARTUP_BANNER=1` fast banner).
  - **`BRANDING.md`** — two new category-1 rows for the above; the old category-2
    row (`DEFAULT_AGENT_IDENTITY`, `DEFAULT_SOUL_MD` — "untouched") replaced with a
    narrower one covering only what still must stay Hermes-verbatim
    (`HERMES_AGENT_HELP_GUIDANCE`, the legacy-SOUL detection strings). The
    command name, env vars, and distribution name rows in category 2 are
    unchanged.
  - **Tests** updated for the new literal: `tests/hermes_cli/test_startup_fast_guards.py`
    (`"Hermes Agent v"` → `"North Forge v"`, 2 assertions),
    `tests/hermes_cli/test_banner.py` (1 assertion).
    `tests/agent/test_prompt_builder.py::test_empty_dir_loads_seeded_global_soul`
    still passes unchanged — the new persona line still contains "Hermes Agent"
    (in the engine-attribution clause).
  - **Deliberately left as "Hermes"** (BRANDING.md category 3 / out of this pass's
    named scope, flagged for a follow-up decision): the `update` / `uninstall` /
    `acp` subcommand help blurbs ("Update Hermes Agent…"), the `/version` REPL
    command description (`hermes_cli/commands.py`), `acp_adapter/commands.py`'s ACP
    version string, and the `⚕ NOUS HERMES` / "AI Agent Framework" startup splash
    art in `cli.py`. `website/**` docs that quote the old fallback text are
    category 3 and stay.
  - Paths: `hermes_cli/default_soul.py`, `agent/prompt_builder.py`,
    `hermes_cli/banner.py`, `hermes_cli/_parser.py`, `hermes_cli/_startup_fast.py`,
    `cli.py`, `BRANDING.md`, `tests/hermes_cli/test_startup_fast_guards.py`,
    `tests/hermes_cli/test_banner.py`. Ref: `DECISION-2026-09-06-001`,
    `FIRST-LAUNCH-WITNESS_2026-09-07`. Run: RUN-2026-09-07-002.

### Fixed

- **CHG-2026-09-07-008** — `scripts/bootstrap-north-forge.ps1` first-run time on a
  drive-native checkout: **~6.5 min → 5–7 s**. Root cause (`ERR-2026-09-07-001`):
  uv installs by hardlinking packages from its cache into the venv, but uv's
  default cache sits under `%LOCALAPPDATA%` on `C:` while a drive-native checkout
  builds its venv on the checkout's own drive — cross-volume, so uv fell back to a
  full byte copy of all 67 packages ("Failed to hardlink files; falling back to
  full copy"). Fix: before the `uv venv` / `uv pip install` calls, set
  `$env:UV_CACHE_DIR = Join-Path $parent '.uv-cache'` (a sibling of the venv, so
  always the same volume) unless the operator already set `UV_CACHE_DIR`.
  Measured on `D:\`: install/link step `Installed 67 packages in 740ms` (was
  `6m 25s` on the `E:\` witness run); total bootstrap 7.4 s cold cache / 5.0 s
  warm; the hardlink-failure warning is gone; sibling venv `hermes.exe --version`
  works and shows the new North Forge banner. Paths:
  `scripts/bootstrap-north-forge.ps1`. Ref: `ERR-2026-09-07-001`,
  `FIRST-LAUNCH-WITNESS_2026-09-07`. Run: RUN-2026-09-07-002.

## [NF-v0.3.0] — 2026-09-07 — hermes@922c0d670c (0 behind upstream/main)

One consolidated pass (`RUN-2026-09-07-001`): a content secret-scanning layer, a
mandatory handoff-redaction gate, a generic-persona + editions-overlay split with
a `BRANDING.md` source of truth, an open install-model decision, a minimal
install-and-launch path, and the final brand art. **MINOR** — new North Forge
capability and workflow on top of upstream; no application code, no `.env`, no
`pyproject.toml` distribution-name change; the engine runs exactly as before
(images, docs, `.githooks/`, `scripts/`, `editions/`, ledger, and the repo-root
`SOUL.md` example file only). While this pass was in flight a GitHub-side "sync
fork" merged `upstream/main` into `origin/main` (`45b2795865`); this block was
**rebased onto it** (clean — disjoint paths), so the fork is now **0 behind
`upstream/main`** at `hermes@922c0d670c`. **Pushed to `origin/main`** — the
review-before-push hold that ran across `NF-v0.1.2`..`NF-v0.2.2` ended here, so
`NF-v0.2.0`..`NF-v0.3.0` are now public together.

### Added

- **CHG-2026-09-07-001** — Content secret-scanning layer alongside the filename
  guard. New `.githooks/content-scan` (`git grep -P` based): three high-confidence
  vendor families — AWS access key ids (`AKIA`/`ASIA`+16; the two AWS-doc
  `…EXAMPLE` placeholders excluded), GitHub tokens (`gh[pousr]_`+36,
  `github_pat_`+82; a ≥20-char single-char run excluded), structured Slack tokens
  (`xox[bpars]-<digits>-<digits>-<secret>`, `xapp-1-…`, `xoxe.xox[bp]-…`). Modes:
  `--commits <range>` (per-commit changed-file content — the gate; catches a key
  added in one fork commit even if a later commit renames/deletes the file, and
  does **not** re-flag upstream's own credential-shaped test fixtures),
  `--tree` / `--worktree` (full manual audits). Inline `# nf-scan: allow <reason>`
  marker for confirmed fakes on the exact line — no whole-file/path exclusions.
  Wired into `.githooks/pre-push` (after the existing `secret-guard` calls) and
  mirrored by a new CI workflow `.github/workflows/nf-secret-scan.yml`
  (`content-scan --commits` on every push/PR + the scanner self-tests), so a push
  is checked even where local hooks aren't installed. Tests:
  `.githooks/tests/run.sh` (6 cases, throwaway repos) — a secret added-then-removed
  in a range still fails `--commits`; the final tree alone is clean; an
  allowlisted fixture passes while the same value without the marker is blocked;
  token shapes are caught; placeholders/prose are not. `.githooks/install` and
  `.githooks/README.md` updated. Paths: `.githooks/content-scan`,
  `.githooks/pre-push`, `.githooks/install`, `.githooks/README.md`,
  `.githooks/tests/run.sh`, `.github/workflows/nf-secret-scan.yml`. Ref: — .
  Run: RUN-2026-09-07-001.
- **CHG-2026-09-07-002** — Mandatory handoff redaction before
  `scripts/collect-logs.*` zips the bundle. New `scripts/redact_handoff.py`
  **reuses the agent's production redactor** (`agent.redact.redact_sensitive_text`,
  `force=True`) — no new vocabulary. `collect-logs.ps1` / `.sh` now: resolve a
  Python that can import `agent.redact` (repo `.venv`, a sibling `-venv`, or
  PATH); copy `D:\logs\` to a **throwaway staging dir**; redact the copy (the
  durable session reports in `D:\logs\` are left byte-for-byte intact); write
  `redaction-report.txt` (into the bundle and back into `D:\logs\`) noting what
  was touched with a "pattern-matching is a backstop, not a guarantee" caveat;
  zip the staging copy. **Fail closed:** no importable redactor, or a likely
  secret that still matches a strict AWS/GitHub/Slack recheck after redaction ⇒
  the zip is NOT created, the run exits non-zero, names the `file:line`, and moves
  any previous zip to `<zip>.stale`. `repo-runtime-logs/*.log` / `*.jsonl` are
  **excluded from the bundle by default** (highest-risk, lowest-value); opt-in
  with `-IncludeRuntimeLogs` (ps) / `--include-runtime-logs` (sh) — still
  redacted. Paths: `scripts/redact_handoff.py`, `scripts/collect-logs.ps1`,
  `scripts/collect-logs.sh`. Ref: — . Run: RUN-2026-09-07-001.
- **CHG-2026-09-07-004** — `logs/ledger/decisions/DECISION-LOG.md`: opened
  **`DECISION-2026-09-06-003`** (Install model — drive-native run-in-place vs
  machine-local managed install), status **OPEN**. Provisional lean: drive-native
  (matches the portable-first principle) — **not ratified**. The hardened form
  (sealed drive, `NORTHFORGE` / `NORTHFORGE-DATA` two-volume split,
  certify/verify/audit) is explicitly out of scope and blocks on ratification;
  the minimal bootstrap (`CHG-2026-09-07-005`) is not blocked and ships now. Id
  keeps the `2026-09-06` date at the owner's request for continuity with the
  rebrand batch (opened 2026-09-07). Register row added. Paths:
  `logs/ledger/decisions/DECISION-LOG.md`. Ref: `DECISION-2026-09-06-003`.
  Run: RUN-2026-09-07-001.
- **CHG-2026-09-07-005** — Minimal install-and-launch path (single-drive,
  single-folder-tree; **not** the hardened install — see
  `DECISION-2026-09-06-003`). `scripts/bootstrap-north-forge.ps1`: creates a
  Python venv as a **sibling** of the checkout (`<parent>\<leaf>-venv`, never
  inside the tree the agent operates on), a sibling data folder
  (`<parent>\<leaf>-data`, reported as `HERMES_HOME`), and an **editable** install
  of the checkout into that venv (`uv pip install -e .`, or
  `python -m pip install -e .`); writes `<venv>\.nf-bootstrapped` as the ready
  marker. `north-forge.cmd` (repo root): double-click launcher — runs the
  bootstrap on first run, then sets `HERMES_HOME` and starts `hermes` with args
  passed through. Tested on a fresh detached checkout: bootstrap → `hermes`
  importable + `hermes.exe` present in ~7 s (uv cache); `hermes --version` /
  `--help` work. No `NORTHFORGE`/`NORTHFORGE-DATA` split, no seal/verify. Paths:
  `scripts/bootstrap-north-forge.ps1`, `north-forge.cmd`. Ref:
  `DECISION-2026-09-06-003`. Run: RUN-2026-09-07-001.
- **CHG-2026-09-07-006** — Final (non-placeholder) brand art from
  `north-forge-brand-assets.zip` (confirmed to contain exactly `banner.png`,
  `north-forge.ico`, `icon-512.png`, `icon-256.png`, `splash-alt.png`).
  `assets/banner.png` replaced with the final banner (1510×724 PNG, compass +
  anvil + flame, "NORTH FORGE"); this **retires the `CHG-2026-09-06-025`
  placeholder** — the new file carries no `PLACEHOLDER` `tEXt` chunk. `README.md`
  needed no edit — it had no placeholder note or "temporary artwork" callout
  (that note lived only in the old PNG's metadata). Added `assets/icons/` —
  `north-forge.ico` (Windows multi-resolution), `icon-512.png`, `icon-256.png`,
  and `splash-alt.png` (held in reserve for a future loading screen; not wired).
  **Icon wiring:** North Forge's own launcher (`north-forge.cmd`) and bootstrap
  create no Start-Menu/desktop shortcut, and upstream `scripts/install.ps1`'s
  shortcut step is application code and out of scope — so the `.ico` is staged
  and `BRANDING.md` records that a future shortcut should point at
  `assets/icons/north-forge.ico`; **shortcut-icon wiring is pending real shortcut
  creation.** Images + one doc-of-record only; agent behaviour unaffected. Paths:
  `assets/banner.png`, `assets/icons/north-forge.ico`, `assets/icons/icon-512.png`,
  `assets/icons/icon-256.png`, `assets/icons/splash-alt.png`. Ref: — .
  Run: RUN-2026-09-07-001.

### Changed

- **CHG-2026-09-07-003** — Generic root persona + editions overlay + branding
  source of truth. `SOUL.md` (repo root) is now the **industry-neutral** chassis
  voice (the general-purpose draft: adaptive tone, honest about uncertainty,
  direct — no field technicians / sales reps / industry). The
  field-service-flavoured persona moved to `editions/field-service/SOUL.md` with
  `editions/field-service/README.md` explaining it is an **optional profile
  overlay, applied on top of the base persona at deploy time, never a
  replacement**; `editions/README.md` describes the `editions/` concept (additive,
  optional, non-proprietary — vertical skill-sets stay in admin-gated repos).
  New **`BRANDING.md`** — the source of truth for which surfaces are
  North-Forge-owned, which are intentionally Hermes-compatible (the `hermes-agent`
  distribution name, the `hermes` commands, workspace package names, persona code
  constants, `HERMES_*` env vars), and which are upstream documentation left
  pointing at Nous. The three translated READMEs (`README.es.md`,
  `README.zh-CN.md`, `README.ur-pk.md`) reduced from full (stale, still
  Hermes-branded) translations to **short landing pages** — North Forge banner, a
  2–3 sentence description in the target language, and links to the canonical
  English `README.md` and upstream's docs (incl. the `zh-Hans` docs for Chinese).
  Repo-root `SOUL.md` is a seed/example file read by no test / installer /
  packaging path (installers seed from `hermes_cli/default_soul.py`, untouched),
  so this has **zero functional effect**. Paths: `SOUL.md`,
  `editions/field-service/SOUL.md`, `editions/field-service/README.md`,
  `editions/README.md`, `BRANDING.md`, `README.es.md`, `README.zh-CN.md`,
  `README.ur-pk.md`. Ref: `DECISION-2026-09-06-001` (branding continuation).
  Run: RUN-2026-09-07-001.

### Unchanged (called out)

- **Application code, `.env`, `pyproject.toml` distribution name, the attic
  clone, `agent/prompt_builder.py` / `hermes_cli/default_soul.py` persona
  constants** — untouched. Same carve-outs as `NF-v0.2.0` (`DECISION-2026-09-06-001`).
- **`docker/SOUL.md`, `AGENTS.md`, `CONTRIBUTING*.md`, `SECURITY*.md`,
  `website/**`** — upstream documentation, left as-is (see `BRANDING.md` §3).
- **`assets/icons/splash-alt.png`** — added but deliberately not wired into
  anything (future loading-screen use).
- **The hardened install** (sealed drive / two-volume split / certification) —
  not built; blocked on `DECISION-2026-09-06-003`.

### Housekeeping

- `D:\logs\GIT_HARDENING_2026-09-06.md` line 173 — an in-place redaction run
  (before `CHG-2026-09-07-002` switched to a staging copy) had mangled
  `secret-guard: clean` → `secret-guard: ***` (a config-key false positive on the
  `secret-guard:` label). Restored by hand; the staging-copy design prevents
  recurrence. `D:\logs\` is outside the repo — no tracked-file change.

---

## [NF-v0.2.2] — 2026-09-06 — hermes@693641aa8b (0 behind upstream/main)

Handoff tooling + a standing agent-conduct policy. **PATCH** per the version
table ("housekeeping" + "docs / ledger-only"). No application code, no `.env`,
no packaging change; agent runtime behaviour untouched. All entries in this
block are `RUN-2026-09-06-003`. **Committed locally; not pushed** — held for
review, same as everything since `NF-v0.1.1`.

### Added

- **CHG-2026-09-06-027** — `scripts/collect-logs.ps1` + `scripts/collect-logs.cmd`
  (double-click wrapper) + `scripts/collect-logs.sh` (POSIX mirror): a
  one-command builder for the review/handoff bundle. It (a) copies
  `logs/ledger/**` into `D:\logs\ledger\`, (b) writes a git snapshot to
  `D:\logs\repo-state.txt` (HEAD, branch, ahead/behind `origin` + `upstream`,
  uncommitted files, recent log), (c) generates `D:\logs\HANDOFF-INDEX.md`
  (manifest + check results), (d) copies any `logs/*.log` / `*.jsonl` if present,
  (e) removes superseded `north-forge-agent-logs-*.zip` nested bundles, (f) zips
  `D:\logs\` → `D:\logs.zip` (overwrite) + writes `D:\logs.zip.sha256`, and
  (g) runs a completeness self-check (ledger copied in full; key ledger files
  present + non-empty; audits/templates counts; local not behind `origin/main`;
  uncommitted-tree warning; **session-report-vs-ledger date freshness**; zip
  entry-set matches staging; zip newer than inputs). Exit 0 unless a check
  FAILs. Existing `D:\logs\*.md` session reports are never touched. Paths
  outside the repo (`D:\logs\`, `D:\logs.zip`) are **not** tracked — only the
  three scripts are. Runnable by hand any time and as end-of-task practice; no
  dependency on being invoked by an agent. Paths: `scripts/collect-logs.ps1`,
  `scripts/collect-logs.cmd`, `scripts/collect-logs.sh`. Ref: — . Run:
  RUN-2026-09-06-003.

### Changed

- **CHG-2026-09-06-028** — `logs/ledger/README.md`: new **"## Agent conduct"**
  section (between "Workflow — the discipline" and "Optional automation").
  States the standing policy: incidental **minor** bugs found during any task
  are fixed and logged in the same run without stopping for approval; anything
  touching **architecture, access-tier logic, secrets, or an already-flagged
  `OPEN` `DECISION-`/`ERR-`** still escalates on the existing path; when unsure
  whether something is "minor", escalate rather than guess. Also records the
  end-of-task handoff step (run `scripts/collect-logs.*`, drop a
  `D:\logs\<TOPIC>_<date>.md` write-up). Paths: `logs/ledger/README.md`.
  Ref: — . Run: RUN-2026-09-06-003.

### Unchanged (called out)

- **`D:\logs\` and `D:\logs.zip`** — regenerated artifacts, outside the repo,
  not committed (same as before). Prior hand-built `D:\logs\*.md` session
  reports are kept as-is by the script.
- **Application code, `.env`, packaging** — untouched.

---

## [NF-v0.2.1] — 2026-09-06 — hermes@693641aa8b (0 behind upstream/main)

Finishes the branding pass started by **`DECISION-2026-09-06-001` Option B (full
rebrand)** — the two items the `NF-v0.2.0` commit left open. **PATCH** per the
version table ("docs" + one image asset; no new capability, no further reshape).
Branding / docs only — no application code, no `.env`, no `pyproject.toml`
distribution-name change, no attic-clone change; agent behaviour is identical
before and after. All entries in this block are `RUN-2026-09-06-003`.
**Committed locally; not pushed** — held for review, same as `NF-v0.1.2` /
`NF-v0.2.0`.

### Changed

- **CHG-2026-09-06-025** — `assets/banner.png`: replaced the upstream Hermes Agent
  banner (blocky gold "HERMES-AGENT" wordmark) with a North Forge equivalent —
  same envelope (1145×196, PNG, 8-bit RGB, non-interlaced): "NORTH FORGE"
  wordmark, an anvil mark, an amber accent rule, and the strap-line "A brandable
  AI-agent chassis on the Hermes engine" on a dark ground. **This is a
  PLACEHOLDER, not final brand art** — an auto-generated wordmark (Pillow +
  system fonts); a `PLACEHOLDER - replace with final art` note is carried in the
  PNG `tEXt` chunks (`Title` / `Comment`). Restored the banner reference in
  `README.md` (removed by `CHG-2026-09-06-020`): a centred
  `<img src="assets/banner.png" alt="North Forge" width="100%">` directly under
  the `# North Forge` H1. The three translated READMEs
  (`README.es.md` / `README.zh-CN.md` / `README.ur-pk.md`) already reference the
  same path with `alt="Hermes Agent"` — left as-is here (out of scope; see the
  known-issues note below). Paths: `assets/banner.png`, `README.md`. Ref:
  `DECISION-2026-09-06-001`. Run: RUN-2026-09-06-003.

### Added

- **CHG-2026-09-06-026** — `README.md`: two new top-level sections between
  "Getting Started" and the Nous Portal section. **"Drive class"** (one sentence)
  — acknowledges that a deployed drive carries a class label in its volume name
  so a recipient or support person can identify the drive at a glance;
  deliberately documents *no* naming scheme, codes, or access mechanics.
  **"Customizing your agent"** (two short paragraphs) — plain-language statement
  that some North Forge editions allow full customization of the underlying AI
  (model/provider, persona, configuration) via the drive's setup menu while
  others ship pre-configured; describes only the user-facing difference, no
  passcode / admin / unlock mechanism. Paths: `README.md`. Ref:
  `DECISION-2026-09-06-001`. Run: RUN-2026-09-06-003.

### Unchanged (called out)

- **Application code, `.env`, `pyproject.toml` distribution name, the attic
  clone, the translated `README.*.md` files** — untouched, same carve-outs as
  `NF-v0.2.0`.

### Known / carried forward

- The translated READMEs (`README.es.md`, `README.zh-CN.md`, `README.ur-pk.md`)
  now render the new North Forge banner but still carry `alt="Hermes Agent"`.
  Pre-existing (they were left upstream-branded at the rebrand); a follow-up if
  those files are kept rather than dropped.
- `ERR-2026-09-06-001` (**OPEN**, HIGH) — live `ANTHROPIC_API_KEY`; unchanged,
  user-owned.
- `DECISION-2026-09-06-002` (**OPEN**) — attic-clone keep-or-delete; unchanged.

---

## [NF-v0.2.0] — 2026-09-06 — hermes@693641aa8b (0 behind upstream/main)

First fork-identity commit — **`DECISION-2026-09-06-001` Option B (full rebrand)**,
implemented. **MAJOR** bump per the version table ("incompatible change to the
fork's shape — rebrand"): the repo now presents as **North Forge**, a generic,
brandable agent chassis on the Hermes Agent engine, not as Hermes Agent itself.
Branding / identity only — no application code, no `.env`, no attic-clone change.
Agent behaviour is identical before and after: the repo-root `SOUL.md` is an
example file read by no test / installer / packaging path, and the persona code
constants (`agent/prompt_builder.py`, `hermes_cli/default_soul.py`) were not
touched. All entries in this block are `RUN-2026-09-06-002`. **Committed locally;
not pushed** — held for review before going public, same as the last hardening
commit.

### Changed

- **CHG-2026-09-06-020** — `README.md`: rewrote the top-level identity. H1
  (`Hermes Agent ☤` → `North Forge`), centre tagline, shields, and the lead
  description now present North Forge as its own project built on the Hermes
  Agent engine by Nous Research; added a one-paragraph provenance note (fork of
  `NousResearch/hermes-agent`, kept rebased, engine used unmodified, maintainer
  Kenneth C. Walker Jr., fork-issue URL) and both copyright lines in the License
  footer. The upstream banner image (`assets/banner.png`, alt "Hermes Agent") is
  no longer referenced. All mechanics — install one-liners, `hermes …` commands,
  `hermes-agent.nousresearch.com/docs` links, feature table, Contributing,
  Community — left verbatim. Paths: `README.md`. Ref: `DECISION-2026-09-06-001`.
  Run: RUN-2026-09-06-002.
- **CHG-2026-09-06-021** — `SOUL.md`: replaced the upstream default persona with
  North Forge's own voice (finalised 2026-09-06 from owner-supplied text, before
  any push; the intermediate draft in this commit's first local revision was
  never published). First person ("I'm North Forge"), written for field
  technicians and sales reps: plain words, short sentences, no corporate filler,
  no fake enthusiasm, no "I apologize for the confusion" — own a mistake and fix
  it. Matches the user's register (terse ↔ chatty ↔ formal) rather than imposing
  one fixed personality; honest about what it does not know ("here's what I know,
  here's what I'm not sure of, here's what would confirm it"); does not talk down
  and does not assume unseen expertise; direct out of respect for the reader's
  time, flags risk and faster paths up front. No Kyocera / Blacksmith / mode
  scaffolding — this is the industry-generic chassis file. Repo-root example file
  only — not read by any test, installer, or packaging path (installers seed
  `$HERMES_HOME/SOUL.md` from `hermes_cli/default_soul.py`, untouched), so it
  diverges from the `DEFAULT_SOUL_MD` / `DEFAULT_AGENT_IDENTITY` code constants by
  design, with zero functional effect. Paths: `SOUL.md`. Ref:
  `DECISION-2026-09-06-001`. Run: RUN-2026-09-06-002.
- **CHG-2026-09-06-022** — `package.json`: `name` `hermes-agent` →
  `north-forge-agent`; `repository.url`, `homepage`, and `bugs.url` repointed
  from `NousResearch/Hermes-Agent` to `kwalker7631/north-forge-agent`. Root
  `name` in `package-lock.json` synced to match (2 lines: `.name` and
  `.packages[""].name`) — identity mirror only, no dependency-tree change, so
  `npm ci` stays consistent with the manifest. Root package is `"private": true`
  and never published; workspace package names (`hermes-tui`, `hermes`,
  `@hermes/root-tests`) left as-is — internal build ids, same blast-radius
  rationale as the pyproject distribution name. Paths: `package.json`,
  `package-lock.json`. Ref: `DECISION-2026-09-06-001`. Run: RUN-2026-09-06-002.
- **CHG-2026-09-06-023** — `pyproject.toml`: added `[project.urls]` (`Homepage` +
  `Repository` → `kwalker7631/north-forge-agent`). The Python **distribution name
  deliberately stays `hermes-agent`** per `DECISION-2026-09-06-001`: never
  published (`setup.py` blocks wheel builds), referenced ~19× by the
  self-referential `hermes-agent[...]` extras, and pinned in `uv.lock` / the
  installed `.venv` — renaming it is blast radius with no outward benefit.
  `authors = [{ name = "Nous Research" }]` also left as-is (engine authorship;
  outside the decision's enumerated fields — attribution is carried by
  `LICENSE`). Paths: `pyproject.toml`. Ref: `DECISION-2026-09-06-001`. Run:
  RUN-2026-09-06-002.

### Added

- **CHG-2026-09-06-024** — `LICENSE`: added `Copyright (c) 2026 Kenneth C. Walker
  Jr.` beneath the existing `Copyright (c) 2025 Nous Research` line. MIT — both
  attributions coexist; Nous Research's line and the permission / warranty body
  are unchanged. Paths: `LICENSE`. Ref: `DECISION-2026-09-06-001`. Run:
  RUN-2026-09-06-002.

### Unchanged (called out)

- **`pyproject.toml` `name = "hermes-agent"`** — NOT renamed (see
  CHG-2026-09-06-023). This is the explicit carve-out in `DECISION-2026-09-06-001`
  Option B.
- **Application code, `.env`, the attic clone, `docker/SOUL.md`, the translated
  `README.zh-CN.md` / `README.es.md` / `README.ur-pk.md` files, workspace
  `package.json` names** — untouched.
- **`agent/prompt_builder.py` / `hermes_cli/default_soul.py`** — the persona code
  constants are upstream's and stay upstream's; only the repo-root `SOUL.md`
  example file carries the North Forge voice.

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
