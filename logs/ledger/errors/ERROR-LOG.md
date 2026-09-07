# North-Forge Error & Anomaly Register

Append-only. One row per **fault**: build break, failed test batch, bad merge, upstream
regression, secret exposure, unexpected repo state — anything that is *wrong* and needs
fixing. An open *judgment call* (a choice between defensible options) is not a fault —
it goes in [`../decisions/DECISION-LOG.md`](../decisions/DECISION-LOG.md) as a
`DECISION-` instead. See [`README.md`](../README.md) § `ERR-` vs `DECISION-`.

Never delete a row — close it by moving status to `RESOLVED` (or `WONTFIX` /
`ACCEPTED-RISK` / `SUPERSEDED`, naming the record that replaces it) with a resolution
note and the resolving `CHG-` id.

`ERR-` id scheme, severities, the `Confidence` vocabulary, and `Run:` ids are defined
in [`README.md`](../README.md). Severity: **CRITICAL** · **HIGH** · **MEDIUM** ·
**LOW** · **INFO**.

> Entries opened before `ledger-schema v2` (everything below dated 2026-09-06 up to
> and including `ERR-2026-09-06-005`) predate the `Confidence` and `Run:` fields.
> Read their claims as **Confirmed Fact** unless the entry says otherwise, and their
> run as the `AUDIT-2026-09-06-001` / `-002` sessions. Not backfilled.

---

## Open

### ERR-2026-09-06-001 — HIGH — Secret hygiene

- **Opened:** 2026-09-06 · **Base:** hermes@820106d4a5 (2 behind upstream/main)
- **Source:** `AUDIT-2026-09-06-001` F-05
- **What:** `.env` in the repo root contains a real, non-placeholder `ANTHROPIC_API_KEY`
  (108-char value, line 545) plus 11 benign upstream debug/timeout toggles copied from
  `.env.example`.
- **Exposure:** `.env` is matched by `.gitignore` (`.env` + `.env.*`); it is **not tracked**
  and `git log --all -- .env` is empty (re-verified `AUDIT-2026-09-06-002`). No leak path
  found in this checkout. Risk is local (disk, shoulder-surf, accidental paste into an
  issue/PR/screenshot/log).
- **Update 2026-09-06 (`AUDIT-2026-09-06-002`):** the same key value also sits at
  **`D:\.env`** (portable-drive root, outside every git repo) — same rotation applies
  there. The in-repo mitigation (`.gitignore` broadening + `.githooks/`) is now
  **committed** to local `main` (`CHG-2026-09-06-010`), not just working-tree state.
- **Status:** **OPEN** — exposure structurally mitigated and committed; user action still required on the key.
- **Mitigation shipped 2026-09-06** (the "never push `.env`" rule):
  - `CHG-2026-09-06-007` — `.gitignore` broadened to `.env` + `.env.*` + `.op.env`
    (`!*.example` / `!*.sample`). Previously `.env.production`, `.env.staging`, etc.
    were **not** ignored — see `ERR-2026-09-06-003`.
  - `CHG-2026-09-06-008` — `.githooks/pre-commit` + `.githooks/pre-push` refuse to
    commit or push any real `.env`; `core.hooksPath = .githooks` set in this checkout.
  - History re-verified: `git log --all -- .env` is empty — `.env` has never been
    committed on any ref.
- **Required action (still open):**
  1. Confirm this key is meant to live on this machine.
  2. If its provenance or exposure history is at all uncertain, **rotate it** at
     `console.anthropic.com` and update `.env`.
  3. Never paste `.env` contents into commits, PRs, screenshots, or chat.
  4. On a fresh clone, run `sh .githooks/install` to re-arm the guard.
- **Do NOT** run `git clean -x`/`-X` in this repo — it would delete `.env`.

### ERR-2026-09-07-002 — LOW — Upstream test suite fails collection on Windows

- **Opened:** 2026-09-07 · **Base:** hermes@a7198a8855 (0 behind upstream/main)
- **Run:** RUN-2026-09-07-003
- **Source:** post-rebase test run for `CHG-2026-09-07-010` (the `upstream/main` sync).
- **Confidence:** Confirmed Fact — reproduced this session: `uv run --extra dev
  python -m pytest tests/hermes_cli/` aborts with
  `ERROR collecting tests/hermes_cli/test_doctor_journal_modes.py … AttributeError:
  module 'os' has no attribute 'geteuid'` and `Interrupted: 1 error during collection`.
- **What:** several upstream test modules evaluate `os.geteuid()` as an **eager
  argument** to a `@pytest.mark.skipif(...)` decorator, which runs at import /
  collection time — before the companion `@pytest.mark.skipif(os.name == "nt", …)`
  can suppress it. `os.geteuid` does not exist on Windows, so collection of the whole
  directory aborts. Known modules: `tests/hermes_cli/test_doctor_journal_modes.py`
  (last touched upstream by `b818085298`), and by grep also
  `test_ensure_acp_launcher.py`, `test_ssh_ownership_endpoint.py`,
  `test_update_autostash.py`, `tests/plugins/platforms/photon/test_sidecar_paths.py`,
  `tests/test_hermes_state_readonly_preflight.py`,
  `tests/tools/test_local_cwd_permission_fallback.py`,
  `tests/tools/test_stage2_hook_api_server_keygen.py` (not all confirmed to fail at
  collection — some may call `geteuid()` inside a function body, which is fine).
- **Not North Forge's:** every listed file is **byte-identical to `upstream/main`**
  (`git diff upstream/main HEAD -- <file>` empty). NF touches none of them. This is a
  pre-existing upstream Windows-portability defect that the `CHG-2026-09-07-010` sync
  simply pulled in; it is **not** a regression from `RUN-2026-09-07-003` steps 2/4,
  and North Forge's own targeted suites are green (238 passed / 4 skipped, see
  `CHG-2026-09-07-010`).
- **Impact:** an unfiltered `pytest` on Windows can't collect. Targeted runs
  (`pytest <file>::<node>`) and non-Windows CI are unaffected. Low.
- **Options for the owner (not defaulted this run):**
  1. Carry a small local test-compat shim (e.g. a `conftest.py` `getattr(os,
     "geteuid", lambda: -1)` fallback, or `--ignore` the offending files on Windows)
     — keeps a full local Windows run possible, adds fork drift on every merge.
  2. Wait for upstream to fix it; rely on Linux CI + targeted Windows runs meanwhile.
  3. Report upstream.
- **Status:** OPEN — documented so the `D:`→`E:` test-bed flow isn't surprised by it;
  no NF code change made. Owner to pick an option.

---

## Resolved

### ERR-2026-09-06-002 — MEDIUM — Fork identity / version drift

- **Opened:** 2026-09-06 · **Base:** hermes@820106d4a5 (2 behind upstream/main)
- **Source:** `AUDIT-2026-09-06-001` F-06, F-08
- **What:** two things were bundled under one id — (a) a **fault**: `origin/main` was
  0 ahead / 2 behind `upstream/main`, a stale pristine mirror; (b) a **choice**:
  `README*.md`, `SOUL.md`, `LICENSE`, and package metadata are all still
  upstream-branded, and whether to rebrand or stay a thin downstream was undecided.
- **Resolved (fault half) 2026-09-06:** `git rebase upstream/main` replayed the
  ledger commit onto `693641aa8b` (no conflicts); `git push origin main`
  fast-forwarded `origin/main` `820106d4a5` → `e6c97b43ef` (the 2 upstream commits +
  the ledger commit). Fork is now 0 behind `upstream/main`. Resolving change:
  `CHG-2026-09-06-014`.
- **Superseded-by:** `DECISION-2026-09-06-001` — the identity/rebrand **choice** was
  migrated to the decision register (`ledger-schema v2`, `CHG-2026-09-06-015`). It was
  never a fault; it does not belong here. The maintainer has since chosen **full
  rebrand** (2026-09-06); tracking of that now lives on `DECISION-2026-09-06-001`
  until `NF-v0.2.0` lands.
- **Status:** RESOLVED (fault half) / SUPERSEDED by `DECISION-2026-09-06-001` (choice half).

### ERR-2026-09-06-003 — MEDIUM — Secret hygiene (`.gitignore` gap)

- **Opened:** 2026-09-06 · **Base:** hermes@820106d4a5 (2 behind upstream/main)
- **Source:** manual check while implementing the "never push `.env`" rule
- **What:** `.gitignore` enumerated `.env`, `.env.local`, `.env.*.local`,
  `.env.development`, `.env.test`, `.op.env` — but **not** `.env.production`,
  `.env.staging`, `.env.ci`, or any other `.env.<name>`. Such a file would not have
  been ignored and a plain `git add .` would have staged it.
- **Exposure:** none realised — no such file exists in the working tree or history.
- **Resolved:** 2026-09-06 — `.gitignore` now uses `.env` + `.env.*` + `.op.env`
  with `!*.example` / `!*.sample`. Verified with `git check-ignore` across
  `.env.production` / `.env.staging` / `deep/nested/.env` / `app/.env.prod`.
  Resolving change: `CHG-2026-09-06-007`.
- **Status:** RESOLVED

### ERR-2026-09-06-004 — LOW — Stray Windows cache tree in repo root

- **Opened:** 2026-09-06 · **Base:** hermes@820106d4a5 (2 behind upstream/main)
- **Source:** `git status` after hook setup showed `?? %SystemDrive%/`
- **What:** a literal directory `%SystemDrive%/ProgramData/Microsoft/Windows/Caches/`
  containing Windows icon-cache DB files (`cversions.2.db`, `*.ver0x*.db`) appeared
  in the repo root (birth 2026-09-06 17:55). Created by some Windows process running
  with cwd = the repo and an **unexpanded** `%SystemDrive%` env var. Not produced by
  git — not reproducible from `git add` / `git commit`.
- **Exposure:** none — untracked; deleted before any commit.
- **Resolved:** 2026-09-06 — directory removed; `.gitignore` guards added
  (`/%SystemDrive%/`, `Thumbs.db`, `ehthumbs.db`, `[Dd]esktop.ini`, `$RECYCLE.BIN/`).
  Resolving change: `CHG-2026-09-06-009`.
- **Recurred 2026-09-06 (`AUDIT-2026-09-06-002` F-03):** the tree reappeared
  (birth 18:01, ~6 min after the first deletion). Deleted again (`CHG-2026-09-06-012`).
  The `/%SystemDrive%/` guard held — it never became git-visible. Stays RESOLVED;
  chasing the offending process is open-item #6 on `AUDIT-2026-09-06-002`.
- **Status:** RESOLVED (recurrence is expected and harmless while the guard stands).

### ERR-2026-09-06-005 — LOW — pytest / mock artifacts in the working tree

- **Opened:** 2026-09-06 · **Base:** hermes@820106d4a5 (2 behind upstream/main)
- **Run:** — (pre-`RUN-` tracking; `AUDIT-2026-09-06-002` session — retro-note added under `ledger-schema v2`)
- **Source:** `AUDIT-2026-09-06-002` F-02 (`git status` after a test run)
- **Confidence:** Confirmed Fact — the paths were listed by `git status` and inspected on disk before deletion.
- **What:** untracked, non-ignored paths written into the `north-forge-agent` repo
  root by test runs on this Windows checkout:
  - `MagicMock/mock._session_db.db_path/{3165824711312,3165827767312}` (+ `.fts_rebuild.lock`
    / `.quarantine.lock`) — a test left `_session_db.db_path` as a `MagicMock` and
    code opened `str(mock)` as a real path.
  - `C:UserskwalkAppDataLocalTemphermes-pytest-tmproot-2oumyv71…` / `…root-iyod624l…`
    (4 files, 0 B) — pytest tmp paths materialised as literal filenames in cwd.
  - `logs.zip` (462 KB) — a zipped copy of `logs/` dropped in the root.
- **Exposure / impact:** none realised — all untracked, deleted before any commit.
  Risk was a `git add -A` sweep and name-shadowing of real Windows paths.
- **Resolved:** 2026-09-06 — deleted (`CHG-2026-09-06-012`); `.gitignore` guards
  added — `MagicMock/`, `*hermes-pytest-tmp*`, `*pytest-tmproot*`, `*pytest-of-*`,
  `/logs.zip` (`CHG-2026-09-06-011`). Verified `git status` clean afterward.
- **Status:** RESOLVED (may recur; guards make recurrence harmless. Root-causing the
  test/tool is `AUDIT-2026-09-06-002` open-item #6).

---

### ERR-2026-09-07-001 — LOW — `bootstrap-north-forge.ps1` cross-volume slow path

- **Opened:** 2026-09-07 · **Base:** hermes@61d30533f7 (13 behind upstream/main)
- **Run:** RUN-2026-09-07-002
- **Source:** `FIRST-LAUNCH-WITNESS_2026-09-07` (AGENT E's first-launch run on `E:\`) —
  measured, not fixed by that agent (witness scope, no edit authority).
- **Confidence:** Confirmed Fact — timed on two drives: `E:\` first run 395.7 s
  wall (uv reported `Installed 67 packages in 6m 25s`); `D:\` after the fix 7.4 s
  cold / 5.0 s warm (`Installed 67 packages in 740ms`).
- **What:** `scripts/bootstrap-north-forge.ps1` let uv's package cache stay at its
  default location under `%LOCALAPPDATA%` on `C:` while building the venv on the
  checkout's own drive (`E:\north-forge-agent-venv`, `D:\north-forge-agent-venv`).
  uv installs by hardlinking from cache into the venv; across volumes the hardlink
  fails and uv full-copies every package instead (`warning: Failed to hardlink
  files; falling back to full copy`). Net: a first `north-forge.cmd` double-click
  took **~6.5 min** instead of the "~7 s" `CHG-2026-09-07-005` advertised.
- **Exposure / impact:** performance and first-impression only — the bootstrap
  produced a correct venv, just slowly, with a warning that reads like a failure.
- **Resolved:** 2026-09-07 — `CHG-2026-09-07-008`: the script now sets
  `$env:UV_CACHE_DIR` to a sibling of the venv (`<parent>\.uv-cache`, same volume)
  before the uv calls, unless the operator already set it. Re-measured on `D:\`:
  link step `740ms`, total 5–7 s, warning gone, sibling venv `hermes.exe
  --version` works.
- **Status:** RESOLVED. Note: `scripts/collect-logs.*` and the `.sh` bootstrap
  variant are unaffected (no `.sh` bootstrap exists); if one is added it needs the
  same `UV_CACHE_DIR` line.

---

## Register (quick scan)

| ID | Date | Sev | Area | Summary | Status | Resolved by |
| --- | --- | --- | --- | --- | --- | --- |
| ERR-2026-09-06-001 | 2026-09-06 | HIGH | Secret hygiene | Live `ANTHROPIC_API_KEY` in `.env` (also `D:\.env`); mitigation committed, key decision pending | OPEN | mitig. CHG-007/008/009/010 |
| ERR-2026-09-06-002 | 2026-09-06 | MEDIUM | Fork identity | Fault half (2 behind upstream) fixed by CHG-014; choice half migrated to DECISION-2026-09-06-001 | RESOLVED / SUPERSEDED | CHG-2026-09-06-014 → DECISION-2026-09-06-001 |
| ERR-2026-09-06-003 | 2026-09-06 | MEDIUM | Secret hygiene | `.gitignore` missed `.env.production` / `.env.<name>` | RESOLVED | CHG-2026-09-06-007 |
| ERR-2026-09-06-004 | 2026-09-06 | LOW | Repo hygiene | Stray `%SystemDrive%` Windows cache tree in root (recurred; guard held) | RESOLVED | CHG-2026-09-06-009 / -012 |
| ERR-2026-09-06-005 | 2026-09-06 | LOW | Repo hygiene | pytest/mock artifacts (`MagicMock/`, `C:Users…`, `logs.zip`) in working tree | RESOLVED | CHG-2026-09-06-011 / -012 |
| ERR-2026-09-07-001 | 2026-09-07 | LOW | Bootstrap tooling | `bootstrap-north-forge.ps1` let uv cache sit on `C:` while venv built on the checkout drive → cross-volume full-copy, ~6.5 min first run | RESOLVED | CHG-2026-09-07-008 |
| ERR-2026-09-07-002 | 2026-09-07 | LOW | Upstream test compat | Upstream test files call `os.geteuid()` in an eager `skipif` decorator arg → `pytest tests/` aborts at collection on Windows. Pre-existing upstream, pulled in by the `CHG-2026-09-07-010` sync; NF touches none of the files; targeted runs green | OPEN | — (owner to pick: local shim / wait upstream) |
