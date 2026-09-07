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

### ERR-2026-09-07-004 — HIGH — Ledger completeness check compares dates, not RUN IDs

- **Opened:** 2026-09-07 · **Base:** hermes@233757037d (6 behind upstream/main)
- **Run:** RUN-2026-09-07-006
- **Source:** Codex audit **F-02** (see `logs/CODEX-AUDIT-2026-09-07.md`).
- **Confidence:** Confirmed Fact (Codex read the code; not independently re-run this session).
- **What:** `scripts/collect-logs.sh` (~lines 183–203) and the equivalent
  PowerShell logic prove only *"a report exists dated ≥ the newest ledger ID's
  date."* They take the newest date embedded in any top-level report filename and
  compare it to the newest date in any ledger ID. One unrelated same-day report
  makes every run that day look covered; a report can also omit its `RUN-` id
  entirely and still satisfy the check. The deliberately off-date decision id
  (`DECISION-2026-09-06-003`, opened 2026-09-07) shows date ≠ run identity.
- **Impact:** the collector does **not** close the evidence gap it was built for —
  it detects "no report of at least this date," not "every run has a report."
- **Fix sketch (not done — R-01):** make completeness *relational* — a required
  `Run:` / `Covers:` header on each report, a `logs/ledger/reports/REPORT-MANIFEST`
  (basename, sha256, run id, covered ids), both collectors extract every `RUN-*`
  touched since the last handoff and match against report headers/manifest
  (missing run ⇒ **FAIL**, not WARN), reject unresolved `[[name]]` links, add
  parity tests (two runs one date, one report ⇒ flagged).
- **Status:** OPEN — logged for scheduling; no change this run.

### ERR-2026-09-07-006 — MEDIUM — Bootstrap readiness marker is not validated

- **Opened:** 2026-09-07 · **Base:** hermes@233757037d (6 behind upstream/main)
- **Run:** RUN-2026-09-07-006
- **Source:** Codex audit **F-05**.
- **Confidence:** Confirmed Fact (code inspection).
- **What:** `.nf-bootstrapped` records `repo=<path>`, but the bootstrap
  early-return and `north-forge.cmd` only test that the marker and `hermes.exe`
  *exist*. Neither checks that the editable install still points at the current
  checkout. A renamed checkout, a copied launcher, or a reused sibling venv can
  silently run code from a different/old repository — likeliest on portable
  drives whose folder name or letter changes.
- **Impact:** confusing stale-code execution; no data loss.
- **Fix sketch (not done — R-03 step 3):** parse `.nf-bootstrapped`; verify its
  `repo=` equals the current `$RepoRoot`, that `hermes.exe` belongs to the
  selected venv, and that `import hermes_cli` resolves under the current repo;
  offer a safe rebuild when it doesn't.
- **Status:** OPEN — logged for scheduling; no change this run.

### ERR-2026-09-07-007 — MEDIUM — Secret defenses narrower than "credential protection" implies

- **Opened:** 2026-09-07 · **Base:** hermes@233757037d (6 behind upstream/main)
- **Run:** RUN-2026-09-07-006
- **Source:** Codex audit **F-09** (residual scope beyond the `ERR-2026-09-07-005`
  bypass).
- **Confidence:** Confirmed Fact for implemented coverage; Field-Reasoned for
  residual exposure likelihood.
- **What:** the secret controls work but are narrow — commit scanning recognizes
  only selected AWS / GitHub / Slack token shapes; filename blocking is
  essentially `.env`-family only (`credentials.json`, `id_rsa`, `*.pfx`, `*.p12`,
  service-account JSON, arbitrary token exports are not uniformly blocked);
  `.gitignore` is not a security boundary and `git add -f` bypasses it; the
  handoff redactor skips binary/large files; both hooks yield to `--no-verify`
  (CI must stay authoritative).
- **Impact:** "credential protection" oversells the current net; several common
  secret-bearing file types can be staged without a hook objecting.
- **Fix sketch (not done — R-02 steps 4–5):** add a maintained pinned CI scanner
  (e.g. Gitleaks) for broad provider coverage alongside the fast local hook; add
  filename policy for private-key / container formats with line-scoped, reviewed
  allowlists; emit `file:line` + rule name, never the secret text.
- **Status:** OPEN — logged for scheduling; no change this run.

---

## Resolved

### ERR-2026-09-07-005 — HIGH — `.githooks/content-scan` whitespace-path bypass

- **Opened:** 2026-09-07 · **Base:** hermes@233757037d (6 behind upstream/main)
- **Run:** RUN-2026-09-07-006 (opened) · RUN-2026-09-07-007 (fixed)
- **Source:** Codex audit **F-03** — reproduced by Codex in a scratch repo.
- **Confidence:** Confirmed Fact — reproduced **and** verified fixed this session
  (`RUN-2026-09-07-007`): a `ghp_`-shaped token in `"my dir/config file.txt"`,
  scanned via `content-scan --commits HEAD~1..HEAD` — the pre-fix script
  (`git show HEAD~1:.githooks/content-scan`) returned exit 0 "clean"; the fixed
  script exits 1 and reports `…:my dir/config file.txt:1:GITHUB_TOKEN = "ghp_…"`.
- **What:** in `.githooks/content-scan` the `--commits` gate captured the
  changed-path list as newline text and expanded it **unquoted** into `_scan`
  (`_scan "$c" $paths`). A legal path like `dir/file name.txt` word-split into two
  non-existent pathspecs, so the blob was never scanned. `content-scan --commits`
  returned exit 0 / "clean" for a planted GitHub token in such a file. Both the
  pre-push hook and the `nf-secret-scan.yml` CI job use this mode, so an ordinary
  filename with a space defeated the content gate. The six existing self-tests
  covered only ASCII/no-space paths.
- **Impact:** live gap in an advertised security mechanism — a secret in a
  space-containing filename passed both local and CI content scanning.
- **Resolved:** 2026-09-07 (`RUN-2026-09-07-007`, `CHG-2026-09-07-019`,
  `NF-v0.5.3`). The `--commits` path no longer stores paths in shell variables at
  all — new `_scan_commit` reads `git diff-tree --no-commit-id -r --no-renames
  --diff-filter=d -z` (NUL-delimited, quoting disabled) one raw record at a time
  and hands `git grep` the **post-image blob OID** (`_scan_blob`); no path is ever
  passed to git as a pathspec on the gate path, so word-splitting cannot happen.
  `--tree` / `--worktree` (which only ever pass the literal pathspec `.`) are
  unchanged. `.githooks/tests/run.sh` gains 7 cases: secret in a space / tab /
  leading-dash / non-ASCII filename, a rename-into-a-spaced-path in one commit, an
  add-then-delete of a spaced path across the range, and a negative control
  (spaced filename, no secret ⇒ not flagged). 13/13 pass under both `bash` and
  `dash` (CI's shell); `sh -n` / `dash -n` clean.
- **Not touched:** `.githooks/secret-guard` (the `.env` *filename* guard) — out of
  scope for this fix.
- **Status:** RESOLVED.

### ERR-2026-09-07-003 — HIGH — `bootstrap-north-forge.ps1` `-Force` can delete the checkout

- **Opened:** 2026-09-07 · **Base:** hermes@233757037d (6 behind upstream/main)
- **Run:** RUN-2026-09-07-005 (opened + resolved same run)
- **Source:** Codex audit finding **F-04** (data-loss).
- **Confidence:** Confirmed Fact — reproduced this session: a fake checkout
  (`pyproject.toml` + a canary file) passed as **both** `-RepoRoot` and
  `-VenvDir` with `-Force`. Before the fix the guard let it through; the script
  would then reach `Remove-Item -LiteralPath $VenvDir -Recurse -Force`. After the
  fix the run exits non-zero at the guard with a "Refusing to bootstrap" error
  and the checkout is untouched. Covered by
  `tests/test_bootstrap_north_forge_path_safety.py` (11 cases; the
  script-execution ones are Windows-only, ran green here).
- **What:** the path-safety guard rejected a venv/data path **strictly inside**
  the repo root — `"$full".TrimEnd('\').ToLower().StartsWith($RepoRoot.TrimEnd('\').ToLower() + '\')`
  — but a path **equal** to the repo root does not start with `repoRoot + '\'`,
  so `-VenvDir <repo root>` passed the check. On `-Force`,
  `Remove-Item -LiteralPath $VenvDir -Recurse -Force` then recursively deletes the
  working tree. `repo-root-inside-venv` (e.g. `-VenvDir <parent of checkout>`)
  was also unguarded. The equality gap applied to `-DataDir` too.
- **Exploitability:** the default path is **safe** — `north-forge.cmd` calls
  `bootstrap-north-forge.ps1` with **no** `-VenvDir` / `-DataDir`, so it always
  uses the computed `<parent>\<leaf>-venv` / `-data` siblings, which the guard
  (old and new) accepts. The bug bites only a caller that explicitly passes a
  venv/data path equal to (or containing, or contained by) the checkout. No such
  caller exists in-repo. Marked HIGH (unrecoverable local data loss if hit;
  recoverable from `origin/main` since the tree is pushed) — the audit framed it
  "critical".
- **Resolved:** 2026-09-07 (`RUN-2026-09-07-005`, `CHG-2026-09-07-015`,
  `NF-v0.5.1`). New helpers `Get-CanonicalDir` (`[IO.Path]::GetFullPath` + trim
  `\` `/` + keep a bare drive root) and `Test-PathOverlap` (ordinal-ignore-case
  `[string]::Equals`, then a `StartsWith(other + '\')` both directions). The
  guard now rejects venv/data **equal to**, **inside**, or **containing** the
  checkout, for both dirs. Genuine siblings (`<leaf>-venv`, `<leaf>-data`) are
  still accepted — verified by `test_accepts_genuine_siblings`.
- **Status:** RESOLVED.

### ERR-2026-09-07-002 — LOW — Upstream test suite fails collection on Windows

- **Opened:** 2026-09-07 · **Base:** hermes@a7198a8855 (0 behind upstream/main)
- **Run:** RUN-2026-09-07-003 (opened) · RUN-2026-09-07-004 (accepted)
- **Source:** post-rebase test run for `CHG-2026-09-07-010` (the `upstream/main`
  sync). Already flagged informally in the `RUN-2026-09-07-002` handoff note
  (`IDENTITY-RUNTIME_2026-09-07.md`, "two pre-existing Windows test issues") at the
  old base `hermes@61d30533f7` — so it **predates the sync**; logged as its own
  `ERR-` now because the 333-commit sync widened it (upstream `b818085298`
  "repoint … 13 tests" touched `test_doctor_journal_modes.py`) and it now aborts
  collection of the whole `tests/hermes_cli/` directory, not just one file.
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
- **Options considered:**
  1. Carry a small local test-compat shim (e.g. a `conftest.py` `getattr(os,
     "geteuid", lambda: -1)` fallback, or `--ignore` the offending files on Windows)
     — keeps a full local Windows run possible, adds fork drift on every merge.
  2. Wait for upstream to fix it; rely on Linux CI + targeted Windows runs meanwhile.
  3. Report upstream.
- **Resolved (ACCEPTED-RISK):** 2026-09-07 (`RUN-2026-09-07-004`) — owner decision:
  **accept as-is, option 2.** No local `conftest.py` shim and no `--ignore` list is
  added — that would put fork drift on `tests/` (a surface NF otherwise keeps
  byte-identical to upstream) on every merge, to paper over a defect that is
  upstream's to fix. North Forge relies on upstream's Linux CI plus targeted
  Windows runs (`pytest <file>::<node>`), which are unaffected. **Revisit trigger:**
  upstream fixes the eager-`skipif` pattern (drop this note), *or* a full
  unfiltered local Windows `pytest` run becomes necessary for NF work (then add the
  minimal `conftest.py` `getattr` shim under a new `CHG-`). Reporting upstream
  (option 3) is encouraged but not tracked here. Recorded by `CHG-2026-09-07-014`;
  no code change.
- **Status:** ACCEPTED-RISK (owner, `RUN-2026-09-07-004`). Pre-existing upstream
  Windows-portability defect; NF touches none of the affected files.

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
| ERR-2026-09-07-002 | 2026-09-07 | LOW | Upstream test compat | Upstream test files call `os.geteuid()` in an eager `skipif` decorator arg → `pytest tests/` aborts at collection on Windows. Pre-existing upstream, pulled in by the `CHG-2026-09-07-010` sync; NF touches none of the files; targeted runs green | ACCEPTED-RISK | CHG-2026-09-07-014 (owner: accept as-is, no shim; rely on Linux CI + targeted runs; revisit if upstream fixes or a full local Windows run is needed) |
| ERR-2026-09-07-003 | 2026-09-07 | HIGH | Bootstrap tooling | Codex F-04 (data-loss): `bootstrap-north-forge.ps1` path guard rejected venv/data *inside* the repo but not *equal to* it → `-VenvDir <repo>` + `-Force` runs `Remove-Item -Recurse` on the checkout. Default `north-forge.cmd` path unaffected (no `-VenvDir` passed). Canonicalize + reject equal/inside/contains for venv AND data | RESOLVED | CHG-2026-09-07-015 (+ `tests/test_bootstrap_north_forge_path_safety.py`) |
| ERR-2026-09-07-004 | 2026-09-07 | HIGH | Ledger tooling | Codex F-02: `collect-logs.{sh,ps1}` completeness check compares newest report-filename *date* to newest ledger-ID date, not `RUN-` id to report. One same-day report covers every run that day; run id can be absent entirely | OPEN | — (R-01: RUN-to-report manifest + FAIL on missing run) |
| ERR-2026-09-07-005 | 2026-09-07 | HIGH | Secret scanning | Codex F-03 (reproduced): `.githooks/content-scan` expands changed paths unquoted → a filename with a space word-splits into non-existent pathspecs; a planted `ghp_` token in `dir/file name.txt` passed `--commits` clean. Pre-push + CI both affected | RESOLVED | CHG-2026-09-07-019 — `--commits` gate now reads `git diff-tree -z` and scans by post-image **blob OID**, never by path string; +7 `run.sh` cases (space/tab/dash/Unicode/rename/add-delete/negative). Verified before/after |
| ERR-2026-09-07-006 | 2026-09-07 | MEDIUM | Bootstrap tooling | Codex F-05: `.nf-bootstrapped` / `north-forge.cmd` only check the marker + `hermes.exe` exist, never that the editable install points at the current checkout → renamed/copied checkout can launch stale code | OPEN | — (R-03.3: verify marker `repo=` == `$RepoRoot`, venv owns `hermes.exe`, `import hermes_cli` resolves in-repo) |
| ERR-2026-09-07-007 | 2026-09-07 | MEDIUM | Secret scanning | Codex F-09: coverage is narrow — 3 provider token shapes only; filename block is `.env`-family only (`credentials.json` / `id_rsa` / `*.pfx` / SA-JSON unblocked); redactor skips binary/large; `--no-verify` bypasses hooks | OPEN | — (R-02.4/5: pinned maintained CI scanner + private-key/container filename policy) |
