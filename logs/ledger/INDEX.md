# North-Forge Ledger Index

Master pointer for the project ledger. Newest first. Rules: [`README.md`](./README.md).

- **Changelog:** [`CHANGELOG.md`](./CHANGELOG.md) — every tracked-file / config change.
- **Error register:** [`errors/ERROR-LOG.md`](./errors/ERROR-LOG.md) — faults (breaks, regressions, leaks).
- **Decision register:** [`decisions/DECISION-LOG.md`](./decisions/DECISION-LOG.md) — open judgment calls.
- **Templates:** [`templates/`](./templates/) — copy when adding an audit / change / error / decision.

## Current coordinates

| Coordinate | Value | As of |
| --- | --- | --- |
| Upstream base | `hermes@233757037d` — **6 behind `upstream/main`** as of `RUN-2026-09-07-004`. `origin/main` advanced +15 mid-run: the owner's `c4e88d2ab6` `Merge branch 'NousResearch:main'` (upstream `agent/` + `hermes_cli/` model/pricing/codex changes). This run's commit was rebased **clean** onto `c4e88d2ab6` (no file overlap). Prior: `RUN-2026-09-07-003` rebase replayed the 8 NF commits (333→0 behind) onto `a7198a8855`. | 2026-09-07 |
| North-Forge version | `NF-v0.5.3` (`RUN-2026-09-07-007`, `CHG-2026-09-07-019`): **security fix** — `.githooks/content-scan` `--commits` gate no longer word-splits changed paths (Codex `F-03`, `ERR-2026-09-07-005`); it reads `git diff-tree -z` and scans by post-image blob OID. A secret in a space/tab/dash/Unicode filename was passing pre-push **and** CI; +7 `run.sh` cases. `NF-v0.5.2` (`RUN-2026-09-07-006`, `CHG-2026-09-07-016..018`): two cheap Codex-audit fixes — README stale-MinGit prose removed / linked upstream (`F-08`); `nf-secret-scan.yml` `actions/checkout` pinned to SHA (`F-10`) — plus ledger back-logging of Codex `F-02`/`F-05`/`F-07`/`F-09` (`ERR-2026-09-07-004`/`006`/`007`, `DECISION-2026-09-07-002`). No behavioural code change. `NF-v0.5.1` (`RUN-2026-09-07-005`, `CHG-2026-09-07-015`): **data-loss fix** — `bootstrap-north-forge.ps1` path guard rejects a venv/data dir equal to / inside / containing the checkout (Codex `F-04`, `ERR-2026-09-07-003`). `NF-v0.5.0` (`RUN-2026-09-07-004`, `CHG-2026-09-07-011..014`): README Quick Install restructure (drive-native `north-forge.cmd` first); North Forge CLI skin (`skins/north-forge.yaml`, `DECISION-2026-09-07-001` **DECIDED — skin swap stands**); self-healing `<drive>:\Start North Forge.lnk` launcher. `NF-v0.4.2` (`CHG-2026-09-07-010`: `upstream/main` sync). `NF-v0.4.1` (`CHG-2026-09-07-009`). `NF-v0.4.0` first application-code identity pass. `NF-v0.2.0`..`NF-v0.5.2` public on `origin/main`. | 2026-09-07 |
| Ledger schema | `v2` | 2026-09-06 |
| Latest run | `RUN-2026-09-07-007` — **security fix, landed alone**: `CHG-2026-09-07-019` / `ERR-2026-09-07-005` (Codex `F-03`) — `.githooks/content-scan` `--commits` gate rewritten to read `git diff-tree -z` NUL-delimited and scan by **post-image blob OID**, so a changed path with a space / tab / leading dash / Unicode can no longer word-split and skip the scan (it was defeating pre-push **and** `nf-secret-scan.yml` CI). `.githooks/tests/run.sh` +7 cases (13/13 under `bash` + `dash`); `--tree`/`--worktree` and `secret-guard` untouched. Verified before/after with the exact Codex repro. `NF-v0.5.3` cut + pushed. **Next queued:** the docs-only batch (`DOC-01..DOC-10`, `logs/CODEX-DOCS-AUDIT-2026-09-07.md`). Prior: `RUN-2026-09-07-006` (Codex cheap fixes + back-log, `CHG-2026-09-07-016..018`, `NF-v0.5.2`); `RUN-2026-09-07-005` (F-04 data-loss fix, `NF-v0.5.1`); `RUN-2026-09-07-004` (drive-native onboarding, `NF-v0.5.0`); `RUN-2026-09-07-003` (push/sync); `RUN-2026-09-07-002`; `RUN-2026-09-07-001`. | 2026-09-07 |
| First fork commits | `NF-v0.1.0` + `NF-v0.1.1` on `origin/main` (`CHG-2026-09-06-010`); fork synced to upstream (`CHG-2026-09-06-014`) | 2026-09-06 |
| Fork identity | **full rebrand** — `NF-v0.2.0` (`CHG-2026-09-06-020..024`); branding-pass finish `NF-v0.2.1`; `NF-v0.3.0` completed the branding (final banner + `assets/icons/`, generic `SOUL.md` + `editions/field-service/` overlay, `BRANDING.md` source of truth, translated-README landing pages). `DECISION-2026-09-06-001` DECIDED. Pushed. | 2026-09-07 |
| Security guards | `.githooks/secret-guard` (filenames) + `.githooks/content-scan` (AWS/GitHub/Slack content, `--commits` gate + CI) + `scripts/redact_handoff.py` (mandatory handoff redaction, fail-closed). `CHG-2026-09-07-001/002`. | 2026-09-07 |

## Audits

| Audit | Date | Version | Scope | Open findings |
| --- | --- | --- | --- | --- |
| [AUDIT-2026-09-06-001](./audits/AUDIT-2026-09-06-001-repository-baseline.md) | 2026-09-06 | `NF-v0.1.0` / `hermes@820106d4a5` | Baseline: provenance, divergence, hygiene, secrets, CI, full README review. F-01–F-10 (F-09/F-10 added same session). | `ERR-2026-09-06-001` (HIGH), `ERR-2026-09-06-002` (MEDIUM) |
| [AUDIT-2026-09-06-002](./audits/AUDIT-2026-09-06-002-hardening-commit-and-hygiene.md) | 2026-09-06 | `NF-v0.1.1` / `hermes@820106d4a5` | Commit the baseline hardening set to `main`; clear recurred Windows/pytest cruft; git-freshness check on all 3 `D:\` repos; `.git` gc. F-01–F-07. | `ERR-2026-09-06-001` (HIGH), `ERR-2026-09-06-002` (MEDIUM) — both carried, none new open |

## Open incidents (faults)

| ID | Sev | Summary |
| --- | --- | --- |
| [ERR-2026-09-06-001](./errors/ERROR-LOG.md#err-2026-09-06-001--high--secret-hygiene) | HIGH | Live `ANTHROPIC_API_KEY` in `.env` (and `D:\.env`). Exposure mitigated **and committed** (`.githooks/` + `.gitignore`); confirm/rotate still pending. |
| [ERR-2026-09-07-004](./errors/ERROR-LOG.md#err-2026-09-07-004--high--ledger-completeness-check-compares-dates-not-run-ids) | HIGH | Codex `F-02` — `collect-logs.{sh,ps1}` completeness check compares report/ledger **dates**, not `RUN-` ids. Logged for scheduling (R-01). |
| [ERR-2026-09-07-006](./errors/ERROR-LOG.md#err-2026-09-07-006--medium--bootstrap-readiness-marker-is-not-validated) | MEDIUM | Codex `F-05` — `.nf-bootstrapped` never checked against the current checkout → stale-code launch on renamed/copied drives. Logged (R-03.3). |
| [ERR-2026-09-07-007](./errors/ERROR-LOG.md#err-2026-09-07-007--medium--secret-defenses-narrower-than-credential-protection-implies) | MEDIUM | Codex `F-09` — secret coverage narrow (3 token shapes; `.env`-only filename block). Logged (R-02.4/5). |

## Open decisions (judgment calls)

| ID | Area | Question | State |
| --- | --- | --- | --- |
| [DECISION-2026-09-06-002](./decisions/DECISION-LOG.md#decision-2026-09-06-002--attic-clone--keep-or-delete-it) | Repo hygiene | Keep or delete the attic clone (~869 MB)? | Leaning **A (delete)** — `origin/main` now carries the work; disk only |
| [DECISION-2026-09-07-002](./decisions/DECISION-LOG.md#decision-2026-09-07-002--rebrand-claim-wording--engine-used-unmodified--full-rebrand) | Branding wording | "engine used unmodified" / "full rebrand" broader than the code (Codex `F-07`)? | Leaning **A** — tighten wording, no identifier churn. Logged only. |
| [DECISION-2026-09-06-003](./decisions/DECISION-LOG.md#decision-2026-09-06-003--install-model--drive-native-run-in-place-vs-machine-local-managed-install) | Install model | Drive-native run-in-place vs machine-local managed install? | Leaning **A (drive-native)**, **not ratified**. Minimal bootstrap ships (`CHG-2026-09-07-005`); hardened form (seal / dual-volume / certify) blocked on ratification. |

## Resolved decisions

| ID | Area | Question | Outcome |
| --- | --- | --- | --- |
| [DECISION-2026-09-07-001](./decisions/DECISION-LOG.md#decision-2026-09-07-001--splash-art--keep-the-stock-hermes-launch-mark-or-swap-it) | Branding | Keep the stock Hermes launch splash, or swap it? | **DECIDED** — C (swap via a North Forge skin), `RUN-2026-09-07-004` / `CHG-2026-09-07-012`. `skins/north-forge.yaml` carries `banner_logo`/`banner_hero`; `hermes_cli/banner.py` unchanged. |
| [DECISION-2026-09-06-001](./decisions/DECISION-LOG.md#decision-2026-09-06-001--fork-identity--rebrand-vs-thin-downstream) | Fork identity | Rebrand vs thin downstream? | **DECIDED** — B (full rebrand), implemented `RUN-2026-09-06-002` / `NF-v0.2.0` (`CHG-2026-09-06-020..024`). `pyproject.toml` distribution name kept as `hermes-agent` per the entry's carve-out. |

## Resolved incidents

| ID | Sev | Summary | Fixed by |
| --- | --- | --- | --- |
| ERR-2026-09-06-002 | MEDIUM | Fault half (fork 2 behind upstream) fixed; choice half → `DECISION-2026-09-06-001` | CHG-2026-09-06-014 / migrated |
| ERR-2026-09-06-003 | MEDIUM | `.gitignore` missed `.env.production` / `.env.<name>` | CHG-2026-09-06-007 |
| ERR-2026-09-06-004 | LOW | Stray `%SystemDrive%` Windows cache tree in repo root (recurred; guard held) | CHG-2026-09-06-009 / -012 |
| ERR-2026-09-06-005 | LOW | pytest/mock artifacts (`MagicMock/`, `C:Users…`, `logs.zip`) in working tree | CHG-2026-09-06-011 / -012 |
| ERR-2026-09-07-001 | LOW | `bootstrap-north-forge.ps1` uv cache on `C:` vs venv on checkout drive → cross-volume full-copy, ~6.5 min first run | CHG-2026-09-07-008 |
| ERR-2026-09-07-002 | LOW | Upstream test files call `os.geteuid()` in an eager `skipif` decorator arg → unfiltered `pytest` aborts at collection on Windows. Pre-existing upstream; NF touches none of the files; targeted runs green | **ACCEPTED-RISK** — CHG-2026-09-07-014 (owner: no shim; Linux CI + targeted runs; revisit if upstream fixes or a full local Windows run is needed) |
| ERR-2026-09-07-003 | HIGH | Codex F-04 (data-loss): `bootstrap-north-forge.ps1` path guard missed venv/data *equal to* the checkout → `-VenvDir <repo>` + `-Force` = `Remove-Item -Recurse` on the tree. Default `north-forge.cmd` path unaffected | CHG-2026-09-07-015 (canonicalize + reject equal/inside/contains, both dirs; + `tests/test_bootstrap_north_forge_path_safety.py`) |
| ERR-2026-09-07-005 | HIGH | Codex F-03: `.githooks/content-scan` `--commits` gate word-split space/tab/dash/Unicode paths → a planted `ghp_` token in `dir/file name.txt` passed pre-push **and** `nf-secret-scan.yml` CI | CHG-2026-09-07-019 (`--commits` reads `git diff-tree -z`, scans by post-image blob OID, never by path string; `.githooks/tests/run.sh` +7 cases; verified before/after) |
