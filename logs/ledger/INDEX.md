# North-Forge Ledger Index

Master pointer for the project ledger. Newest first. Rules: [`README.md`](./README.md).

- **Changelog:** [`CHANGELOG.md`](./CHANGELOG.md) — every tracked-file / config change.
- **Error register:** [`errors/ERROR-LOG.md`](./errors/ERROR-LOG.md) — faults (breaks, regressions, leaks).
- **Decision register:** [`decisions/DECISION-LOG.md`](./decisions/DECISION-LOG.md) — open judgment calls.
- **Templates:** [`templates/`](./templates/) — copy when adding an audit / change / error / decision.

## Current coordinates

| Coordinate | Value | As of |
| --- | --- | --- |
| Upstream base | `hermes@421c72f3bc` (= `NF-v0.4.0` HEAD; merge-base with upstream is `hermes@08b140d14e`) — **333 behind `upstream/main`** (tip `a7198a8855`). The last handoff's "13 behind" is stale — upstream moved ~320 commits since `NF-v0.3.0` was cut. Mechanical sync attempted `RUN-2026-09-07-003`: see `CHANGELOG.md` and the session report. | 2026-09-07 |
| North-Forge version | `NF-v0.4.1` (`RUN-2026-09-07-003`: legacy-SOUL auto-upgrade closes the `CHG-2026-09-07-007` gap — homes seeded `NF-v0.2.0`..`NF-v0.3.x` now converge to the North Forge `DEFAULT_SOUL_MD`). `NF-v0.4.0` (first application-code identity pass: seeded/fallback persona + `hermes --version` / `--help` display name → North Forge; `BRANDING.md` recategorized; bootstrap cross-volume slow path fixed) **is now pushed** to `origin/main`. `NF-v0.2.0`..`NF-v0.4.1` public. | 2026-09-07 |
| Ledger schema | `v2` | 2026-09-06 |
| Latest run | `RUN-2026-09-07-003` — consolidated push/sync pass: pushed `NF-v0.4.0`; `CHG-2026-09-07-009` legacy-SOUL auto-upgrade (`NF-v0.4.1`); mechanical `upstream/main` sync attempt (333 behind); `⚕ NOUS HERMES` splash art **left open** for the owner (not defaulted). Prior: `RUN-2026-09-07-002` (`CHG-2026-09-07-007`/`-008`, `ERR-2026-09-07-001`); `RUN-2026-09-07-001` (`CHG-2026-09-07-001..006`). | 2026-09-07 |
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

## Open decisions (judgment calls)

| ID | Area | Question | State |
| --- | --- | --- | --- |
| [DECISION-2026-09-06-002](./decisions/DECISION-LOG.md#decision-2026-09-06-002--attic-clone--keep-or-delete-it) | Repo hygiene | Keep or delete the attic clone (~869 MB)? | Leaning **A (delete)** — `origin/main` now carries the work; disk only |
| [DECISION-2026-09-06-003](./decisions/DECISION-LOG.md#decision-2026-09-06-003--install-model--drive-native-run-in-place-vs-machine-local-managed-install) | Install model | Drive-native run-in-place vs machine-local managed install? | Leaning **A (drive-native)**, **not ratified**. Minimal bootstrap ships (`CHG-2026-09-07-005`); hardened form (seal / dual-volume / certify) blocked on ratification. |

## Resolved decisions

| ID | Area | Question | Outcome |
| --- | --- | --- | --- |
| [DECISION-2026-09-06-001](./decisions/DECISION-LOG.md#decision-2026-09-06-001--fork-identity--rebrand-vs-thin-downstream) | Fork identity | Rebrand vs thin downstream? | **DECIDED** — B (full rebrand), implemented `RUN-2026-09-06-002` / `NF-v0.2.0` (`CHG-2026-09-06-020..024`). `pyproject.toml` distribution name kept as `hermes-agent` per the entry's carve-out. |

## Resolved incidents

| ID | Sev | Summary | Fixed by |
| --- | --- | --- | --- |
| ERR-2026-09-06-002 | MEDIUM | Fault half (fork 2 behind upstream) fixed; choice half → `DECISION-2026-09-06-001` | CHG-2026-09-06-014 / migrated |
| ERR-2026-09-06-003 | MEDIUM | `.gitignore` missed `.env.production` / `.env.<name>` | CHG-2026-09-06-007 |
| ERR-2026-09-06-004 | LOW | Stray `%SystemDrive%` Windows cache tree in repo root (recurred; guard held) | CHG-2026-09-06-009 / -012 |
| ERR-2026-09-06-005 | LOW | pytest/mock artifacts (`MagicMock/`, `C:Users…`, `logs.zip`) in working tree | CHG-2026-09-06-011 / -012 |
| ERR-2026-09-07-001 | LOW | `bootstrap-north-forge.ps1` uv cache on `C:` vs venv on checkout drive → cross-volume full-copy, ~6.5 min first run | CHG-2026-09-07-008 |
