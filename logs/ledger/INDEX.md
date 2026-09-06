# North-Forge Ledger Index

Master pointer for the project ledger. Newest first. Rules: [`README.md`](./README.md).

- **Changelog:** [`CHANGELOG.md`](./CHANGELOG.md) — every tracked-file / config change.
- **Error register:** [`errors/ERROR-LOG.md`](./errors/ERROR-LOG.md) — faults (breaks, regressions, leaks).
- **Decision register:** [`decisions/DECISION-LOG.md`](./decisions/DECISION-LOG.md) — open judgment calls.
- **Templates:** [`templates/`](./templates/) — copy when adding an audit / change / error / decision.

## Current coordinates

| Coordinate | Value | As of |
| --- | --- | --- |
| Upstream base | `hermes@693641aa8b` (**0 behind** `upstream/main`) | 2026-09-06 |
| North-Forge version | `NF-v0.1.2` (`NF-v0.1.1` pushed; `v0.1.2` local, see CHANGELOG) | 2026-09-06 |
| Ledger schema | `v2` | 2026-09-06 |
| Latest run | `RUN-2026-09-06-001` — ledger-schema v2 additions | 2026-09-06 |
| First fork commits | `NF-v0.1.0` + `NF-v0.1.1` on `origin/main` (`CHG-2026-09-06-010`); fork synced to upstream (`CHG-2026-09-06-014`) | 2026-09-06 |
| Fork identity | decided **full rebrand** (`DECISION-2026-09-06-001`), not yet implemented — `NF-v0.2.0` pending | 2026-09-06 |

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
| [DECISION-2026-09-06-001](./decisions/DECISION-LOG.md#decision-2026-09-06-001--fork-identity--rebrand-vs-thin-downstream) | Fork identity | Rebrand vs thin downstream? | Leaning **B (full rebrand)**, chosen 2026-09-06 — not yet implemented; blocks `NF-v0.2.0` |
| [DECISION-2026-09-06-002](./decisions/DECISION-LOG.md#decision-2026-09-06-002--attic-clone--keep-or-delete-it) | Repo hygiene | Keep or delete the attic clone (~869 MB)? | Leaning **A (delete)** — `origin/main` now carries the work; disk only |

## Resolved incidents

| ID | Sev | Summary | Fixed by |
| --- | --- | --- | --- |
| ERR-2026-09-06-002 | MEDIUM | Fault half (fork 2 behind upstream) fixed; choice half → `DECISION-2026-09-06-001` | CHG-2026-09-06-014 / migrated |
| ERR-2026-09-06-003 | MEDIUM | `.gitignore` missed `.env.production` / `.env.<name>` | CHG-2026-09-06-007 |
| ERR-2026-09-06-004 | LOW | Stray `%SystemDrive%` Windows cache tree in repo root (recurred; guard held) | CHG-2026-09-06-009 / -012 |
| ERR-2026-09-06-005 | LOW | pytest/mock artifacts (`MagicMock/`, `C:Users…`, `logs.zip`) in working tree | CHG-2026-09-06-011 / -012 |
