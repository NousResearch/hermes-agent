# North-Forge Project Ledger

`ledger-schema v1` · single source of truth for **what changed, what broke, and what was audited** in this repository.

This checkout is a fork of [`NousResearch/hermes-agent`](https://github.com/NousResearch/hermes-agent).
The ledger records **north-forge's own history on top of upstream** — every tracked-file
or config change, every incident, and every audit — in one place, under strict naming
and version rules so the record stays greppable and unambiguous.

---

## Location rules

| Path | Purpose | Git |
| --- | --- | --- |
| `logs/` | Hermes **runtime** logs (session logs, `*.log`). Transient. | ignored (`.gitignore` `logs/*`) |
| `logs/ledger/` | **This ledger.** Audits, changelog, error register, templates. Permanent. | **tracked** (`.gitignore` `!logs/ledger/`) |

Nothing else goes in `logs/ledger/`. Do not put runtime output, scratch files, or
data dumps here. If it is not an audit, a change record, or an incident record, it
does not belong.

```
logs/ledger/
├── README.md            ← this file (the rules)
├── INDEX.md             ← chronological index of every audit + pointers
├── CHANGELOG.md         ← every project change, newest first, under version headings
├── audits/             ← one file per audit
│   └── AUDIT-YYYY-MM-DD-NNN-<slug>.md
├── errors/
│   └── ERROR-LOG.md     ← append-only incident / error / anomaly register
└── templates/          ← copy these when adding an entry
    ├── AUDIT-TEMPLATE.md
    ├── CHANGE-ENTRY-TEMPLATE.md
    └── ERROR-ENTRY-TEMPLATE.md
```

---

## Naming — strict

| Record | Where | Identifier / filename | Example |
| --- | --- | --- | --- |
| **Audit** | `audits/` (one file each) | `AUDIT-YYYY-MM-DD-NNN-<kebab-slug>.md` | `AUDIT-2026-09-06-001-repository-baseline.md` |
| **Change** | bullet in `CHANGELOG.md` | `CHG-YYYY-MM-DD-NNN` | `CHG-2026-09-06-004` |
| **Error / incident** | row in `errors/ERROR-LOG.md` | `ERR-YYYY-MM-DD-NNN` | `ERR-2026-09-06-001` |

Rules:

- **Date** — ISO 8601 `YYYY-MM-DD`, local time (America/Los_Angeles, same clock as `git log`).
- **NNN** — zero-padded 3-digit sequence. Resets each day. Counted **per record type**.
- **`<kebab-slug>`** — lowercase, hyphen-separated, ≤ 6 words, no dates inside it.
- IDs are **immutable and never reused**. A wrong entry is corrected by a new entry that supersedes it (`Supersedes: <id>` / `Superseded-by: <id>`), never by rewriting history.
- Every record **cross-links by ID**: an audit finding names the `ERR-`/`CHG-` it produced; a `CHG-` names the `ERR-`/`AUDIT-` that motivated it; an `ERR-` names its resolving `CHG-`.

---

## Versioning — strict

Every change record and audit carries **three version coordinates**. No entry is
complete without all three.

### 1. Upstream base — `hermes@<sha> (N behind)`

The `NousResearch/hermes-agent` commit the work sits on.

```bash
git rev-parse --short HEAD                 # -> <sha>
git rev-list --count HEAD..upstream/main   # -> N   (fetch upstream first)
```

Record as `hermes@820106d4a5 (2 behind upstream/main)`.

### 2. North-Forge version — `NF-vMAJOR.MINOR.PATCH`

Semantic version for **north-forge's own deltas** on top of upstream. Independent of
Hermes' own versioning.

| Bump | When |
| --- | --- |
| **MAJOR** | Incompatible change to the fork's shape — rebrand, a subsystem removed or replaced, layout overhaul. |
| **MINOR** | A new north-forge capability, integration, tool, skill, or workflow added on top of upstream. |
| **PATCH** | Fixes, housekeeping, dependency/upstream syncs, docs, and ledger-only changes. |

- Current line: **`NF-v0.1.0`** (ledger established + baseline repo cleanup).
- `0.y.z` = pre-identity. The first real rebrand/identity commit cuts **`NF-v0.2.0`**;
  declaring the fork stable cuts **`NF-v1.0.0`**.
- One version = one dated block in `CHANGELOG.md`. Accumulate `CHG-` bullets under an
  `## [NF-vX.Y.Z] — Unreleased` block, then stamp the date when you cut it.

### 3. Ledger schema — `ledger-schema vN`

Bump only when these rules or the directory layout change. Currently **v1**. Record the
schema version in an audit only if it changed since the previous audit.

### CHANGELOG heading format

```
## [NF-v0.1.0] — 2026-09-06 — hermes@820106d4a5 (2 behind upstream/main)
```

---

## Workflow — the discipline

**Start of a work session**
1. Read `INDEX.md`, the newest `AUDIT-`, and every `OPEN` row in `errors/ERROR-LOG.md`.
2. Note the upstream base: `git fetch upstream && git rev-list --count HEAD..upstream/main`.

**When you change tracked files or config**
- Add a `CHG-YYYY-MM-DD-NNN` bullet to `CHANGELOG.md` under the current
  `Unreleased` version block. State *what* and *why*, name affected paths, link the
  motivating `ERR-`/`AUDIT-` if any.
- Bump the `NF-v` version per the table when you cut a release block.

**When something breaks or an anomaly appears**
- Build break, failed test batch, bad merge, upstream regression, secret exposure,
  unexpected state — add an `ERR-` row **immediately**, status `OPEN`.
- On fix: set `RESOLVED`, add a resolution note and the resolving `CHG-` id. Never
  delete a row.

**Run an audit** — weekly, before a rebrand, before a large upstream merge, and after
any incident:
1. Copy `templates/AUDIT-TEMPLATE.md` → `audits/AUDIT-<today>-NNN-<slug>.md`.
2. Fill every section. Assign `F-NN` finding ids.
3. For each unresolved finding, open an `ERR-` row.
4. Add the audit to `INDEX.md`.

**Every entry cross-links by ID.** That is what makes the ledger navigable.

---

## Optional automation

The discipline above is manual by design (an entry needs judgement). If you want a
nudge, a Claude Code `Stop` hook or a git `post-commit` hook can append a skeleton
`CHG-` line for you to fill in — configure via the `update-config` skill. Do **not**
auto-generate finished entries; a half-true record is worse than none.

---

## History

- **2026-09-06** — ledger created (`ledger-schema v1`), `NF-v0.1.0`. See
  `audits/AUDIT-2026-09-06-001-repository-baseline.md` and `CHANGELOG.md`.
