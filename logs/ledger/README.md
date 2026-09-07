# North-Forge Project Ledger

`ledger-schema v2` · single source of truth for **what changed, what broke, what
was decided, and what was audited** in this repository.

This checkout is a fork of [`NousResearch/hermes-agent`](https://github.com/NousResearch/hermes-agent).
The ledger records **north-forge's own history on top of upstream** — every tracked-file
or config change, every fault, every open decision, and every audit — in one place,
under strict naming and version rules so the record stays greppable and unambiguous.

---

## Location rules

| Path | Purpose | Git |
| --- | --- | --- |
| `logs/` | Hermes **runtime** logs (session logs, `*.log`). Transient. | ignored (`.gitignore` `logs/*`) |
| `logs/ledger/` | **This ledger.** Audits, changelog, error + decision registers, templates. Permanent. | **tracked** (`.gitignore` `!logs/ledger/`) |

Nothing else goes in `logs/ledger/`. Do not put runtime output, scratch files, or
data dumps here. If it is not an audit, a change record, an incident record, or a
decision record, it does not belong.

```
logs/ledger/
├── README.md            ← this file (the rules)
├── INDEX.md             ← chronological index of every audit + pointers
├── CHANGELOG.md         ← every project change, newest first, under version headings
├── audits/             ← one file per audit
│   └── AUDIT-YYYY-MM-DD-NNN-<slug>.md
├── errors/
│   └── ERROR-LOG.md     ← append-only register of faults (breaks, regressions, leaks)
├── decisions/
│   └── DECISION-LOG.md  ← append-only register of open judgment calls
└── templates/          ← copy these when adding an entry
    ├── AUDIT-TEMPLATE.md
    ├── CHANGE-ENTRY-TEMPLATE.md
    ├── ERROR-ENTRY-TEMPLATE.md
    └── DECISION-ENTRY-TEMPLATE.md
```

---

## Naming — strict

| Record | Where | Identifier / filename | Example |
| --- | --- | --- | --- |
| **Audit** | `audits/` (one file each) | `AUDIT-YYYY-MM-DD-NNN-<kebab-slug>.md` | `AUDIT-2026-09-06-001-repository-baseline.md` |
| **Change** | bullet in `CHANGELOG.md` | `CHG-YYYY-MM-DD-NNN` | `CHG-2026-09-06-004` |
| **Error / fault** | row in `errors/ERROR-LOG.md` | `ERR-YYYY-MM-DD-NNN` | `ERR-2026-09-06-001` |
| **Decision** | row in `decisions/DECISION-LOG.md` | `DECISION-YYYY-MM-DD-NNN` | `DECISION-2026-09-06-001` |
| **Run** | defined in this file (§ Run IDs) | `RUN-YYYY-MM-DD-NNN` | `RUN-2026-09-06-001` |

Rules:

- **Date** — ISO 8601 `YYYY-MM-DD`, local time (America/Los_Angeles, same clock as `git log`).
- **NNN** — zero-padded 3-digit sequence. Resets each day. Counted **per record type**
  (`AUDIT-`, `CHG-`, `ERR-`, `DECISION-`, and `RUN-` each keep their own daily count).
- **`<kebab-slug>`** — lowercase, hyphen-separated, ≤ 6 words, no dates inside it.
- IDs are **immutable and never reused**. A wrong entry is corrected by a new entry that supersedes it (`Supersedes: <id>` / `Superseded-by: <id>`), never by rewriting history. This holds for `DECISION-` exactly as for `ERR-`, including when an entry is **migrated** between the two registers — keep the old id, add the link both ways.
- Every record **cross-links by ID**: an audit finding names the `ERR-`/`DECISION-`/`CHG-` it produced; a `CHG-` names the `ERR-`/`DECISION-`/`AUDIT-` that motivated it; an `ERR-`/`DECISION-` names its resolving `CHG-`.

### `ERR-` vs `DECISION-` — which register

| Use `ERR-` when… | Use `DECISION-` when… |
| --- | --- |
| Something is **wrong** and needs fixing — build break, failed test batch, bad merge, upstream regression, secret exposure, unexpected repo state. | A **choice** between defensible options is open and unmade — rebrand vs thin-downstream, keep vs delete a backup, rename a command or not, take on a dependency or not. |
| Closure = **RESOLVED** (the fault is gone), or WONTFIX / ACCEPTED-RISK / SUPERSEDED. | Closure = **DECIDED** (a choice was made and implemented), or DEFERRED (with a revisit trigger) / DROPPED (no longer relevant). |

If an item has both a fault half and a choice half (as `ERR-2026-09-06-002` did),
split it: the fault stays and closes in `ERROR-LOG.md`; the choice migrates to a
new `DECISION-` id with `Supersedes:` / `Superseded-by:` links both ways.

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

Bump only when these rules or the directory layout change. Currently **v2**. Record the
schema version in an audit only if it changed since the previous audit.

- **v1** (2026-09-06) — initial: `AUDIT-`/`CHG-`/`ERR-`, three version coordinates.
- **v2** (2026-09-06) — added the `decisions/` register and `DECISION-` id; the
  `Confidence` field on findings and `ERR-`/`DECISION-` entries; `RUN-` ids and the
  `Run:` field; the mandatory "Since last handoff" section on audits.

### CHANGELOG heading format

```
## [NF-v0.1.0] — 2026-09-06 — hermes@820106d4a5 (2 behind upstream/main)
```

---

## Confidence tags

Every audit finding and every `ERR-` / `DECISION-` entry carries a **Confidence**
field. Fixed vocabulary — use one of these three exactly, nothing else:

| Tag | Meaning |
| --- | --- |
| **Confirmed Fact** | Directly verified this session — command output, file contents, a reproduced failure. Someone re-running the same check would see the same thing. Quote or cite the evidence. |
| **Field-Reasoned** | Not directly verified, but a well-supported inference from things that were — consistent with the code, the docs, the observed behaviour, and prior entries. State the reasoning so a reviewer can check it. |
| **Unverified** | Plausible but unchecked — a suspicion, a second-hand claim, a "this looks wrong" with no reproduction yet. Says clearly what would need to be done to confirm it. |

Rules:

- Pick the tag for **what the entry asserts**, not for how worried you are. A
  low-severity finding can be a Confirmed Fact; a scary one can be Unverified.
- If different parts of one entry have different confidence, tag the entry at its
  **weakest load-bearing claim** and note which part is stronger.
- Never round **Unverified** up to **Field-Reasoned** to sound more certain. A
  half-true record is worse than an honest "not checked".
- These tags let a later reader — including a Codex session with no memory of this
  one — know what to trust and what to re-check.

---

## Run IDs

One `RUN-YYYY-MM-DD-NNN` per **agent invocation** — one Claude Code prompt worked
to completion, or one Codex session. Assign it at the **start** of that session's
ledger work, before the first entry.

- Every `CHG-` / `ERR-` / `DECISION-` entry created during that run carries a
  `Run:` field naming it. Audits carry it in their header table.
- This is **traceability only** — it groups entries by the session that produced
  them so a reviewer can see "everything that one run touched" at a glance. It
  does **not** replace the `CHG-`/`ERR-`/`DECISION-` ids and is never referenced
  in place of them.
- Entries created **before** `RUN-` ids existed (anything before `RUN-2026-09-06-001`)
  have no `Run:` field; treat their run as "the `AUDIT-2026-09-06-001` / `-002`
  sessions" and do not backfill.
- No separate run log — the id is defined by its use. To see a run's footprint,
  `grep -rn 'RUN-2026-09-06-001' logs/ledger/`.

---

## Workflow — the discipline

**Start of a work session**
1. Read `INDEX.md`, the newest `AUDIT-`, every `OPEN` row in `errors/ERROR-LOG.md`,
   and every `OPEN` row in `decisions/DECISION-LOG.md`.
2. Note the upstream base: `git fetch upstream && git rev-list --count HEAD..upstream/main`.
3. Assign this session's **`RUN-YYYY-MM-DD-NNN`** id (next free `NNN` for today).
   Every entry you create this session carries `Run: <that id>`.

**When you change tracked files or config**
- Add a `CHG-YYYY-MM-DD-NNN` bullet to `CHANGELOG.md` under the current
  `Unreleased` version block. State *what* and *why*, name affected paths, link the
  motivating `ERR-`/`DECISION-`/`AUDIT-` if any, and end with `Run: <RUN- id>`.
- Bump the `NF-v` version per the table when you cut a release block.

**When something breaks or an anomaly appears** (a **fault**)
- Build break, failed test batch, bad merge, upstream regression, secret exposure,
  unexpected state — add an `ERR-` row **immediately**, status `OPEN`, with a
  `Confidence` tag and `Run:` id.
- On fix: set `RESOLVED`, add a resolution note and the resolving `CHG-` id. Never
  delete a row.

**When a judgment call is open** (a **decision**, not a fault)
- A choice between defensible options with no obvious right answer — add a
  `DECISION-` row to `decisions/DECISION-LOG.md`, status `OPEN`, with the options,
  a leaning, what it blocks, the owner, a `Confidence` tag, and `Run:` id.
- On resolution: set `DECIDED` (or `DEFERRED` / `DROPPED`), name the choice and the
  implementing `CHG-` id(s), move the block to `## Resolved`. Never delete a row.
- See § `ERR-` vs `DECISION-` above for the boundary.

**Run an audit** — weekly, before a rebrand, before a large upstream merge, and after
any incident:
1. Copy `templates/AUDIT-TEMPLATE.md` → `audits/AUDIT-<today>-NNN-<slug>.md`.
2. **Write section 1 "Since last handoff" first** — diff this audit's open items
   against the previous audit's: Closed / New / Unchanged. It is not optional.
3. Fill every remaining section. Assign `F-NN` finding ids, each with a
   `Confidence` tag.
4. For each unresolved finding, open an `ERR-` row (fault) or `DECISION-` row
   (open choice).
5. Add the audit to `INDEX.md`.

**Every entry cross-links by ID**, carries a **Confidence** tag (audits / `ERR-` /
`DECISION-`), and names its **`Run:`**. That is what makes the ledger navigable and
trustworthy to a reader who wasn't there.

---

## Agent conduct

Standing rules for any agent (Claude Code, Codex, …) doing work in this repo.

**Incidental minor fixes — fix and log, don't stop to ask.**
A small defect noticed in passing during some other task is fixed in the same run
and recorded here — a `CHG-` bullet, or an `ERR-` + resolving `CHG-` if it was a
genuine fault — without pausing the task for approval. "Minor" means: local,
reversible, no design or policy content, and no change to user-visible behaviour
beyond the fix itself (a typo, a broken link, a wrong path in a comment, a
mis-scoped `.gitignore` line, a crashing edge case in a helper script).

**Always escalate instead — do _not_ self-fix — when the thing touches:**
- **Architecture** — the chassis/vertical split, repo layout, a subsystem's shape.
- **Access-tier logic** — the two-tier Full/Basic model, passcode gating, the
  pinned front-door skill, anything that decides what a recipient can reach.
- **Secrets** — `.env`, credentials, keys, tokens; anything under
  `ERR-2026-09-06-001`'s subject.
- **An already-flagged pending decision** — any `OPEN` row in
  `decisions/DECISION-LOG.md` or `errors/ERROR-LOG.md`. Add to that record; do
  not pre-empt it.

These keep the existing escalation path: surface it to the owner, open or update
the `DECISION-`/`ERR-` row, stop.

**When genuinely unsure which side something falls on, escalate.** A wrong guess
that quietly changes architecture or access logic is far more expensive than a
question. Use judgement on "minor"; when the judgement isn't clear, don't guess.

**End-of-task handoff.** At the end of any session that changed tracked files or
this ledger: run `scripts/collect-logs.ps1` (or `scripts/collect-logs.sh`) to
rebuild `D:\logs\` + `D:\logs.zip`, and drop a plain-language write-up at
`D:\logs\<TOPIC>_<YYYY-MM-DD>.md` alongside the ledger entries. The script's
completeness check will warn if a run left ledger entries but no such report.

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
- **2026-09-06** — `ledger-schema v2` (`RUN-2026-09-06-001`, `NF-v0.1.2`): added the
  `decisions/` register + `DECISION-` id, `Confidence` tags, `RUN-` ids, and the
  mandatory audit "Since last handoff" section. `ERR-2026-09-06-002`'s identity
  half migrated to `DECISION-2026-09-06-001`; `DECISION-2026-09-06-002` (attic
  clone) opened. See `CHANGELOG.md` `CHG-2026-09-06-015..019`.
