# AUDIT-2026-09-06-002 — Land the hardening set; clear recurred working-tree cruft

| | |
| --- | --- |
| **Audit ID** | AUDIT-2026-09-06-002 |
| **Date** | 2026-09-06 (America/Los_Angeles) |
| **Author** | Claude Sonnet 5 — session `https://claude.ai/code/session_01NNbXHcJ2jGDT5kehWPkQjd` |
| **Scope** | `north-forge-agent` working-tree hygiene + committing the `AUDIT-2026-09-06-001` hardening set; git-freshness check across all three repos on `D:\` (`north-forge-agent`, `hermes-webui`, `north-forge-hermes-edition`); `.git` maintenance. Application code was **not** reviewed. |
| **Upstream base** | `hermes@820106d4a5` (2 behind `upstream/main` — tip `693641aa8b`) |
| **North-Forge version** | `NF-v0.1.1` (this audit cuts it) |
| **Ledger schema** | v1 (unchanged) |
| **Supersedes** | — (builds on `AUDIT-2026-09-06-001`) |
| **Commit** | the `chore(ledger): land secret-guard + project ledger` commit on local `main`, 2026-09-06 (`git log --grep 'chore(ledger)'`) — **not pushed** |
| **Method** | `git` (fetch / rev-list / status / check-ignore / ls-tree / gc / pull --ff-only), `find`, filesystem inspection, `.githooks/secret-guard --worktree`. `.env` values never printed. Full output in the appendix. |

---

## 1. Summary

Follow-up to the baseline audit. The `AUDIT-2026-09-06-001` remediation
(`.gitignore` hardening, `.githooks/` secret-guard, the `logs/ledger/` itself) had
been sitting **uncommitted in the working tree** — a fresh clone or a
`hermes update` would have had neither the secret guard nor the ledger. That set
is now **committed** (`CHG-2026-09-06-010`).

> **Update (2026-09-06, later same day — "sync only" chosen for `ERR-2026-09-06-002`):**
> the ledger commit was rebased onto `upstream/main` `693641aa8b` (clean — no file
> overlap with the 2 upstream commits) and **pushed to `origin/main`** as a
> fast-forward (`CHG-2026-09-06-014`). The fork is now **0 behind `upstream/main`**
> and `NF-v0.1.0`/`v0.1.1` are public. The **fork-identity** decision (rebrand vs
> thin-downstream, first identity commit, `NF-v0.2.0`) was deliberately deferred;
> `ERR-2026-09-06-002` stays OPEN, narrowed to identity only. References below to
> "not pushed" / "820106d4a5" / "2 behind" describe the audit-time state before
> this update.

Since the baseline audit the working tree had re-accumulated Windows/pytest junk —
the `%SystemDrive%/` cache tree recurred exactly as `ERR-2026-09-06-004` predicted,
plus a `MagicMock/` mock-path leak, four literal `C:Users…hermes-pytest-tmp…` pytest
files, a `logs.zip`, and the unrelated `marguerite-and-penny-suno.txt` back in the
root again. All removed; `.gitignore` now guards each pattern so recurrence is
harmless (`CHG-2026-09-06-011`, `CHG-2026-09-06-012`, `ERR-2026-09-06-005`).

**Git freshness:** all three repos' local branches are level with their `origin`
(`north-forge-agent` `main` == `origin/main` `820106d4a5`; `hermes-webui` `master`
== `origin/master` `e168b67e`; `north-forge-hermes-edition` `main` == `origin/main`
`1c47b60`). Nothing was stale; `git pull --ff-only` on each was a no-op.
`north-forge-agent` is still 2 behind `upstream/main` — unchanged, `ERR-2026-09-06-002`.

**`.git` maintenance:** `git gc` run on all three (housekeeping, no tracked-file
change). Still open and **user-owned**: the `ANTHROPIC_API_KEY` decision
(`ERR-2026-09-06-001`) and the fork sync / identity commit (`ERR-2026-09-06-002`).

---

## 2. Repository state

- **Identity / provenance:** unchanged from `AUDIT-2026-09-06-001` — `origin` =
  `github.com/kwalker7631/north-forge-agent.git` (public fork), `upstream` =
  `github.com/NousResearch/hermes-agent.git`.
- **Fork vs upstream** (`git rev-list --left-right --count HEAD...upstream/main`):
  `0  2` — local/`origin` still a strict mirror, now 2 behind (`upstream/main`
  advanced to `693641aa8b`). No north-forge commits existed on `origin` at audit
  start.
- **Local checkout vs `origin/main`:** `0  0` before the commit — was **not**
  stale (contrast `AUDIT-2026-09-06-001`, which found it 245 behind). After
  `CHG-2026-09-06-010` the local `main` is **1 ahead / 0 behind** `origin/main`
  (the hardening commit, unpushed).
- **Working tree** (`git status --porcelain`):
  - **Before:** `M .gitignore`; `?? .githooks/{README.md,install,pre-commit,pre-push,secret-guard}`;
    `?? logs/ledger/**` (8 files); `?? MagicMock/**`; `?? "C:Users…"` ×4;
    `?? logs.zip`; `?? marguerite-and-penny-suno.txt`; `%SystemDrive%/` present on
    disk (git-ignored, so not listed).
  - **After:** **clean.** Everything intended is committed; every junk path is
    deleted and `.gitignore`-guarded.
- **Other two repos:** `git status --porcelain` empty for both; both level with
  `origin` and (for `hermes-webui`) with `upstream/master` too.

---

## 3. Findings

Severity: **CRITICAL · HIGH · MEDIUM · LOW · INFO**

### F-01 — MEDIUM — `AUDIT-2026-09-06-001` hardening set was never committed
- **Observed:** `git status` showed `M .gitignore`, `?? .githooks/`, `?? logs/ledger/`
  — the entire baseline remediation was working-tree-only. `core.hooksPath` was set
  locally (`.githooks`), so the guard worked *in this checkout*, but a fresh
  `git clone` or a re-provision would ship with no secret guard and no ledger.
- **Impact:** the "never push `.env`" rule and the project history record were one
  `rm -rf` / re-clone away from being lost; the baseline audit's own open item #4
  ("commit this session's work") was unactioned.
- **Status:** **RESOLVED (this session)** — `CHG-2026-09-06-010`. Committed to local
  `main`; **not pushed** (see `ERR-2026-09-06-002`). Pre-commit `secret-guard`
  passed (no `.env`-shaped path staged).
- **Linked:** `CHG-2026-09-06-010`.

### F-02 — LOW — pytest / `unittest.mock` artifacts recurred in the working tree
- **Observed:** four items, all untracked, none `.gitignore`-matched:
  - `MagicMock/mock._session_db.db_path/{3165824711312,3165827767312}` — two 240 KB
    SQLite files + `.fts_rebuild.lock` / `.quarantine.lock` siblings. Created when a
    test left `_session_db.db_path` as a `MagicMock` and code opened `str(mock)` as a
    real path.
  - `C:UserskwalkAppDataLocalTemphermes-pytest-tmproot-2oumyv71…` and
    `…root-iyod624l…` (×2 each, 0 bytes) — pytest tmp paths written as literal
    filenames in the repo cwd (the `:` survives as a name char under Git-Bash).
  - `logs.zip` (462 KB) — a zipped copy of `logs/` dropped in the root.
- **Impact:** clutter; `git add -A` sweep risk; two of them shadow-name real Windows
  paths.
- **Status:** **RESOLVED (this session)** — deleted (`CHG-2026-09-06-012`);
  `.gitignore` guards added: `MagicMock/`, `*hermes-pytest-tmp*`, `*pytest-tmproot*`,
  `*pytest-of-*`, `/logs.zip` (`CHG-2026-09-06-011`).
- **Linked:** `ERR-2026-09-06-005`, `CHG-2026-09-06-011`, `CHG-2026-09-06-012`.

### F-03 — LOW — `%SystemDrive%/` Windows cache tree recurred
- **Observed:** `%SystemDrive%/ProgramData/Microsoft/Windows/Caches/` present on
  disk again (birth 2026-09-06 18:01), after `AUDIT-2026-09-06-001` `CHG-2026-09-06-009`
  deleted it at 17:55. Exactly the recurrence `ERR-2026-09-06-004` said was possible.
- **Impact:** none — the `/%SystemDrive%/` guard from the baseline audit already
  keeps it out of git. Disk clutter only.
- **Status:** **RESOLVED (this session)** — deleted again (`CHG-2026-09-06-012`).
  Guard confirmed still effective (`git check-ignore -v '%SystemDrive%/x'` →
  `.gitignore:43`). `ERR-2026-09-06-004` stays RESOLVED with a recurrence note.
- **Linked:** `ERR-2026-09-06-004`, `CHG-2026-09-06-012`.

### F-04 — LOW — `marguerite-and-penny-suno.txt` back in the repo root
- **Observed:** the stray Suno-lyrics file (`AUDIT-2026-09-06-001` F-02, moved to
  the attic as `CHG-2026-09-06-003`) was in the repo root again, byte-identical to
  `D:\north-forge-agent-attic\marguerite-and-penny-suno.txt` (`diff -q` clean).
  Never tracked on any ref.
- **Impact:** commit-sweep risk; unrelated content.
- **Status:** **RESOLVED (this session)** — deleted the root copy (attic copy is
  the canonical one); added a named `.gitignore` guard
  `/marguerite-and-penny-suno.txt` (`CHG-2026-09-06-011`) since a plain move has
  now failed to hold twice.
- **Linked:** `CHG-2026-09-06-011`, `CHG-2026-09-06-012`.

### F-05 — INFO — Git freshness across all three `D:\` repos
- **Observed** (after `git fetch` on each):
  | Repo | Branch | Local == origin? | vs upstream |
  | --- | --- | --- | --- |
  | `north-forge-agent` | `main` | **yes** — `820106d4a5` | 2 behind `upstream/main` (`693641aa8b`) |
  | `hermes-webui` | `master` | **yes** — `e168b67e` | level with `upstream/master` |
  | `north-forge-hermes-edition` | `main` | **yes** — `1c47b60` | (no upstream remote) |
- `git pull --ff-only` on each: `Already up to date.` No stale checkout anywhere.
- **Status:** **INFO / no action.** The only gap — `north-forge-agent` 2 behind
  `upstream/main` — is the deliberately-deferred `ERR-2026-09-06-002` (merging it
  is a fork-identity decision and would touch the uncommitted `.gitignore` hunk,
  now committed).

### F-06 — INFO — `.git` maintenance
- **Observed:** `.git` sizes before gc — `north-forge-agent` ≈ 1,099 MB,
  `north-forge-agent-attic/nested-clone-2026-09-06` ≈ 713 MB, `hermes-webui`
  ≈ 188 MB. Large because they carry upstream's full history (32,268 / 7,957
  commits).
- **Status:** **INFO.** `git gc` run on the three live repos this session
  (housekeeping; no tracked-file change; see appendix for reclaimed sizes). The
  attic clone was left untouched — it is a delete-when-confirmed backup
  (`AUDIT-2026-09-06-001` F-01, open item #5), not a working repo.

### F-07 — INFO — Carried-forward open incidents (unchanged, user-owned)
- `ERR-2026-09-06-001` (**HIGH, OPEN**) — live `ANTHROPIC_API_KEY` in
  `north-forge-agent/.env` (line 545) **and** in `D:\.env` (drive root, outside any
  repo). Exposure is structurally mitigated in-repo (`.gitignore` + `.githooks/`,
  now committed). Rotation / confirmation is the user's to do at
  `console.anthropic.com`. `git log --all -- .env` re-verified empty.
- `ERR-2026-09-06-002` (**MEDIUM, OPEN**) — fork 2 behind `upstream/main`, no
  north-forge identity commit on `origin`. Needs `git push` and a rebrand-vs-thin-
  downstream decision.

---

## 4. Remediation performed this session

| Change | Finding | Effect |
| --- | --- | --- |
| `CHG-2026-09-06-010` | F-01 | Committed the baseline hardening set (`.gitignore` hunk, `.githooks/`, `logs/ledger/`) to local `main`. Not pushed. |
| `CHG-2026-09-06-011` | F-02/F-03/F-04 | `.gitignore`: added `MagicMock/`, `/C:Users*`, `*hermes-pytest-tmp*`, `/logs.zip`, `/marguerite-and-penny-suno.txt`. |
| `CHG-2026-09-06-012` | F-02/F-03/F-04 | Deleted from the working tree: `%SystemDrive%/`, `MagicMock/`, 4× `C:Users…` pytest files, `logs.zip`, `marguerite-and-penny-suno.txt`; also cleared 3 stale `logs/full_suite_run*.log` (≈ 6.6 MB, git-ignored — housekeeping). |
| housekeeping | F-06 | `git gc` on `north-forge-agent`, `hermes-webui`, `north-forge-hermes-edition`. |
| verification | F-05 | `git fetch` + `git pull --ff-only` on all three — all `Already up to date`. |

**Commit:** the hardening set is the single `chore(ledger)` commit on local `main`
(2026-09-06), 1 ahead of `origin/main` — **unpushed**. The `.githooks/pre-push`
guard will scan the whole tree when the user pushes.

---

## 5. Open items / recommendations

| # | Item | Owner | Ref |
| --- | --- | --- | --- |
| 1 | Rotate or positively confirm `ANTHROPIC_API_KEY`. It is in `north-forge-agent/.env` **and** `D:\.env`. Rotate at `console.anthropic.com`, set a spend cap, and prefer keeping the real `.env` off the portable drive. | user | `ERR-2026-09-06-001` |
| 2 | `git push origin main` to publish `NF-v0.1.0` + `NF-v0.1.1` (the ledger + secret-guard). `pre-push` will scan on the way out. | user | F-01 |
| 3 | ~~Sync the fork (close the 2-commit gap)~~ — **DONE** 2026-09-06 (`CHG-2026-09-06-014`, rebased onto `693641aa8b` + pushed). Remaining: decide rebrand vs thin-downstream, land the first identity commit → cut `NF-v0.2.0`. | user | `ERR-2026-09-06-002` |
| 4 | On any fresh clone / re-provision, run `sh .githooks/install` to arm the secret guard (`core.hooksPath` is per-clone). | user | F-01 |
| 5 | Once satisfied the nested clone is redundant, delete `D:\north-forge-agent-attic\nested-clone-2026-09-06\` to reclaim ≈ 869 MB. | user | `AUDIT-2026-09-06-001` F-01 |
| 6 | If the pytest / `%SystemDrive%` artifacts keep recurring, find the offending test/tool (runs with `cwd` = repo root, unexpanded `%SystemDrive%`, mock path leaks) rather than relying on the `.gitignore` guards. | user | `ERR-2026-09-06-005` |

---

## 6. Notes

- **No application code touched.** Every change is `.gitignore`, `.githooks/`, or
  `logs/ledger/`. Upstream-merge risk stays minimal: the only tracked hunk outside
  the ledger is the `.gitignore` block near lines 25–61.
- **Why `main`, not a branch:** `AUDIT-2026-09-06-001` open item #4 floated a
  `chore/project-ledger` branch. Committed to `main` instead because the fork has
  zero personal commits, the changes are pure safety/hygiene, and a clean `main`
  working tree *is* the "strong base" asked for. Nothing was pushed, so this is
  fully reversible (`git reset --soft HEAD~1`).
- **`hermes-webui` / `north-forge-hermes-edition`:** clean, level with `origin`,
  no ledger of their own and none added — out of scope for this fork's ledger.
- **Next audit triggers:** before any `git push origin main`; before a rebrand
  commit; before a large `upstream/main` merge; after any `ERR-`; otherwise weekly.

---

## Appendix — raw data

```
# git freshness (after fetch), 2026-09-06
north-forge-agent            HEAD 820106d4a5  == origin/main 820106d4a5   | 2 behind upstream/main 693641aa8b
hermes-webui                 HEAD e168b67e    == origin/master e168b67e   | == upstream/master e168b67e
north-forge-hermes-edition   HEAD 1c47b60     == origin/main 1c47b60      | (no upstream)

$ git -C /d/north-forge-agent pull --ff-only origin main       -> Already up to date.
$ git -C /d/hermes-webui pull --ff-only origin master          -> Already up to date.
$ git -C /d/north-forge-hermes-edition pull --ff-only origin main -> Already up to date.

# working tree before / after (north-forge-agent)
before:  M .gitignore
         ?? .githooks/{README.md,install,pre-commit,pre-push,secret-guard}
         ?? logs/ledger/**  (8 files)
         ?? MagicMock/**  ?? "C:Users…"x4  ?? logs.zip  ?? marguerite-and-penny-suno.txt
         (%SystemDrive%/ on disk, git-ignored)
after:   clean  (hardening set committed; junk deleted + guarded)

# deleted this session
%SystemDrive%/ProgramData/Microsoft/Windows/Caches/   (recurred; git-ignored)
MagicMock/mock._session_db.db_path/{3165824711312,3165827767312}(+ .fts_rebuild.lock/.quarantine.lock)
C:Userskwalk…hermes-pytest-tmproot-2oumyv71…  x2
C:Userskwalk…hermes-pytest-tmproot-iyod624l…  x2
logs.zip                                              (462 KB)
marguerite-and-penny-suno.txt                        (3,606 B; identical copy kept in attic)
logs/full_suite_run{,2,3}.log                        (2.63 + 2.59 + 1.53 MB; git-ignored)

# .gitignore guards added (CHG-2026-09-06-011)
MagicMock/
*hermes-pytest-tmp*
*pytest-tmproot*
*pytest-of-*
/logs.zip
/marguerite-and-penny-suno.txt

$ sh .githooks/secret-guard --worktree
secret-guard: clean — no .env-style files tracked or stage-able.

$ git log --all --oneline -- .env
(empty)

# .git sizes before git gc
north-forge-agent/.git                                  ~1,099 MB
north-forge-agent-attic/nested-clone-2026-09-06/.git    ~713 MB   (not gc'd — backup)
hermes-webui/.git                                       ~188 MB
<gc results recorded in the D:\logs report>
```
