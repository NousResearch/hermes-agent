# AUDIT-2026-09-06-001 — Repository baseline audit

| | |
| --- | --- |
| **Audit ID** | AUDIT-2026-09-06-001 |
| **Date** | 2026-09-06 (America/Los_Angeles) |
| **Author** | Claude Sonnet 5 — session `https://claude.ai/code/session_01REK1NdYvLgKzEZ6i9CGND8` |
| **Scope** | Whole repository: git provenance & divergence, working-tree hygiene, secrets, CI inheritance, and a full review of `README.md`. Application code correctness was **not** reviewed. |
| **Upstream base (at audit end)** | `hermes@820106d4a5` (2 behind `upstream/main`) |
| **North-Forge version** | `NF-v0.1.0` |
| **Ledger schema** | v1 (established by this audit) |
| **Supersedes** | — (first audit) |
| **Method** | `git` (rev-list / log / diff / check-ignore / ls-files), `gh repo view`, `.env` key-name analysis (values never printed), filesystem inspection. Full commands in the appendix. |

> **Continuation note (2026-09-06, same session):** while implementing the "never
> push `.env`" rule, findings **F-09** and **F-10** were added and remediated, and
> **F-05**'s status was updated. New incidents: `ERR-2026-09-06-003`,
> `ERR-2026-09-06-004`. See `CHANGELOG.md` `CHG-2026-09-06-007..009`.

---

## 1. Summary

`D:\north-forge-agent` is a **fork of [`NousResearch/hermes-agent`](https://github.com/NousResearch/hermes-agent)** ("Hermes Agent" — a self-improving CLI/messaging AI agent, MIT, ~11.8k tracked files). As of this audit the fork has **zero north-forge code**: `origin/main` is 0 ahead / 2 behind `upstream/main`, and every tracked file — `README*.md`, `SOUL.md`, `LICENSE`, `package.json`, `pyproject.toml` — is still upstream's, fully branded "Hermes Agent / Nous Research". The only customization anywhere is the GitHub repo **name** and its **description** ("The agent that grows with you").

The local checkout was healthy in its tracked content but had accumulated hygiene problems: it was **245 commits stale**, a **complete second clone** was nested inside it (~897 MB), an unrelated stray file sat in the root, and a **live `ANTHROPIC_API_KEY`** is present in `.env` (gitignored, untracked — no leak path found).

**Actioned this session:** stale checkout fast-forwarded; nested clone and stray file moved out to `D:\north-forge-agent-attic\`; `.gitignore` hardened; this ledger established. **Left open:** the API-key hygiene decision (`ERR-2026-09-06-001`) and the fork-identity / upstream-sync decision (`ERR-2026-09-06-002`).

---

## 2. Repository state

### Identity & provenance

| Item | Value |
| --- | --- |
| Local path | `D:\north-forge-agent` — git repo, branch `main` |
| `origin` | `https://github.com/kwalker7631/north-forge-agent.git` — **public fork** |
| `upstream` | `https://github.com/NousResearch/hermes-agent.git` |
| GitHub description | "The agent that grows with you" (upstream's is different — this is a deliberate change) |
| GitHub `diskUsage` | ~711 MB · `pushedAt` 2026-09-06T21:17:22Z · `visibility` PUBLIC · `isFork` true |
| License | MIT © 2025 Nous Research (`LICENSE`, unchanged) |
| Composition | 11,815 tracked files — 5,567 `.py`, 2,033 `.ts`, 1,604 `.md`, 822 `.tsx`, plus `web/`, `website/`, `ui-tui/`, `docs/` |
| Other local disk | `D:\north-forge-agent` tree ≈ 2.2 GB (`.git` ≈ 1.1 GB) |

### Fork vs upstream

- `git rev-list --left-right --count upstream/main...origin/main` → **`2  0`** — `origin/main` is a strict mirror of `upstream/main`, currently 2 commits behind, **no unique commits**.
- `git log --all --author="kwalker…"` → **empty**. No personal commits on any branch, ever.
- Branches: `main`; `origin/bot/js-autofix` (one automated `npm run fix` commit — upstream tooling).

### Local checkout vs `origin/main`

- **Before this session:** local `HEAD` `245e48008f` — 0 ahead / **245 behind** `origin/main` (~1 day stale). `.bytecode-fingerprint` pinned the stale sha.
- **After `CHG-2026-09-06-001`:** `HEAD` `820106d4a5`, **== `origin/main`**, 2 behind `upstream/main`.

### Working tree

- Before: tracked files unmodified; 2 untracked non-ignored entries (`marguerite-and-penny-suno.txt`, `north-forge-agent/`); gitignored cruft present (`.env`, `__pycache__/`, `hermes_agent.egg-info/`, `.bytecode-fingerprint`).
- After: `git status --porcelain` → `M .gitignore` and `?? logs/` only (both are this session's ledger work, uncommitted).

---

## 3. Findings

Severity: **CRITICAL · HIGH · MEDIUM · LOW · INFO**

### F-01 — MEDIUM — Full clone nested inside the checkout
- **Observed:** `D:\north-forge-agent\north-forge-agent\` was a complete independent clone (own `.git`, same `origin`/`upstream` remotes), created 2026-09-06 17:18, `HEAD` `820106d4a5`, working tree pristine. ≈897 MB (714 MB of it `.git`). Not matched by any `.gitignore` rule → a `git add -A` from the parent would have staged it.
- **Impact:** ~900 MB wasted; real risk of editing / committing the wrong working copy; confuses tooling.
- **Status:** **RESOLVED** — `CHG-2026-09-06-002` (moved to attic), `CHG-2026-09-06-005` (`.gitignore` guard `/north-forge-agent/`).

### F-02 — LOW — Unrelated stray file in repo root
- **Observed:** `marguerite-and-penny-suno.txt` (3,606 bytes — Suno song-prompt lyrics, "Marguerite & Penny"), untracked, not ignored.
- **Impact:** commit-sweep risk; clutter; unrelated to the project.
- **Status:** **RESOLVED** — `CHG-2026-09-06-003` (moved to `D:\north-forge-agent-attic\`).

### F-03 — MEDIUM — Local checkout 245 commits stale
- **Observed:** local `main` at `245e48008f` (2026-09-05 17:00), `origin/main` at `820106d4a5` (2026-09-06 14:04). 0 local commits ahead → clean fast-forward available.
- **Impact:** developing against ~1-day-old framework code; misleading diffs; `hermes update` / bytecode-fingerprint churn.
- **Status:** **RESOLVED** — `CHG-2026-09-06-001` (fast-forwarded).

### F-04 — LOW — No home for a version-controlled project record
- **Observed:** upstream `.gitignore` line 33 was a bare `logs/` (Hermes writes runtime logs there). No `CHANGELOG.md`, no audit trail, nowhere tracked to record fork history.
- **Impact:** change/incident history would live only in shell scrollback.
- **Status:** **RESOLVED** — `CHG-2026-09-06-004` (`.gitignore` `logs/` → `logs/*` + `!logs/ledger/`), `CHG-2026-09-06-006` (ledger established). Verified: `logs/x.log` ignored, `logs/ledger/**` tracked.

### F-05 — HIGH — Live API key in `.env`
- **Observed:** `.env` (repo root) holds a real, non-placeholder `ANTHROPIC_API_KEY` (108-char value — consistent with a genuine `sk-ant-…` key) plus 11 benign upstream debug/timeout toggles that match `.env.example` defaults. **Value never printed during this audit.**
- **Exposure:** `.env` is matched by `.gitignore:15`; **not tracked**; `git log --all -- .env` empty. No leak path in this checkout. Residual risk is local (disk theft, shoulder-surf, accidental paste into issue/PR/screenshot/log).
- **Impact:** a leaked Anthropic key is billable and must be rotated.
- **Status:** **OPEN** — `ERR-2026-09-06-001`.
  - **Mitigation shipped (2026-09-06):** the "never push `.env`" rule is now
    enforced — `.gitignore` broadened (F-10 / `CHG-2026-09-06-007`) and
    `.githooks/pre-commit` + `pre-push` block any real `.env` from a commit or
    push (`CHG-2026-09-06-008`). History re-verified clean.
  - **Still on the user:** confirm the key is intended on this machine; rotate if
    provenance/history is uncertain; run `sh .githooks/install` on fresh clones.
  - **Never run `git clean -x`/`-X` in this repo — it deletes `.env`.**

### F-09 — LOW — Stray `%SystemDrive%` Windows cache tree in repo root
- **Observed:** a literal `%SystemDrive%/ProgramData/Microsoft/Windows/Caches/` directory (Windows icon-cache DBs: `cversions.2.db`, `*.ver0x*.db`; birth 2026-09-06 17:55) appeared in the repo root. Written by a Windows process running with cwd = the repo and an unexpanded `%SystemDrive%` env var; **not** a git artefact, not reproducible from `git add`/`commit`.
- **Impact:** clutter; commit-sweep risk on `git add -A`.
- **Status:** **RESOLVED** — `ERR-2026-09-06-004`, `CHG-2026-09-06-009` (deleted; `.gitignore` guards `/%SystemDrive%/`, `Thumbs.db`, `ehthumbs.db`, `[Dd]esktop.ini`, `$RECYCLE.BIN/`). May recur; guard makes recurrence harmless.

### F-10 — MEDIUM — `.gitignore` `.env` coverage gap
- **Observed:** the `.env` ignore block enumerated `.env`, `.env.local`, `.env.*.local`, `.env.development`, `.env.test`, `.op.env` — but **not** `.env.production`, `.env.staging`, `.env.ci`, or any other `.env.<name>`. A plain `git add .` would have staged such a file.
- **Impact:** silent path for a secret file to enter git.
- **Status:** **RESOLVED** — `ERR-2026-09-06-003`, `CHG-2026-09-06-007` (`.env` + `.env.*` + `.op.env`, `!*.example` / `!*.sample`; verified with `git check-ignore`).

### F-06 — MEDIUM — Fork is behind upstream and carries no identity
- **Observed:** `origin/main` 0 ahead / 2 behind `upstream/main`; no north-forge commit exists.
- **Impact:** "the base for my ongoing project" has not actually been started; drift from upstream will only grow.
- **Status:** **OPEN** — `ERR-2026-09-06-002`. Sync the fork (`git push origin upstream/main:main` or GitHub "Sync fork"); decide rebrand vs thin-downstream (see §6); land the first identity commit and cut `NF-v0.2.0`.

### F-07 — INFO — Build cruft in the working tree (no action)
- **Observed:** `__pycache__/`, `hermes_agent.egg-info/`, `.bytecode-fingerprint` present. All correctly covered by `.gitignore` (lines 65, 66, 179). Harmless.
- **Status:** INFO / no action. Do not `git clean` (would take `.env` with it).

### F-08 — INFO — All branding is still upstream's
- **Observed:** `README.md` title "Hermes Agent ☤"; `README.es/zh-CN/ur-pk.md`; `SOUL.md` ("You are Hermes Agent, built by Nous Research"); `LICENSE` copyright; badges and every doc link → `hermes-agent.nousresearch.com` / `NousResearch/hermes-agent`.
- **Status:** INFO — tracked under `ERR-2026-09-06-002`; details in §6.

---

## 4. Remediation performed this session

| Change | Finding | Effect |
| --- | --- | --- |
| `CHG-2026-09-06-001` | F-03 | Local `main` fast-forwarded `245e48008f` → `820106d4a5` (245 commits). |
| `CHG-2026-09-06-002` | F-01 | Nested clone → `D:\north-forge-agent-attic\nested-clone-2026-09-06\` (intact, deletable). |
| `CHG-2026-09-06-003` | F-02 | `marguerite-and-penny-suno.txt` → `D:\north-forge-agent-attic\`. |
| `CHG-2026-09-06-004` | F-04 | `.gitignore`: `logs/` → `logs/*` + `!logs/ledger/`. |
| `CHG-2026-09-06-005` | F-01/F-04 | `.gitignore`: added `/north-forge-agent/` guard. |
| `CHG-2026-09-06-006` | F-04 | `logs/ledger/` established (`ledger-schema v1`) + this audit. |
| `CHG-2026-09-06-007` | F-05/F-10 | `.gitignore`: `.env` block → `.env` + `.env.*` + `.op.env`, `!*.example` / `!*.sample`. |
| `CHG-2026-09-06-008` | F-05 | `.githooks/` — `secret-guard` + `pre-commit` + `pre-push`; `core.hooksPath = .githooks`. Blocks committing/pushing a real `.env`. |
| `CHG-2026-09-06-009` | F-09 | Deleted stray `%SystemDrive%` cache tree; `.gitignore` Windows-junk guards. |

**Uncommitted.** Working tree after remediation: `M .gitignore`, `?? .githooks/`, `?? logs/`. `core.hooksPath` is set locally, so the guard is already active in this checkout. See §5 for the commit decision.

---

## 5. Open items / recommendations

| # | Item | Owner | Ref |
| --- | --- | --- | --- |
| 1 | `.env` exposure is now mitigated (F-05 / F-10 / `.githooks/`). Remaining: confirm `ANTHROPIC_API_KEY` is intended on this machine; rotate if provenance uncertain; never paste `.env` anywhere. | user | `ERR-2026-09-06-001` |
| 2 | Sync fork: `git fetch upstream && git push origin upstream/main:main` (closes the 2-commit gap). | user | `ERR-2026-09-06-002` |
| 3 | Decide fork identity — **rebrand** vs **thin downstream** (§6) — then land the first identity commit and cut `NF-v0.2.0`. | user | `ERR-2026-09-06-002` |
| 4 | Commit this session's work. On `main` (default branch); recommend a branch, e.g. `chore/project-ledger`, covering `.gitignore` + `.githooks/` + `logs/ledger/`. Not done automatically. After cloning elsewhere, run `sh .githooks/install`. | user | — |
| 5 | Once satisfied, delete `D:\north-forge-agent-attic\nested-clone-2026-09-06\` to reclaim ~897 MB. | user | F-01 |
| 6 | Add a root `CLAUDE.md` (or keep `AGENTS.md`) noting this is a hermes-agent fork and what north-forge adds, so future sessions have context. | user | — |
| 7 | If you want CI (OSV scanner, lockfile checks, tests) on your commits, enable Actions on `kwalker7631/north-forge-agent`. | user | §7 |

---

## 6. `README.md` review (full)

**What it is:** upstream's Hermes Agent README, verbatim, 265 lines. Flow: banner → capability table → Quick Install (Linux/macOS + native Windows PowerShell) → `hermes` command list → Nous Portal pitch → CLI-vs-Messaging table → docs index → OpenClaw migration → Contributing → Community → MIT.

**Quality as a document (upstream's work):** strong. Value proposition is up front and concrete; the capability table is scannable; install is copy-paste for every platform; there is an honest native-Windows "heads up" and a genuinely useful antivirus / `uv.exe` false-positive section; the two entry points (`hermes` TUI vs `hermes gateway`) are explained clearly; the docs map is complete. Relevant here: it claims **native Windows is fully supported** (CLI, gateway, TUI, tools) — matches this environment.

**Problem for a fork owner — every pointer targets the wrong project:**

| Line(s) | Issue |
| --- | --- |
| 1–2 | Banner `assets/banner.png` is the Hermes banner. |
| 5–17 | Title "Hermes Agent ☤"; all badges → `hermes-agent.nousresearch.com`, `NousResearch/hermes-agent`, Nous Discord. |
| 19–31 | Body + capability table credit "Nous Research" as the builder. |
| 40, 50 | Install one-liners fetch `hermes-agent.nousresearch.com/install.sh` / `install.ps1` — these install **upstream**, not north-forge. |
| 45, 254 | "File issues" → `NousResearch/hermes-agent/issues` (a fork has Issues off by default; your bugs don't belong upstream). |
| 107–120 | Every command is `hermes …`; the repo entrypoint script is literally `hermes`. Decide whether the command stays `hermes` or becomes `north-forge` / `nfa`. |
| 124–139, 163–184 | Nous Portal section and the entire docs table link to `hermes-agent.nousresearch.com/docs/…` — no north-forge equivalent exists. |
| 221–246 | Contributor setup assumes the `$HERMES_HOME/hermes-agent` install layout. |
| 262–265 | "Built by Nous Research." |

Same branding runs through `README.es.md`, `README.zh-CN.md`, `README.ur-pk.md`, the `.es` `SECURITY`/`CONTRIBUTING` variants, and `SOUL.md`.

**Recommendation:** do **not** scrub every "Hermes" mention — you depend on that codebase. Instead:

- Replace the **top** of `README.md` with a short north-forge section: new title, one paragraph on the project's goal, and a clear "Built on **[Hermes Agent](https://github.com/NousResearch/hermes-agent)** by Nous Research (MIT)" credit line.
- Provide **your own** install/run steps (or: "clone this fork, then follow upstream setup").
- Keep `LICENSE` (MIT requires retaining the Nous copyright) and **add** your own copyright line — don't replace theirs.
- Update `SOUL.md`, and the `name` / `repository` / `homepage` fields in `package.json` and `pyproject.toml`, in the same identity commit.
- Decide whether to keep the translated READMEs (they will drift from upstream immediately) or delete them.

---

## 7. Notes

- **CI inheritance:** upstream ships a strong CI/security posture you inherit but which will not run until Actions is enabled on the fork — `osv-scanner.yml`, `supply-chain-audit.yml`, `uv-lockfile-check.yml`, `lockfile-diff.yml`, Dependabot, `SECURITY.md`, plus ~40 workflows under `.github/workflows/`.
- **`.gitignore` hygiene:** `.env`, `.op.env`, `.env.*.local`, `__pycache__/`, `hermes_agent.egg-info/`, `.bytecode-fingerprint` are all correctly ignored. No secrets in tracked files.
- **Upstream-merge risk from this session:** the only tracked change is a 7-line `.gitignore` hunk near line 33 — trivial to reconcile on the next `git merge upstream/main`.
- **Next audit triggers:** before any rebrand commit; before a large `upstream/main` merge; after any `ERR-` incident; otherwise weekly.

---

## Appendix — raw data

```
$ git rev-list --left-right --count upstream/main...origin/main
2	0

$ git log --all --author="kwalker" --oneline        # (also tried kwalker138@gmail.com, "north")
(empty)

$ git pull --ff-only origin main
Updating 245e48008f..820106d4a5
Fast-forward   (245 commits)

$ git rev-parse --short HEAD ; git rev-list --count HEAD..upstream/main
820106d4a5
2

$ git status --porcelain
 M .gitignore
?? logs/

$ git diff .gitignore
-logs/
+# Runtime logs stay ignored, but the north-forge project ledger under
+# logs/ledger/ (audits, changelog, error register) is version-controlled.
+logs/*
+!logs/ledger/
+# Guard against a repo being cloned inside this checkout again (see
+# logs/ledger/audits/AUDIT-2026-09-06-001).
+/north-forge-agent/

$ git check-ignore -v logs/x.log        -> .gitignore:35:logs/*   logs/x.log      (ignored, correct)
$ git check-ignore    logs/ledger/.probe -> (no match: tracked, correct)

$ du -sh north-forge-agent  north-forge-agent/.git      # the nested clone, pre-move
897M ...   714M ...

$ gh repo view kwalker7631/north-forge-agent --json ...
isFork=true  parent=NousResearch/hermes-agent  defaultBranch=main
description="The agent that grows with you"  diskUsage=711252  visibility=PUBLIC

.env : 12 assignments — 11 match .env.example defaults; 1 user-supplied: ANTHROPIC_API_KEY (len 108). Values not printed.
```
