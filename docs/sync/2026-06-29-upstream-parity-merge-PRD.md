# PRD — Upstream Parity Merge (fork ← NousResearch/main, 2026-06-29)

**Status:** IN REVIEW (Opus pass-1: BLOCK → folded; pass-2: APPROVE WITH CHANGES → folded) · **Version:** v0.3
**Owner:** Apollo · **Reviewer gate:** prd-review-pipeline (Opus, 2 passes done) → Ace sign-off
**Type:** High-blast-radius infra merge. The fleet runs `fork/main`.
**Frozen upstream target SHA:** `929dd9c0d` (origin/main @ 2026-06-29 12:09 CT) — pinned for the whole build/CI/review per D-1/OQ-1.

> **Review history:** Opus pass-1 (claude-bpp) **BLOCK** — 5 blockers, all on *verification
> completeness* (strategy endorsed); all 5 folded into v0.2. Opus pass-2 (claude-bpp) **APPROVE
> WITH CHANGES** — confirmed the 5 folds are real gates, raised 7 foldable tightenings (re-aim
> rename+delta audits at the CODE surface not git-metadata/log; close Blocker-#4 to named tests +
> reconcile the revert-to-RED tiering; correct the deploy vector `fleet.sh`→`hermes update` +
> host-1 self-restart hazard; Phase-5 writable-copy mechanics; phase renumbering). All 7 folded
> into v0.3. See `docs/sync/review/pass1.md`, `pass2.md`.

---

## 1. Summary & Goal

Bring `Kyzcreig/hermes-agent` (`fork/main`) up to parity with `NousResearch/hermes-agent`
(`origin/main`) by **merging** upstream into the fork — absorbing **1,734 upstream commits**
(1,546 non-merge) while **preserving all 220 fork-only commits** that the running fleet depends
on. Land it as a single reviewed merge commit on a branch, prove it green in an isolated
worktree, and only then deploy to the fleet behind Ace's sign-off.

**Why now:** drift is accelerating and compounding the eventual reconciliation cost.
Measured cadence: 164/483 (06-20) → **220 ahead / 1,734 behind** (06-29). Merge-base is
`c6b0eb4` (2026-06-15) — ~2 weeks stale. The last successful sync (`d8c2836f`, 2026-06-16)
proved the pattern works; we've simply let it lapse.

**Strategy = MERGE, not rebase, not reset.** A reset/hard-takeover would brick the fleet
(~60% of our 220 commits are fleet-load-bearing systems that **do not exist upstream**). A
rebase replays 210 commits against the same hot files (3–5× the conflict labor). A merge keeps
honest history, resolves the conflict surface once, and matches the proven 06-16 precedent.

### Ground-Truth (measured 2026-06-29, fresh fetch — load-bearing, not assumed)

| Metric | Value | How measured |
|---|---|---|
| Fork ahead of upstream | **220** commits (210 non-merge) | `git rev-list --count origin/main..fork/main` |
| Fork behind upstream | **1,734** commits (1,546 non-merge) | `git rev-list --count fork/main..origin/main` |
| Merge-base | `c6b0eb4` (2026-06-15 23:50) | `git merge-base fork/main origin/main` |
| Upstream HEAD | `929dd9c0d` (2026-06-29 12:09 CT) | `git log -1 origin/main` |
| Files upstream touched since base | 2,070 | `git diff --name-only c6b0eb4..origin/main` |
| Files fork touched since base | 356 | `git diff --name-only c6b0eb4..fork/main` |
| **Real conflict files (test-merge)** | **35** (22 core, 13 test, 1 modify/delete) | `git merge --no-commit origin/main` in scratch worktree |
| **Real conflict hunks** | **83** | `grep -c '^<<<<<<<'` across conflicted files |
| Conflict concentration | `plugins/memory/mem0/__init__.py` = **27 of 83 hunks** | per-file marker count |
| Prior sync precedent | `d8c2836f` (2026-06-16): 1,084 behind, 12 conflicts, 3 arch ports — **LANDED** | `git merge-base --is-ancestor` = true |

**The test-merge was run on a throwaway detached worktree, aborted, and removed. Nothing live
was touched.** The 35-file / 83-hunk number is a *real* `git merge` result, not an estimate.

### Conflict surface (measured, ranked)

| File | Hunks | Nature |
|---|---|---|
| `plugins/memory/mem0/__init__.py` | 27 | Our hybrid-retrieval Wave-2 vs upstream's mem0 lazy-install. **Dominant.** |
| `gateway/run.py` | 4 | Our F1/F2 restart-cascade + send_message routing vs upstream gateway refactor |
| `agent/chat_completion_helpers.py` | 4 | Both heavily edited (12 upstream / 11 fork commits) |
| `tools/delegate_tool.py` / `tests/tools/test_delegate.py` | 3+3 | Subagent routing vs upstream delegate changes |
| `hermes_cli/kanban_db.py` | 3 | Kanban per-task-model + audit log |
| `cron/scheduler.py` | 3 | Cron per-job api-max-retries |
| `agent/context_compressor.py` | 3 | LCM/compression vs upstream compressor work (13 upstream commits) |
| `agent/auxiliary_client.py` | 3 | Aux-client route-scoped max_retries |
| 14 more core files | 1–2 each | agent_init, background_review, turn_context, usage_pricing, session*, etc. |
| 13 test files | 1 each | mirror the core conflicts |
| `tests/plugins/memory/test_mem0_v2.py` | modify/delete (UD) | upstream **deleted** it; we modified it |

### Architectural relocations (the genuinely hard part — not raw hunks)

Upstream ran a **platform-adapter refactor**: `gateway/platforms/*.py` → `plugins/platforms/*/adapter.py`
(16 renamed files: telegram, slack, whatsapp, matrix, sms, email, feishu, wecom, dingtalk…).
Our fork modified `gateway/platforms/telegram.py` (network-reconnect ladder); upstream
relocated it to `plugins/platforms/telegram/adapter.py`. Git surfaces this as a conflict at the
**new** path — **our fix must be ported onto upstream's relocated module**, not line-merged.
This is the same class as the 3 architectural ports in the 06-16 sync (turn_finalizer
extraction, slash_commands mixin, agent_init config merge).

**⚠️ Rename-detection is itself a risk (pass-1 Blocker #1).** Git flagged the telegram rename as
a conflict because similarity stayed above threshold. If a fork-touched file's similarity to its
upstream-relocated counterpart drops **below** git's rename threshold, git treats it as
*delete-old + add-new* — the fork edit lands as a **clean ADD at the dead old path** while
runtime calls the new module. Result: fork behavior silently absent, **and a naive symbol-grep
passes on the dead duplicate.** The clean-merge audit gate (§6 Phase 1b) exists specifically to
catch this: it enumerates every rename, and for each one the fork also touched, proves
*old-absent AND new-present*, not just symbol-present.

### Upstream behavior delta (measured — Blocker #5: the merge is NOT "zero behavior change")

Absorbing 1,546 upstream commits **is** 1,546 commits of upstream behavior change. Only the
conflicted slice gets line-reviewed; the rest merges clean and unaudited. Measured facts:

| Check | Base `c6b0eb4` | Upstream `929dd9c0d` | Implication |
|---|---|---|---|
| `_config_version` | **29** | **32** | **3 migrations (30→31→32)** run on first launch against each host's live `config.yaml` carrying fork-only keys → Blocker #3 gate required |
| Core tool count (`toolsets.py`) | 2 | 2 | No new every-call core tool detected at the schema entrypoint (good — but Phase-0b audit must confirm across the whole tool-registration surface, not just `toolsets.py`) |
| Renamed files (R) | — | 16+ platform adapters | Architectural-port surface (Class A) |

The Phase-0b upstream-diff audit (new core tools on the every-call schema, new outbound/
telemetry, new `HERMES_*` env vars, major dependency bumps) is now a required gate, per the
fleet's `external-code-ingest-audit` posture — this is the largest single code ingest of the year.

---

## 2. Non-Goals

- **NOT a rebase** of fork commits onto upstream (history churn, repeated conflicts). Merge only.
- **NOT a reset-to-upstream** or hard takeover (would delete fleet-critical fork-only systems).
- **NOT upstreaming our fork commits** to NousResearch. This is inbound parity only.
- **NOT a refactor or feature change.** Zero behavior changes beyond what the merge resolution
  strictly requires. Every conflict resolution preserves *both* sides' intent or is a named port.
- **NOT touching the live shared checkout** (`~/.hermes/hermes-agent`) during build. All work in
  an isolated worktree (per LIVE TREE memory rule).
- **NOT auto-deploying to the fleet** on green. Deploy is a separate, gated, Ace-approved step.
- **NOT resolving the intentional `.5`/`.208` DNS split or any infra** — code merge only.

---

## 3. Constitution / Invariants (non-negotiable; become closeout checks)

- **INV-1 — Fleet-critical fork systems survive intact.** LCM/compaction-announce,
  mem0-selfhost hybrid retrieval, F1/F2 restart-cascade breaker, send_message in-turn routing,
  blackbox cost telemetry, kanban per-task-model, fallback reasoning_effort — all present and
  functional after merge.
  - *Why:* none exist upstream (LCM = 0 upstream matches; mem0 = 2 trivial upstream commits).
    A silent clobber = fleet regression in production.
  - *Closeout proof (VERIFIED-EXISTING NAMED test modules — Blocker #4 resolved):* each system's
    suite passes against a temp `HERMES_HOME` via `scripts/run_tests.sh`:
    - LCM/compaction → `tests/agent/test_compaction_announce.py`, `…_announce_lcm.py`,
      `…_stats_reconcile.py`, `…_hygiene_replay.py`, `…_stats_degrade_observability.py`
    - mem0 hybrid → `tests/plugins/memory/` (dedup + hybrid-retrieval modules)
    - F1/F2 restart-cascade → `tests/gateway/test_restart_cascade.py`, `…_drain.py`,
      `…_redelivery_dedup.py`, `tests/hermes_cli/test_gateway_restart_loop.py`
    - send_message routing → `tests/tools/test_send_message_origin.py`,
      `tests/tools/test_subagent_send_origin.py`, `…_target_parse.py`
    - blackbox per-class cost → `tests/plugins/blackbox/test_cost_perclass.py`,
      `…/test_backfill_perclass.py` (named, verified-existing — not a directory citation)
    - kanban per-task-model → `tests/hermes_cli/test_kanban_model_override.py` (named,
      verified-existing — the actual per-task-model override test)
  - **Revert-to-RED tiering (pass-2: INV/Phase contradiction reconciled).** The 4 highest-risk
    systems get a **mutation proof** (revert the fork hunk → the named test goes RED): mem0
    hybrid, F2 breaker, send_message in-turn, LCM/compaction-reconcile. The remaining 3 (blackbox
    per-class, kanban model-override, fallback reasoning_effort) get **existence + green-pass**
    verification of their named test; their teeth are honestly labeled "green-pass verified, not
    mutation-proven" rather than over-claimed as mutation-proof.
  - **Verify-then-cite rule:** before execution, Phase-0a confirms each cited test EXISTS
    (`git cat-file -e`) and, for the 4 mutation-tiered systems, FAILS on revert. A gate citing a
    nonexistent test is vapor.
- **INV-2 — Architectural ports preserve fork behavior on upstream's new structure.** The
  telegram network-reconnect fix lands on `plugins/platforms/telegram/adapter.py`; any other
  fork feature whose host file upstream relocated/extracted is re-threaded, not dropped.
  - *Closeout proof:* `tests/gateway/test_telegram_network_reconnect.py` +
    `…_noise_filter.py` pass on the new path (both VERIFIED-EXISTING); grep shows the fork
    symbol present in the relocated module **AND absent from the dead old path** (INV-10).
- **INV-3 — Prompt cache / role-alternation / byte-stable system prompt preserved.** (Upstream
  AGENTS.md sacred rule.) The merge introduces no mid-conversation context mutation, no
  same-role-twice, no system-prompt rebuild.
  - *Closeout proof:* relevant context/alternation tests pass; no resolution edits the
    system-prompt assembly to inject state.
- **INV-4 — CI is fully green on the merge branch before merge.** All required workflows:
  `tests`, `typecheck`, `lint`, `contributor-check`, `history-check`, `fleet-secret-scan`,
  `fleet-sast`, `osv-scanner`, `supply-chain-audit`, `uv-lockfile-check`,
  `skills-index-freshness`. No skips, no `--admin` over red.
  - *Closeout proof:* `gh pr checks` all green, 0 pending.
- **INV-5 — Config-version migration (29→32) preserves every fork-only key against REAL fleet
  configs.** Upstream bumped `_config_version` 29→32 (3 migrations). The migration runs on first
  post-deploy launch against each host's live `config.yaml` carrying fork-only keys (kanban
  per-task-model, fallback reasoning_effort, F1/F2 settings, compression thresholds). CI runs
  against a temp `HERMES_HOME` and **structurally cannot see this** — a migration that drops or
  rejects a fork-only key is a silent fleet config regression with green CI.
  - *Closeout proof:* a **config-migration dry-run** (§6 Phase 5) runs the merged `migrate_config`
    against a **read-only snapshot copy** of each of the 7 hosts' real `config.yaml`; asserts
    `check_config_version()` reaches 32 AND every fork-only key survives byte-equal (diff the
    pre/post key set). No host is restarted until its config snapshot passes.
  - **Lockfiles:** `uv.lock` / `package-lock.json` regenerated (not hand-merged) so
    `uv-lockfile-check` passes; `uv sync` clean. **Major-version dependency bumps** in the
    absorbed delta are enumerated and reviewed (resolution-clean ≠ runtime-compatible) — see INV-9.
- **INV-6 — No fork-only test deleted to dodge a conflict.** The modify/delete on
  `test_mem0_v2.py` is resolved by a deliberate decision (keep-and-port vs accept-upstream-delete
  *because the covered behavior moved*), recorded — never silently dropped to make merge easy.
  - *Closeout proof:* decision recorded in merge commit; mem0 coverage still asserts hybrid
    retrieval somewhere.
- **INV-7 — LCM vendoring provenance intact.** `scripts/check_lcm_upstream_drift.py` +
  `VENDORED_FROM.txt` metadata still validate post-merge (the LCM security-drift guard).
  - *Closeout proof:* `python scripts/check_lcm_upstream_drift.py` (offline) passes.
- **INV-8 — Live working-tree changes parked, not lost.** The uncommitted live-tree edits
  (`toolsets.py`, `package-lock.json`, `tests/plugins/memory/test_mem0_remember.py`,
  `_bgr_mem0_proof.py`) are committed/stashed/triaged before any merge, and consciously
  re-applied or discarded.
  - *Closeout proof:* `git status` clean reconciliation recorded; nothing orphaned.
- **INV-9 — The inherited upstream delta is behavior-audited, not blindly trusted (Blocker #5).**
  The 1,546 absorbed upstream commits are audited by **scanning the code diff** `c6b0eb4..929dd9c0d`
  (NOT commit-message grep, which misses un-labeled additions — pass-2 fix): new `os.getenv("HERMES_`
  / `os.environ[` reads (non-secret config-as-env), new outbound call sites
  (`requests.`/`httpx.`/`urllib`/`socket`), new entries in the **tool-registration surface** (named
  sites, not just `toolsets.py`), and major-version dependency bumps in `uv.lock`. Findings
  recorded; any AGENTS.md violation inherited from upstream is flagged (we deploy upstream's
  choices knowingly).
  - *Closeout proof:* Phase-0b audit artifact (`docs/sync/review/upstream-delta-audit.md`) lists
    each category's diff-scan findings or an explicit "none found"; major dep bumps enumerated
    with a runtime-compat note. Commit-message grep may be used as a *supplement*, never the
    primary instrument.
- **INV-10 — The clean-merge surface is audited at the CODE level, not just the 35 conflicts
  (Blocker #1).** Two enumerations (pass-2: walk deletes too, since a rename-MISS surfaces as
  delete-old + add-new, NOT as `R`): (a) `git diff --diff-filter=R` renames ∩ fork-touched AND
  (b) `git diff --diff-filter=D` upstream **deletes** ∩ fork-touched — each reconciled to its new
  home (or confirmed a clean upstream-only relocation), proving old-absent AND new-present. Plus a
  fork-only **import smoke** importing every INV-1 module to catch dead-duplicate and semantic
  (signature-drift) breaks the conflict set is blind to. **Grep proves text/wiring, not behavior**
  — the behavioral net is the per-system revert-to-RED (INV-1) + host-1 soak, not the grep.
  - *Closeout proof:* the R∩fork and D∩fork checklists complete (each ported or confirmed
    clean-relocation); the import-smoke (§6 Phase 1b) passes over all INV-1 modules.

---

## 4. Resolved Decisions

- **D-1 — Merge, not rebase/reset.** (See §1.) Keeps history honest, resolves conflicts once,
  matches proven 06-16 precedent. Ace approved the merge strategy in-thread (2026-06-29).
- **D-2 — Isolated worktree build.** New worktree under `~/.hermes/worktrees/upstream-parity-<date>`,
  branch `sync/upstream-2026-06-29`. Live shared checkout untouched until deploy. Prove
  `__file__` resolves to the worktree (LIVE TREE memory rule).
- **D-3 — mem0 file resolved FIRST, in isolation.** It's 27/83 hunks (1/3 of the work) and is
  fleet-critical (Wave-2 hybrid retrieval). Resolve it as its own focused sub-task with the mem0
  test module as the gate, before touching the other 34 files.
- **D-4 — Architectural ports handled like 06-16.** For each file upstream relocated/extracted
  that we modified, take upstream's new structure and **re-thread** the fork behavior onto it,
  documented per-port in the merge commit body (telegram adapter is the known one; surface any
  others during resolution).
- **D-5 — One merge commit, descriptive body.** Model the commit message on `d8c2836f`: state
  commits-behind, conflict count, and enumerate every genuine architectural port with what moved
  where. Mechanical keep-both / take-upstream resolutions listed in a second block.
- **D-6 — Deploy is separate and gated.** Green CI + Ace sign-off on the **mem0 + LCM/compaction
  reconciliation specifically** (where fleet-critical logic could be silently clobbered) is the
  precondition. Deploy via `fleet.sh deploy claude-bridge --restart` only after. Staged: one
  host first, verify, then fan out.
- **D-7 — Full PRD pipeline.** spec → prd-review-pipeline (Opus ≥2 passes) → fold ALL findings →
  prd-plan → Ace go-ahead. (This document is step 1.)
- **D-8 — Establish a catch-up cadence (policy, not a build step here).** A recurring
  upstream-merge cron so we never face an 83-hunk wall again. Scoped as a **follow-up roadmap
  item** (v0.2), not part of this merge — see roadmap.

---

## 5. Architecture / Approach

The merge is executed as a sequence of **bounded, ordered phases** in an isolated worktree.

**⚠️ Executability constraint (pass-1 Blocker #2 — the spine is re-ordered around this).**
Conflict markers (`<<<<<<<`) are syntax errors in `.py` files. While *any* of the 35 conflicts
is unresolved, `pytest` fails at **collection** (import error) for every test that transitively
imports a still-conflicted core module — not on logic. Therefore a "resolve-one-file-then-run-
its-subsystem-test-green" loop is **physically impossible** mid-merge. The spine is split into
two distinct stages:

- **Stage 1 — RESOLUTION (tree not yet importable).** Clear all 35 conflicts in risk order. The
  only valid checkpoints here are **marker-free** (`git diff --check`, `git grep` for all three
  marker types) and, once markers are gone, a **byte/AST-level import smoke** (`python -c "import
  <module>"` over the INV-1 modules). No subsystem pytest runs in Stage 1.
- **Stage 2 — VERIFICATION (tree importable).** Only after the **whole** tree is marker-free and
  imports cleanly do the per-subsystem test gates + full suite run. This is where INV-1/INV-2
  revert-to-RED proofs live.

Resolution is still *ordered by risk* (mem0 first, then ports, then bulk) so a human-reviewable
sub-commit history exists — but the **test gates are batched at the Stage-1→Stage-2 boundary**,
not interleaved per file.

### 5A. Layer / resolution-class analysis (every conflict file gets a class)

Each of the 35 conflict files is pre-classified so resolution is mechanical-by-class, not
ad-hoc:

| Class | Meaning | Resolution rule | Files (examples) |
|---|---|---|---|
| **A — Architectural port** | Upstream relocated/extracted the host file | Take upstream structure; re-thread fork behavior onto it; test on new path | `plugins/platforms/telegram/adapter.py`, possibly `gateway/run.py`, `agent/conversation_loop.py` |
| **B — Fleet-critical keep-both** | Both sides changed a fork-load-bearing file | Preserve BOTH intents; fork behavior must survive; gate with the system's own test | `plugins/memory/mem0/__init__.py`, `agent/context_compressor.py`, `agent/auxiliary_client.py`, `cron/scheduler.py`, `hermes_cli/kanban_db.py` |
| **C — Mechanical keep-both** | Independent additions on each side | Union the changes | `hermes_state.py`, `hermes_cli/config.py`, `hermes_cli/commands.py`, `hermes_cli/backup.py` |
| **D — Take-upstream** | Upstream evolved; fork edit is stale/superseded | Accept upstream; re-verify fork need is gone | `tests/hermes_cli/test_update_check.py`, some test files |
| **E — Modify/delete** | Upstream deleted a file we modified | Deliberate decision per INV-6 | `tests/plugins/memory/test_mem0_v2.py` (UD), `tests/run_agent/test_run_agent.py` (DU) |

Every file's class is decided by reading **both** sides' diffs since merge-base (`git log -p
c6b0eb4..origin/main -- <f>` and `…fork/main…`), never by picking a side blind. The DU/UD pair
is spelled out: **UD** (`test_mem0_v2.py`) = we modified, upstream deleted → INV-6 decision
(keep-and-port vs accept-delete-because-coverage-moved). **DU** (`test_run_agent.py`) = we
deleted, upstream modified → default keep-upstream's version unless our delete was deliberate.

### 5B. Verification spine (two-stage, executable)

| Stage | Tree state | Valid checks | Gate to advance |
|---|---|---|---|
| **1 — Resolution** | NOT importable (markers present) | `git diff --check`; `git grep -nE '^(<<<<<<<\|=======\|>>>>>>>)'`; once clear → `python -c "import <INV-1 module>"` import-smoke | Zero markers anywhere + every INV-1 module imports |
| **2a — Per-subsystem** | importable | `scripts/run_tests.sh <subsystem dirs>` (hermetic wrapper, NOT bare pytest) + revert-to-RED proofs | All INV-1/INV-2 suites green |
| **2b — Whole** | importable + committed | `scripts/run_tests.sh` full suite (no `-x` — full failure set); lockfile + LCM-drift checks; then CI on the PR | Full CI green, 0 pending |
| **3 — Deploy (gated)** | merged | config-migration dry-run (§6 Phase 5) → staged host rollout + soak + live smoke | Ace sign-off + per-host smoke pass |

**Marker grep uses all three marker types** (`<<<<<<<`, `=======`, `>>>>>>>`) — grepping only
`<<<<<<<` misses a resolved-wrong file that left a `=======`/`>>>>>>>` behind.

---

## 6. Implementation Phases

### Phase 0 — Park the live tree + stand up the isolated worktree
Commit/stash/triage the 4 uncommitted live-tree changes (INV-8). Create
`~/.hermes/worktrees/upstream-parity-<date>` on a new branch `sync/upstream-2026-06-29` from
`fork/main`. Fetch both remotes fresh. **Re-confirm divergence numbers still match this spec**
(drift grows daily; if materially larger than 220/1734, re-measure the conflict set before
proceeding). Pin the frozen target SHA `929dd9c0d`.
- *Unit/script check:* `git worktree list` shows the new worktree; `git rev-parse --abbrev-ref HEAD` = `sync/upstream-2026-06-29`.
- *E2E/integration check:* in the worktree, `PYTHONPATH=$PWD venv/bin/python -c "import run_agent, sys; print(run_agent.__file__)"` resolves to the **worktree path**, not the live checkout.
- *Negative/adversarial:* `git -C ~/.hermes/hermes-agent status` shows the live tree's parked state is intentional (no orphaned uncommitted edits silently carried in).
- *Verify with:* `git status` clean in worktree on a fresh branch + `__file__` proof above.

### Phase 0a — Verify-then-cite: confirm every gate test EXISTS and bites (Blocker #4)
Before any resolution, confirm each INV-1/INV-2 cited test file exists on `fork/main`, and (for
the load-bearing systems) that it FAILS when the fork behavior is reverted (revert-to-RED). Any
system lacking a real behavior-covering test gets one written here, or its INV is honestly
downgraded to "manual verification."
- *Unit/script check:* `git cat-file -e fork/main:<each cited test>` all succeed (already spot-verified: telegram, restart-cascade, send_message_origin, compaction, blackbox, kanban suites all present).
- *E2E/integration check:* for the 3 highest-risk systems (mem0 hybrid, F2 breaker, send_message in-turn), a revert-the-fork-then-run shows RED on `fork/main` before merge — proving the gate has teeth.
- *Negative/adversarial:* a system whose "test" passes even with the fork behavior reverted is flagged as vapor and gets a real test or an honest downgrade.
- *Verify with:* a checklist artifact mapping each INV system → existing test path → revert-to-RED confirmed.

### Phase 0b — Upstream-delta behavior audit (Blocker #5, INV-9)
Audit the inherited behavior the merge ships unreviewed by **scanning the code diff**
`c6b0eb4..929dd9c0d` (not commit-message grep — pass-2 fix): new `os.getenv("HERMES_`/`os.environ[`
reads, new outbound call sites (`requests.`/`httpx.`/`urllib`/`socket`), new entries in the
tool-registration surface (named sites), major dependency bumps in `uv.lock`. Record in
`docs/sync/review/upstream-delta-audit.md`.
- *Unit/script check:* `git diff c6b0eb4..929dd9c0d -- '*.py' | grep -nE '^\+.*(os\.getenv\("HERMES_|os\.environ\[|requests\.|httpx\.|urllib|socket\.)'` enumerated; tool-registration sites diffed; `uv.lock` major-version deltas listed.
- *E2E/integration check:* `Not applicable` (audit/inspection phase) — findings feed the deploy go/no-go.
- *Negative/adversarial:* any inherited AGENTS.md violation (un-gated telemetry, non-secret `HERMES_*`, new always-on core tool) is flagged for Ace's knowing-deploy decision, not silently absorbed.
- *Verify with:* the audit artifact exists with per-category diff-scan findings or explicit "none found."

### Phase 1 — STAGE 1 resolution: mem0 monster first (27 hunks), then ports, then bulk
Start the merge (`git merge --no-commit --no-ff 929dd9c0d`), materializing all 35 conflicts.
Resolve in risk order — **but run no subsystem pytest yet** (tree is non-importable):
1. `plugins/memory/mem0/__init__.py` (Class B, 27 hunks) — preserve Wave-2 hybrid retrieval
   (param-drop fix, rerank gate, temporal) AND upstream's lazy-install; make the `test_mem0_v2.py`
   UD decision (INV-6).
2. Class A architectural ports — telegram ladder → `plugins/platforms/telegram/adapter.py`;
   inspect `gateway/run.py` + `agent/conversation_loop.py` for extraction-relocations
   (slash_commands/turn_finalizer pattern). Port, don't line-merge.
3. Class B/C/D remaining core (19 files) + 13 test conflicts + DU/UD pair.
- *Unit/script check:* after each sub-step, `git diff --check` clean for the touched files; at end of Stage 1, `git grep -nE '^(<<<<<<<|=======|>>>>>>>)'` returns **nothing** tree-wide.
- *E2E/integration check:* `Not applicable in Stage 1 (tree non-importable)` — the only Stage-1 integration check is the **import smoke** once markers are gone (Phase 1b).
- *Negative/adversarial:* a sub-step that "resolves" a file but leaves a `=======`/`>>>>>>>` is caught by the all-three-marker grep before advancing.
- *Verify with:* zero markers tree-wide (all three types) + `git diff --check` clean.

### Phase 1b — Clean-merge + import-smoke audit (Blocker #1, INV-10) — STILL STAGE 1
With markers cleared, before any pytest: (a) enumerate **both** the rename set
(`git diff -M --diff-filter=R --name-status c6b0eb4..929dd9c0d`) **and the upstream-delete set**
(`git diff --diff-filter=D --name-only c6b0eb4..929dd9c0d`), intersect each with fork-touched
files, and reconcile each to its new home (or confirm clean upstream-only relocation) — proving
old-absent AND new-present; (b) run a fork-only **import smoke** — `python -c "import <m>"` over
every INV-1 module — to catch dead-duplicate and semantic (signature-drift) breaks the conflict
set is blind to. **One empirical check (pass-2 OQ):** confirm whether any fork-touched
missed-rename actually lands as a silent clean-ADD vs surfacing as modify/delete (telegram's
rename WAS detected as a conflict) — target the audit at the real residual.
- *Unit/script check:* R∩fork-touched AND D∩fork-touched lists, each marked ported (old-path absent, symbol present new-path) or confirmed clean-relocation.
- *E2E/integration check:* `venv/bin/python -c "import <each INV-1 module>"` all succeed (mem0 plugin, gateway restart, context_compressor, blackbox, send_message tool, kanban_db).
- *Negative/adversarial:* deliberately point the smoke at the dead old telegram path → it must NOT resolve the fork symbol there (proving the audit detects a rename-miss). Grep proves wiring, not behavior — behavior is netted by INV-1 revert-to-RED + soak.
- *Verify with:* import smoke all-green + both rename-port and delete-port checklists complete.

### Phase 2 — STAGE 2a: per-subsystem test gates (tree now importable)
Now that the tree imports, run each INV-1/INV-2 subsystem suite via the **hermetic wrapper**
against temp `HERMES_HOME`, with revert-to-RED proofs for the load-bearing systems.
- *Unit/script check:* `scripts/run_tests.sh tests/plugins/memory` green (mem0 hybrid).
- *E2E/integration check:* `scripts/run_tests.sh tests/agent tests/gateway tests/tools tests/cron tests/hermes_cli` — compaction/LCM, F1/F2 restart-cascade, send_message routing, delegate, blackbox, kanban all green.
- *Negative/adversarial:* per INV-1 system, revert the fork hunk → its named test goes RED (mem0 reranked path; F2 breaker fires; send_message in-turn → current channel not home; blackbox per-class cost emitted; telegram ladder on new path).
- *Verify with:* every subsystem suite green + each revert-to-RED demonstrated.

### Phase 3 — STAGE 2b: lockfiles, LCM provenance, commit, full suite
Reconcile `uv.lock` + `package-lock.json` (regenerate, don't hand-merge). Validate LCM
provenance (INV-7). Write the merge commit (D-5 body). Run the **full** hermetic suite (no `-x`).
- *Unit/script check:* `uv lock` clean + lockfile-check passes; `python scripts/check_lcm_upstream_drift.py` (offline) passes; merge commit has 2 parents (`git rev-list --parents -n1 HEAD`).
- *E2E/integration check:* `scripts/run_tests.sh` **full suite, no `-x`** → complete pass set (a huge merge needs the full failure list, not first-failure-abort).
- *Negative/adversarial:* confirm no fork-only test was silently dropped (INV-6); `git diff --stat` of deleted test files reviewed.
- *Verify with:* full hermetic suite green; lockfile + LCM-drift + 2-parent checks pass.

### Phase 4 — Push branch, open PR, drive CI to full green
Push `sync/upstream-2026-06-29` to `fork`, open a PR (NOT auto-merge). Drive all required
workflows green; address any Greptile threads on merit (per merge reference doc). Resolve
`contributor-check` / `history-check` (large merge — confirm author-map + history gates pass).
- *Unit/script check:* `gh pr checks <N> --repo Kyzcreig/hermes-agent` → all green, 0 pending.
- *E2E/integration check:* the full CI matrix (tests shards, typecheck, lint, SAST, secret-scan, OSV, supply-chain) green — the real cross-host proof, not just local.
- *Negative/adversarial:* confirm no required check was skipped or made to pass via `--admin` over red; `fleet-secret-scan` + `fleet-sast` specifically green (large merge = large secret-scan surface).
- *Verify with:* `gh pr view <N> --json statusCheckRollup` → every required check SUCCESS.

### Phase 5 — Config-migration dry-run against REAL fleet configs (Blocker #3, INV-5) — pre-deploy gate
Before ANY host restarts: copy each of the hermes-agent hosts' live `config.yaml` to a **writable**
scratch `HERMES_HOME` (pass-2 fix: `migrate_config` writes in place, so the copy can't be
read-only), run the merged `migrate_config` (29→32) against each, and assert every fork-only key
survives. This is the one regression class CI structurally cannot see (CI uses temp `HERMES_HOME`).
- *Unit/script check:* per host, `check_config_version()` pre=29 → post=32; the dry-run invokes the **same code path the fleet's real first-launch migration takes** (`migrate_config(interactive=False)` must be the same branch production hits, or the dry-run validates a branch production won't).
- *E2E/integration check:* diff the pre/post key set of each host's writable snapshot — every fork-only key (`kanban` per-task-model, `fallback_model`/reasoning_effort, F1/F2 gateway settings, `compression.*` thresholds) present and value-preserved post-migration.
- *Negative/adversarial:* inject a synthetic fork-only key into a snapshot → migration must NOT drop or reject it (a migration that strips unknown keys is a BLOCK).
- *Verify with:* a per-host migration report; all pass before Phase 6 proceeds. **No host restarts until its snapshot passes.**

### Phase 6 — Ace review gate + staged fleet deploy (GATED — not auto)
Present the green PR to Ace with the per-system reconciliation summary (esp. mem0 + LCM/
compaction) **and the Phase-0b upstream-delta audit**. **PAUSE for explicit go-ahead** (high-blast,
fleet-wide). On approval: merge via `--admin` (own-fork green policy, **only after Greptile
threads resolved** — `--admin` does NOT bypass `required_conversation_resolution`; it bypasses the
require-approving-review rule on the sole-maintainer fork), then **staged** deploy.

**⚠️ Deploy vector (pass-2 corrected — was wrong in v0.1).** `fleet.sh deploy` ships **only
`claude-bridge` / `claude-api-proxy`**, NOT hermes-agent core code (confirmed:
`~/.hermes/fleet/fleet.sh` only knows those two repos). The real core-update vector is
**`hermes update`** per host (`git pull --ff-only` of the hermes-agent checkout + gateway relaunch
via `hermes update --gateway`, which has built-in auto-rollback on a failed pull). Each
hermes-agent-running host is updated with `hermes update`, not `fleet.sh deploy`.

**⚠️ Host-1 self-amputation hazard (pass-2).** Host-1 must NOT be the gateway host carrying
**this** deploying session — a synchronous `hermes update --gateway` there kills the very session
driving the deploy and its rollback (memory: NEVER synchronous restart → double-reboot; skill
`safe-gateway-restart`). Two valid options: (a) pick an **equally-representative gateway host that
is not this session's host** as host-1; or (b) drive host-1's gateway relaunch through
`safe-gateway-restart` from a **detached** context (`systemd-run`/launchd-detached, never a
synchronous tool call). State which before deploying.

Staged sequence:
1. **Host-1 = a representative gateway host that is NOT this deploying session's host** (proves the real interactive path). `hermes update` it; relaunch via `safe-gateway-restart` discipline.
2. **Soak window** on host-1: not a single smoke — a **bake period** spanning multiple real turns + ≥1 compaction event + ≥1 mem0 recall + ≥1 cron tick. The regression threshold below is a **floor, not a closed set** — any anomalous fork-system behavior (blackbox miscount, kanban model-override ignored, send_message mis-route) is also a rollback trigger even if not enumerated.
3. Only after host-1 soaks clean, **fan out to the remaining hosts**, each `hermes update`'d + smoke-checked.
- *Unit/script check:* post-merge `git rev-list --count fork/main..929dd9c0d` = 0 (parity to the frozen target).
- *E2E/integration check:* host-1 soak — real live turns succeed; `hermes -p apollo -q "ping"` returns; a real compaction fires and reconciles (no `COMPACTION_STATS_RECONCILE_FAILED`); mem0 recall returns the reranked path; a cron tick runs. Then each subsequent host smoke-checked.
- *Negative/adversarial:* **objective regression FLOOR (not exhaustive)** — any of {a turn errors, a `COMPACTION_STATS_RECONCILE_FAILED` marker, mem0 recall falls back to non-reranked, gateway restart-loops, a cron fails, OR any observed fork-system semantic anomaly} on host-1 = **rollback host-1** to its recorded pre-merge SHA (`git reset --hard <pre-sha>` + `hermes update`) before any fan-out. Blast radius = 1 host.
- *Verify with:* all hermes-agent hosts on the merged SHA, host-1 soak clean by the threshold above, each later host smoke-passing; per-host rollback SHA recorded and the rollback path tested on host-1.

---

## 7. Security, Privacy, Ops, Observability

- **Secrets:** large merge → large `fleet-secret-scan` surface. INV-4 requires it green; no
  redaction-mangled commits (use literal-bytes / `brand-safe-write.py` if the redactor bites a
  token in the merge body — per memory).
- **Rollback:** every deployed host's pre-merge `fork/main` SHA recorded before its restart.
  Staged deploy means a bad merge harms 1 host, recoverable by `git reset --hard <pre-sha>` +
  fleet redeploy — **this per-host reset is the real rollback.** ⚠️ Do **not** use
  `git revert -m 1 <merge-sha>` as the rollback: reverting a merge commit poisons the *next*
  upstream merge (git treats the reverted commits as already-merged, so the re-merge silently
  drops them — the classic revert-of-merge footgun). If the whole merge must be undone on the
  branch, **reset the branch to its pre-merge SHA**, don't revert the merge.
- **Major dependency bumps:** the absorbed delta is enumerated for major-version bumps (INV-9).
  `uv sync` clean proves *resolution*, not *runtime compatibility* — a transitive major bump
  (e.g. pydantic) can pass OSV + lockfile-check and still break the fleet. Any major bump gets a
  runtime-compat note and is part of host-1's soak watch.
- **Observability:** deploy alerts via house `fleet_alert` envelope (skill `notify`). LOUD-FAIL
  to #alerts only on regression; quiet/healthy → #logs. No success-noise in #alerts.
- **Provenance:** LCM vendoring drift metadata (INV-7) re-validated so the security-drift guard
  stays meaningful after absorbing upstream LCM-adjacent changes.

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **mem0 hybrid retrieval silently clobbered** by accepting upstream lazy-install | Med-High | D-3 resolve-first-in-isolation + INV-1 + Phase-1 negative test (reranked path, not upstream default) |
| **Architectural port drops a fork feature** (telegram, gateway extraction) | Med | Class-A discipline + Phase-1b/2 revert-to-RED proof the test covers the ported behavior |
| **Rename-MISS → fork edit lands as a dead duplicate at old path** (Blocker #1) | Med | INV-10 + Phase-1b clean-merge audit (old-absent AND new-present, not just symbol-present) + import smoke |
| **Semantic conflict** — fork caller in an unconflicted file vs upstream-changed callee signature, merges clean, fails at runtime | Med | Phase-1b import smoke over all INV-1 modules; host-1 soak catches runtime-only breaks CI's mocks miss |
| **Config migration (29→32) drops a fork-only key** (Blocker #3) | Med-High | INV-5 + Phase-6a dry-run against real fleet config snapshots before any restart |
| **Inherited upstream behavior change** (new core tool / un-gated telemetry / new HERMES_* / major dep bump) ships unreviewed (Blocker #5) | Med | INV-9 + Phase-0b upstream-delta audit; Ace knowing-deploy decision |
| **Hidden upstream relocation** beyond telegram (god-file extraction we didn't pre-spot) | Med | Phase-1 inspects gateway/run.py + conversation_loop.py for extraction pattern; any new port documented in commit body |
| **Drift grows between spec and execution** (1734 → more) | High (it's daily) | Phase-0 re-measures; if materially larger, re-confirm conflict count before proceeding |
| **Lockfile churn breaks runtime deps** | Med | INV-5 regenerate (not hand-merge) + `uv sync` clean + major-bump enumeration (INV-9) + host-1 soak |
| **`test_mem0_v2.py` modify/delete resolved by silently dropping coverage** | Med | INV-6 — deliberate recorded decision; coverage must live somewhere |
| **Phase gates non-executable mid-merge** (markers = syntax errors) (Blocker #2) | — (resolved in design) | §5B two-stage spine: marker-free + import-smoke in Stage 1; pytest only in Stage 2 |
| **Fleet regression after deploy** | Low-Med | D-6 staged 1-host-first + soak window + recorded rollback SHA + objective regression threshold |
| **CI red on a huge merge masks a real fork regression** | Med | Don't `--admin` over red (INV-4); per-system suites in Stage 2a catch fork regressions before CI |
| **Greptile blocks merge** (conversation-resolution gate) | High (always reviews) | Merge reference doc: address on merit, resolve thread, re-green; `--admin` does NOT bypass it |

## 9. Open Questions (elaborated for Ace in chat, per prd-spec rule)

1. **Cut-off snapshot vs chase-to-HEAD?** Upstream commits daily. Merge to a **frozen** target
   SHA (`929dd9c0d`, pinned in the header) for the whole build/CI/review, accepting it's hours
   stale at deploy — or re-merge to absorb net-new commits before deploy? *Recommendation: freeze
   at `929dd9c0d`; a tiny follow-up catch-up merge later is cheaper than a moving target. (Pass-1
   sharpened: the frozen SHA is now the single pinned value used in §1, §10 AC, and Phase 6 parity
   check — no "snapshot" ambiguity.)*
2. **`test_mem0_v2.py` (modify/delete): keep-and-port or accept-upstream-delete?** Depends on
   whether the behavior it covered moved to a still-present module. *Recommendation: decide in
   Phase 1 after reading why upstream deleted it; default to keep-and-port if it covers
   fork-only hybrid retrieval; accept-delete only if the coverage genuinely relocated.*
3. **Deploy cadence policy (D-8) — build it now or after?** *Recommendation: ship the merge
   first; spec the recurring-sync cron as a v0.2 follow-up so we don't gold-plate the critical
   path.*
4. **Are the 4 parked live-tree changes (INV-8) part of the "220 fork-only preserved" count, or
   additive uncommitted work?** They're *uncommitted*, so they are NOT in the 220 and §10's
   `git log` AC won't cover them. *Recommendation: triage them in Phase 0 — each is either (a)
   already superseded by a landed commit (discard), or (b) genuine un-landed work (commit to a
   fork branch first, so it's covered by history), before the merge starts. Record the disposition.*

## 10. Acceptance Criteria

- [ ] Merge commit on `sync/upstream-2026-06-29` has 2 parents (`fork/main` + `929dd9c0d`); `git rev-list --count <branch>..929dd9c0d` = 0. Evidence: `git rev-list --parents -n1 HEAD`.
- [ ] Zero conflict markers anywhere (**all three types**). Evidence: `git grep -nE '^(<<<<<<<|=======|>>>>>>>)'` empty; `git diff --check` clean.
- [ ] All 220 fork-only commits preserved (none lost to resolution). Evidence: `git log --oneline 929dd9c0d..<branch>` still contains the fleet-critical commits; INV-1 grep + per-system tests pass.
- [ ] **Clean-merge surface audited (INV-10):** every rename ∩ fork-touched file confirmed ported (old-absent + new-present); import smoke over all INV-1 modules green. Evidence: Phase-1b checklist + `python -c "import …"` output.
- [ ] mem0 hybrid retrieval intact. Evidence: `scripts/run_tests.sh tests/plugins/memory` green + revert-to-RED proves reranked path active.
- [ ] Telegram fix ported to new adapter path. Evidence: `scripts/run_tests.sh tests/gateway/test_telegram_network_reconnect.py` green; grep shows symbol on `plugins/platforms/telegram/adapter.py` AND absent from `gateway/platforms/telegram.py`.
- [ ] **All INV-1 systems gated by VERIFIED-EXISTING tests with revert-to-RED proof.** Evidence: Phase-0a checklist (each system → real test path → RED-on-revert confirmed).
- [ ] **Upstream-delta audit complete (INV-9).** Evidence: `docs/sync/review/upstream-delta-audit.md` lists new-core-tools / telemetry / `HERMES_*` / major-dep-bumps findings or explicit "none."
- [ ] Full CI green, 0 pending, nothing skipped, no `--admin` over red. Evidence: `gh pr checks` all SUCCESS.
- [ ] LCM provenance + lockfiles valid. Evidence: `check_lcm_upstream_drift.py` passes; `uv-lockfile-check` green.
- [ ] Live tree reconciled (INV-8). Evidence: the 4 parked changes consciously committed-to-branch or discarded, disposition recorded.
- [ ] **(GATED) Config-migration dry-run passes for all 7 hosts (INV-5).** Evidence: per-host report — `_config_version` 29→32, every fork-only key value-preserved; no host restarted before its snapshot passed.
- [ ] (GATED) Fleet on merged SHA, host-1 soak clean by the objective threshold, each later host smoke-passing, per-host rollback SHA recorded + rollback path tested on host-1. Evidence: per-host smoke + `fleet.sh` status.

---

## Roadmap (post-merge)

| Version | What ships | Trigger | Maps to |
|---|---|---|---|
| v0.1 | This merge → parity, fleet deployed | Ace go-ahead | §6 Phases 0–6 |
| v0.2 | Recurring upstream-sync cron (weekly/biweekly merge so drift never re-walls) | After v0.1 lands clean | D-8 |
| v0.3 | Optional: automate the conflict-class triage (the Class A/B/C/D pre-classifier) if v0.2 reveals it's worth it | If manual triage is the bottleneck at v0.2 cadence | — |
