# Sync forgeguard-ai/hermes-agent with upstream v2026.8.31 (Hermes 0.21.0 "Pantheon") + silence no-op workflow runs

## Context

The fork last synced at upstream **v2026.8.16.2** (per `FORK_UPSTREAM_BASE`; the user
guessed ~v2026.8.13 — it's actually one release later, so the delta is smaller than
feared). Since then the fork shipped v0.20.4 → v0.20.8 (TTS chunking modes, OpenAI
streaming-TTS hardening, electron 40.10.6 + advisory bumps, cli-image retirement,
Linux icon set, etc.). Upstream has since cut v2026.8.18, .19, .27 and the major
**v2026.8.31 = product 0.21.0 "Pantheon"** (Bot Mode in desktop, cron memory,
steerable subagents, MCP command center, agent-driven in-app browser; rolls up the
0.20.1–0.20.6 windows).

The repo has a mandated, load-bearing runbook — `docs/maintainers/upstream-sync/sync-policy.md`
with companions `patch-inventory.md` and `conflict-resolution.md` — and this plan follows
it step by step, enriched with concrete research findings below.

Side task: the fork's Actions tab has **523 schedule-event runs, nearly all "skipped"**
(~10/day) plus per-PR/push/release skipped runs from upstream-guarded workflows. User
chose to silence them by **editing workflow files** (removing the unwanted triggers).

### Research findings (verified against a real trial merge in a scratch worktree)

- **Delta**: 464 upstream commits, ~3,722 files (475k+/64k-). Merge base confirmed =
  `7339f5f` (v2026.8.16.2's head), so `git merge v2026.8.31` computes only the delta
  since the last sync, as the runbook designed.
- **Trial merge produced exactly 28 conflicted files** (list in Phase B) — all inside
  the patch-inventory's expected surface. Everything else auto-merges.
- **ci.yml → ci.yaml rename**: upstream renamed CI and added a `detect` path-filter
  job (cost saver), `rust-tests` (new Tauri bootstrap-installer at
  `apps/bootstrap-installer/src-tauri`), and its own PR-cancel `concurrency` block
  (absorbs the fork's concurrency patch). Git rename-detection **auto-carried the
  fork's `contributor-check` repo guard into ci.yaml** in the trial merge — verify,
  don't re-add. Workflow `name: CI` is unchanged, so `ci-review-comment.yml`'s
  `workflow_run: workflows: [CI]` keeps working.
- **New upstream workflows**: `nix.yml` (pull_request/push, but path-gated via
  detect → only runs when nix files change; leave as-is), `rust-tests.yml`
  (workflow_call only), `windows-venv-e2e.yml` (only `wine2e/**` branches; inert).
  No new external reusable-workflow calls beyond osv-scanner's existing one.
- **Fork patches NOT superseded upstream** (carry forward, verify by their tests):
  auth.py "no-key" sentinel; `canonicalize_provider_slug` cluster; mem0
  embedder-bearer scoping; OpenAI streaming-TTS hardening (upstream still reads
  `OPENAI_BASE_URL` env at `tools/tts_streaming.py:280`); TTS chunking modes;
  desktop connection cluster; Linux icon set; desktop `homepage` field;
  gateway-settings `localDesc` wording.
- **Superseded / partially superseded**:
  - PR concurrency in CI → absorbed by upstream ci.yaml (drop fork delta).
  - `gateway.ping` → upstream added a ws-transport fast-path heartbeat in
    `tui_gateway/ws.py:474` (replies before dispatch). The fork's `@method("gateway.ping")`
    in `tui_gateway/server.py:8901` **composes** with it (non-ws transports still hit
    dispatch) — keep both, note "partially superseded" in the inventory, keep
    `tests/tui_gateway/test_gateway_ping.py` green.
  - `nanoid` 3.3.18 pin → upstream moved to **6.0.0**; take upstream, then verify no
    vulnerable 3.x copy remains nested in either npm lockfile.
- **Fork stays ahead on security pins** (keep fork side): electron **40.10.6**
  (upstream 40.10.2 carries CVE-2026-70606; keep devDependency + `build.electronVersion`
  + root `package.json` `allowScripts` exact-version key, guarded by
  `apps/desktop/electron/desktop-electron-pin.test.ts` and
  `tests-js/allow-scripts-sync.test.ts`); `h2` **4.4.1** (upstream 4.3.0).
- **Dockerfile**: upstream change is +14/−6 (comment wording + build-info tweak) — port
  into the fork's `base/toolchain/venv-runtime/runtime` multi-stage layout; do NOT
  resurrect the retired `cli` stage.
- **Versioning** (user confirmed): fork version → **0.21.0**; hand-set in
  `pyproject.toml`, `hermes_cli/__init__.py`, `uv.lock`, `apps/desktop/package.json`,
  and the `apps/desktop` entry in root `package-lock.json`. `FORK_UPSTREAM_BASE` →
  `v2026.8.31`. Add the mapping row to `docs/site/fork/compatibility.md`.
- **Branch** (user confirmed): run on the standing **`dev`** branch per the runbook
  (explicit permission granted to push `dev`; this supersedes the session's
  `claude/hermes-agent-upstream-sync-b5d7xm` designation).
- **Noise inventory** (user confirmed: silence ALL of these by editing files):
  crons — `skills-index-freshness.yml` (0 */4), `skills-index.yml` (6,18 daily),
  `install-e2e.yml` (7,19 daily), `osv-scanner.yml` (weekly); per-event —
  `docker.yml` (every PR/push/release; build/publish/merge all guarded to
  NousResearch), `js-autofix.yml` (main pushes touching JS), `publish-e2e-evidence.yml`
  (after every CI run), `deploy-site.yml` (each fork release + site-path pushes).

## Step 0 (AGENTS.md plan-saving rule)

Copy this plan (with the checkboxes below) to
`docs/agent-plans/2026-09-03-upstream-sync-v2026.8.31-plan.md` before starting, and
update checkboxes in place as work completes.

## Phase A — Prep (runbook steps 1–2)

- [x] Save the plan copy (Step 0 above).
- [x] Un-shallow enough history if needed (clone is shallow; the trial merge already
      worked, so this is only a fallback: `git fetch origin main --deepen=200`).
- [x] Fetch upstream tag properly: add `upstream` remote per the runbook
      (`git remote add upstream https://github.com/NousResearch/hermes-agent.git`),
      `git fetch upstream --tags` (tags v2026.8.16.2–v2026.8.31 already fetched locally).
- [x] `git checkout main && git pull origin main`, then `git checkout -B dev main`,
      `git push -u origin dev`.

## Phase B — Merge + conflict resolution (runbook steps 3–4)

- [x] `git merge v2026.8.31 --no-edit -m "Merge upstream v2026.8.31 into fork dev"`
      (real merge; never squash/rebase).
- [x] Resolve the 28 known conflicts. Per-file guidance (rule of thumb from
      `conflict-resolution.md`: keep upstream's substantive change, re-apply the
      fork-only delta on top):
  - **`apps/desktop/src/app/gateway/hooks/use-gateway-boot.ts` (15 hunks) and
    `apps/desktop/electron/main.ts` (11 hunks)** — the hard part. Fork side: TLS-bypass
    threading (`probeConnectionConfig(url, allowInvalidCertificate)`,
    `hostAllowsInvalidCertificate`, `installCertificateBypass()`), savedRemotes,
    first-run choice IPC + gates, heartbeat liveness + bounded initial-connect retries.
    Upstream side: Bot Mode / 0.21 restructuring. Take upstream structure, re-thread each
    fork behavior; the fork tests are the acceptance gate
    (`use-gateway-boot.test.tsx` fork cases, `first-run-choice.test.ts`,
    `connection-config.test.ts`, `tests/tui_gateway/test_gateway_ping.py`).
  - **`boot-failure-overlay.tsx` (3), `use-gateway-boot.test.tsx` (3)** — same cluster.
  - **Lockfiles — never hand-merge**:
    - `package-lock.json` (2 hunks): take either side to unblock, then regenerate with a
      real resolve — `npm install --package-lock-only` at root; explicitly re-resolve
      electron 40.10.6 (`npm install electron@40.10.6 --package-lock-only -w apps/desktop`)
      so `@electron-internal/extract-zip` stays in the tree (hand-edited entries broke
      cold-cache `npm ci` before — patch-inventory warning).
    - `uv.lock` (4 hunks): set pyproject version first, then relock with the uv version
      the **merged** CI pins (upstream now uses uv 0.11.6 in Docker; check the merged
      `uv-lockfile-check.yml`/`tests.yml` pin and use that exact version). Keep `h2` ≥ 4.4.1.
  - **`pyproject.toml` (2), `hermes_cli/__init__.py` (1)** — upstream 0.21.0 changes +
    fork version: set `0.21.0` per Phase E.
  - **`plugins/memory/mem0/_backend.py` (1)** — keep fork's scoped Ollama-client
    Authorization patch inside upstream's updated code.
  - **`apps/desktop/src/app/settings/gateway-settings.test.tsx` (1)** — keep fork
    `localDesc` wording assertions ("… Works offline.", no "This is the default").
  - **i18n `en/ja/zh/zh-hant.ts` (1 each) + `settings/constants.ts` area** — merge
    upstream's new keys, keep fork's TTS-chunking + client-mode-first strings.
  - **`Dockerfile` (1)** — comment/build-info wording; port upstream's text into the
    fork's staged layout.
  - **`.gitignore`, `agent/agent_init.py`, `agent/agent_runtime_helpers.py`,
    `apps/shared/src/json-rpc-gateway.ts`, `updates.test.ts`, `hardening.test.ts`,
    `gateway-menu-panel.tsx`, `model-settings.tsx`, `gateway-settings.tsx`,
    `config-settings.tsx`, `wiring.tsx`, `use-desktop-integrations.ts`,
    `website/docs/user-guide/desktop.md`** — single-hunk, additive; standard resolution.
- [x] Grep the merged tree for regressions the auto-merge may have silently caused:
      no `cli-latest`/`venv-cli` references, no second `test:` block in
      `apps/desktop/vite.config.ts`, only `ci.yaml` (no stray `ci.yml`).

## Phase C — Patch-inventory re-verification (runbook step 4b)

Walk `docs/maintainers/upstream-sync/patch-inventory.md` top to bottom on the merged
branch. Highlights this sync:

- [x] `ci.yaml`: fork `contributor-check` guard present (trial merge carried it — verify);
      upstream's own concurrency block retained; `docker-lint`/`osv-scanner` still called
      as fork-owned files (their direct-invocation contents were untouched upstream).
- [x] Audit **every job** of the 3 new workflows (`nix.yml`, `rust-tests.yml`,
      `windows-venv-e2e.yml`) for anything that fires or costs money on the fork, and for
      `uses: <external>/.github/workflows/...` calls (org policy). Finding: none needed
      guards as of the trial merge — nix is path-gated, rust-tests is call-only,
      windows-venv-e2e triggers only on `wine2e/**`. Re-verify on the real merge.
- [x] Confirm the guarded set survived untouched (upstream didn't modify them):
      `deploy-site.yml`, `skills-index*.yml`, `install-e2e.yml`, `js-autofix.yml`,
      `publish-e2e-evidence.yml`, `osv-scanner.yml` fork guards.
- [x] Run every "Carried runtime patches" test file listed in the inventory
      (auth no-key, custom-slug, mem0, TTS hardening + chunking, connection cluster,
      Linux icons, electron-pin + allowScripts guards, voice re-arm coverage).
- [x] Update `patch-inventory.md` itself: mark concurrency **retired-superseded**;
      add the gateway.ping "partially superseded, both kept" note; update nanoid
      entry (upstream 6.0.0 adopted); record the new Phase D trigger-stripping
      entries; note the ci.yml→ci.yaml rename.

## Phase D — Silence the no-op workflows (side task; same PR)

Strip the fork-useless triggers, keeping `workflow_dispatch` and `workflow_call`
so reusable/manual paths stay valid (a workflow needs at least one trigger):

- [x] `skills-index-freshness.yml`: drop `schedule`.
- [x] `skills-index.yml`: drop `schedule` + `push`.
- [x] `install-e2e.yml`: drop `schedule` + `push` (v* tag trigger matches fork release tags).
- [x] `osv-scanner.yml`: drop `schedule` only (keep `workflow_call` from ci.yaml + dispatch).
- [x] `docker.yml`: drop `pull_request`/`push`/`release`; add `workflow_dispatch` so the
      file keeps a valid trigger.
- [x] `js-autofix.yml`: drop `push` (keep dispatch).
- [x] `publish-e2e-evidence.yml`: drop `workflow_run`; add `workflow_dispatch`.
- [x] `deploy-site.yml`: drop `release` + `push` (keep dispatch).
- [x] Each edit: one comment line above `on:` referencing the fork policy, e.g.
      `# ForgeGuard fork: <original triggers> removed — upstream-infra only; see docs/maintainers/upstream-sync/patch-inventory.md`.
- [x] Add a new patch-inventory checklist entry: "**Trigger-stripped upstream-infra
      workflows**" listing all 8 files and the rule (fork removes event triggers whose
      jobs are entirely upstream-guarded; on every sync, re-strip if upstream's trigger
      blocks merge back in, and triage any NEW upstream workflow for the same treatment).
- [x] Note for the PR body: existing job-level `if:` guards stay (defense in depth if a
      trigger block ever slips back in during a future sync).

## Phase E — Version + marker (runbook step 6)

- [x] Hand-set fork version **0.21.0**: `pyproject.toml`, `hermes_cli/__init__.py`,
      `apps/desktop/package.json` `"version"`, root `package-lock.json`'s
      `apps/desktop` entry, and `uv.lock` (via the relock).
- [x] `echo "v2026.8.31" > FORK_UPSTREAM_BASE`; commit
      `chore: bump FORK_UPSTREAM_BASE to v2026.8.31`.
- [x] Add the 0.21.0 ↔ v2026.8.31 row to `docs/site/fork/compatibility.md`.

## Phase F — Test the merged branch (runbook step 5)

- [x] Python: create the venv (`uv sync` with the CI-pinned uv; container has Python 3.11 —
      match the project's `requires-python` and CI setup), then `scripts/run_tests.sh`
      (never bare pytest).
- [x] JS: `npm ci` then the repo's JS test entry points (root workspaces + `apps/desktop`
      unit tests; upstream's new `run-workspace-checks.mjs` path in `js-tests.yml` shows
      the canonical commands).
- [x] Lint/format/typecheck fast checks that CI runs.
- [x] Triage every failure per `conflict-resolution.md`: reproduce suspicious ones on a
      clean upstream-v2026.8.31 checkout (`/home/user/nousresearch/hermes-agent` read
      clone is available) — pre-existing upstream debt gets noted in the PR, merge-only
      regressions get fixed before the PR opens.

## Phase G — PR, merge, release verification (runbook steps 7–8)

- [x] Push `dev`; open PR `sync: merge upstream v2026.8.31 into fork main`
      (base `main`, head `dev` — never an upstream branch as head). Body: upstream
      release summary, conflict/resolution notes, patch-inventory outcome table,
      workflow-silencing changes, test triage. End with the required attribution footer.
- [ ] Subscribe to PR activity and drive CI to green.
- [ ] Land with a **real merge** (`--merge`, not squash) — preserves the two-parent
      structure the next sync depends on.
- [ ] After merge, verify: ci checks green on the merge commit; `release-on-merge.yml`
      runs fully (check **step**-level conclusions — installers actually uploaded, the
      `runtime-*` image actually pushed; the cli leg is retired, so expect runtime only,
      i.e. the release-process doc's "both image variants" phrasing is stale — update it
      if touched); GitHub Release tagged `v0.21.0` with "Upstream release: v2026.8.31"
      and all 7 installers attached.
- [ ] Confirm the Actions tab: no new scheduled/skipped runs appear after the merge
      (crons stop within their next window).

## Verification summary

1. All patch-inventory test files pass on the merged branch (the inventory lists each).
2. `scripts/run_tests.sh` + JS suites triaged clean (upstream-debt exceptions documented).
3. Trial-merge parity: `git diff --name-only --diff-filter=U` during the real merge
   should match the 28-file list above; investigate any surprise per the runbook.
4. Post-merge release artifacts verified per Phase G.
5. Next-day check: zero new skipped scheduled runs in Actions.

## Notes / risks

- The desktop connection-cluster files (`use-gateway-boot.ts`, `main.ts`) are the only
  genuinely hard conflicts — 26 hunks against upstream's Bot Mode restructuring. Budget
  most of the resolution time there; the fork's test files define "done".
- Branch protection: if any required check names changed with ci.yaml's new job set
  (`detect`, `rust-tests`), update the required-checks list on `main` (surfaces
  immediately on the PR as a stuck required check).
- Silencing edits mean `workflow_dispatch` on those 8 workflows still shows them in the
  Actions sidebar (harmless); repo-level disabling remains available later as an extra
  step if desired.


## Execution log (2026-09-03)

- Merge commit `6998ca32a7` (parents: fork main `b091ab88ed` + upstream tag `29112bef09`).
- Merge-only test regressions found and fixed: session-model-source provider healing
  (named custom slugs kept), agent_runtime_helpers indentation, desktop 0.21-seam
  re-threading (initial-connect retries, post-boot ticket gate carve-out, heartbeat
  test adaptation, voice-playback mocks, voiceFieldVisible move, runQuitAsyncTeardown
  reconstruction, oauthLogoutConnectionConfig URL arg, import order).
- Pre-existing upstream debt (reproduced on clean v2026.8.31): 28 failures / 17 files.
- Local full-check caveat: run-workspace-checks runs all checks concurrently; on a
  4-core box use `--concurrency 1` or the ui suite alone reports 30+ minutes of
  CPU-starvation overhead.

## Execution log (2026-09-08) — Phase G, the release

- PR #35 merged 2026-09-04 (real merge, `637e01d9bb`); ci green on the merge commit.
- `release-on-merge.yml` run 33832767206 did NOT ship v0.21.0: the macOS desktop
  job failed its main-process tests on ONE upstream test — the POSIX managed
  launcher test hard-codes `hermesPath: '/bin/true'`, and macOS has `true` only
  at `/usr/bin/true`, so `env … /bin/true update --yes` exits 127 and the
  launcher publishes that as the status. Linux passed; the other jobs were
  cancelled behind the failure. Upstream never sees it: its only macOS lane is
  the Python suite. Not a merge regression — reproduces on clean v2026.8.31.
- Fix: the test resolves `/bin/true` else `/usr/bin/true` at runtime
  (`TRUE_BINARY`); recorded in the patch inventory under Desktop app. Upstream
  has not touched either file since the base tag (checked against upstream/main
  2026-09-08), so this is a fork-carried patch and an upstream PR candidate.
- The re-cut: merging the fix `dev` → `main` fires the release again; the test
  path is release-relevant to the gate (`apps/desktop/electron/` is not in its
  skip list), so expect `v0.21.0` (first cut of the product version — no
  `-forgeguard.N` suffix, since no v0.21.0 release exists yet).
