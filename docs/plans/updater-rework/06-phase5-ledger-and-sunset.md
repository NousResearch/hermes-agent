# Phase 5 — Feature Ledger, Docker Unification, and the Sunset

> **For Hermes:** subagent-driven-development, task-by-task.
> Read `docs/updater-world.md` §2.10, §2.11, §2.12, §2.13 (sunset) first.

**Goal:** (a) lazy-feature activation survives venv replacement via a
data-dir ledger, (b) the Docker image is built from the release bundle,
(c) a dated, checklisted sunset plan for deleting the legacy machinery.

---

## Task 5.1: `features.json` ledger — TDD (§2.10)

**Files:**
- Modify: `tools/lazy_deps.py`
- Test: `tests/tools/test_feature_ledger.py` (extend
  `tests/tools/test_lazy_deps.py` patterns — read it first)

**Step 1 (failing tests):**
- `record_feature(name, via)` writes/updates
  `$HERMES_HOME/state/features.json` (atomic: tmp + replace; schema
  `{"schema":1, "features": {name: {"activated_at": iso, "via": str}}}`).
- `ledger_features() -> list[str]`.
- `remove_feature(name)`.
- One-time seed: when the state file is absent and the venv probe
  (`active_features()`, unchanged) finds features, the first
  `ledger_features()` call writes them with `via="venv-probe-migration"`.
  Also consume `state/features.pending.json` if present (written by
  phase-2 adopt) and merge it in.
- `apply_ledger(venv_python) -> dict[str,str]` — for each ledger feature,
  run the existing `ensure()` machinery against the given venv; statuses
  mirror `refresh_active_features` (`current/refreshed/failed:/skipped:`);
  honors `security.allow_lazy_installs`.

**Step 2-4:** red → implement → green:
`scripts/run_tests.sh tests/tools/test_feature_ledger.py -q`.

**Step 5:** `ensure()` gains the `record_feature` call on first successful
install. `hermes features` CLI: `list` / `disable <name>` (small argparse
wiring; `enable` is just using the feature).

**Step 6:** Commit: `feat(features): data-dir activation ledger`.

## Task 5.2: Wire the ledger into all three worlds

**Files:**
- Modify: `apps/hermes-launcher/src/apply.rs` — post-flip, pre-restart:
  run `<new slot>/bin/hermes features apply-ledger --json`, report
  failures as warnings (never fail the flip for a feature install).
- Modify: `hermes_cli/dev_sync.py` — step 5 placeholder from phase 3 now
  calls `apply_ledger`.
- Modify: `hermes_cli/main.py` — `_refresh_active_lazy_features()` body
  becomes `apply_ledger(sys.executable)` (§2.10: the ledger REPLACES the
  probe-based refresh; keep the function name/callsite — it is NOT in the
  frozen contract? **CHECK `updater_compat` FIRST** — it IS listed there,
  so keep the symbol + signature and only change internals).

**Verification:** phase-1 E2E extended: activate a fake lazy feature in v1
(fixture feature pointing at a tiny wheel in the file:// fixture), apply
v2, assert the feature is importable in v2's venv. Same in the phase-3 E2E
for a second worktree.

**Commit:** `feat(features): ledger applied on flip and dev sync`.

## Task 5.3: Docker image from bundle (§2.12)

**Files:**
- Modify: `Dockerfile`
- Test: existing `tests/docker/test_immutable_install.py` must stay green

**Step 1:** Replace the image's build stages with: fetch (or COPY from CI
artifact) the `linux-<arch>` bundle → verify → unpack to `/opt/hermes` as
a single baked slot (`current.txt` naming the one version). Entrypoint =
`/opt/hermes/bin/hermes`. Keep: the baked `docker`
`.install_method` stamp, `/opt/data` as `HERMES_HOME`.

**Step 2:** `hermes update` / `/update` in-container behavior is already
correct (redirect to `docker pull`) — add a regression test asserting the
updater's `apply` refuses inside a container (probe via the existing
`is_container()`).

**Step 3:** Build + run the image locally; run the docker test suite:
`scripts/run_tests.sh tests/docker/ -q`.

**Step 4:** Commit: `feat(docker): image built from release bundle`.

## Task 5.4: Default flip for new installs

**Files:**
- Modify: `scripts/install.sh`, `scripts/install.ps1`, website install docs

**Gate (maintainer sign-off required):** phases 1-4 E2E all green in CI for
2+ consecutive weeks; no open P1s against adopt/apply.

**Step 1:** `--bundle` becomes the default; `--source` = the old path
(clone + eject formalization). The desktop bootstrap runner requests the
bundle path via its stage protocol (the updater serves the stage manifest —
phase 1 task 1.8 note).

**Step 2:** Update `website/docs/getting-started/installation.md` (+ zh-Hans
mirror) describing managed vs ejected.

**Commit:** `feat(install): bundle install is the default`.

## Task 5.5: The sunset checklist (write now, execute later)

**Files:**
- Create: `docs/plans/updater-rework/sunset-checklist.md`

Write the dated checklist; each item has a precondition and a verification.
Contents (from §2.11's fate table and §2.13):

```
PRECONDITION for all: adoption prompts shipping ≥ 12 months; maintainer
declares legacy population negligible (support-channel signal / opt-in
diagnostics).

[ ] Delete hermes_cli/updater_compat.py + tests/test_updater_compat_fence.py
[ ] Delete _cmd_update_impl git flow (keep the thin dispatcher:
    slot→updater, checkout→dev_update, docker/nix/brew→messages)
[ ] Delete: _UvResult (managed_uv.py), rebuild_venv tombstone,
    _update_via_zip, _quarantine_running_hermes_exe + friends,
    _pause/_resume_windows_gateways_for_update,
    _detect_concurrent_hermes_instances, _detect_venv_python_processes,
    .update-incomplete recovery (_recover_from_interrupted_install),
    _install_hangup_protection + _UpdateOutputStream (updater owns logs),
    gateway/code_skew.py, retry-once in any remaining Tauri path,
    sourceDeclaresServe + dashboardFallbackArgs (desktop)
[ ] Shrink install.sh/install.ps1 to: fetch updater + PATH setup + --source
    ⚠ PRECONDITION beyond the global one: hermes_cli/dep_ensure.py uses
    install.sh AS ITS RUNTIME BACKEND for lazy native-dep installs
    (ffmpeg, chromium/agent-browser, system packages — see its module
    docstring: "1900 lines of battle-tested OS detection"). Before
    shrinking, extract the OS-detection + package-manager logic into a
    standalone script the bundle carries (e.g. scripts/install-native-dep.sh,
    shipped in app/scripts/ of every bundle) and re-point dep_ensure.py
    at it. Verify: `hermes doctor` on a slot install with no system
    ffmpeg can still prompt-install it.
[ ] Remove run_tests.sh third venv probe (deprecated in phase 3)
[ ] Each deletion lands as its own PR with the E2E gates green
```

**Commit:** `docs: legacy updater sunset checklist` — **phase 5 complete**
(sunset execution excluded — it is time-gated).

## Pitfalls

- 5.2's `_refresh_active_lazy_features` note is the pattern for EVERY
  frozen symbol: internals may change freely; name + signature may not,
  until sunset.
- The ledger applies features with the NEW slot's pins — a feature that no
  longer resolves must degrade to `failed:` + warning, never block a flip
  (a user's update must not hostage on an optional extra).
- Docker: keep `/opt/data` and the bind-mount contract EXACTLY —
  documented user setups depend on it.
