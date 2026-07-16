# Phase 4 — Desktop Unification

> **For Hermes:** subagent-driven-development, task-by-task.
> Read `docs/updater-world.md` §2.3, §2.4, §1.6 first. Also read
> `apps/desktop/AGENTS.md` before touching any desktop code.

**Goal:** The desktop app's update/bootstrap flows call `hermes-updater`;
delete `applyUpdatesPosixInApp`, the relaunch-watcher script generation,
and the Tauri `run_update` orchestration. The desktop GUI becomes a bundle
artifact that updates via the flip like everything else.

**Prereqs:** phases 1 AND 2 E2E gates green; phase 3 for the dev-shell
routing (task 4.4).

**Definition of done:** `scripts/e2e/test-desktop-update.sh` (packaged-app
smoke via the existing electron-playwright E2E harness — see the
`electron-playwright-e2e` skill and `apps/desktop`'s existing e2e setup):
managed desktop detects update → applies via updater → relaunches into the
new version. Windows path verified manually on a Windows machine with a
written checklist (task 4.6).

---

## Task 4.0: Update detection reads updater status

**Files:**
- Modify: `apps/desktop/electron/main.ts` — `checkUpdates()`
- Test: `apps/desktop/electron/update-status.test.ts` (new, extract logic)

**Step 1:** Extract a pure `interpretUpdaterStatus(json)` module (per the
desktop test rules: never regex source, extract + DI). For a SLOT install
(`resolveUpdateRoot()` under `versions/`), `checkUpdates()` runs
`hermes-updater status --check --json` and maps to the existing
`DesktopUpdateStatus` shape (`behind` = releases behind, `commits` =
release notes list from the manifest changelog field). For a CHECKOUT,
keep the existing git-based detection UNCHANGED (it becomes the dev-shell
path).

**Step 2:** vitest: `cd apps/desktop && npx vitest run
electron/update-status.test.ts` → green.

**Step 3:** Commit: `feat(desktop): update detection via hermes-updater status`.

## Task 4.1: Apply = spawn updater, quit

**Files:**
- Modify: `apps/desktop/electron/main.ts` — `applyUpdates()`

**Step 1:** For slot installs, ALL platforms converge (this replaces both
the Windows Tauri handoff AND `applyUpdatesPosixInApp`):

1. `writeUpdateMarker(HERMES_HOME, spawnedPid)` (KEEP — byte-compatible,
   the marker pre-write race fix at `update-marker.ts:106-127` still
   applies);
2. spawn detached: `hermes-updater apply --relaunch-app <execPath>
   --report json --notify-file <tmp>`;
3. `isQuittingForHandoff = true; app.quit()` (existing mechanism).

The updater does: wait-for-exit (it already has the lock-wait code
inherited from Tauri) → apply → flip → relaunch the app from the NEW
slot's `desktop/` — the relaunch honesty ladder (`update-relaunch.ts`)
collapses to nothing because the GUI binary is IN the slot: the relaunched
app is definitionally the new version. AppImage/.deb/.rpm shells remain
the guiSkew exception and keep their current message (they're not slot
installs; detection in 4.0 reports them `supported: false` with the
package-manager message).

**Step 2 (checkout installs):** `applyUpdates()` routes to a plain spawn of
`hermes update` (which after phase 3 does the worktree flow) and renders
its choice prompt in the overlay — reuse the existing
`runStreamedUpdate` streaming.

**Step 3:** vitest for the pure routing decision (slot/checkout/package →
updater/dev-update/guiSkew).

**Step 4:** Commit: `feat(desktop): apply via hermes-updater handoff`.

## Task 4.2: Delete the superseded paths

**Files:**
- Delete code from: `apps/desktop/electron/main.ts`
  (`applyUpdatesPosixInApp`, the mac bundle-swap script generation, the
  detached bash relauncher construction)
- Delete: `apps/desktop/electron/update-relaunch.ts` + its test,
  `apps/desktop/electron/update-rebuild.ts` + its test
- Modify: `apps/bootstrap-installer/src-tauri/src/update.rs` — `run_update`
  becomes: exec staged `hermes-updater apply` and stream its `--report
  json` events onto the existing `BootstrapEvent` channel (the Tauri app
  stays as the WINDOWS GUI SHELL for the updater — progress window — but
  owns no orchestration logic anymore)

**Step 1:** Before deleting each function: `search_files '<name>'
apps/` — remove every call site and its dead imports/tests together.

**Step 2:** Full desktop test suite: `cd apps/desktop && npm test` → green;
`npm run typecheck` → green. Rust: `cargo test` in bootstrap-installer.

**Step 3:** Commit: `refactor(desktop): delete in-app apply + Tauri
orchestration (superseded by hermes-updater)`.

## Task 4.3: Backend spawn simplification

**Files:**
- Modify: `apps/desktop/electron/backend-command.ts`, `main.ts`

**Step 1:** For slot installs the backend is ALWAYS
`$HERMES_HOME/bin/hermes serve ...` — the stable launcher resolves
`current.txt` and the env,
so `pathWithHermesManagedNode` / venv-path assembly for the child is
unnecessary on this path. KEEP `sourceDeclaresServe`/`dashboardFallbackArgs`
for now (legacy checkouts still exist until sunset) but route: slot →
always `serve`, no sniffing.

**Step 2:** vitest on the routing. Commit:
`feat(desktop): slot backends spawn via launcher`.

## Task 4.4: Dev-shell routing (needs phase 3)

**Files:**
- Modify: `apps/desktop/src/store/updates.ts` + overlay copy

**Step 1:** When the running app is a checkout build (execPath under a git
tree's `release/<plat>-unpacked` — `resolveUnpackedRelease` logic, now
inverted to detect rather than gate), label the update UI "source install"
and the apply button drives the phase-3 worktree flow (4.1 step 2),
including rendering the switch/merge choice.

**Step 2:** vitest for the labeling/routing atoms. Commit:
`feat(desktop): dev-shell update routing`.

## Task 4.5: E2E gate (POSIX)

**Files:**
- Create: `scripts/e2e/test-desktop-update.sh`

**Contract:** using the phase-1 fixture release server with v1/v2 bundles
(desktop included): install v1 → launch the packaged app (xvfb) → poke the
update check IPC (playwright driver, reuse the desktop e2e harness) →
apply → assert the app exits, the updater flips, and the relaunched app's
`getVersion()` reports v2. Marker file removed at the end.

**Verification:** green locally under xvfb; CI linux job (nightly — slow).

**Commit:** `test(e2e): desktop update via updater gate`.

## Task 4.6: Windows verification checklist (manual, written artifact)

**Files:**
- Create: `docs/plans/updater-rework/windows-verification.md`

Write the step-by-step manual checklist (fresh Windows VM): bundle install
via install.ps1 `--bundle`-equivalent → desktop launch → update available →
apply → Tauri progress window shows updater events → relaunch on v2 →
`hermes-updater.old.exe` swept on next run. Each step has an expected
observable. Run it (or hand to the maintainer) and record results in the
doc. **Phase 4 is not complete until this doc has a filled-in PASS column.**

**Commit:** `docs: windows desktop update verification checklist` —
**phase 4 complete.**

## Pitfalls

- Desktop rules from `apps/desktop/AGENTS.md` apply: no source-regex tests,
  extract logic for DI, nanostores conventions in the renderer.
- `REQUIRED_BACKEND_CONTRACT` and remote-gateway update UI are UNTOUCHED —
  they're the honest remote-skew path (§2.9).
- Keep `readLiveUpdateMarker` byte-compat until sunset: mid-migration a NEW
  desktop may coexist with an OLD Tauri updater writing the same marker.
- The Tauri bootstrap (first-install stage runner) is NOT in scope here —
  first-install UX migrates when install.ps1's bundle path is default
  (phase 5 decision).
