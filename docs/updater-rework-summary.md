# Hermes updater rework, in brief

## What's wrong with the install / update process?

Hermes has [five separate install/update surfaces](./updater-world.md#11-the-cast-of-characters): a 3,100-line `install.sh`, a 3,300-line `install.ps1`, the ~1,000-line `hermes update`, the gateway's detached `/update`, and the desktop app's Electron/Tauri updater. There is no binary release. Every user machine builds everything from source: venv, TUI, dashboard, and (for desktop users) a full local Electron build.

The deeper problem, laid out in [§1.4](./updater-world.md#14-hermes-update--anatomy-of-the-cli-apply-path): `hermes update` is a long-lived process that pulls new code onto disk while old code is still running in memory. Every lazy import after the pull crosses that boundary. Incidents from exactly this have left [permanent scar tissue](./updater-world.md#17-failure-modes-the-current-design-has-accumulated-a-partial-list): the `_UvResult` polymorphic return, retry-once loops, `code_skew.py`, bytecode purges, two Windows process guards, autostash recovery, post-pull syntax rollback. My personal favorite is the [`rebuild_venv` tombstone function](./updater-world.md#appendix-b--the-rebuild_venv-tombstone-update-boundary-reverse-direction), whose entire body is `True # dont remove me. ask ethernet`. It has no callers and can't be deleted, because some install parked on an old version will call it mid-update and crash half-surgeried. Each guard is a correct fix for a real incident. Together they are the cost of letting a process update itself, times five surfaces that each carry their own copy. The repo lands ~100 commits a day, so the scar tissue only accumulates faster from here.

## The redesign
Most users won't ever edit their own codebase, so, let's split the userbase into two explicit modes (one way easier to support).

**Managed.** CI builds a [self-contained bundle per platform](./updater-world.md#21-the-release-artifact): relocatable Python, a fully resolved venv, Node, pre-built TUI/web/desktop, and a signed manifest. Bundles unpack into [versioned slots](./updater-world.md#22-on-disk-layout-versioned-slots--atomic-flip) under `$HERMES_HOME/versions/`, and a small external Rust binary commits the switch by atomically replacing a one-line `current.txt` naming the active version. That's the whole flip: file rename-over is atomic on every platform, so POSIX and Windows share one commit mechanism instead of maintaining a symlink path and a junction path with different failure modes (a `current` symlink still exists for humans, refreshed after the commit, but nothing load-bearing reads it). The [one hard invariant](./updater-world.md#23-the-updater-as-a-separate-tiny-program): no Hermes Python process is alive while the updater mutates anything. New code only ever runs in a fresh process. That single rule deletes the update-boundary bug class, and with it most of the Windows lock war and the interrupted-update recovery. [§2.11](./updater-world.md#211-what-happens-to-the-partial-install-recovery-machinery).
Rollback is rewriting `current.txt` from `previous.txt`.

The updater updates itself with a [bootstrap hop](./updater-world.md#231-who-updates-the-updater): when a bundle's `min_updater_version` exceeds the staged binary, the old updater extracts the new one from the already-verified bundle and re-execs into it, once, with a guard against loops. this is just like rustup and deno, which do the same dance.

**Ejected.** A plain git checkout. [Activation is just where your PATH symlink points](./updater-world.md#251-ejected-mode--the-dev-clone-the-launcher-is-the-activation): at a slot's launcher for managed, at a checkout's in-repo launcher for ejected. No registry, no mode state, no stamp archaeology. And because several checkouts plus a managed install can coexist, [running `hermes` from inside any checkout requires saying which one you mean](./updater-world.md#251a-the-cwd-guard-inside-a-checkout-always-say-which-hermes): `--dev` for the tree you're standing in, `--global` for the installed one — so muscle-memory `hermes update` can never hit the wrong tree. `hermes dev sync` owns all provisioning and builds; [app surfaces refuse to launch stale artifacts](./updater-world.md#29-app-surfaces-in-ejected-mode-hermes-desktop--hermes-web--tui) rather than surprise-building for five minutes because you ran `hermes desktop` after a pull. Updating a tree with local changes [defaults to a git worktree switch](./updater-world.md#252-ejected-updates-worktree-instead-of-stash) instead of the autostash dance, so the failure mode where stash machinery eats someone's work can't occur.

Around the edges: [one `runtime-deps.json`](./updater-world.md#26-one-dependency-manifest) replaces the version floors currently copy-pasted across bash, PowerShell, and TypeScript. A [data-dir `features.json` ledger](./updater-world.md#210-optional-features-lazy_deps-across-updates) records which optional features the user activated, so they survive venv replacement instead of silently vanishing on the first flip. [Docker barely changes](./updater-world.md#212-docker); it already is the immutable-artifact model, and the image just becomes a thin wrapper around the same bundle.

## Getting existing installs there

This is the hard part. We can't push code to users' existing installs. every current install migrates by running its own old updater, which is the exact bug class we're trying to escape. The rule in [§2.13](./updater-world.md#213-the-migration-chicken-and-egg-getting-every-old-version-through): old code never performs the migration. Its final job will be to perform one more ordinary update.

We do this in three hops:

1. Any old version updates to current main through the git flow it already has. This requires main to stay a valid target for arbitrarily old updaters, enforced by a frozen `updater_compat` registry with a CI fence over every symbol historical updaters touch post-pull. This is a bit of a nightmare, and i'm down to negotiate it. We can remove it when we deem "old enough users have to re-install."
2. The next launch is a fresh process running new code! It detects the legacy layout and offers adoption of the new process. (`updates.adopt: auto|prompt|never`, dirty trees never auto-adopt).
3. `hermes adopt` fetches the verified updater binary, execs it, and fully exits. The updater installs a slot, flips, re-points the PATH symlink, and leaves the old checkout untouched as both the rollback artifact and a ready-made ejected tree. Undo at any point is one symlink re-point.

The compat fence is time-boxed at.. some amount of time, maybe a couple months, then the frozen symbols and the giant `_cmd_update_impl` get deleted in one commit :D

## Phases

| Phase | Deliverable | Closing gate |
|---|---|---|
| 0 | CI release bundles: manifest, signing, relocatable venv | bundle boots in a bare `debian:stable-slim` container |
| 1 | updater/launcher binary, slots, atomic flip | install → apply → rollback → tamper → interrupt, all E2E |
| 2 | compat fence + adoption funnel | a real 6-month-old release updates itself to main, then adopts |
| 3 | ejected mode: in-repo launcher, `dev sync`, worktree updates | worktree switching on a dirty tree, changes untouched |
| 4 | desktop calls the updater; in-app apply + Tauri orchestration deleted | packaged-app update E2E, plus a manual Windows checklist |
| 5 | feature ledger, docker-from-bundle, dated sunset checklist | deletions are time-gated on the legacy population shrinking |

Every phase closes on an E2E gate against real artifacts, not unit mocks.

The single most valuable test in the project is phase 2: an actual old release, updating itself with its own code, against today's main. When it breaks, the fix is always widening the compat registry. We can't patch a tag that's already on user machines.
