# Updater Rework — Implementation Spec (Index)

> **For Hermes:** Use the subagent-driven-development skill to implement these
> plans task-by-task. Each phase doc is self-contained.

**Goal:** Replace the five overlapping install/update surfaces with: CI-built
release bundles + versioned slots + atomic activation for managed users, and
an explicit launcher-based "ejected" source mode for developers — per the
design in `docs/updater-world.md` (read it FIRST; this spec implements that
document and cites its section numbers throughout).

**Architecture (one paragraph):** CI builds a self-contained bundle per
platform (python runtime + resolved venv + node + pre-built TUI/web/desktop).
A tiny Rust binary (`hermes-updater`, same crate as the `hermes` launcher)
downloads/verifies/unpacks bundles into `$HERMES_HOME/versions/<v>/` and
atomically flips a `current` symlink. Running processes are never mutated;
new code only ever runs in fresh processes. Source checkouts ("ejected" mode)
carry their own in-repo launcher and per-checkout venv; activation is which
launcher the PATH `hermes` symlink points at. Legacy installs migrate through
a single three-hop funnel (§2.13).

## Documents

| Doc | Phase | Deliverable | Depends on |
|---|---|---|---|
| `01-phase0-bundles.md` | 0 | CI release-bundle pipeline; nothing consumes it yet | — |
| `02-phase1-updater.md` | 1 | `hermes-updater`/launcher binary; slots + flip; new installs use it | 0 |
| `03-phase2-compat-and-adoption.md` | 2 | frozen `updater_compat` contract + CI fence; adoption funnel for legacy installs | 1 |
| `04-phase3-ejected-dev.md` | 3 | in-repo launcher, `hermes dev sync`, worktree-based ejected updates | 1 |
| `05-phase4-desktop.md` | 4 | desktop app calls the updater; delete in-app POSIX apply + Tauri run_update | 2, 3 |
| `06-phase5-ledger-and-sunset.md` | 5 | `features.json` ledger; docker unification; sunset/deletion checklist | 2 |

Phases 0/1 can be developed in parallel with 3 (they share only the launcher
crate). Phase 2 must not start until phase 1's updater passes its E2E gate.
Phase 5's sunset tasks are time-gated (§2.13) — implement the ledger now,
schedule the deletions.

## Ground rules (apply to every task in every doc)

1. **Never break hop-1 updatability of `main`.** Until the phase-5 sunset,
   every commit must keep `main` a valid update target for old updaters.
   The CI fence from `03` task 2.1 enforces this — land that fence EARLY
   (it is deliberately the first task of phase 2 and may be landed during
   phase 0/1 work).
2. **Tests:** always `scripts/run_tests.sh <path>` — never bare pytest.
   New tests must be behavior/invariant tests; no change-detector tests, no
   reading source files in tests (AGENTS.md rules — read that section before
   writing any test).
3. **No new `HERMES_*` env vars for non-secret config**; new settings go in
   `DEFAULT_CONFIG` (`hermes_cli/config.py`) under the existing `updates:`
   section.
4. **Paths:** `get_hermes_home()` / `display_hermes_home()` from
   `hermes_constants` — never hardcode `~/.hermes`.
5. **Commit per task** with conventional-commit messages. Do not push or
   merge without maintainer review.
6. **When a verification step fails, STOP.** Do not improvise around a
   failed gate; report what failed and the exact output.

## Verification philosophy

Every phase ends with an **E2E gate** exercising the real artifact:
a temp `$HERMES_HOME`, real downloads (or a local file:// release fixture),
real process spawns. Unit tests alone never close a phase. The E2E gates are
specified at the end of each doc, and each has a script committed under
`scripts/e2e/` so it is repeatable and CI-runnable.

## Vocabulary (used consistently in all docs)

- **bundle** — the per-platform release archive (`hermes-<v>-<plat>.tar.zst`).
- **slot** — an unpacked bundle at `$HERMES_HOME/versions/<v>/`.
- **flip** — atomically re-pointing `$HERMES_HOME/current` at a slot.
- **launcher** — the Rust `hermes` binary that sets env and execs the tree's
  venv python (§2.5.1). Same crate as the updater.
- **staged updater** — `$HERMES_HOME/bin/hermes-updater`, the copy that runs
  before any slot is current (§2.3.1).
- **adoption** — migrating a legacy git-checkout install to slots (§2.13).
- **ejected** — a source checkout activated via its in-repo launcher (§2.5).
- **hop 1/2/3** — the migration funnel stages (§2.13).
