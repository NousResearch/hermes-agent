# Caduceus v1.0 ⚕ — deep-planning mode + multi-agent workflows for Hermes

**Caduceus turns Hermes into a senior-engineer planner with one switch.** Turn
it on and the agent lays out a visible to-do plan and drives it methodically —
raising reasoning effort, delegating where it helps, verifying before it claims
done — and escalates to a deterministic multi-agent **workflow engine** (the
*Loom*) only when you ask. An optional **Auto Router** sends each delegated
worker to the cheapest configured model that can actually do that subtask.

> **Off by default · session-scoped · additive · fully reversible · fully tested.**
> With Caduceus off, this fork behaves exactly like upstream Hermes.

---

## Install in one command

```bash
git clone -b caduceus https://github.com/OnlyTerp/Hermes-caduceus.git
python3 Hermes-caduceus/install_caduceus.py --with-desktop   # safe, reversible overlay
```

Then restart Hermes and run **`/caduceus on`**. The installer auto-detects your
Hermes install (CLI, TUI, or the desktop app), backs up every file it touches
with a restore manifest, and can be fully undone with `--uninstall`. Pure-Python
stdlib — nothing to `pip`.

- Backend (CLI/TUI) works after a restart; `--with-desktop` also rebuilds the
  desktop UI (status-bar toggle + Orchestration Theater).
- `--dry-run` previews every change, `--list-targets` shows detected installs,
  `--uninstall` reverts everything.

> Built for **Hermes v0.15.1**. The installer warns (and stops, unless `--force`)
> on a different version, since it overlays a few shared core files — but every
> write is backed up, so it's always reversible.

---

## What's in this release

| Layer | What it does | How to use |
|---|---|---|
| **Deep-planning loop** | Plans with a live `todo` list and drives it one item at a time (exactly one in-progress, marked done as it goes), verifies before claiming done, and right-sizes — trivial asks are just done, no ceremony. | `/caduceus on` |
| **The Loom** | A deterministic async workflow engine: `agent()` / `parallel()` / `pipeline()`, validated structured output, a shared token budget, and per-run caching + resume (edit the script, re-run, unchanged calls return instantly). | say **"workflow"** |
| **Auto Router** | Scores every configured candidate on capability (never price) and routes each *worker* to the cheapest one that clears the bar. The orchestrator always keeps your session model. | `/caduceus auto on` |

Workflow progress streams live to the desktop **Orchestration Theater**, redesigned
in this release to match the app's native design system.

---

## Why

Hermes already had the raw pieces — a `todo` tool, `delegate_task`, auxiliary
models, a great desktop UI. Caduceus is the *mode* that composes them into the
disciplined plan-and-execute loop that makes top-tier coding agents feel
reliable, plus a way to fan work out deterministically and spend the right model
on the right subtask. It's built as an additive fork: every hot-path edit is
small, guarded, and a no-op when the mode is off.

---

## Safety & testing

- **Off by default, everywhere.** `agent_init` forces the mode off; every prompt
  injection checks `is_active()`; tiering/routing return the base path when off.
- **77 new feature tests + 280 existing regression tests — all green.** Plus a
  line-by-line behavioral-parity matrix (15/15 rules) against a reference
  deep-planning loop, a runnable offline parity eval, and an Auto-Router
  self-test.
- **Fully reversible.** `--uninstall` restores every backed-up original and
  removes the files Caduceus added.

---

## Changelog (this build)

- **Native Caduceus mode** — `/caduceus on|off|status` deep-planning loop with
  auto-tuned effort/tiering/budgets and power-user `caduceus.*` config overrides.
- **The Loom** workflow engine + the `Workflow` tool (opt-in; only fires on
  explicit "workflow"/orchestrate intent or clear large fan-out).
- **Auto Router** — per-task worker model selection (`/caduceus auto on`).
- **Orchestration Theater** redesigned to the app's design tokens (single accent,
  no gradients/emoji) for a clean, native look.
- **1-click installer** (`install_caduceus.py`) — auto-detect, backup + restore
  manifest, `--dry-run` / `--list-targets` / `--uninstall` / `--force`, and
  `--with-desktop` for a full desktop deploy.
- **Node-free desktop repack** — `--with-desktop` now swaps the freshly-built
  renderer into the packaged `app.asar` with a pure-Python asar reader/writer
  that recomputes per-file SHA256 integrity, so it works even where `node` is a
  Windows binary reached from WSL (which breaks the `npx` shim) or where the
  network is unavailable. It builds + validates the archive before atomically
  replacing the original (keeping `app.asar.precaduceus.bak` + a timestamped
  backup), and falls back to `@electron/asar` only if needed.

---

## Docs

[`docs/caduceus/`](docs/caduceus/) — [install](docs/caduceus/INSTALL.md) ·
[user guide](docs/caduceus/USER_GUIDE.md) · [design](docs/caduceus/DESIGN.md) ·
[parity](docs/caduceus/PARITY.md) · [contribution summary](docs/caduceus/PR_DESCRIPTION.md)

## Verify locally

```bash
pytest tests/caduceus/ tests/workflow/test_loom_offline.py -q   # feature + engine tests
python3 docs/caduceus/eval/parity_eval.py                       # to-do-loop discipline rubric
python3 docs/caduceus/eval/auto_router_selftest.py              # router selection core
```
