# Caduceus — native deep-planning mode + dynamic multi-agent workflows for Hermes

> One switch turns Hermes into a senior-engineer planner: it lays out a live
> to-do plan and drives it methodically, raising reasoning effort and delegating
> where it helps — and escalates to a deterministic multi-agent **workflow
> engine** (the "Loom") only when you ask for it. An optional **Auto Router**
> sends each delegated worker to the cheapest model that can actually do that
> subtask. Off by default; additive; fully tested.

This PR adds **Caduceus**, a self-contained mode that makes Hermes plan and
execute like a top-tier coding agent, with an opt-in parallel workflow engine
and per-task model routing. It is **off by default**, **session-scoped**, and
built as an **additive fork** — every hot-path edit is small, guarded, and a
no-op when the mode is off.

---

## 1. Why

Hermes already has the raw pieces — a `todo` tool, `delegate_task`, auxiliary
models, a great desktop UI. What it lacks is a *mode* that composes them into
the disciplined plan-and-execute loop that makes agents like Devin/Claude Code
feel reliable, plus a way to fan work out deterministically and to spend the
right model on the right subtask. Caduceus is that mode.

Three problems it solves:

1. **Inconsistent planning.** Without a nudge, models dive straight into
   multi-step work and lose the thread. Caduceus drives a **visible, to-do-first
   loop** with strict discipline (one step in-progress, mark done as you go,
   verify before claiming done) — and *right-sizes*: trivial asks are just done.
2. **No deterministic orchestration.** `delegate_task` is great for a handful of
   children, but there's no way to express "fan this out across 30 files,
   pipeline through verify, dedupe, re-run only what changed." The **Loom** is a
   small async workflow engine with caching/resume that does exactly that.
3. **One model for everything.** Users configure many models but pay frontier
   prices on trivial subtasks. The **Auto Router** scores each candidate on
   capability (never price) and sends each worker to the cheapest one that
   clears the bar.

---

## 2. What it does (user-visible)

- **`/caduceus on|off|status`** (bare toggles). When on, the agent plans with the
  `todo` tool and drives it; the desktop shows the live plan and an
  **Orchestration Theater** when a workflow runs.
- **Say "workflow"** (or ask to fan out) → the agent authors a small Python
  workflow script and runs it on the **Loom**: `agent()/parallel()/pipeline()`,
  structured output, budgets, per-run caching + resume.
- **`/caduceus auto on|off`** → the Auto Router routes each worker per task.
- Everything else (effort, model tiers, budgets, concurrency) is **auto-tuned**
  with sane defaults and lives as power-user overrides in `caduceus.*` config —
  *not* as command knobs. The whole feature is one switch by design.

---

## 3. Architecture & integration map

The feature is overwhelmingly **new, isolated modules**; the core touch-points
are tiny, additive, and guarded.

### New modules (isolated; nothing imports them unless the mode is used)
| Module | Purpose |
|---|---|
| `agent/caduceus.py` | Single source of truth: per-session state, the prompt stack (standing + lifecycle reminders), effort mapping, role-aware tiering, the Auto-Router bridge. |
| `agent/auto_router.py` | Pure, stdlib, network-free model-selection core (classifier injected). |
| `agent/workflow/` | The Loom: `engine`, `scheduler`, `runner`, `dsl`, `sandbox`, `structured`, `journal`, `budget`, `events`, `reliability`. |
| `tools/workflow_tool.py` | The `Workflow` tool (wires the Loom to the agent). |

### Core touch-points (every one is additive + guarded)
| File | LOC | What | Off-state behavior |
|---|---|---|---|
| `agent/agent_init.py` | +16 | seed `agent.caduceus` (always created **OFF**) | inert |
| `agent/system_prompt.py` | +14 | inject standing reminder *iff* active | no-op |
| `agent/conversation_loop.py` | +26 | inject per-turn reminder *iff* active | no-op |
| `agent/agent_runtime_helpers.py` | +2 | dispatch `Workflow` like `delegate_task` | unreachable unless called |
| `agent/tool_executor.py` | +19 | `Workflow` spinner/dispatch (mirrors `delegate_task`) | unreachable |
| `run_agent.py` | +19 | `_dispatch_workflow()` (mirrors `_dispatch_delegate_task`) | unreachable |
| `model_tools.py` | +1 | add `"Workflow"` to existing `_AGENT_LOOP_TOOLS` | harmless |
| `tools/delegate_tool.py` | +204 | role-aware tiering + Auto-Router worker selection | returns base creds when off |
| `toolsets.py` | +11 | register `Workflow` tool + `caduceus` toolset | tool present, gated by opt-in policy |
| `hermes_cli/commands.py` | +4 | one `CommandDef` (drives help/autocomplete/etc.) | command exists, no-op until used |
| `hermes_cli/config.py` | +~100 | `caduceus:` config block (all defaults off/auto) | defaults are off |
| `cli.py` | +172 | `/caduceus` handler + apply-to-agent | only runs on command |
| `gateway/run.py` | +42 | gateway `/caduceus` handler | only on command |
| `tui_gateway/server.py` | +135 | `caduceus.*` RPCs + `workflow.*` event forwarding | only on RPC |
| `apps/desktop/...` | new + small edits | statusbar toggle, settings, Orchestration Theater | hidden unless active |

**Total committed diff: 42 files, +5,320 / −11.** No line-ending or formatting
churn — the diff is exactly the feature.

### Design principles a reviewer can verify quickly
- **Off by default, everywhere.** `agent_init` forces `enabled=False`; every
  injection checks `is_active()`; tiering/routing return the base path when off.
- **Reuses Hermes patterns**: the central `COMMAND_REGISTRY`, the toolset system,
  `_AGENT_LOOP_TOOLS`, `delegate_task`'s child construction, `get_text_auxiliary_client`,
  the config/`save_config_value` path, the TUI JSON-RPC event bus.
- **The orchestrator is never routed/downgraded** — the Auto Router only selects
  *worker* models; the planning model stays the user's session model.
- **Safe degradation**: the router never raises; a missing/failed classifier
  falls back deterministically; unknown candidates are dropped.

---

## 4. The Auto Router (per-task worker model selection)

A small classifier model scores every configured candidate on the probability it
nails *this* subtask on the first try (0–1). The selector picks the **cheapest
candidate that clears `threshold`** (capability and cost are kept separate, so
the classifier can't be biased toward expensive models), hard-zeroes
image-incapable models on image tasks, and caches one decision per (worker,
task). Candidates can be listed by model id alone — capability cards auto-fill
for known families (MiniMax, MiMo, GPT-5.5/Codex, Claude Opus/Sonnet/Haiku,
Gemini, Grok, DeepSeek, Qwen, Llama…).

The core is pure and stdlib-only; the classifier call is injected, so it's fully
unit-testable offline. Ported from a standalone proxy implementation and adapted
to select Hermes worker models via `delegate_task`.

---

## 5. Safety, reliability, testing

**Tests — 77 new feature tests + 280 existing regression tests, all green.**

- `tests/caduceus/test_caduceus_state.py` — state, config parsing, activate/
  deactivate, role-aware tiering, the reminder lifecycle (enter once → cadence →
  exit once), standing-reminder gating.
- `tests/caduceus/test_auto_router.py` — clamp, lenient score parsing (clean/
  fenced/prose/partial/garbage), cheapest-among-viable + escalation + image
  hard-zero, fallbacks, capability cards (incl. the `mini` substring trap that
  must not match `gemini`/`minimax`), and `select()` (cache hit, single
  candidate, exception-safety).
- `tests/caduceus/test_route_worker_model.py` — the delegate bridge: off /
  router-off / orchestrator-never-routed / no-candidates guards + happy paths.
- `tests/workflow/test_loom_offline.py` — the engine end-to-end with a mocked
  leaf (scheduler, sandbox, structured output, budget, resume, quality patterns).
- **Regression:** `tests/hermes_cli/test_commands.py`, `test_config_drift.py`,
  `tests/tools/test_delegate.py` — **280 passed**, confirming the command
  registry, config defaults, and delegation paths are unaffected.

**Behavioral parity & evidence** (see `docs/caduceus/`):
- A line-by-line parity matrix vs the reference to-do-loop contract (15/15
  behavioral rules matched) plus a runnable parity-eval harness.
- A live A/B (baseline Hermes vs Caduceus, same model) showing Caduceus reliably
  plans and right-sizes (skips ceremony on trivial tasks).
- A live router smoke: easy task → cheap model, hard task → strong model.

---

## 6. Rollout

- Ships **disabled**. No behavior change for any existing user until they run
  `/caduceus on` (or set `caduceus.enabled: true`).
- The `Workflow` tool is present but governed by an explicit opt-in policy
  (only fires on "workflow"/orchestrate intent), so it never surprises.
- The Auto Router requires an explicit `caduceus.router.enabled` + candidates.

We'd love feedback on: (a) whether the prompt stack should live behind a small
formal "prompt-contributor" hook rather than the `system_prompt`/`conversation_loop`
injection points, and (b) whether a `delegate`-model-resolution hook would be a
welcome general extension point (the Auto Router is its first consumer).

---

## 7. Try it

```bash
/caduceus on                 # plan-and-drive mode
# ... ask for a multi-step task; watch the live to-do plan
# say "workflow" on something big to fan out on the Loom
/caduceus auto on            # (optional) per-task worker routing
/caduceus status             # see current state
```
