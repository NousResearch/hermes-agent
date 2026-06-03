# Caduceus — user guide

Caduceus is a **deep-planning mode** for Hermes. Turn it on and the agent works
like a senior engineer: it lays out a visible to-do plan and drives it
methodically, raising reasoning effort and delegating where it helps. Ask for a
"workflow" and it fans out across many subagents on the **Loom** engine. It's a
single switch — everything underneath is auto-tuned.

---

## Turn it on

**CLI / TUI**
```
/caduceus            # toggle on/off (one keystroke)
/caduceus on         # force on
/caduceus off        # force off
/caduceus status     # show current state
```

**Desktop**
- Click the **⚕ Caduceus** chip in the status bar to toggle it, or
- Settings → **Caduceus** → enable (also sets the default for new sessions).

Caduceus is **session-scoped** and **off by default**. Toggling it does not
change anything for other sessions, and it takes effect on your next message.

---

## What changes when it's on

1. **It plans first — and right-sizes.** For anything multi-step or hard, the
   first thing it does is write a `todo` plan, then works it one item at a time
   (exactly one *in-progress*, marked done the moment it's finished). For a
   trivial one- or two-step ask, it just does it — no ceremony.
2. **It verifies.** Before claiming something is done it checks its work (lint /
   typecheck / build / tests as appropriate); when tests exist it prefers
   writing a failing test first.
3. **It parallelizes.** Independent work is fanned out (batched tool calls,
   parallel `delegate_task`) instead of run serially.
4. **It's honest about progress.** A step is only marked complete when it's
   truly done; a blocked step stays in-progress and a follow-up to-do is added.

The desktop renders the live to-do list as you go.

---

## Workflows (the Loom)

When you say **"workflow"**, ask to **"fan out / orchestrate with subagents"**,
or hand it a large map-reduce ("audit every module", "research these 20
topics"), Caduceus authors a small **Python workflow script** and runs it on the
Loom — a deterministic async engine with:

- `agent(prompt, *, schema=…, phase=…)` — run a subagent (optionally forced to
  return validated JSON),
- `pipeline(items, *stages)` — flow each item through stages with no barrier,
- `parallel(thunks)` — run concurrently with a barrier,
- structured output, a shared token **budget**, per-run **caching + resume**
  (edit the script and re-run; unchanged calls return instantly).

Progress streams live to the desktop **Orchestration Theater**. You don't have
to write scripts yourself — the agent authors them; saying "workflow" is enough.

---

## Auto Router (optional) — the right model per task

With many models configured, you don't want to pay frontier prices on trivial
subtasks. The Auto Router sends each **worker** (delegated subtask) to the
cheapest configured model a quick classifier judges can do *that* subtask. The
**orchestrator always keeps your session model** — only workers are routed.

Enable it:
```
/caduceus auto on    # toggle the router (also: /caduceus auto off)
```
or Settings → Caduceus → Auto Router.

Then add a few candidates once in `~/.hermes/config.yaml`:
```yaml
caduceus:
  router:
    enabled: true
    classifier: ""          # empty = use the fast auxiliary model
    threshold: 0.7          # lower = save more; higher = escalate sooner
    candidates:
      - {model: "google/gemini-3-flash-preview", provider: "openrouter", cost: 0.3}
      - {model: "gpt-5.5", provider: "codex", cost: 5.0, supports_images: true}
```
You can list models by id alone — capability cards auto-fill for known families.
If the router can't run for any reason it falls back safely to the cheapest
candidate and never breaks a request.

---

## Configuration reference (`caduceus:` in config.yaml)

Everything here is **optional** — the defaults are sensible and the mode is one
switch. These are power-user overrides, not command knobs.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Persisted default for new sessions. |
| `effort` | `high` | Orchestrator reasoning effort (`low`/`medium`/`high`/`xhigh`). `high` is the fast-but-strong default. |
| `apply_effort_to_worker` | `false` | Apply the effort to workers too (else workers stay fast/cheap). |
| `orchestrator` | `{}` | `{provider, model}` for the heavy tier (empty = session model). |
| `worker` | `{}` | `{provider, model}` for the worker tier (empty = solo). |
| `router.enabled` | `false` | Turn on per-task worker routing. |
| `router.classifier` | `""` | Scorer model id (empty = fast auxiliary model). |
| `router.threshold` | `0.7` | Capability bar a candidate must clear. |
| `router.default` | `""` | Fallback candidate (empty = cheapest). |
| `router.candidates` | `[]` | `{model, provider, cost, supports_images, card}` list. |
| `workflow.max_concurrency` | `auto` | Loom parallelism (`auto` = `min(16, cpu-2)`). |
| `workflow.max_agents` | `1000` | Runaway backstop. |
| `workflow.agent_timeout_seconds` | `600` | Per-leaf wall-clock timeout. |
| `reminders.turns_between_maintenance` | `8` | Cadence of the "still on" nudge. |

(Delegation parallelism is shared with Hermes's `delegation.max_concurrent_children`.)

---

## FAQ

**Does turning it on slow simple tasks?** No — it right-sizes; trivial asks skip
the plan entirely.

**Will it run a workflow on everything?** No. The Loom is opt-in: it only fires
when you say "workflow"/ask to orchestrate, or a task clearly needs large
parallel fan-out.

**Is anything on by default?** No. Caduceus and the Auto Router are both off
until you enable them.

**Does it change my normal Hermes?** No. With Caduceus off, behavior is
identical to stock Hermes.
