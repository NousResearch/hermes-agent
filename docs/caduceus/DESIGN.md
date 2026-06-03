# Caduceus — Hermes-Native Dynamic-Workflow Mode

*(A native deep-planning + dynamic multi-agent workflow mode for Hermes —
inspired by the best agentic coding loops, built entirely on Hermes primitives.)*

> **Status: implemented.** This is the design record; the feature is built,
> tested, and off-by-default. For the contribution summary and integration map
> see [`PR_DESCRIPTION.md`](PR_DESCRIPTION.md); for usage see
> [`USER_GUIDE.md`](USER_GUIDE.md). The behavioral-parity evidence and eval
> harness live alongside this doc.

Goal: a **native deep-planning mode** for Hermes — a disciplined, visible
to-do-driven loop; an opt-in **dynamic multi-agent workflow engine** (the
"Loom"); role-aware orchestrator/worker model tiering; an optional per-task
**Auto Router**; and a live **"Orchestration Theater"** desktop view. Built as a
small, additive integration on top of Hermes's existing `todo` tool,
`delegate_task`, auxiliary models, and TUI/desktop event bus. Targets
`NousResearch/hermes-agent` (base commit `b34ee8074`).

---

## Table of contents

1. [What "UltraCode" actually is (RE summary)](#1-what-ultracode-actually-is-re-summary)
2. [What Hermes already gives us](#2-what-hermes-already-gives-us)
3. [Architecture overview](#3-architecture-overview)
4. [Component A — Activation & the session "mode"](#4-component-a--activation--the-session-mode)
5. [Component B — Orchestrator/Worker model tiering](#5-component-b--orchestratorworker-model-tiering)
6. [Component C — The Loom: dynamic-workflow engine](#6-component-c--the-loom-dynamic-workflow-engine)
7. [Component D — The prompt stack](#7-component-d--the-prompt-stack)
8. [Component E — Orchestration Theater (the live UI)](#8-component-e--orchestration-theater-the-live-ui)
9. [Event protocol (tui_gateway ↔ desktop)](#9-event-protocol-tui_gateway--desktop)
10. [Reliability engineering](#10-reliability-engineering)
11. [Latency engineering](#11-latency-engineering)
12. [File-by-file change list (core fork)](#12-file-by-file-change-list-core-fork)
13. [Testing & verification plan](#13-testing--verification-plan)
14. [Phased rollout](#14-phased-rollout)
15. [Risks & open questions](#15-risks--open-questions)
16. [Appendix — exact ported prompt text](#16-appendix--exact-ported-prompt-text)

---

## 1. What "UltraCode" actually is (RE summary)

From the reverse-engineering in `claude-ultracode-re` and the working
reproduction in `UltraCode-Shim`, UltraCode is **three layers**, not a model:

1. **The envelope** — at the API boundary it is simply `output_config.effort =
   "xhigh"` + `thinking.type = "adaptive"` + a large `max_tokens` (64k) + one
   model-visible reminder. There is *no secret model or field*.
   (`claude-ultracode-re/evidence/ULTRACODE_REQUEST_CONTEXT.md`.)

2. **The dynamic-workflow engine** — the `Workflow` tool. The model authors a
   **deterministic script** that orchestrates many subagents via
   `agent() / parallel() / pipeline() / phase() / log() / workflow() / budget`,
   with structured-output schemas, token budgets, worktree isolation, and
   resume/caching. Quality patterns: adversarial-verify, perspective-diverse
   verify, judge-panel, loop-until-dry, multi-modal sweep, completeness critic.
   (`claude-ultracode-re/evidence/WORKFLOW_TOOL_PROMPT.md`.)

3. **The standing opt-in prompt stack** — a model-visible reminder ("Ultracode
   is on: optimize for the most exhaustive, correct answer… Use the Workflow
   tool on every substantive task") plus a Workflow-tool description that treats
   that reminder as a *standing* opt-in. A lifecycle of enter/sparse/exit
   reminders keeps the behavior alive across long sessions.
   (`claude-ultracode-re/docs/PROMPT_STACK_RE.md`.)

`UltraCode-Shim` adds a fourth idea that we adopt natively:

4. **Orchestrator + Worker** — the main interactive loop runs on one ("heavy")
   model; every spawned subagent/workflow leaf runs on a ("fast") worker model.
   In the shim this was a proxy hack keyed on interactive-only tools
   (`proxy.py:_request_tier`, `_select_target`). In Hermes we make it
   **first-class config**, because Hermes already supports per-child model
   overrides.

The shim also hardened three real failure modes on long runs that we port as
first-class engine features (see §10): **empty-turn auto-retry**, **stalled-stream
idle timeout**, and **tool-call sequence repair**.

---

## 2. What Hermes already gives us

Hermes is unusually well-suited to host UltraCode — it already ships ~80% of the
primitives. Verified during RE:

| UltraCode piece | Native Hermes equivalent | File evidence |
|---|---|---|
| Worker-model split | `delegate_task(model=…, override_provider=…)`; `delegation.provider`; `role='orchestrator'` for nested delegation | `tools/delegate_tool.py:738, 879-898, 913, 967` |
| Subagents / `agent()` | `delegate_task` (single + **parallel batch**); child `AIAgent` w/ isolated context, own `task_id`, restricted toolset | `tools/delegate_tool.py:1-17`; `toolsets.py:243-247` |
| Worktree isolation | kanban `workspace_kind: "worktree"` | `hermes_cli/kanban_db.py` (Task dataclass) |
| Subagent UI surfacing | `subagent.spawn_requested/start/thinking/tool/progress/complete` events + `$subagents` store | `apps/desktop/src/store/subagents.ts`; `use-message-stream.ts:87-94` |
| `execute_code` (Python that calls tools) | precedent + sandbox for a script-driven engine | `toolsets.py:237-241` |
| Prompt/guidance injection | three-tier system prompt (stable/context/volatile); per-feature `*_GUIDANCE` constants | `agent/system_prompt.py`, `agent/prompt_builder.py` |
| Slash-command framework | central `COMMAND_REGISTRY`; auto-propagates to CLI/gateway/TUI/autocomplete | `hermes_cli/commands.py` |
| Per-task side-model config | `auxiliary` config section (per-task provider/model/effort) | `agent/auxiliary_client.py::_resolve_auto` |
| Tool registration | `registry.register()` + wire into `toolsets.py` | `tools/registry.py`; `AGENTS.md` "Adding New Tools" |
| Desktop ↔ backend bridge | `tui_gateway` JSON-RPC over WS: `prompt.submit`, `model.options`, `slash.exec`, streaming `message.delta`/`tool.*`/`subagent.*` | `tui_gateway/server.py:3848, 6598`; desktop spawns `hermes dashboard --tui` (`apps/desktop/electron/main.cjs:3004-3158`) |

**Gaps to build:** (1) a session "UltraCode mode" toggle + the prompt stack;
(2) first-class orchestrator/worker wiring + a two-slot picker; (3) the
deterministic workflow runtime ("the Loom"); (4) the live Orchestration Theater
UI and its event protocol.

> **Naming (final).** The user-facing mode is **Caduceus** — Hermes's two-snake
> winged staff (the two snakes = orchestrator + worker; the wings = parallel
> speed). Command: `/caduceus`. Config key: `caduceus:`. The workflow engine is
> **the Loom** (it weaves many agents into one result); the UI is the
> **Orchestration Theater**.
>
> **Convention used in this doc:** "**Caduceus**" = our native Hermes mode;
> "**UltraCode**" = Claude Code's original feature (the RE source we port from).
> Where you see `ultracode_*` identifiers below they are renamed to `caduceus_*`
> in code; model-visible reminder strings are rebranded to "Caduceus is on:"
> while preserving the exact behavioral wording (the behavior depends on the
> instruction, not the literal word).

---

## 3. Architecture overview

```
┌──────────────────────────── Hermes Desktop (Electron + React) ───────────────────────────┐
│  Statusbar ▸ [⚡ UltraCode]  +  Orchestrator/Worker model pickers                           │
│  Orchestration Theater  ◀── $workflow store ◀── use-message-stream (workflow.* WS events)  │
└───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                             │  tui_gateway JSON-RPC / WS
┌───────────────────────────────────────────▼──────────────────────────────────────────────┐
│  tui_gateway/server.py                                                                     │
│    methods: ultracode.set / ultracode.status / model.tiers.set                             │
│    events : workflow.start/phase/agent.*/parallel.barrier/pipeline.stage/budget/log/done   │
├────────────────────────────────────────────────────────────────────────────────────────  │
│  AIAgent (orchestrator model)  ── run_conversation() ──┐                                   │
│    system prompt += UltraCode reminder + Workflow desc  │  tool_call: Workflow(script=…)    │
│                                                         ▼                                   │
│  tools/workflow_tool.py  ──▶  agent/workflow/ (THE LOOM)                                    │
│    ├ sandbox.py     restricted Python exec of the script body                              │
│    ├ dsl.py         agent() parallel() pipeline() phase() log() workflow() budget          │
│    ├ scheduler.py   asyncio dataflow scheduler + concurrency semaphore                     │
│    ├ runner.py      each agent() → delegate_task child on the WORKER model                 │
│    ├ structured.py  schema → forced StructuredOutput tool-call + validate + retry          │
│    ├ journal.py     resume/cache (hash of prompt,opts,phase,idx) under session dir         │
│    ├ reliability.py empty-turn retry · idle timeout · tool-call repair · keepalive         │
│    └ events.py      emit workflow.* to tui_gateway (and CLI/TUI renderers)                 │
└───────────────────────────────────────────────────────────────────────────────────────── ┘
```

Two execution surfaces share the same engine: the **desktop** (Theater UI) and
the **CLI/TUI** (a compact text renderer of the same `workflow.*` events). Both
are driven by the engine's event emitter, so there's a single source of truth.

---

## 4. Component A — Activation & the session "mode"

UltraCode is **session-scoped**, exactly like Claude Code (`this session only`).
We mirror that.

### 4.1 Config schema (`config.yaml`)

Add a top-level `caduceus:` section to `DEFAULT_CONFIG` in
`hermes_cli/config.py` (deep-merge handles new keys without a version bump):

```yaml
caduceus:
  enabled: false                 # default-off; session toggle flips a runtime flag, not this
  effort: xhigh                  # mapped per-provider (see §5.3)
  orchestrator:                  # the "heavy" main-loop model
    provider: ""                 # "" = use the session's current model
    model: ""
  worker:                        # the "fast" model every subagent/leaf uses
    provider: ""                 # "" = same as orchestrator
    model: ""
  workflow:
    max_concurrency: auto        # auto = min(16, cpu-2); int overrides
    max_agents: 1000             # runaway backstop (UltraCode parity)
    default_budget_tokens: null  # null = unbounded; or an int hard-ceiling
    isolation_default: none      # none | worktree
    persist_scripts: true        # write each run's script under the session dir
  reminders:
    enter: full                  # full | sparse
    turns_between_maintenance: 8 # sparse reminder cadence (UltraCode parity)
```

### 4.2 Slash command (`/ultracode`)

Add one `CommandDef` to `COMMAND_REGISTRY` (`hermes_cli/commands.py`) — it then
auto-propagates to CLI dispatch, gateway, TUI, Telegram/Slack menus, and
autocomplete:

```python
CommandDef("caduceus",
           "Toggle Caduceus dynamic-workflow mode (xhigh effort + standing Workflow opt-in)",
           "Session", aliases=("cad", "uc"),
           args_hint="[on|off|status|orch <model>|worker <model>]")
```

Handlers: `cli.py::process_command` (interactive), `gateway/run.py` (messaging
parity). `on`/`off` flip a **session runtime flag** (not persisted by default);
`status` prints current orchestrator/worker tiers + budget; `orch`/`worker` set
the tiers live (and re-emit `model.options`-style state).

### 4.3 Session state plumbing

`AIAgent.__init__` / `agent/agent_init.py` gain:

```python
ultracode_mode: bool = False
orchestrator_model: Optional[dict] = None   # {provider, model}
worker_model: Optional[dict] = None          # {provider, model}
ultracode_budget: Optional[int] = None
```

The TUI/desktop session carries these in its `SessionState.config_overrides`
(already supported per RE) so a toggle takes effect on the next turn without a
process restart. Mode on/off emits the **enter/exit reminder** (see §7.4).

### 4.4 Desktop activation

- A **statusbar toggle** (`apps/desktop/src/app/.../statusbar-items.tsx`):
  an `⚡ UltraCode` chip that flips `$ultracodeEnabled` and calls
  `gw.request('ultracode.set', { enabled, session_id })`.
- A **settings tab** (`apps/desktop/src/app/settings/ultracode-settings.tsx`):
  default orchestrator/worker tiers, concurrency, budget, isolation default.

---

## 5. Component B — Orchestrator/Worker model tiering

### 5.1 The principle

- The **orchestrator** model runs the main `run_conversation()` loop — planning,
  reading results, deciding the next phase, writing workflow scripts.
- The **worker** model runs **every** delegated leaf (`agent()` in a workflow,
  and any plain `delegate_task`). This is where the bulk fan-out happens.

This is the UltraCode-Shim split, but native: instead of classifying requests at
a proxy by interactive-only tools, we set it explicitly because we *own* the
spawn path.

### 5.2 Wiring (low-touch, reuses existing machinery)

`delegate_task` already accepts `model` and `override_provider`
(`tools/delegate_tool.py:738, 879-898`). The Loom's `runner.py` passes the
**worker tier** into every `delegate_task` call by default. A workflow `agent()`
call may override per-call via `opts.model` (UltraCode parity), but the default
is the worker tier — "use MiniMax everywhere" really means MiniMax for all
leaves, not the stock model.

Plain (non-workflow) `delegate_task` calls in UltraCode mode also default to the
worker tier, so the behavior is consistent.

Two useful presets:
- **Solo**: orchestrator == worker (one model runs everything). Equivalent to
  the shim's "Same as orchestrator."
- **Split**: e.g. orchestrator = a strong reasoner (Opus/GPT‑5.5/MiniMax‑M3),
  worker = a cheaper/faster model (Haiku/DeepSeek/MiMo).

### 5.2a Role-aware tiering (the "smartest integration" — DECIDED)

Rather than a blunt "all delegates use the worker tier" switch, tiering follows
the **recursion structure of the work**, with explicit override and escalation.
This mirrors how UltraCode actually behaves and is strictly smarter:

| Caller | Default tier | Rationale |
|---|---|---|
| Main `run_conversation()` loop | **orchestrator** | plans, reads results, writes scripts, synthesizes |
| Workflow `agent()` **leaf** (`role='leaf'`) | **worker** | the bulk fan-out — cheap/fast model |
| Nested `role='orchestrator'` delegate | **orchestrator** | it *itself* plans + delegates further, so it needs the heavy model |
| Plain `delegate_task` (non-workflow) | **worker** | so "use MiniMax everywhere" really holds end-to-end |

On top of the role default, two intelligent overrides:

- **Explicit per-call override.** Any `agent(..., model=…)` /
  `delegate_task(model=…)` wins — the orchestrator can *escalate* one hard leaf
  (e.g. a tricky synthesis or a high-stakes verify) to the heavy model, or
  *de-escalate* trivial leaves to an even cheaper model. (UltraCode's `opts.model`
  parity — "default to omitting it; set only when highly confident a different
  tier fits.")
- **Optional auto-escalation (off by default).** A cheap heuristic/router can
  promote a leaf flagged high-difficulty (e.g. `schema` depth, prompt length, or
  an explicit `tier:'auto'`) to the orchestrator model. Kept opt-in
  (`ultracode.workflow.auto_escalate: false`) because it adds a routing decision
  and we prioritize latency/determinism by default.

Net effect: the strong model is spent only where structure says it matters
(planning, nested orchestration, explicitly-escalated leaves); everything else
runs on the fast worker. This is the maximally-intelligent reading of the user's
"smartest integration" ask and supersedes the earlier all-or-nothing question.

### 5.3 Effort/`xhigh` mapping

Hermes already threads `reasoning_effort` into requests
(`run_agent.py:4181-4266`, `build_api_kwargs`). UltraCode's `xhigh` is
provider-specific:
- **Anthropic adaptive-thinking models** → `output_config.effort=xhigh` +
  `thinking:{type:"adaptive"}` (verbatim envelope from
  `ULTRACODE_REQUEST_CONTEXT.md`).
- **OpenAI/Codex** → `reasoning_effort: "high"` (+ `service_tier` if set).
- **MiniMax/DeepSeek/others** → highest supported effort; otherwise no-op.

We add an `effort: xhigh` resolution table in `agent/model_metadata.py` /
`build_api_kwargs` that maps the abstract `xhigh` to each provider's real knob,
falling back gracefully. UltraCode mode sets the orchestrator (and optionally
worker) to `xhigh` while active; exiting restores the prior effort.

### 5.4 Desktop two-slot picker

Extend the existing model picker (`apps/desktop/src/components/model-picker.tsx`)
with an **Orchestrator/Worker** two-column variant
(`OrchestratorWorkerPicker.tsx`) shown when UltraCode mode is on — directly
inspired by the shim's pre-launch selector, but native and live-switchable.
Selections call a new `model.tiers.set` JSON-RPC method.

---

## 6. Component C — The Loom: dynamic-workflow engine

This is the heart and the "smartest possible system" ask. Design goals, in
priority order: **correctness/determinism → reliability → latency → developer
ergonomics (for the orchestrator model) → observability**.

### 6.1 Surface: the `Workflow` tool

`tools/workflow_tool.py` registers a `Workflow` tool whose schema mirrors
UltraCode's (`WORKFLOW_TOOL_PROMPT.md`): `script`, `name`, `args`, `scriptPath`,
`resumeFromRunId`. The handler launches a Loom run and — like UltraCode — returns
**immediately** with a `runId` and streams progress; a `<workflow-notification>`
tool-result arrives on completion (Hermes supports long/async tool patterns; we
implement the notification as a follow-up tool result + event).

### 6.2 The DSL (Python, not JS)

We keep UltraCode's exact mental model and API names, but the script body is
**sandboxed Python** (Hermes is Python; `execute_code` already establishes the
pattern). The orchestrator writes:

```python
meta = {
    "name": "review-changes",
    "description": "Review changed files across dimensions, verify each finding",
    "phases": [{"title": "Review"}, {"title": "Verify"}],
}

DIMENSIONS = [{"key": "bugs", "prompt": "…"}, {"key": "perf", "prompt": "…"}]

async def main():
    results = await pipeline(
        DIMENSIONS,
        lambda d: agent(d["prompt"], label=f"review:{d['key']}", phase="Review", schema=FINDINGS),
        lambda review, d, i: parallel([
            (lambda f=f: agent(f"Adversarially verify: {f['title']}", phase="Verify", schema=VERDICT)
                          .then(lambda v: {**f, "verdict": v}))
            for f in review["findings"]
        ]),
    )
    confirmed = [f for r in flatten(results) if r for f in [r] if r["verdict"]["isReal"]]
    return {"confirmed": confirmed}
```

DSL hooks (1:1 with UltraCode semantics — see `WORKFLOW_TOOL_PROMPT.md` lines
96–105):

| Hook | Semantics |
|---|---|
| `agent(prompt, *, label, phase, schema, model, isolation, agent_type)` | spawn one subagent (a `delegate_task` child on the worker model). With `schema`, force a structured-output tool-call and return the validated object; else return final text. Returns `None` if skipped. |
| `pipeline(items, *stages)` | **default**. Each item flows through all stages independently — *no barrier between stages* (item A in stage 3 while B in stage 1). Wall-clock = slowest single chain. |
| `parallel(thunks)` | **barrier** — gather all; a thrower resolves to `None` (never rejects). Use only when stage N needs all of N‑1. |
| `phase(title)` | start a progress group; subsequent `agent()`s group under it. |
| `log(msg)` | narrator line to the UI. |
| `workflow(nameOrRef, args)` | run a saved/scripted child workflow inline (one level deep), sharing the concurrency cap + budget. |
| `budget` | `{total, spent(), remaining()}` shared token pool across the main loop + all leaves; a **hard** ceiling. |
| `args` | the verbatim `args` input. |

**Determinism guards (resume-critical):** `time`, `random`, wall-clock, and
network are unavailable in the *script body* (UltraCode forbids
`Date.now`/`Math.random` for the same reason). Side effects happen only inside
delegated agents, which run under Hermes's normal tool guardrails/approvals.

> **Why a script, not a static DAG?** UltraCode's power is *deterministic
> control flow with model-driven leaves* — `while budget.remaining() > 50k`,
> loop-until-dry, dedup-vs-seen. A static DAG can't express those. The script is
> the right abstraction; we make it safe via the sandbox (§6.6) and reliable via
> the journal (§6.5).

### 6.3 Scheduler (asyncio dataflow)

`scheduler.py` runs the script body as a coroutine. `agent()` returns an
awaitable backed by a task submitted to a bounded worker pool.

- **Concurrency cap** = `min(16, cpu-2)` (UltraCode parity), tunable via
  `ultracode.workflow.max_concurrency`. Excess `agent()` calls queue; total
  agents across a run capped at `max_agents` (1000 backstop).
- **`pipeline()` = per-item coroutine chains** (no barrier): item *i*'s stage
  *k+1* is scheduled the moment its stage *k* resolves. This is the latency win —
  fast items finish without waiting for slow siblings.
- **`parallel()` = `asyncio.gather(return_exceptions=True)`** mapped to `None`.
- The orchestrator stays in the loop: `Workflow` returns a `runId` immediately;
  the engine streams `workflow.*` events; on completion the tool-result is
  delivered so the orchestrator reads the result and decides the next phase.

**Bridging to sync `delegate_task`:** Hermes' delegate runs children in a
`ThreadPoolExecutor`. The Loom keeps an asyncio event loop on its own thread and
drives delegate calls via `loop.run_in_executor` (or wraps the existing batch
path). This gives asyncio's clean cancellation/streaming for scheduling while
reusing the *battle-tested* child-agent lifecycle (approval callbacks, toolset
restriction, terminal isolation) unchanged.

### 6.4 Structured output (`schema`)

`structured.py`: when `agent(..., schema=S)` is set, the child is given a forced
`StructuredOutput(S)` tool and a system instruction to return via it; the result
is validated with `jsonschema`. On mismatch, retry up to N times with the
validator error fed back (UltraCode "validation happens at the tool-call layer so
the model retries on mismatch"). Returns the parsed object — no brittle parsing.

### 6.5 Resume & caching (journal)

`journal.py`: every `agent()` result is written to
`<session_dir>/workflows/<runId>/journal.jsonl`, keyed by a stable hash of
`(normalized_prompt, opts, phase, call_index)`. On
`Workflow(scriptPath, resumeFromRunId=…)`:
- replay the longest **unchanged prefix** of `agent()` calls from cache instantly;
- the first edited/new call and everything after runs live.
Same script + same args ⇒ 100% cache hit. Per-agent transcripts persist as
`agent-<id>.jsonl` (UltraCode parity) so the UI timeline scrubber (§8) can replay.

### 6.6 Sandbox

`sandbox.py`: the script body executes with a restricted global namespace —
only the DSL hooks + a safelist of pure builtins (`len`, `range`, `sorted`,
`min`, `max`, `sum`, `enumerate`, `zip`, `map`, `filter`, `dict/list/set/str/int/float`,
`json`, `math`, comprehensions). No `import`, no file/network, no `eval/exec`, no
`__builtins__` escape. AST-validate before exec (reject `Import`, `Attribute`
access to dunders, etc.). Per-run wall-clock + per-agent timeouts. This is
strictly *more* constrained than `execute_code` because the script only needs to
orchestrate.

### 6.7 Quality patterns as a stdlib

Ship the UltraCode quality patterns (`WORKFLOW_TOOL_PROMPT.md` lines 172–204) as
both (a) prompt guidance and (b) optional named helpers the script can call:
`adversarial_verify(claim, n=3)`, `judge_panel(attempts)`,
`loop_until_dry(finders, k=2)`, `multimodal_sweep(searches)`,
`completeness_critic(state)`. These compile to the same `agent/parallel/pipeline`
primitives but reduce boilerplate and standardize the "exciting" UI moments
(verify duels, dryness counters).

---

## 7. Component D — The prompt stack

We port the UltraCode prompt stack **verbatim where it's model-visible**, adapted
only for Python/Hermes tool names. All injection uses Hermes's existing
mechanisms (`agent/prompt_builder.py` guidance constants + `agent/system_prompt.py`
assembly). Exact strings in the Appendix (§16).

### 7.1 The enter reminder (verbatim)

Injected when UltraCode mode is active (after the skills inventory, mirroring the
captured placement in `ULTRACODE_REQUEST_CONTEXT.md`):

> Ultracode is on: optimize for the most exhaustive, correct answer — not the
> fastest or cheapest. Use the Workflow tool on every substantive task; token
> cost is not a constraint. See the Workflow tool's **Ultracode** section and
> quality patterns. Solo only on conversational/trivial turns.

### 7.2 The Workflow tool description (ported)

The full `Workflow.description` from `WORKFLOW_TOOL_PROMPT.md` becomes the Hermes
`Workflow` tool's description, edited for: Python (not JS) script body; "subagent"
= Hermes delegate child; MCP/tool reachability via Hermes toolsets; the
**Ultracode** standing-opt-in clause kept intact.

### 7.3 Opt-in policy

Same policy as UltraCode: only orchestrate when the user opted in — and
**UltraCode mode is itself the standing opt-in**. We also keep the keyword path:
a `workflow`/`workflows` keyword in the user's message injects the keyword
reminder (`PROMPT_STACK_RE.md` §"Workflow keyword reminder").

### 7.4 Reminder lifecycle

Reproduce UltraCode's attachment lifecycle as Hermes meta-reminders:
- **enter (full)** on activation;
- **sparse maintenance** every `turns_between_maintenance` turns
  ("Ultracode is still on — use the Workflow tool; see its Ultracode section.");
- **exit** on deactivation
  ("Ultracode is off — the Workflow tool's standard opt-in rule applies again.").

Implemented as a small per-session reminder scheduler invoked from the
conversation loop's pre-turn hook.

---

## 8. Component E — Orchestration Theater (the live UI)

The differentiator. When a workflow runs, the desktop opens a **Theater** panel
(right rail or full-screen overlay) that makes the fan-out *feel alive*. It
consumes the `workflow.*` event stream into a `$workflow` nanostore and renders
with React + a lightweight canvas/SVG layer.

### 8.1 Layout

```
┌─ Orchestration Theater ───────────────────────────────  ⏱ 00:42  ⚡ xhigh ─┐
│  Narrator:  "scouting changed files… 7 dimensions fanning out"             │
│                                                                            │
│   PHASE: Review            PHASE: Verify           PHASE: Synthesize        │
│   ┌────────┐  ┌────────┐    ┌────────┐  ┌────────┐    ┌──────────────┐      │
│   │bugs  ● │─▶│perf  ● │    │verify ◐│  │verify ◷│    │  synthesize  │      │
│   │MiniMax │  │MiniMax │    │Haiku   │  │Haiku   │    │  Opus (orch) │      │
│   │1.2k tok│  │ 980 tok│    │streaming…           │    │  pending      │      │
│   └────────┘  └────────┘    └────────┘  └────────┘    └──────────────┘      │
│       │            └─────────┐                                              │
│       ▼ (fan-out 5)          ▼                                              │
│   ◔ ◔ ◔ ◔ ◔  verify duel: 4/5 confirm · 1 refute → REAL                     │
│                                                                            │
│  Concurrency ▕████████░░░░░░▏ 8/14 active · 23 queued                       │
│  Budget      ▕██████████░░░░▏ 312k / 500k  (62%)  ~3m left at current burn  │
└────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Components (`apps/desktop/src/components/workflow/`)

- **`WorkflowTheater.tsx`** — container; subscribes to `$workflow`.
- **`PhaseLane.tsx`** — one column per `phase()`; agents flow left→right through
  phases (pipeline) or cluster (parallel).
- **`AgentCard.tsx`** — per-agent node: label, **model badge** (orchestrator =
  gold, worker = accent color), live status ring (queued → running → streaming →
  done ✓ / failed ✗), a 1–2 line live tail of its current thinking/tool action,
  token count, elapsed. Animates in on spawn (burst), pulses while streaming,
  settles with a check, shakes red on failure.
- **`FanOut.tsx`** — animated edges from a parent node to N children when
  `parallel()`/pipeline stage spawns a batch (the "burst").
- **`VerifyDuel.tsx`** — for adversarial/perspective-diverse verify: shows N
  skeptics voting, a running tally, and the verdict (REAL/REJECTED) with a snap
  animation. This is the single most fun thing to watch.
- **`BudgetGauge.tsx`** — burn-down bar with live `spent/total`, % and a naive
  ETA from current burn rate; turns amber→red near the ceiling.
- **`ConcurrencyMeter.tsx`** — `active/cap` + queued depth.
- **`DrynessCounter.tsx`** — for loop-until-dry: round number, fresh finds this
  round, dryness streak (k/K).
- **`Narrator.tsx`** — `log()` ticker above the graph.
- **`TimelineScrubber.tsx`** — after completion, scrub the run; click any node to
  open its full transcript (`agent-<id>.jsonl`). Replays animations.

### 8.3 Feel / motion

- Framer-motion (or CSS transitions) for spawn/settle; respect
  `prefers-reduced-motion`.
- Color/skin-aware: tie node accents to the active Hermes skin (e.g. Ares
  crimson) so it matches the app.
- Throttle: coalesce `agent.delta` events to ~10–15 fps per node; virtualize the
  graph when >150 nodes (render off-screen nodes as dots).
- A compact **"mini-theater"** strip is always visible inline in the chat while a
  workflow runs; clicking expands to the full Theater.

### 8.4 CLI/TUI parity

The same events render in the Ink TUI as a compact live tree (phases as headers,
agents as spinner rows with token counts, a budget bar) — so `hermes --tui` and
headless/cron runs still show progress. One emitter, two renderers.

---

## 9. Event protocol (tui_gateway ↔ desktop)

New JSON-RPC methods on `tui_gateway/server.py`:

| Method | Params | Returns |
|---|---|---|
| `ultracode.set` | `{session_id, enabled, budget?}` | `{enabled, orchestrator, worker, budget}` |
| `ultracode.status` | `{session_id}` | full mode state |
| `model.tiers.set` | `{session_id, orchestrator?, worker?}` | resolved tiers |

New streaming events (namespaced `workflow.*`, siblings of the existing
`subagent.*`), each carrying `{run_id, session_id, ts}` plus:

| Event | Payload |
|---|---|
| `workflow.start` | `{name, description, phases[], budget_total, concurrency_cap}` |
| `workflow.phase` | `{phase, index}` |
| `workflow.agent.spawn` | `{agent_id, label, phase, model, parent_id?}` |
| `workflow.agent.status` | `{agent_id, status}` (queued/running/streaming/done/failed/skipped) |
| `workflow.agent.delta` | `{agent_id, kind: text|reasoning|tool, text}` (throttled) |
| `workflow.agent.tokens` | `{agent_id, in, out}` |
| `workflow.agent.done` | `{agent_id, summary?, tokens, ms}` |
| `workflow.parallel.barrier` | `{phase, count}` |
| `workflow.pipeline.stage` | `{item_index, stage_index}` |
| `workflow.verify` | `{finding_id, votes:[{lens, verdict}], result}` |
| `workflow.budget` | `{spent, total}` |
| `workflow.log` | `{message}` |
| `workflow.complete` | `{result_summary, agents, tokens, ms, runId}` |
| `workflow.error` | `{message, agent_id?}` |

`events.py` in the engine emits these via the existing callback bridge used for
`subagent.*` (`acp_adapter/events.py` pattern / tui_gateway emit path), so no new
transport is needed. The desktop adds handlers in
`apps/desktop/src/app/session/hooks/use-message-stream.ts` that reduce into
`$workflow`.

---

## 10. Reliability engineering

Long, fan-out runs fail in specific ways; the shim already proved the fixes
(`UltraCode-Shim/README.md` "Built for long, dynamic workflows"). We port them as
first-class engine behavior in `reliability.py`:

1. **Empty-turn auto-retry.** A worker turn with no text and no tool call (a
   transient blip or budget-exhausted reasoning turn) is transparently re-issued.
   Buffer only until the first real token ⇒ zero added latency on normal turns;
   never retry after real output or a fatal error.
2. **Stalled-stream idle timeout.** If a worker stream opens then goes silent
   mid-turn, a bounded idle timeout converts the stall into a quick retry instead
   of a multi-minute hang — so one stuck leaf can't freeze the whole run.
3. **Tool-call sequence repair.** A declined/partial tool call no longer wedges
   strict backends — synthesize stub results for unanswered calls (parity with
   shim `#3`). Hermes already has a `message_sequence_repair` test surface to
   build on.
4. **Reasoning keepalive.** Keep the connection live during a model's pre-answer
   thinking so a step looks busy, without leaking chain-of-thought.
5. **Per-agent + per-run timeouts**, with `failed→None` semantics so
   `.filter(Boolean)`/`[x for x in … if x]` patterns degrade gracefully.
6. **Budget hard ceiling** prevents runaway token spend; `max_agents` backstops
   runaway loops.
7. **Crash-safe journal** ⇒ a killed run resumes from the last completed agent.
8. **Rate-limit awareness** via Hermes's existing `agent/nous_rate_guard.py` /
   credential pool, so high concurrency backs off instead of 429-storming.

All knobs are config-driven and covered by an offline self-test (parity with the
shim's CI self-test).

---

## 11. Latency engineering

- **Pipeline-by-default** (no barriers) — the biggest win; fast items don't wait
  on slow siblings. Barriers only where genuinely needed (§6.2).
- **Async scheduler** drives many lightweight leaves without thread-per-agent
  overhead; the bounded executor reuses warm child contexts where safe.
- **Prompt caching**: the orchestrator's stable+context prompt tiers are cached
  (`agent/prompt_caching.py`); worker leaves reuse a cached shared preamble.
- **Streaming-first**: results stream into the Theater immediately; the
  orchestrator's `Workflow` call returns a `runId` without blocking the UI.
- **Worker tier = fast model** by design — the cheap/fast model does the bulk
  fan-out while the strong orchestrator only plans/synthesizes.
- **Event coalescing** (server-side) keeps the WS from flooding under high
  fan-out; deltas batched per ~50–100 ms per agent.
- **Lazy structured-output retries** only on validation failure, not by default.

---

## 12. File-by-file change list (core fork)

### Python — `hermes-agent/`

**New**
- `agent/workflow/__init__.py`
- `agent/workflow/engine.py` — top-level `run_workflow(script, args, tiers, budget, emit)`.
- `agent/workflow/dsl.py` — `agent/parallel/pipeline/phase/log/workflow/budget`, quality-pattern helpers.
- `agent/workflow/scheduler.py` — asyncio dataflow scheduler + concurrency semaphore.
- `agent/workflow/runner.py` — leaf execution via `delegate_task` on the worker tier.
- `agent/workflow/structured.py` — schema-forced structured output + validation/retry.
- `agent/workflow/journal.py` — resume/caching + per-agent transcripts.
- `agent/workflow/sandbox.py` — restricted Python exec + AST validation.
- `agent/workflow/reliability.py` — empty-turn retry, idle timeout, tool repair, keepalive.
- `agent/workflow/events.py` — `workflow.*` emitter.
- `tools/workflow_tool.py` — the `Workflow` tool (register + schema + handler).
- `tests/workflow/…` — unit + integration + offline self-test.

**Modified**
- `hermes_cli/config.py` — `ultracode:` block in `DEFAULT_CONFIG`; loaders.
- `hermes_cli/commands.py` — `/ultracode` `CommandDef`.
- `cli.py` — `/ultracode` handler + session flag + `save_config_value`.
- `gateway/run.py` — `/ultracode` parity for messaging.
- `agent/agent_init.py`, `run_agent.py` — accept `ultracode_mode`,
  `orchestrator_model`, `worker_model`, `ultracode_budget`; default delegated
  children to the worker tier; pre-turn reminder scheduler.
- `agent/prompt_builder.py` — `ULTRACODE_ENTER/SPARSE/EXIT_REMINDER`,
  `WORKFLOW_TOOL_DESCRIPTION`, orchestrator/worker guidance constants.
- `agent/system_prompt.py` — inject the reminder + Workflow description when mode on.
- `agent/model_metadata.py` / `agent/chat_completion_helpers.py` — `xhigh`
  effort resolution table per provider.
- `toolsets.py` — register `workflow` tool; new `ultracode` toolset; worker
  toolset restrictions; orchestrator guidance wiring.
- `tools/delegate_tool.py` — minor: accept the workflow run/emit context so leaf
  activity emits `workflow.agent.*` (keeps `subagent.*` for plain delegation).
- `tui_gateway/server.py` — `ultracode.set/status`, `model.tiers.set` methods;
  forward `workflow.*` events.

### TypeScript — `apps/desktop/`

**New**
- `src/store/ultracode.ts` — `$ultracodeEnabled`, `$orchestratorTier`, `$workerTier`.
- `src/store/workflow.ts` — `$workflow` run graph + reducers for `workflow.*`.
- `src/components/workflow/WorkflowTheater.tsx` (+ `PhaseLane`, `AgentCard`,
  `FanOut`, `VerifyDuel`, `BudgetGauge`, `ConcurrencyMeter`, `DrynessCounter`,
  `Narrator`, `TimelineScrubber`).
- `src/components/OrchestratorWorkerPicker.tsx` — two-slot model picker.
- `src/app/settings/ultracode-settings.tsx` — settings tab.

**Modified**
- `src/app/session/hooks/use-message-stream.ts` — handle `workflow.*` → store.
- `src/app/.../statusbar-items.tsx` — `⚡ UltraCode` toggle + mini-theater strip.
- `src/app/settings/index.tsx` + `constants.ts` — register the new tab.
- `src/hermes.ts` / `src/types/hermes.ts` — `ultracode.*`, `model.tiers.set`,
  `workflow.*` types.

Electron `main.cjs`/`preload.cjs`: **no change** (everything rides the existing
WS bridge).

---

## 13. Testing & verification plan

- **Engine unit tests**: `pipeline` no-barrier ordering; `parallel` barrier +
  `None`-on-throw; concurrency cap; budget hard-ceiling stop; `max_agents`
  backstop; sandbox rejects `import`/`eval`/dunder access; determinism guards.
- **Resume tests**: unchanged prefix → 100% cache hit; edited call → live from
  that point; crash mid-run → resume from journal.
- **Reliability tests** (offline, no network): empty-turn retry adds zero latency
  on normal turns; idle-timeout converts a stall to a retry; tool-repair stub for
  declined/partial calls; failed leaf → `None`.
- **Structured output**: schema mismatch retries then validates.
- **Prompt tests**: enter/sparse/exit reminder lifecycle; Workflow description
  present only when mode on; keyword path.
- **Tiering**: delegated children run on the worker tier; `opts.model` override
  respected; `xhigh` maps correctly per provider (mock adapters).
- **tui_gateway**: `ultracode.set/status`, `model.tiers.set` round-trip;
  `workflow.*` events emitted in order.
- **Desktop**: store reducers from a recorded event stream; Theater renders
  spawn→stream→done→verify; reduced-motion path; >150-node virtualization.
- **End-to-end smoke**: `Solo` and `Split` tiers; a real `review-changes`
  workflow on a sample repo; watch the Theater; confirm tokens, budget, verdicts.
- **Offline self-test** (`scripts/`/CI) mirroring the shim's, provable without
  keys.

Run via `scripts/run_tests.sh` (probes `.venv`/`venv`). Desktop: `npm run
type-check`, `npm test`, `npm run build` in `apps/desktop`.

---

## 14. Phased rollout

- **Phase 0 — Dev env.** Decide dev location (see §15.1); get the desktop app
  running from our checkout via `HERMES_DESKTOP_HERMES_ROOT`; baseline tests
  green.
- **Phase 1 — Mode + tiering.** `/ultracode`, config, prompt stack (reminders +
  Workflow description), orchestrator/worker wiring through `delegate_task`,
  `xhigh` mapping. No engine yet — workflows fall back to plain delegation. Ship
  the statusbar toggle + two-slot picker.
- **Phase 2 — The Loom (core).** `Workflow` tool, DSL, scheduler, runner,
  structured output, sandbox, budget. CLI/TUI text renderer of `workflow.*`.
- **Phase 3 — Reliability + resume.** `reliability.py`, journal/resume, rate-limit
  backoff, offline self-test.
- **Phase 4 — Orchestration Theater.** Full desktop UI + animations + timeline
  scrubber + verify duels + budget burn-down.
- **Phase 5 — Polish.** Quality-pattern stdlib, saved/named workflows
  (`.hermes/workflows/`), perf hardening, docs.

Each phase is independently demoable.

---

## 15. Risks & open questions

### 15.1 Dev location — DECIDED: edit the live install in place
We edit the installed tree at
`/mnt/c/Users/User/AppData/Local/hermes/hermes-agent/` directly. Verified facts:
- The tree is **writable** from WSL.
- The packaged desktop app runs the **Python backend from this source tree**
  (`ACTIVE_HERMES_ROOT` + its venv; `apps/desktop/electron/main.cjs:1045-1050,
  1526, 1579-1581`). ⇒ **Python edits go live on the next backend start** — no
  rebuild needed.
- The **desktop UI** is served from the packaged bundle (`dist/` → asar), *not*
  from `src/` at runtime. ⇒ UI changes require `npm run build` in
  `apps/desktop` (and, if the running exe loads from `win-unpacked/resources`,
  repackaging or running `npm run dev`). Plan UI iteration via dev mode, then a
  build to land it in the live app.
- The updater is a **manual** "Check for Updates" menu item that spawns
  `hermes-setup.exe` (`main.cjs:1190-1232, 2525-2601`); it is *not* aggressive
  on launch (`.update_check` shows `behind: 0`). **Caution:** do not click
  "Check for Updates" or re-run `hermes-setup.exe` while our edits are
  unsaved/uncommitted — it can overwrite the tree. Mitigation: `git init` (or use
  the existing `.git`) a working branch and commit early/often so any overwrite
  is recoverable, and consider stashing a copy of our diff under
  `~/repos/hermes-ultracode/patches/`.

### 15.2 `xhigh` semantics vary by provider
Not every provider exposes an `xhigh`-equivalent. The mapping table degrades
gracefully (no-op where unsupported), but the *behavioral* lift then comes purely
from the prompt stack + workflow engine. Confirm per provider (MiniMax‑M3, Codex,
DeepSeek, local) during Phase 1.

### 15.3 Sandbox safety
Executing model-written Python is the main security surface. Mitigations:
AST allowlist, no imports/IO/network in the script body, restricted builtins,
per-run/per-agent timeouts, and the fact that real side effects only happen
inside delegated agents (which keep Hermes's approval/guardrail flow). Worth a
focused security review before enabling by default.

### 15.4 Async ↔ sync bridge
`delegate_task` is thread-pool based; the Loom is asyncio. The bridge
(`run_in_executor`) is straightforward but must preserve delegate's approval
callbacks (`tools/delegate_tool.py:56-111`), interrupt propagation, and terminal
isolation. Validate with the existing concurrency stress tests
(`tests/stress/test_concurrency*.py`).

### 15.5 Upstream drift
A core fork diverges from upstream. Mitigation: keep changes modular (new
`agent/workflow/` package + additive hooks), minimize edits to hot files, and
maintain a rebase/patch script. Consider proposing the feature upstream.

### 15.6 Naming / scope confirmations
- **Tiering — DECIDED:** role-aware tiering with explicit override + opt-in
  auto-escalation (§5.2a). Supersedes the all-or-nothing question.
- **Naming — DECIDED: Hermes-native.** Pending the specific pick (lead
  proposal: **Caduceus** — Hermes's two-snake winged staff: the two snakes =
  orchestrator + worker, the wings = parallel speed). The chosen name threads
  through the config key (`caduceus:`/`ultracode:`), the slash command, and UI
  copy; a single rename pass lands it once confirmed.
- Default-off mode (recommended) vs always-available toggle — still open
  (recommend default-off, session-scoped activation).

---

## 16. Appendix — exact ported prompt text

These are reproduced from the RE evidence; model-visible strings are kept
verbatim except for JS→Python and tool-name adaptations.

> These are the UltraCode strings (`PROMPT_STACK_RE.md`) **rebranded to
> Caduceus** — the wording that drives behavior is preserved verbatim; only the
> mode's proper noun changes. The original UltraCode strings are kept in the RE
> repo for reference.

### 16.1 Enter reminder
```
Caduceus is on: optimize for the most exhaustive, correct answer — not the fastest or cheapest. Use the Workflow tool on every substantive task; token cost is not a constraint. See the Workflow tool's **Caduceus** section and quality patterns. Solo only on conversational/trivial turns.
```

### 16.2 Sparse maintenance reminder
```
Caduceus is still on — use the Workflow tool; see its Caduceus section.
```

### 16.3 Exit reminder
```
Caduceus is off — the Workflow tool's standard opt-in rule applies again.
```

### 16.4 Workflow-keyword reminder (verbatim)
```
The user included the keyword "workflow" or "workflows", which means you should use the Workflow tool to fulfill their request.
```

### 16.5 Workflow tool description
Ported from `claude-ultracode-re/evidence/WORKFLOW_TOOL_PROMPT.md` (the full
`Workflow.description`, lines 51–207) with these edits:
- "Scripts are plain JavaScript" → "Scripts are restricted Python (async
  `main()`); the DSL hooks are injected — no imports."
- `agent()/parallel()/pipeline()/phase()/log()/workflow()/budget` semantics kept
  identical (§6.2).
- "subagent" = a Hermes `delegate_task` child running on the worker tier.
- MCP/tool reachability described via Hermes toolsets instead of ToolSearch.
- The **Ultracode** standing-opt-in clause (lines 76 / 228–230) kept intact.

> The canonical source for the description, schema, quality patterns, and resume
> semantics is `claude-ultracode-re/evidence/WORKFLOW_TOOL_PROMPT.md` — Phase 2
> will copy it into `agent/prompt_builder.py::WORKFLOW_TOOL_DESCRIPTION` with the
> edits above.

---

*End of design v1. Next step: confirm §15 decisions (esp. dev location), then
begin Phase 0/1.*
