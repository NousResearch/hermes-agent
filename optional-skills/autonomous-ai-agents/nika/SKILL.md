---
name: nika
description: "Runs repeatable AI work with checks, budgets and traces."
version: 1.2.2
author: Thibaut Melen (@ThibautMelen) · SuperNovae Studio (github.com/supernovae-st)
license: MIT
platforms: [linux, macos]
prerequisites:
  commands: [nika]
metadata:
  hermes:
    tags: [Workflow, Automation, Deterministic, Cost-Control, Audit, Local-First, MCP]
    category: autonomous-ai-agents
    related_skills: [opencode, claude-code, codex]
    homepage: https://nika.sh
    requires_toolsets: [terminal]
---

# Nika Skill

Use [Nika](https://nika.sh) as a deterministic workflow worker orchestrated by
the Hermes `terminal` tool. Nika is an open-source (AGPL) Rust engine that captures
a repeatable AI task as a plain-text `*.nika.yaml` file, audits it **before a
single token is spent** (plan, cost floor, secret flows, types), executes it
against local or cloud providers (Ollama/llama.cpp/vLLM included), and records
a tamper-evident trace.

Division of labor: **Hermes orchestrates · Nika captures repeatable work as a
checkable file and runs it with receipts.** Nika is NOT another coding agent —
for autonomous coding, use the `opencode` skill. Delegate to Nika when the
work should be *repeatable, budgeted, and auditable*.

## When to Use

- The user asks to run, check, or author a `*.nika.yaml` workflow
- A task will be repeated (daily digest, triage, ETL, report, multi-step LLM
  pipeline) — capture it as a workflow instead of re-prompting
- The user wants a hard cost cap, a cost estimate before running, or
  receipts/audit of what ran
- A pipeline mixes models/providers (local + cloud) or mixes LLM steps with
  shell/HTTP/file steps
- The user wants a run they can replay, verify, or reproduce later

### When NOT to use

- One-off questions or single tool calls — just answer or use a tool
- Autonomous code implementation/refactoring/PR review — use the `opencode` skill
- Interactive back-and-forth tasks — workflows are non-interactive by design

## Prerequisites

- Nika installed: `brew install supernovae-st/tap/nika` — other install
  paths (script, manual download) are documented at https://nika.sh: installing
  is a human step, not something this skill runs
- Verify: `terminal(command="nika --version")`
- Zero keys needed for local/offline work: `--model mock/echo` (offline) and
  `--model ollama/...` (local) run without any API key
- Cloud providers read standard env vars from the shell;
  `terminal(command="nika doctor")` diagnoses and prints exact fix commands

## How to Run

Prove the toolchain offline first (no key, no network):

```
terminal(command="nika try 01-hello")
```

Run a real workflow — local model first:

```
terminal(command="nika run flow.nika.yaml --model ollama/qwen3.5:4b", workdir="~/project")
```

Cloud model with a hard budget (always set one for paid models):

```
terminal(command="nika run flow.nika.yaml --model mistral/mistral-small-latest --max-cost-usd 0.25", workdir="~/project")
```

Supply the declared `inputs:` (repeatable · JSON when it parses · an
undeclared key is refused, never silently ignored):

```
terminal(command="nika run report.nika.yaml --var city=Paris --var days=7 --max-cost-usd 0.50", workdir="~/project")
```

Long runs: launch in background and poll — do not block the turn:

```
terminal(command="nika run long.nika.yaml --max-cost-usd 1.00", workdir="~/project", background=true)
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")
```

### The check-before-run law

Never run a workflow you have not checked. `nika check` is a static pre-flight
(no tokens spent, no network): plan shape, cost floor, secret-flow analysis,
type checks, tool args.

```
terminal(command="nika check flow.nika.yaml --json", workdir="~/project")
```

Findings carry `NIKA-XXXX` codes that explain themselves via
`nika explain NIKA-XXXX`. Exit 0 = green, safe to run. Fix findings before
running — never suppress them.

### Authoring a workflow

Turn a repeated task into a file. List templates, then instantiate:

```
terminal(command="nika new '?'")
terminal(command="nika new chain flow.nika.yaml", workdir="~/project")
```

The first argument also accepts plain-words intent. Edit the skeleton, then
**check it**. `nika explain flow.nika.yaml` narrates what it will do, the
waves, the cost floor, and what it touches — before anything runs.

The artifact you are producing looks like this (W1 map form — the task
key IS the identity):

```yaml
nika: daily-brief
model: ollama/qwen3.5:4b
permits:                       # absent = ZERO authority · an effect without a grant refuses
  exec: false
  net:
    http: ["hn.algolia.com"]
  tools: ["nika:fetch"]
tasks:
  fetch:
    invoke:
      tool: "nika:fetch"
      args: { url: "https://hn.algolia.com/api/v1/search?tags=front_page" }
  brief:
    with:
      hn: ${{ tasks.fetch.output }}
    infer:
      max_tokens: 300
      prompt: |
        Five bullet points, most signal first: ${{ with.hn }}
outputs:
  brief: ${{ tasks.brief.output }}
```

One file, plain YAML: tasks, a named wire (the binding IS the edge —
`brief` runs after `fetch` because it reads it), a bounded model step,
a declared output. That file is what gets checked, run, diffed and reused.

### The envelope — nine keys, no more

`additionalProperties: false` at the top level: a key outside this list is
a PARSE refusal, not a warning. Only `nika:` and `tasks:` are required; the
rest earn their place. `nika mcp`'s `nika_schema` tool serves the
machine-readable copy.

```yaml
nika: daily-brief                 # 1 · the mark AND the name · kebab-case
inputs:                           # 2 · what the CALLER supplies — and
  feed_url:                       #     what a DEPLOYMENT supplies, via
    type: string                  #     required: false + a default:
  locale:
    type: string
    required: false
    default: "en"
model: ollama/qwen3.5:4b          # 3 · default model · <provider>/<name>
const:                            # 4 · baked into the file
  max_items: 5
secrets:                          # 5 · store references · never literals
  FEED_TOKEN:
    source: env
    key: FEED_TOKEN
    egress:                       # the sanctioned sinks · absent = default-deny
      - { to: "nika:fetch" }
      - { to: "infer" }
permits:                          # 6 · absent = ZERO authority
  net:
    http: ["hn.algolia.com"]
  tools: ["nika:fetch"]
run:                              # 7 · entropy + clock, declared not ambient
  entropy: { seeded: 42 }
  clock: virtual
tasks:                            # 8 · the work
  fetch:
    timeout: "30s"
    invoke:
      tool: "nika:fetch"
      args:
        url: "${{ inputs.feed_url }}"
        headers: { authorization: "${{ secrets.FEED_TOKEN }}" }
  brief:
    with:
      raw: ${{ tasks.fetch.output }}
      n: ${{ const.max_items }}
      lang: ${{ inputs.locale }}
    returns: string
    infer:
      max_tokens: 400
      prompt: |
        In ${{ with.lang }}, give ${{ with.n }} bullets: ${{ with.raw }}
outputs:                          # 9 · the return value
  brief: ${{ tasks.brief.output }}
```

### Where a value comes from (three authorities)

One question decides the block: **who supplies this?**

| Block | Supplier | Read as | Use for |
|---|---|---|---|
| `inputs:` | the caller (`--var k=v`), or the deployment via `required: false` + a `default:` | `${{ inputs.X }}` | per-run parameters · non-sensitive settings |
| `const:` | the file itself | `${{ const.X }}` | fixed values baked in |
| `secrets:` | a store | `${{ secrets.X }}` | credentials · masked · never inline |

The two older catch-all envelope blocks are retired — `NIKA-VALUES-001`
and `NIKA-VALUES-002` refuse them at PARSE and name the replacement. This
is a classify step, not a rename: a required parameter is an `inputs:`
declaration, a fixed value is a `const:` entry. `nika check <file> --fix`
migrates what it can prove and skips what it cannot.

Secrets are tracked through the graph, not just at the reference: if a
tainted task output flows into a later step, `check` names the path and
refuses until `egress:` sanctions that sink.

### Permits — absent means zero authority

`permits:` is the declared capability boundary. **No block at all is not
"unrestricted" — it is zero.** A task with an effect and no grant refuses
at check with `NIKA-AUTH-006` (`exec` · `net` · `fs` all fire), and once
the block is present every category is default-deny unless listed.

- Pure compute states the zero explicitly: `permits: {}`.
- `nika check <file> --infer-permits` prints the tightest block the body
  actually needs — paste it in rather than guessing wide.
- A grant nothing reaches draws a `NIKA-DRIFT-001` hint: the boundary is
  meant to shrink to the body.

### Beyond the verb — the task modifiers

Exactly one verb per task (`infer` · `exec` · `invoke` · `agent`), plus any
of these on the same task:

| Modifier | What it does |
|---|---|
| `with:` | bind another task's output — **the binding IS the edge** |
| `after:` | pure ordering · `{ producer: success }` (or `failure`) |
| `when:` | a local condition, evaluated after the gate |
| `for_each:` | map the task over a list · `max_parallel:` caps it · `fail_fast:` aborts |
| `retry:` | re-attempt policy |
| `on_error:` / `after: { producer: unwind }` | the failure path · cleanup that always runs |
| `extract:` | named jq bindings, read as `${{ tasks.X.<name> }}` |
| `returns:` | the task's typed output contract |
| `timeout:` | quoted Go-duration · `"30s"` · `"5m"` |
| `lift:` | the one authored door · each entry names its law (`- law: taint` with `from:` · `- law: data-as-code`) and its `because:` |

`nika catalog --tools` lists the builtins an `invoke` reaches without any
MCP server — 28 of them across six families (core · file · data · network ·
introspection · media). Reach for one before writing `exec:`.

### A workflow can call a workflow

`invoke:` carries **exactly one** of `tool:` or `workflow:` — both, or
neither, is a PARSE refusal. The child is an ordinary workflow file:

```yaml
nika: page-title
inputs:
  url:
    type: string
permits:
  net:
    http: ["example.com"]
  tools: ["nika:fetch"]
tasks:
  get:
    invoke:
      tool: "nika:fetch"
      args: { url: "${{ inputs.url }}", mode: text }
outputs:
  text: ${{ tasks.get.output }}
```

The parent calls it by a **static** path — a `${{ }}`-templated target is
refused (`NIKA-COMP-001`), because a call graph you cannot draw before the
run is one you cannot bound:

```yaml
nika: site-report
inputs:
  target:
    type: string
permits:
  net:
    http: ["example.com"]
  tools: ["nika:fetch"]
tasks:
  page:
    invoke:
      workflow: "./page-title.nika.yaml"
      args:
        url: "${{ inputs.target }}"
outputs:
  text: ${{ tasks.page.output }}
```

The one law that surprises people: **a child never gains authority the
parent lacks.** Drop the parent's `permits:` above and check refuses with
`NIKA-COMP-002` — the child's `nika:fetch` is outside the parent boundary.
So a parent that only delegates still declares what its children reach,
and the drift hint on those entries is advisory where the containment
refusal is not. Cycles are refused too (`NIKA-COMP-003`).

### Cost honesty

- When the workflow prices above the budget, `--max-cost-usd` refuses to
  start (exit 2, zero tokens) — and since 0.99 the pre-start floor prices
  the EFFECTIVE model, `--model` override included
- Mid-run, the ledger stops the workflow the moment real spend crosses the
  budget: the crossing call completes, nothing new starts, the run fails
  `NIKA-1704` (exit 1) with spent-vs-budget
- Estimates use LIST RATES from the vendored public catalog. The checker
  estimates output-token cost, not the prompt/input bill or entire invoice.
- An unpriced call has no measured USD amount. The USD budget cannot bound
  that unknown spend; never report it as free or $0. Prefer cataloged ids
  (`nika catalog`) when budget protection matters.
- Report the cost line from the final run card (the summary block `nika
  run` prints last — status, cost, trace path) back to the user verbatim

### Receipts and verification

Execution journals are enabled by default under `.nika/traces/`; recording
can be disabled, a refusal before execution can have no journal, and a lost
run can leave incomplete evidence. Use the recorded trace path; bare, both commands fall
back to the workspace's latest trace and say which one they read:

```
terminal(command="nika trace show .nika/traces/<run>.ndjson", workdir="~/project")
terminal(command="nika trace verify .nika/traces/<run>.ndjson", workdir="~/project")
```

`trace verify` checks the tamper-evidence hash chain, then climbs a proof
ladder and reports **the highest tier honestly attained**:

| Tier | What it proves | How to reach it |
|---|---|---|
| chain intact | the recorded links are internally consistent | a recorded hash-chained journal |
| `SEALED` | a custody key signed the run | `--key <pub>` |
| `ANCHORED` | the `<trace>.anchor.json` sidecar verifies offline | `--anchored` |
| `REPLAYED` | a fresh journal of the same workflow matches | `--replay <trace>` |

Report the exit code and the verifier's actual tier together. **`INCOMPLETE`** means the
journal never reached a terminal frame — the run was killed or crashed, so
the chain still links the complete lines while the lifecycle end is
unattested. Report it as what it is; do not call it a pass or a break.

Compare the chain head with independently trusted evidence, or verify its
trusted seal. An unkeyed chain can be rewritten consistently; even a valid
signature does not prove that its producer told the truth. Verification
reads existing evidence and never creates or seals a journal.

For a run someone else must audit, `nika trace evidence <trace>` writes
`<trace-stem>.evidence/` — `journal.ndjson`, `pack.json` (the manifest and
receipt, which say plainly whether a seal is present) and `VERIFY.md` (the
auditor's three commands). Add `--workflow <file>` to unlock the boundary
and the receipt. Also useful: `nika trace outputs` · `nika trace flow` ·
`nika trace reproduce` · `nika trace export` (OTLP lines).

### Optional: MCP oracle tools

Nika also ships a read-only MCP oracle (`nika mcp`) exposing validation and
learning tools (`nika_check`, `nika_inspect`, `nika_explain`, `nika_schema`,
`nika_examples`, `nika_template`, `nika_canon`, `nika_catalog`, `nika_tools`).
If the user wants those wired into their agent client, point them at the
wiring guide —
https://github.com/supernovae-st/nika-plugins/tree/main/integrations/mcp —
editing the client's own configuration is the user's step, never this
skill's. Without the oracle, everything above still works over the terminal;
running workflows stays there regardless, where the budget flags and traces
live.

## Quick Reference

| Command | Use |
|---------|-----|
| `nika welcome` | What Nika is + what this machine has (offline, exit 0) |
| `nika new <template> <file>` | Scaffold a workflow (`nika new '?'` lists) |
| `nika check <file> --json` | Static pre-flight — ALWAYS before run |
| `nika explain <file>` | Narrate: waves, cost floor, touches |
| `nika run <file> --model <p/m> --max-cost-usd <usd>` | Execute with budget |
| `nika test <file>` | Golden test under the mock provider (offline) |
| `nika trace show/verify/outputs/flow <trace>` | Receipts after a run (path from the run card's `trace:` line) |
| `nika trace evidence <trace>` | Export the auditor pack (journal · manifest · receipt · VERIFY.md) |
| `nika doctor` | Diagnose env/keys — prints exact fixes |
| `nika catalog` | Provider/model ids + required env vars |
| `nika catalog --tools` | The builtins an `invoke` reaches without MCP |

## Procedure

1. Verify readiness: `terminal(command="nika --version")`; install per
   Prerequisites if missing.
2. If the task is new, scaffold: `nika new <template> <file>`.
3. Check: `nika check <file> --json`. Fix every finding
   (`nika explain <code>`). Do not run an unchecked file.
4. Preview offline when useful: `nika run <file> --model mock/echo`.
5. Run with an explicit `--model` and, for any paid model, an explicit
   `--max-cost-usd`.
6. For long runs use `background=true` and poll with
   `process(action="poll"|"log")`.
7. After the run: `nika trace show <trace>` + `nika trace verify <trace>`
   (path from the run card); report outputs, actual cost, and the verify
   verdict to the user — including `INCOMPLETE`, which means the run died
   before a terminal frame. When someone must audit it later,
   `nika trace evidence <trace>` exports the pack.

### Rules

1. NEVER run an unchecked workflow — `nika check` first, every time.
2. ALWAYS pass `--max-cost-usd` when the model is a paid cloud model.
3. Prefer local models (`ollama/...`) or `mock/echo` for drafts; escalate to
   cloud models only when needed. Probe every new builtin on `mock/echo`
   *before* wiring it after a paid infer. The model extracts facts
   (`type: integer` for numeric enums); `nika:jq` or `nika:decide` is
   the law — do not pay a second infer to pick a level
   (`nika try 13-extract-then-law`).
4. Report the final run card honestly: status, actual cost, trace path,
   `trace verify` verdict.
5. One workflow file per delegated task; keep files in the user's repo so
   they are diffable and reusable.
6. If a run fails, read `nika explain <NIKA-code>` before retrying — do not
   blind-retry.

## Pitfalls

- `nika run` renders live on a TTY; when piped (Hermes terminal), output can
  stay quiet until completion — for anything long, prefer `background=true` +
  poll, then read `nika trace show <trace>` for the final card.
- `nika new` with no intent opens a guided TTY flow; in a pipe it fails
  fast naming the missing argument — always pass a template or intent when delegating.
- The budget guard stops NEW admissions: one wide parallel wave can overshoot
  by that wave's spend. Tighten with `max_parallel:` when the budget is strict.
- Unpriced model calls have no measured USD bound. Do not rely on
  `--max-cost-usd` alone for a custom endpoint or promise a hard invoice cap.
- Workflow `outputs:` are not resolved on a budget stop — per-task values
  live in the trace (`nika trace outputs`).
- Deleting a `permits:` block to unblock a refusal does the opposite: absent
  is zero authority, not unrestricted. Widen the grant, or run
  `nika check <file> --infer-permits` and paste what it prints.
- A parent that only delegates still declares what its children reach, and
  then draws a drift hint on grants its own body never uses. The hint is
  advisory; the containment refusal underneath it is not — do not "fix" the
  hint by emptying the block.
- `INCOMPLETE` is not proof of a completed lifecycle. Read the actual verdict
  and available evidence before telling the user what has been proved.

## Verification

Smoke test (offline, zero keys):

```
terminal(command="nika try 01-hello")
```

Success criteria: run completes exit 0 with a final run card · `nika check`
exits 0 before any real run · `nika trace verify` exits 0 after the run.
