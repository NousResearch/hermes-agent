---
name: nika
description: "Delegate repeatable work to Nika — a deterministic workflow runner with pre-flight checks, cost caps, and verifiable receipts."
version: 1.0.2
author: SuperNovae Studio (github.com/supernovae-st)
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

# Nika

Use [Nika](https://nika.sh) as a deterministic workflow worker orchestrated by
Hermes terminal tools. Nika is an open-source (AGPL) Rust engine that captures
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

## When NOT to Use

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

## The Check-Before-Run Law

Never run a workflow you have not checked. `nika check` is a static pre-flight
(no tokens spent, no network): plan shape, cost floor, secret-flow analysis,
type checks, tool args.

```
terminal(command="nika check flow.nika.yaml --json", workdir="~/project")
```

Findings carry `NIKA-XXXX` codes that explain themselves via
`nika explain NIKA-XXXX`. Exit 0 = green, safe to run. Fix findings before
running — never suppress them.

## One-Shot Delegation

Prove the toolchain offline first (no key, no network):

```
terminal(command="nika examples run 01-hello --model mock/echo")
```

Run a real workflow — local model first:

```
terminal(command="nika run flow.nika.yaml --model ollama/qwen3.5:4b", workdir="~/project")
```

Cloud model with a hard budget (always set one for paid models):

```
terminal(command="nika run flow.nika.yaml --model mistral/mistral-small-latest --max-cost-usd 0.25", workdir="~/project")
```

Pass workflow variables:

```
terminal(command="nika run report.nika.yaml --var city=Paris --var days=7 --max-cost-usd 0.50", workdir="~/project")
```

Long runs: launch in background and poll — do not block the turn:

```
terminal(command="nika run long.nika.yaml --max-cost-usd 1.00", workdir="~/project", background=true)
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")
```

## Authoring a Workflow

Turn a repeated task into a file. List templates, then instantiate:

```
terminal(command="nika new --from '?'")
terminal(command="nika new flow.nika.yaml --from chain", workdir="~/project")
```

`--from` also accepts plain-words intent. Edit the skeleton (`vars:`,
`tasks:`, `outputs:`), then **check it**. `nika explain flow.nika.yaml`
narrates what it will do, the waves, the cost floor, and what it touches —
before anything runs.

## Cost Honesty

- When the workflow file's own `model:` prices above the budget,
  `--max-cost-usd` refuses to start (exit 2, zero tokens) — the strongest
  guarantee, so prefer setting the model in the file for budgeted runs
- Today (0.98.x) a `--model` OVERRIDE is not priced by that pre-start gate:
  the run starts and the budget binds mid-run — the crossing call completes,
  nothing new starts, the run fails `NIKA-1704` (exit 1) with spent-vs-budget
- Estimates use LIST RATES from the vendored public catalog; local · mock ·
  unpriced work is never blocked
- A model absent from the catalog meters as $0 — a paid *uncataloged* model
  runs with no budget protection; prefer cataloged ids (`nika catalog`)
- Report the cost line from the final run card back to the user verbatim

## Receipts & Verification

Every run writes a trace under `.nika/traces/`. After a run:

```
terminal(command="nika trace show", workdir="~/project")
terminal(command="nika trace verify", workdir="~/project")
```

`trace verify` checks the tamper-evidence hash chain: exit 0 intact · 2
broken · 3 pre-chain. Also useful: `nika trace outputs` · `nika trace flow` ·
`nika trace reproduce` · `nika trace export` (OTLP lines).

## Optional: MCP Oracle Tools

Nika also ships a read-only MCP oracle (`nika mcp`) exposing validation and
learning tools (`nika_check`, `nika_explain`, `nika_schema`, `nika_examples`,
`nika_template`, `nika_canon`, `nika_catalog`, `nika_tools`). If the user
wants those wired into their agent client, point them at the wiring guide —
https://github.com/supernovae-st/nika-agents/tree/main/integrations/mcp —
editing the client's own configuration is the user's step, never this
skill's. Without the oracle, everything above still works over the terminal;
running workflows stays there regardless, where the budget flags and traces
live.

## Quick Reference

| Command | Use |
|---------|-----|
| `nika welcome` | What Nika is + what this machine has (offline, exit 0) |
| `nika new <file> --from <template>` | Scaffold a workflow (`--from '?'` lists) |
| `nika check <file> --json` | Static pre-flight — ALWAYS before run |
| `nika explain <file>` | Narrate: waves, cost floor, touches |
| `nika run <file> --model <p/m> --max-cost-usd <usd>` | Execute with budget |
| `nika test <file>` | Golden test under the mock provider (offline) |
| `nika trace show / verify / outputs / flow` | Receipts after a run |
| `nika doctor` | Diagnose env/keys — prints exact fixes |
| `nika catalog` | Provider/model ids + required env vars |

## Procedure

1. Verify readiness: `terminal(command="nika --version")`; install per
   Prerequisites if missing.
2. If the task is new, scaffold: `nika new <file> --from <template>`.
3. Check: `nika check <file> --json`. Fix every finding
   (`nika explain <code>`). Do not run an unchecked file.
4. Preview offline when useful: `nika run <file> --model mock/echo`.
5. Run with an explicit `--model` and, for any paid model, an explicit
   `--max-cost-usd`.
6. For long runs use `background=true` and poll with
   `process(action="poll"|"log")`.
7. After the run: `nika trace show` + `nika trace verify`; report outputs,
   actual cost, and the verify verdict to the user.

## Pitfalls

- `nika run` renders live on a TTY; when piped (Hermes terminal), output can
  stay quiet until completion — for anything long, prefer `background=true` +
  poll, then read `nika trace show` for the final card.
- `nika new` with no `--from` opens a guided TTY flow; in a pipe it fails
  fast naming the flag — always pass `--from <template>` when delegating.
- The budget guard stops NEW admissions: one wide parallel wave can overshoot
  by that wave's spend. Tighten with `max_parallel:` when the budget is strict.
- Uncataloged model ids meter as $0 — never rely on `--max-cost-usd` for a
  custom endpoint model.
- Workflow `outputs:` are not resolved on a budget stop — per-task values
  live in the trace (`nika trace outputs`).

## Verification

Smoke test (offline, zero keys):

```
terminal(command="nika examples run 01-hello --model mock/echo")
```

Success criteria: run completes exit 0 with a final run card · `nika check`
exits 0 before any real run · `nika trace verify` exits 0 after the run.

## Rules

1. NEVER run an unchecked workflow — `nika check` first, every time.
2. ALWAYS pass `--max-cost-usd` when the model is a paid cloud model.
3. Prefer local models (`ollama/...`) or `mock/echo` for drafts; escalate to
   cloud models only when needed.
4. Report the final run card honestly: status, actual cost, trace path,
   `trace verify` verdict.
5. One workflow file per delegated task; keep files in the user's repo so
   they are diffable and reusable.
6. If a run fails, read `nika explain <NIKA-code>` before retrying — do not
   blind-retry.
