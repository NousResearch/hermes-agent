---
name: nika
description: "Runs repeatable AI work with checks, budgets and traces."
version: 1.2.3
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

Use [Nika](https://nika.sh) through the Hermes `terminal` tool to capture repeatable
AI work in checked `*.nika.yaml` files. Nika is an AGPL-3.0-or-later workflow engine;
this MIT-licensed skill teaches the Hermes handoff to public Nika 0.118.6. Hermes
owns the user's intent and authorization; Nika checks the file, admits execution
under its boundaries, and produces outputs and evidence when execution records them.

## When to Use

- Author, check, run, or diagnose an existing `*.nika.yaml` workflow.
- Capture repeated model, file, HTTP, or command work in a reusable artifact.
- Inspect costs, declared authority, outputs, or evidence before repeating a job.

For a one-off answer or tool call, use the relevant Hermes tool directly. For
interactive coding work, use the existing coding skill; a workflow becomes useful
when a repeatable part has a clear input, output, and failure policy.

## Prerequisites

Use `terminal(command="nika --version")` to identify the installed release.
Installation and client wiring are separate effects: carry out already authorized
setup within scope, or report the missing prerequisite. Installation routes are at
https://nika.sh; do not assume an installed command matches this skill's release.

Use `terminal(command="nika doctor")` for setup diagnostics and `nika catalog` for
available model names. Keep credentials in the configured stores or environment;
never put secret values in workflow files, tool arguments, logs, or a report.
A local model still needs its runtime and compute resources. `mock/echo` simulates
inference; it does not make arbitrary workflows offline or harmless.

## How to Run

Use `search_files` to find an existing workflow, `read_file` to inspect it, and
`write_file` or `patch` to author or repair it. Keep the artifact in the user's
project and pass that project as the `terminal` working directory.

Discover a template and create a file without overwriting an existing owner:

```
terminal(command="nika new '?'")
terminal(command="nika new chain flow.nika.yaml", workdir="~/project")
```

Read the generated file. For language details, use the installed `nika --help`,
`nika catalog --tools`, and the matching release's
[authoring guide](https://github.com/supernovae-st/nika/blob/v0.118.6/.agents/plugins/nika/skills/nika-authoring/SKILL.md).
If the read-only `nika mcp` oracle is already wired, use `nika_schema`,
`nika_examples`, `nika_template`, `nika_check`, and `nika_explain` as needed.
The oracle proposes or diagnoses; workflow execution remains a separate action.

### Preflight and authority

Check the exact file after every edit, with the model override intended for the
run if there is one:

```
terminal(command="nika check flow.nika.yaml --json --native-strict", workdir="~/project")
terminal(command="nika explain flow.nika.yaml", workdir="~/project")
```

Read the exit code AND the JSON. `clean`, `native_strict_clean`, and `paid_ready`
answer different questions. Require native-strict cleanliness before handing over
a run; before paid inference, require `paid_ready: true` and inspect the access,
capacity, model, and risk findings. Neither a green check nor a ready access lane
is permission from the user, a live connectivity test, or proof that effects ran.
Use `nika explain NIKA-XXXX` for a diagnostic. Fix its cause and check again.

Preserve existing authorization and budgets. Continue work already authorized;
resolve only a genuinely missing business decision, authority, or spend limit.
Do not widen `permits:` or secret `egress:` merely to silence the checker.
`nika check flow.nika.yaml --infer-permits` proposes a boundary to review against
the intended effects; it does not authorize that boundary. Missing permits grant
zero authority. Keep secrets store-backed and admit only intended sinks.

### A small native example

Save this as `hash-abc.nika.yaml`. It needs no model or external service and calls
a native builtin; check it before considering any run:

```yaml
nika: hash-abc
permits:
  tools: [nika:hash]
tasks:
  probe:
    invoke:
      tool: nika:hash
      args: { content: abc, algo: sha256 }
outputs:
  result: ${{ tasks.probe.output }}
```

Use exactly one of four verbs per task: `infer` for model output, `exec` for an
external command, `invoke` for a callable tool or workflow, and `agent` for a
bounded adaptive loop. Prefer an existing native tool before writing shell glue.
A model can propose facts; deterministic tools enforce business rules. Bound
`infer` with `max_tokens` and `agent` with `max_turns` and `max_tokens_total`.
Caller values belong in `inputs:`, fixed values in `const:`, credentials in
`secrets:`. Bind another task's data through `with:`; `after:` orders tasks when
no data flows. Consult the schema rather than inventing fields or arguments.

`nika:compose` checks a draft supplied through an agent's tool access; it does
not execute it. `invoke: { workflow: ... }` calls a child through normal engine
admission. The child must fit the parent's authority; neither drafting nor
checking grants new execution rights.

### Execution and recovery

Once the exact workflow, inputs, model, effects, and budget are ready and already
authorized, execute through `terminal`. This is a command shape: replace every
placeholder with the checked value; retain the user's configured cap.

```
terminal(command="nika run flow.nika.yaml --model <provider/model> --var <input>=<value> --max-cost-usd <authorized-cap>", workdir="~/project")
```

For long work, use the Hermes terminal's supported background/session mechanism
and retain its returned identifier. Inspect that session before starting another
copy. A timeout or lost terminal response does not prove the process stopped.
Before retrying after interruption, reconcile the process, trace, outputs, and
external effects. Preserve negative results and never retry publication blindly.

For a paused `nika:prompt`, use
`nika run flow.nika.yaml --resume <trace> --answer <task>=<value>`
to supply the authorized answer; it is execution, not inspection.
Read `nika run --help` and preserve the original inputs, model, cap, and scope.
Transient errors may use bounded `retry:`; expected errors may use a deliberate
`on_error:` recovery. `after: { producer: unwind }` is best-effort cleanup for a
producer that started, including cancellation or timeout. It does not run for an
unstarted producer, and process death can prevent cleanup.

## Quick Reference

| Command | Use |
|---|---|
| `nika new '?'` / `nika new <template> <file>` | Discover and create a workflow |
| `nika check <file> --json --native-strict` | Read the file's distinct verdicts |
| `nika explain <file>` / `nika explain <code>` | Understand the plan or finding |
| `nika catalog --tools` / `nika catalog` | Discover shipped tools or models |
| `nika run <file> --max-cost-usd <authorized-cap>` | Execute within the reviewed scope |
| `nika test <file>` | Compare a supported simulated golden; not an effects rehearsal |
| `nika trace show <trace>` / `nika trace outputs <trace>` | Read an identified run |
| `nika trace verify <trace>` | Verify existing evidence without rerunning |
| `nika trace evidence <trace>` | Write an auditor pack for an identified trace |
| `nika doctor` | Diagnose installed access and environment |

## Procedure

1. Recover the current artifact and any active run; preserve prior authorization.
2. Choose inputs, deterministic rules, outputs, failure policy, and effect scope.
   Reuse the installed schema, templates, and tool catalog for exact syntax.
3. Write or repair the file with Hermes file tools. Keep model proposals separate
   from deterministic judgment, engine admission, and the resulting effects.
4. Check the final file, including its intended model override. Read all verdicts;
   repair root causes without expanding secrets or authority as a workaround.
5. For a rehearsal, audit every reachable effect and model first. A `mock/echo`
   override changes the envelope model, not task-pinned models or real tools.
   Golden tests simulate a restricted plane and refuse network, subprocess, and
   write effects; use them only when the actual workflow fits that plane.
6. Run only with the already authorized inputs, effects, and configured limits.
   Retain the terminal session and identify the produced trace, if any.
7. Report the real exit status, outputs, metered cost and unknowns, exact trace
   path, and verification verdict. Reconcile partial effects before any retry.

## Pitfalls

- `nika new --from ...` is obsolete here. Use positional intent and destination;
  bare `nika new` needs a TTY. Do not overwrite an existing artifact casually.
- A cost estimate is not an invoice. The checker's output-token estimate omits
  input billing; unpriced models or compute must remain explicitly unpriced.
  A metered cap stops new admissions after crossing it; already admitted calls
  can finish and overshoot, especially in a parallel wave. Do not promise a hard
  invoice ceiling or interpret an unknown price as zero.
- `mock/echo` can accompany real file, network, command, or child-workflow
  effects. Merely adding a mock model or a zero USD cap does not remove them.
- An advisory `clean: true` can coexist with native-strict or paid-readiness
  blockers. Process exit alone is insufficient for the paid-run decision.
- A successful call to a checker is not evidence that a workflow ran. Conversely,
  execution can have effects without a complete journal. Recording is enabled
  by default; pre-execution refusal, disabled recording, or lost ownership can
  leave no complete trace.
- An intact hash chain proves consistency of the recorded lines, not producer
  honesty or a completed lifecycle. `INCOMPLETE` has exit code 5 in this release:
  retain that result, report the missing lifecycle evidence, and investigate the
  producer and effects. Do not infer success, death, or safe retry from it alone.

## Verification

For the native example, first use
`nika check hash-abc.nika.yaml --json --native-strict` through `terminal`.
A clean check proves the file passed that preflight; it is not a run receipt.
If an execution is authorized and performed, compare the output to SHA-256 of
`abc` (`ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`).
Inspect its run card and use the exact returned trace path with `nika trace show`
and `nika trace verify`; compare the verified chain head to the run card.

Report only the proof tier actually verified. Chain verification is distinct
from a signature, anchor, or comparison to a fresh replay journal. Verification
neither creates a missing trace nor re-executes a workflow. An incomplete,
missing, or broken journal remains an explicit limitation, even if other checks
passed; do not present a planned test or an initiated publication as completed.
