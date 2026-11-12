---
name: hermes-cost-meter
description: Use when the user wants live session cost in the CLI.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Cost, CLI, Status-Bar, Pricing, Extension]
---

# Hermes Cost Meter

Show a live spend figure **inside the CLI status bar** for the session you are in. This is
a wrapper CLI (a `HermesCLI` subclass) — it needs no edits to the Hermes source tree, so an
`hermes update` or a rebuilt pm environment never breaks it.

## When to use

- The user asks what a session costs, or wants a running meter while they work.
- The user sets `display.show_cost: true` and reports nothing renders. That key is dead in
  shipped builds (defined in `hermes_cli/config_defaults.py`, read by nothing); upstream
  tracks it as issue #93951 with PRs #99649/#75465 open and unmerged. Do not patch core —
  run this instead, and say why.

## Run

```sh
scripts/hermes-cost              # fresh session with the meter
scripts/hermes-cost --continue   # any hermes flag passes through
```

The bar gains a `~$0.00xx` segment right after the model name, and `/cost` prints the
breakdown (model, session cost + status, input/output/cache tokens, API calls).

Verify without a TTY (allocates its own pty, one cheap model call, exit 0 on pass):

```sh
python3 scripts/selftest.py
```

## How it hooks in

`_status_bar_segments()` is the single method both the classic and the modern layout build
their status line from, so one override lands in either. Insert the segment early (right
after the model) so narrow terminals trim the tail — git branch, elapsed — before the
spend. `process_command()` adds `/cost`.

`_get_extra_tui_widgets()` is the other documented hook, but it renders a **separate row
above** the bar. If the user says "in this bar" and points at the model/context line, that
is `_status_bar_segments`, not the widget hook.

`scripts/_launch.py` re-execs the current runtime (`hermes --print-runtime-command`) with
the wrapper module instead of `hermes_cli.main`, so no venv path is hardcoded.

## Reading the number

Cost = the session's token counters times the model's rate card, via
`agent.usage_pricing.estimate_usage_cost` — the same helpers `/usage` and `state.db` use.
It is a **lower bound**: only main-agent responses carry usage, so auxiliary calls,
retries and cache-write overhead are missed. `/cost` labels it, and the meter says
"no pricing for <model>" instead of a reassuring `$0.00`.

OpenRouter `@preset/<slug>` names have no row in `/api/v1/models`, so they price out as
unknown. Resolve them through `GET /api/v1/presets/<slug>`
(`data.designated_version.config.model`) and cache the answer — presets are account-scoped
aliases, never catalog models.

## Pitfalls

- **Source edits to a git install do not reach the running process.** Each pm environment
  generation runs its own copy at
  `~/.hermes/installs/<id>/environments/<env>/workspace/`, and the venv's editable finder
  maps `agent`/`cli`/`hermes_cli` there. Tests run from the checkout pass while the live CLI
  is unchanged — a `.py` edit never triggers an environment rebuild.
  `hermes pm install` is the supported way to make a source change live; staying out of
  core is better.
- Never render `$0.00` for an unpriced route: it reads as "free" when the spend is simply
  unmetered.
- Display-only. Do not touch context, toolsets or the system prompt from a status render —
  that breaks the cached prefix and multiplies the user's bill.

Long form, including the research trail and the desktop-plugin alternative, is in
`references/why-wrapper-cli.md`.
