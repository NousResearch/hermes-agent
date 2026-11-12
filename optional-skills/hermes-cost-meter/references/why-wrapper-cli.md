# hermes-cost-meter

A live session-cost meter for the Hermes CLI, as a **wrapper CLI**. No edits to the
Hermes source tree, no fork, nothing to re-apply after `hermes update`.

The spend lands **inside the status bar you already have**, right after the model:

```
 ☤ deepseek-v4-1-flash-101... · ~$0.0043 · 17.9K/1M · [░░░░░░░░░░] 2% · ◎ 20.1% · ◷ 1.6s · ↑ 2 t/s · 1m · ⏲ 2s · ✓ 56s
```

## Why this shape (the research)

`display.show_cost` looks like the answer and is not: it is defined in
`hermes_cli/config_defaults.py` and documented in
`website/docs/user-guide/configuration.md`, but **nothing reads it**. Upstream knows —
issue [#93951](https://github.com/NousResearch/hermes-agent/issues/93951) ("show_cost is a
dead config in 0.20.4") is open, and the two PRs that wire it up
([#99649](https://github.com/NousResearch/hermes-agent/pull/99649),
[#75487](https://github.com/NousResearch/hermes-agent/pull/75465)) are open and unmerged.
So there is no `hermes update` that turns the meter on.

Patching core was the wrong shape for three reasons:

1. Hermes' own contributing guide rejects "plugins that touch core files" — the core is a
   narrow waist, capability lives at the edges.
2. A git-install does **not** run `~/.hermes/hermes-agent`. Each pm environment generation
   runs its own copy at `~/.hermes/installs/<id>/environments/<env>/workspace/`, and the
   venv's editable finder maps `agent`, `cli`, `hermes_cli` there. Editing the checkout
   looks like it works and changes nothing (`_layout_digest` only hashes
   `pyproject.toml`/`setup.py`, so a `.py` edit never triggers a rebuild).
3. Every `hermes update` would autostash and re-apply the patch — a permanent conflict
   surface for a display feature.

The sanctioned extension point is documented at
[developer-guide/extending-the-cli](https://hermes-agent.nousresearch.com/docs/developer-guide/extending-the-cli):
`HermesCLI` exposes five protected hooks. Two matter here:

- `_status_bar_segments()` — the single method both the classic and the modern layout build
  their status line from (`cli_modern_layout._status_line_fragments` calls it too). Overriding
  it appends a segment **inside** the existing bar. (`_get_extra_tui_widgets()` is the other
  documented hook, but it adds a *separate row above* the bar — not what we want.)
- `process_command()` — adds `/cost`.

## Run it

```sh
scripts/hermes-cost              # fresh session
scripts/hermes-cost --continue   # resume, flags pass through
```

## How it stays working across updates

`hermes --print-runtime-command` prints the exact argv the launcher would exec for the
current environment generation (right python, right bootstrap). `scripts/_launch.py` reuses
that argv and only swaps the module it runs (`hermes_cli.main` → `hermes_cost_cli`). No
venv path is hardcoded, so an `hermes update` or a rebuilt environment is picked up
automatically.

## What the number is

Session-to-date spend, computed from the session's own token counters times the model's
rate card — the same `agent.usage_pricing` helpers `/usage` and `state.db` use. It is a
**lower bound**: only main-agent responses carry usage, so auxiliary calls, retries and
cache-write overhead are missed. `/cost` prints the breakdown.

OpenRouter `@preset/<slug>` names have no row in `/api/v1/models`, so they price out as
"unknown". `_resolve_preset()` reads the routed model from
`GET /api/v1/presets/<slug>` (`data.designated_version.config.model`) and caches it in
`~/.hermes/cache/cost_meter_presets.json` for 7 days. When a route still has no price row
the meter says so instead of rendering a reassuring `$0.00`.

## Files

| File | Role |
|---|---|
| `scripts/hermes_cost_cli.py` | the wrapper CLI: status-bar segment, `/cost`, preset resolution, pricing |
| `scripts/_launch.py` | re-execs the current Hermes runtime with the wrapper module |
| `scripts/hermes-cost` | the entry point you run |
| `scripts/selftest.py` | pty self-test: asserts the meter lands on the status bar |

## Desktop / TUI alternatives

- **Desktop app**: status-bar items are a first-class registry surface in the
  [Desktop Plugin SDK](https://hermes-agent.nousresearch.com/docs/developer-guide/desktop-plugin-sdk)
  — one ESM file in `~/.hermes/desktop-plugins/<id>/plugin.js`, hot-reloads.
- **Marketplace plugins** meter spend but do not put a number in the status bar:
  `costguard` (per-call metering + a budget brake), `hermes-telemetry` (`/stats`,
  `/budget`; note its default *enforce* mode writes a $5/day, $100/month budget and starts
  blocking tool calls), `hermes-call-ledger`, `session-dashboard` (desktop analytics).

## Status

Verified live in a pty against the real runtime: the meter renders
(`~$0.0001 · /cost`) and `/cost` prints the breakdown, on an `@preset/...` model.
Core-patch alternative kept on branch `reference/core-patch` (not applied) for the case
where upstream wants PR #99649 finished instead.
