---
name: model-auto-router
description: Build and verify Hermes `/model auto` per-task model router wiring.
version: 1.0.0
author: User <user@example.com>, Hermes Agent
license: MIT
platforms: [linux, macos]
metadata.hermes.tags: [models, routing, slash-commands, cli]
metadata.hermes.category: configuration
related_skills: hermes-agent, hermes-themes
---

# Model Auto-Router Skill

Adds a per-task model auto-router to Hermes: `/model auto` (and `/model --auto`,
`/model auto-<alias>`) picks the best configured `model_aliases` entry for the
current turn — vision, deep-research intent, or fast/cheap default — then feeds
the picked alias straight into the existing `_switch_model_from -> confirm -> commit`
chain. No rewrite of the ~900-line switch handler; no GPU re-partitioning.

What it does: scores aliases by capability tier + turn signals and returns one pick.
What it does not do: load or unload models (that stays with `model_switch.py`),
parse the full context window, or read images — those are best-effort soft signals.

## When to Use
- Adding per-task model selection to a Hermes install that already has Stage 1
  aliases (`config.yaml` -> `model_aliases`).
- Wiring a new router module into an existing slash-command handler without
  rewriting the ~900-line mixin.
- Verifying routing picks against real config before shipping.

## Prerequisites
- `models_dev.py` ModelInfo registry is present (capability metadata:
  `attachment`, `reasoning`, tool_call, context_window).
- At least two aliases in `model_aliases` (e.g. `fast`, `researcher`).
- Read `hermes_cli/models_router.py::resolve()` and the handler at
  `hermes_cli/cli_model_switch_mixin.py::_handle_model_switch` before editing — both APIs are small and flat.

## How to Run

### 1. Inspect current wiring
Use `read_file` on `models_router.py`, then grep the mixin for the entry point:

```
search_files pattern='def _resolve_auto_model' target=content path=.
grep -n 'with_target\|is_deep\|MODEL_SCOPE_AUTO' hermes_cli/model_switch.py
```

The router's public API is:
`resolve(aliases_config, *, has_vision=False, context_bytes=0, deep_intent=False, prefer_alias=None) -> RouteDecision(alias, provider_model, tier, reason)`
where `RouteDecision.alias` is the chosen alias key.

### 2. Wire the router into `/model auto`
The handler already parses `/model <target>` via `parse_model_switch_args`. Add a
step **after** `request.errors` guard and before `_switch_model_from`:

```python
auto_pick = self._resolve_auto_model(request)   # returns (alias, reason) or ()
if auto_pick:
    alias, reason = auto_pick
    if _cprint: _cprint(f"✓ Model picked automatically: {alias} ({reason})", "green")
    request = request.with_target(alias)          # frozen-safe reassignment
```

Add the helper on the mixin (reads real turn signals):
- `self._attached_images` -> `has_vision`
- `request.is_deep` flag -> `deep_intent`
- history length -> `context_bytes` (estimate; full window not parsed)
- `prefer_alias` comes from `auto-<alias>` override form.

### 3. Frozen-dataclass reassignment
`ModelSwitchRequest` is `@dataclass(frozen=True)`, so direct attribute assignment
raises `FrozenInstanceError`. Use the provided helper:
```python
def with_target(self, target: str) -> "ModelSwitchRequest":
    return replace(self, target=target)
```
Import `replace` from `dataclasses` in `model_switch.py`.

### 4. Verify green
Run the bundled test and a live resolve against real config (see ## Verification).
Add a test at `tests/skills/test_model_auto_router_skill.py`.

## Quick Reference

| Command | Picks | Signal source |
|---|---|---|
| `/model auto` | `fast` | default tier |
| `/model --auto` | `fast` | `is_auto` flag path |
| `/model auto` (+image) | multimodal-capable alias | `self._attached_images` |
| `/model auto --deep` | `researcher` | `--deep` flag / deep intent |
| `/model auto-researcher` | `researcher` | manual override form |
| `/model fast` | no pick (unchanged) | plain alias flows through |

## Procedure
1. Confirm the router module exists and `resolve()` returns a flat `RouteDecision`.
2. Add `_resolve_auto_model(request)` to `cli_model_switch_mixin.py`; read the four
   turn signals from `self` (`_attached_images`, history) + parsed flags on `request`.
3. Wire the pick into `_handle_model_switch` via `with_target()` before `_switch_model_from`.
4. Add `replace` import + `with_target()` to `model_switch.py`; add `is_auto`/`is_deep`
   fields to the request dataclass and propagate them from parse results.
5. Compile: `python3 -m py_compile hermes_cli/models_router.py hermes_cli/model_switch.py hermes_cli/cli_model_switch_mixin.py`.
6. Run the bundled test; then run a live resolve against real config.

## Pitfalls
- **Frozen dataclass**: never do `request.target = alias`; use `with_target()`. Direct
  assignment raises `FrozenInstanceError` and is silent if not caught early.
- **Signals are soft heuristics**: context budget is estimated from history length, deep
  intent only triggers on `--deep` or known trigger phrases, vision relies on
  `_attached_images` being populated before the handler runs. Do not claim these are exhaustive.
- **Manual override wins everything** in `resolve()`; `/model auto <alias>` maps to
  `prefer_alias`. A bare alias with no prefix must NOT route through the auto path.
- **No `elif` ladder**: commands dispatch via `_SLASH_DISPATCH`; keep additions additive.
- Don't re-read the whole ~905-line mixin — the seam is a single insertion point in
  `_handle_model_switch` right after the errors guard.

## Verification
Run the bundled test (stdlib + pytest, no live network):
```
scripts/run_tests.sh tests/skills/test_model_auto_router_skill.py -q
```
Then confirm picks against real config:
```
python3 scripts/model_auto_router/demo_resolve.py --auto        # -> fast
python3 scripts/model_auto_router/demo_resolve.py --deep        # -> researcher
python3 scripts/model_auto_router/demo_resolve.py auto-researcher  # -> researcher (override)
```
All three should print a pick and exit 0. If `demo_resolve.py` reports no pick for an
`auto` case, check that the parser applies the flag to `is_auto`/`is_deep` before wiring — a
swallowed error in resolve() masks this. Confirm `with_target()` returns a copy (not mutated) by
asserting identity of the original request object after the call.
