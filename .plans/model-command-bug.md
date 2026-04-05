# Bug Investigation: /model Gateway Command Not Working

**Date:** 2026-04-05  
**Reported by:** Tranquil-Flow  
**Status:** Confirmed bug — no handler implemented  
**Priority:** Medium — affects mid-session model switching for all platforms

---

## Summary

The `/model` slash command is registered in the Discord UI and appears in autocomplete, but it does nothing — the gateway has no handler for it. Instead of intercepting the command, the gateway passes it through to the AI as a plain chat message. The AI then (correctly) explains it's a gateway-level command, but that's the wrong outcome.

The same gap likely affects all platforms (Telegram, WhatsApp, etc.) since the dispatch loop is shared.

---

## Reproduction Steps

1. Open Discord, DM the Hermes bot
2. Type `/` → select `/model` from the autocomplete popup
3. Enter a model name e.g. `anthropic:claude-opus-4-5`
4. Send

**Expected:** Gateway intercepts, switches model for the session, returns a system confirmation  
**Actual:** Moon receives it as a regular message and replies explaining she can't process gateway commands

---

## Root Cause

Two-part mismatch:

### Part 1: Slash command registered but routes to AI
`gateway/platforms/discord.py` line 1551 registers `/model` as a Discord slash command:

```python
@tree.command(name="model", description="Show or change the model")
@discord.app_commands.describe(name="Model name (e.g. anthropic/claude-sonnet-4). Leave empty to see current.")
async def slash_model(interaction: discord.Interaction, name: str = ""):
    await self._run_simple_slash(interaction, f"/model {name}".strip())
```

`_run_simple_slash` (line 1512) builds a `MessageEvent` with `text="/model anthropic:claude-opus-4-5"` and calls `handle_message()`. So the event arrives at the gateway's command dispatcher correctly.

### Part 2: Dispatcher has no model handler
`gateway/run.py` ~line 1883 dispatches commands based on `canonical` name. It handles:
`new`, `help`, `commands`, `profile`, `status`, `stop`, `reasoning`, `verbose`, `yolo`, `provider`, `personality`, `plan`, `retry`, `undo`, `sethome`, `compress`, `usage`, `insights`, `reload-mcp`, `approve`, `deny`, `update`, `title`, `resume`, `rollback`, `background`, `btw`, `voice`

**`model` is completely absent from this list.**

When `canonical == "model"` reaches the dispatcher, nothing catches it. It falls through to the plugin handler, fails, and gets passed to the AI agent as a regular message.

### Part 3: model is not in COMMAND_REGISTRY
`hermes_cli/commands.py` has a `GATEWAY_KNOWN_COMMANDS` set derived from `COMMAND_REGISTRY`. Since `model` isn't registered there, it's not even recognized as a command — just unknown text starting with `/`.

---

## Files to Modify

1. **`gateway/run.py`** — Add `_handle_model_command` method + dispatch case
2. **`hermes_cli/commands.py`** — Register `model` in `COMMAND_REGISTRY` / `GATEWAY_KNOWN_COMMANDS`
3. **`tests/gateway/`** — Add tests for the new handler

---

## What the Fix Should Do

### `_handle_model_command(self, event: MessageEvent) -> str`

Parse the argument: `provider:model-name` (e.g. `anthropic:claude-opus-4-5`)

With no argument: show current model (similar to `/provider` but showing model too)

With a model argument:
1. Parse provider and model from the string (split on `:`, first part is provider)
2. Validate the provider exists in `list_available_providers()`
3. Persist the per-session model override — need to decide where (see open questions)
4. Evict the cached agent: `self._evict_cached_agent(session_key)` (line 5343)
5. Return confirmation: `"✓ Switched to \`{provider}:{model}\` for this session. Starting fresh context."`

With `/model` (no args): show current active model for the session.

### Dispatch wiring in `run.py`
```python
if canonical == "model":
    return await self._handle_model_command(event)
```
Add this in the dispatch block around line 1929 (near `provider`).

---

## Open Questions for the PR

**Q1: Where to persist the per-session model override?**

Options:
- `session_store` entry — add a `model_override` field to the session entry and read it in `_run_agent` when building `turn_route`
- In-memory dict keyed by `session_key` — simpler, lost on gateway restart (probably fine)
- `~/.hermes/state.db` sessions table — already has a `model` column (line 65 in hermes_state.py schema), may be the right place

The `state.db` sessions table already stores `model TEXT` and `model_config TEXT` per session. The gateway reads this on session resume. Updating it on `/model` switch would make it truly persistent and visible in session history. This seems like the right approach.

**Q2: Does switching model mid-session clear conversation history?**

Probably should offer a choice: switch silently (same history, new model) OR `/model ... --fresh` (clears history too). The `/new` + model config approach currently forces a history clear. Mid-session switch keeping history is more useful.

**Q3: Should `/provider` and `/model` be unified?**

Currently `/provider` only shows providers, doesn't switch anything. `/model` is supposed to switch. They overlap conceptually. Consider whether `/provider anthropic:claude-opus-4-5` should also work as an alias.

---

## Reference: How Other Commands Handle State

`_handle_reasoning_command` (`/reasoning`) is a good reference — it:
1. Reads args
2. Persists to `session_store` via `update_session()`
3. Evicts cached agent
4. Returns confirmation

The `/model` handler should follow the same pattern.

---

## Test Coverage Needed

- `tests/gateway/test_model_command.py` (new file)
  - `/model` with no args → shows current model
  - `/model anthropic:claude-opus-4-5` → switches and confirms
  - `/model invalid-provider:foo` → returns helpful error
  - `/model` via Discord slash command event → intercepted, not passed to AI
  - Session persists new model across turns after switch

---

## PR Scope

Small, self-contained fix. No new dependencies. Estimated: ~100 lines of new code + tests.

Upstream repo: https://github.com/NousResearch/hermes-agent

Check for existing issue before filing — search: `model command`, `/model`, `slash command model switch`
