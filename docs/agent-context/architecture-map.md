# Hermes Agent Architecture Map

This reference preserves the architecture orientation formerly embedded in `AGENTS.md`.

## Development environment

```bash
# Prefer .venv; fall back to venv if that is what the checkout has.
source .venv/bin/activate   # or: source venv/bin/activate
```

`scripts/run_tests.sh` probes `.venv`, then `venv`, then the shared worktree venv.

## Project structure

```text
hermes-agent/
├── run_agent.py          # AIAgent — core conversation loop
├── model_tools.py        # Tool discovery and dispatch
├── toolsets.py           # Toolset definitions
├── cli.py                # Interactive CLI (HermesCLI)
├── hermes_state.py       # SQLite session store
├── agent/                # prompt builder, compression, memory, routing, skill dispatch
├── hermes_cli/           # CLI subcommands, config, setup, command registry
├── tools/                # one file per tool + registry.py
├── gateway/              # messaging gateway and platform adapters
├── cron/                 # scheduled job system
├── tests/                # pytest suite; use scripts/run_tests.sh
└── website/              # Docusaurus docs site
```

## File dependency chain

High-level flow:

```text
CLI / gateway / cron / desktop / dashboard
→ AIAgent in run_agent.py
→ prompt builder + model routing
→ model_tools.py
→ tools/registry.py + individual tool modules
→ state/config/memory/skills/plugins as needed
```

## AIAgent and agent loop

`run_agent.py` owns the core conversation loop:

1. Build system prompt.
2. Call the selected model with OpenAI-format messages + tool schemas.
3. If the model emits tool calls, dispatch them and append tool results.
4. Continue until a text response is returned or max iterations are reached.
5. Context compression is the only allowed mid-conversation context mutation.

Preserve:

- strict role alternation
- cache-stable system prompts
- bounded tool loops
- tool result ordering and error handling

## CLI architecture

`cli.py` owns interactive CLI behavior. `hermes_cli/commands.py` is the slash command registry; help text, autocomplete, Telegram menu, and Slack mapping derive from the central registry.

Adding slash commands generally touches:

1. `hermes_cli/commands.py` — `CommandDef`
2. `cli.py` — command handler
3. optional gateway handler in `gateway/run.py`

## TUI / dashboard / desktop surfaces

Hermes runs across CLI, TUI, dashboard, desktop, and messaging gateway. Surface-specific behavior should stay at the surface edge when possible; avoid pushing UI-specific logic into the core loop.

Important TUI concepts:

- process model separates UI and backend gateway pieces
- transport bridges user input and agent output
- slash command flow must preserve the same command registry semantics

## TypeScript / desktop style

For frontend/desktop code, keep style consistent with existing app patterns. Prefer small typed modules and avoid cross-surface coupling.

## Skin and theme system

The skin engine is data-driven. Skins should be pure data, not code changes, unless extending the generic theme system itself.

Use this summary in `AGENTS.md`; keep detailed skin/theme specifics close to `hermes_cli/skin_engine.py` and docs.
