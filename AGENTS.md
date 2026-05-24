# Hermes Agent - Development Guide

Concise instructions for agents working in this repo. The old long-form guide was moved to `docs/AGENTS.reference.md`; load that file only when you need deep architecture details. Do not inject the whole cookbook into every session. That was token arson with markdown.

## Environment

```bash
source .venv/bin/activate  # prefer .venv; fall back to venv if needed
```

- Current repo: `~/.hermes/hermes-agent`.
- User config: `~/.hermes/config.yaml`; secrets: `~/.hermes/.env`.
- Logs: `~/.hermes/logs/` (`gateway.log`, `agent.log`, `errors.log`).
- Use `get_hermes_home()` from `hermes_constants.py` for Hermes paths; do not hardcode `~/.hermes` in code.

## Load-bearing files

- `run_agent.py` — `AIAgent`, system prompt assembly, compression, conversation loop.
- `model_tools.py` — tool discovery/dispatch.
- `toolsets.py` — built-in toolset definitions.
- `cli.py` / `hermes_cli/commands.py` — CLI and slash commands.
- `hermes_state.py` — SQLite session store/search.
- `gateway/run.py`, `gateway/session.py`, `gateway/platforms/` — messaging gateway.
- `tools/` — tool implementations registered via `tools.registry`.
- `tests/` — pytest suite.

## Coding rules

1. Keep prompt caching stable: do not mutate system prompts, tool schemas, or context files mid-session unless the user explicitly resets/restarts.
2. Preserve OpenAI message validity: avoid duplicate adjacent assistant/user messages and keep tool call/result pairing intact.
3. New tools need: `tools/<tool>.py`, import/discovery in `model_tools.py`, and a toolset entry in `toolsets.py`. Handlers return JSON strings and include `check_fn`/env gating when applicable.
4. New slash commands go through `COMMAND_REGISTRY` in `hermes_cli/commands.py`; CLI/gateway help derive from it.
5. Gateway platform code must avoid blocking the event loop; use async adapters and proper shutdown cleanup.
6. Config belongs in `config.yaml`; credentials belong in `.env` or auth stores, never source files.
7. Tests redirect `HERMES_HOME` to temp dirs. Do not write tests that touch real user state.

## Testing

```bash
source .venv/bin/activate 2>/dev/null || source venv/bin/activate
python -m pytest tests/ -o 'addopts=' -q
python -m pytest tests/path/to/test_file.py -q
```

`scripts/run_tests.sh` probes `.venv`, then `venv`, then `$HOME/.hermes/hermes-agent/venv` for worktrees sharing the main checkout venv.

For gateway/config/toolset changes, add targeted tests before running broad suites. Use foreground commands with sane timeouts; kill anything stuck over ~60 seconds unless it is an expected long test run.

## Common workflows

- Gateway status: `hermes gateway status` or `hermes status --all`.
- Gateway logs: `tail -n 120 ~/.hermes/logs/gateway.log`.
- Tool resolution: inspect `hermes_cli/tools_config.py::_get_platform_tools` and `toolsets.py`.
- Session bloat: measure fixed prompt pieces with `AIAgent._build_system_prompt_parts()` and tool schema rough tokens via `agent.model_metadata.estimate_tokens_rough`.

## Reference

Full archived/reference version: `docs/AGENTS.reference.md`.
Load it only for details not covered above: deep architecture, plugin internals, platform edge cases, or historical pitfalls.
