# Hermes anchor files

Read these first when building a compact map of the Hermes codebase.

## Primary anchors
- `AGENTS.md` — repo-wide development constraints, path rules, testing guidance, known pitfalls
- `run_agent.py` — agent lifecycle, cached system prompt, session state, tool execution loop
- `model_tools.py` — tool discovery/imports, schema post-processing, dispatch behavior
- `toolsets.py` — shared toolset composition and tool availability
- `agent/prompt_builder.py` — skills index prompt, context-file prompt, prompt-side snapshot caching
- `hermes_cli/config.py` — defaults, migrations, config schema metadata
- `tools/registry.py` — central tool registration contract

## Common secondary anchors
- `cli.py` — interactive CLI loop and slash-command dispatch
- `hermes_cli/main.py` — top-level CLI subcommands
- `agent/prompt_caching.py` — Anthropic cache-control strategy
- `agent/context_compressor.py` — prompt rebuild/compression path
- `gateway/run.py` — messaging gateway dispatch
- `gateway/session.py` — gateway session persistence
- `hermes_state.py` — SQLite-backed session metadata/search

## Fast subsystem routing
- Prompt/caching questions -> `run_agent.py`, `agent/prompt_builder.py`, `agent/prompt_caching.py`, `agent/context_compressor.py`
- Tool availability questions -> `model_tools.py`, `toolsets.py`, `tools/registry.py`
- CLI/setup/config questions -> `cli.py`, `hermes_cli/main.py`, `hermes_cli/config.py`, `hermes_cli/setup.py`
- Gateway/platform questions -> `gateway/run.py`, `gateway/platforms/*`, `gateway/session.py`
- State/search/memory questions -> `hermes_state.py`, `tools/session_search_tool.py`, `tools/memory_tool.py`
- Skill-system questions -> `agent/prompt_builder.py`, `agent/skill_tools.py`, `agent/skill_utils.py`, `skills/`
