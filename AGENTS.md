# Hermes Agent - Development Guide

Instructions for AI coding assistants and developers working on this Hermes Agent
checkout. Keep this file compact: it is always loaded as project context and must
stay under the active `context_file_max_chars` budget. Full historical guidance is
archived at `website/docs/developer-guide/agents-full.md`; load focused docs or
skills only when the task needs them.

**Never give up on the right solution.**

## What Hermes Is

Hermes is a personal AI agent that runs the same core across CLI, messaging
gateways, TUI, desktop, cron, browser/terminal tools, memory, skills, MCP, and
plugins. Two design constraints dominate reviews:

1. **Prompt caching is sacred.** Do not mutate past context, swap toolsets, or
   rebuild the system prompt mid-conversation except through supported
   compression/rebuild paths.
2. **The core is a narrow waist.** New model tools are expensive because they
   are sent on every API call. Prefer capability at the edges.

## Contribution Intent

We want:
- real bug fixes with reproduction and whole-class fixes;
- edge expansion: platforms, providers, models, desktop/TUI/dashboard features;
- refactors that split god-files into focused modules;
- behavior/invariant tests and E2E validation for config, security, I/O, and
  provider resolution paths;
- contributor credit preserved when salvaging external work.

We do not want:
- speculative hooks/infrastructure without a concrete consumer;
- duplicate managers when existing infrastructure can be extended;
- new core model tools unless all smaller-footprint options are exhausted;
- snapshot/change-detector tests for model lists, enum counts, config versions,
  or other routine source data.

## Footprint Ladder

For new capability, choose the smallest viable footprint:

1. extend existing code;
2. CLI command + skill;
3. service-gated tool with `check_fn`;
4. plugin;
5. MCP server in the catalog;
6. new core tool only as last resort.

When several PRs add the same category, design one shared interface instead of
merging one-off integrations.

## Source Map

- `run_agent.py` — core loop.
- `model_tools.py`, `toolsets.py`, `tools/` — tool dispatch/schemas/toolsets.
- `agent/` — prompt assembly, memory, compression, model routing/transports.
- `hermes_cli/` — CLI commands, config, setup, auth, provider UX.
- `cli.py` — interactive CLI shell.
- `gateway/` — platform adapters and gateway runtime.
- `plugins/`, `plugins/kanban/` — plugin system and Kanban dispatcher/worker.
- `models/tools/registry.py` — tool discovery path.
- `tests/` — pytest suite.
- `website/docs/developer-guide/` — long-form developer references.

## Development Commands

Prefer the repo virtualenv when present:

```bash
source .venv/bin/activate 2>/dev/null || source venv/bin/activate
python -m pytest tests/ -o 'addopts=' -q
```

Run focused tests for the touched area before broad suites. Avoid commands that
write to the real `~/.hermes` in tests; use temp `HERMES_HOME` fixtures.

## Profile and Config Safety

- Never hardcode `~/.hermes` in source; use profile-safe helpers such as
  `get_hermes_home()`.
- Profile workers read profile-local config/env/auth. Root config/env may be a
  different profile.
- Secrets belong in `.env` or auth stores, never in docs, skills, tests, or logs.
- Config/provider/tool/MCP/gateway changes are often startup snapshots; verify
  whether a restart or fresh session is required.
- Do not merge an entire root config into a named profile. Promote only the
  specific artifact needed.

## Adding or Changing Hermes Surfaces

- **Tool:** add `tools/<name>.py`, register/discover it, assign toolset
  membership, add readiness checks, and run focused tests. Core tool additions
  must justify why CLI+skill/plugin/MCP is insufficient.
- **Slash command:** update `hermes_cli/commands.py`, CLI handler, gateway/TUI
  handlers if applicable, help/autocomplete, and focused tests.
- **Config option:** update defaults, loaders, docs, migration behavior if
  needed, and tests that exercise the real load path against temp `HERMES_HOME`.
- **Plugin/provider/skill:** prefer existing plugin and skill infrastructure;
  keep runtime gating explicit and avoid importing optional heavy deps at
  startup.

Load the `hermes-agent` skill for exact command sequences and focused reference
files before Hermes setup/config/gateway/source work.

## Prompt and Message Invariants

- Preserve strict role alternation; never inject synthetic user messages into
  the model history.
- Keep the cached prefix byte-stable for a conversation.
- Do not add always-on context when an on-demand doc, skill, plugin, or CLI path
  is enough.
- Large evidence belongs on disk with a short path + summary, not in chat or
  always-loaded context.

## Testing Policy

Write tests for behavior contracts and relationships, not current data snapshots.

Do not write tests like:

```python
assert "gemini-2.5-pro" in _PROVIDER_MODELS["gemini"]
assert DEFAULT_CONFIG["_config_version"] == 21
assert len(models) == 8
```

Do write tests like:

```python
assert "gemini" in _PROVIDER_MODELS
assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
assert not (set(moonshot_models) & coding_plan_only_models)
for model in _PROVIDER_MODELS["huggingface"]:
    assert model.lower() in DEFAULT_CONTEXT_LENGTHS_LOWER
```

For config propagation, security boundaries, remote backends, or file/network
I/O, exercise the real integration path with real imports and temporary state.
Mocks alone are not sufficient.

## Known Pitfalls

- `AGENTS.md` must stay compact; if it grows, move long guidance to docs or a
  procedure skill. The budget invariant is enforced by tests.
- Do not introduce new `simple_term_menu` usage.
- Do not use `\\033[K` erase-to-EOL in spinner/display code.
- `_last_resolved_tool_names` in `model_tools.py` is process-global.
- The gateway has approval/control command bypass guards in more than one path.
- Squash merges from stale branches can silently revert recent fixes.
- Do not wire dead code without E2E validation.

## Where Long Guidance Lives

- Full pre-trim guide: `website/docs/developer-guide/agents-full.md`.
- Prompt/context assembly: `website/docs/developer-guide/prompt-assembly.md`.
- User configuration docs: `website/docs/user-guide/configuration.md`.
- Runtime/source workflow: load the `hermes-agent` skill.
- Context-bloat remediation: load the `context-discipline` skill.

Rule of thumb: if content is executed as a reusable workflow, make or update a
skill; if it is read as reference, put it in docs; if every agent must obey it on
every turn, keep only the compact form here.
