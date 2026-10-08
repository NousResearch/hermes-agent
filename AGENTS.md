# Hermes Agent repository instructions

This file is the always-loaded router and safety contract. It carries rules that must hold for every path. Nested `AGENTS.md` files add area-specific rules. The filesystem, scripts, and configuration are canonical, so this file does not inventory source files. Keep this file below the `python scripts/check` limit of 12,000 characters and keep each root-to-area instruction chain below 30,000 characters.

## Load by task

- **Editing or reviewing files, including documentation and instructions; dependency or environment work; contributions, issues, PRs, and salvage:** read [`CODING_STANDARDS.md`](CODING_STANDARDS.md).
- **A new gateway platform:** read [`gateway/platforms/ADDING_A_PLATFORM.md`](gateway/platforms/ADDING_A_PLATFORM.md) as the procedure, plus [`gateway/AGENTS.md`](gateway/AGENTS.md).
- **A security boundary, vulnerability, or threat report:** read [`SECURITY.md`](SECURITY.md). Its scope and reporting channel control.
- **A dependency or Hermes environment:** read [`pm/AGENTS.md`](pm/AGENTS.md) and the [package-management developer workflow](website/docs/reference/package-management.md#developer-workflow).
- **A contribution, issue, PR, review, or salvage:** read the [developer contribution guide](website/docs/developer-guide/contributing.md) and the contributor workflow in [`skills/autonomous-ai-agents/hermes-agent/SKILL.md`](skills/autonomous-ai-agents/hermes-agent/SKILL.md).
- **Architectural background:** start with the [architecture map](website/docs/developer-guide/architecture.md), then follow its subsystem links.

## Universal invariants

### Prompt cache

Keep the conversation prefix byte-stable. Do not mutate past context, swap toolsets, reload memories, or rebuild the system prompt during a conversation. Context compression is the only exception. A command that changes system-prompt state takes effect in the next session unless the command explicitly offers an opt-in `--now`, such as `/skills install --now`. Content that must arrive during a conversation goes through a user message or tool result, never a system-prompt rewrite.

### Narrow waist

The core is a narrow waist and capability belongs at the edges. Every model tool is sent on every API call. Use the footprint ladder in [`CODING_STANDARDS.md#footprint-ladder`](CODING_STANDARDS.md#footprint-ladder) before adding core surface. A session-specific capability is not a process-wide environment fact; the tool and session-scope rules live in [`tools/AGENTS.md`](tools/AGENTS.md).

### Profile and execution scope

One process can serve several profiles. Bind the owning profile scope for code outside a turn, including boot probes, eviction, tickers, deferred callbacks, RPC methods, thread hops, and child processes. `os.environ`, module globals, and import-time values describe the launch profile; an unbound read can leak the default profile. Read [`gateway/AGENTS.md`](gateway/AGENTS.md) for binding points and execution-scope contracts.

Use `get_hermes_home()` for code paths and `display_hermes_home()` for user-facing text. Never hardcode `~/.hermes`; `_get_profiles_root()` is intentionally HOME-anchored. Tests use disposable homes and never write to the user's Hermes home.

### Process identity

Never infer process identity from an argv substring. Use `gateway.status.looks_like_gateway_command_line` and `hermes_cli.update_cmd._hermes_holder_subcommand` (HX003; details [`hermes_cli/AGENTS.md`](hermes_cli/AGENTS.md)).

### Security reporting

[`SECURITY.md`](SECURITY.md) is the scope authority. A §3.1 finding goes through a GitHub Security Advisory or `security@nousresearch.com`, never a public issue, PR, commit, or comment. §3.2 hardening is ordinary public work. Name the §2 boundary crossed and reproduce the finding on `main`.

## Routing table

Before editing a path in the first column, read its guide in the second column, even if the runtime has not discovered it. The third column states the branch that the guide owns. General implementation policy stays in [`CODING_STANDARDS.md`](CODING_STANDARDS.md); these guides own their local invariants.

| Working area | Read before editing | Guide owns |
|---|---|---|
| `run_agent.py`, `agent/` | [`agent/AGENTS.md`](agent/AGENTS.md) | turn phases, caching and message flow, compression, model and auxiliary resolution |
| `cli.py`, `hermes_cli/` | [`hermes_cli/AGENTS.md`](hermes_cli/AGENTS.md) | CLI mixins, slash registry, config, skins, `hermes update`, profiles and multiplex |
| `gateway/` | [`gateway/AGENTS.md`](gateway/AGENTS.md) | adapters, message guards, streaming, notifications, token locks, profile scope |
| New adapter under `gateway/platforms/` | [`gateway/platforms/ADDING_A_PLATFORM.md`](gateway/platforms/ADDING_A_PLATFORM.md) | adapter procedure |
| `tools/`, `toolsets.py`, `model_tools.py` | [`tools/AGENTS.md`](tools/AGENTS.md) | tool registry, toolsets, delegation, session-scoped capability |
| `plugins/`, `hermes_cli/plugins*.py` | [`plugins/AGENTS.md`](plugins/AGENTS.md) | plugin kinds, compatibility, isolation, in-tree policy |
| `plugin-catalog/` | [`plugin-catalog/README.md`](plugin-catalog/README.md) | catalog admission rules; keep the developer-guide mirror identical |
| `tui_gateway/`, `ui-tui/` | [`tui_gateway/AGENTS.md`](tui_gateway/AGENTS.md) | process model, JSON-RPC transport, slash flow |
| `web/`, `hermes_cli/web_routers/` | [`web/AGENTS.md`](web/AGENTS.md) | dashboard and embedded TUI contract |
| `apps/desktop/` | [`apps/desktop/AGENTS.md`](apps/desktop/AGENTS.md), [`apps/desktop/src/AGENTS.md`](apps/desktop/src/AGENTS.md) | desktop backend, renderer, slash palette, Bot Mode |
| `skills/`, `optional-skills/`, `agent/curator*.py` | [`skills/AGENTS.md`](skills/AGENTS.md) | skill frontmatter, authoring, curator |
| `cron/`, kanban | [`cron/AGENTS.md`](cron/AGENTS.md) | scheduler invariants, job fields, Kanban dispatcher |
| `tests/` | [`tests/AGENTS.md`](tests/AGENTS.md) | runner, placement, OS markers, `wine2e`, banned test shapes |
| `pm/`, `pyproject.toml` | [`pm/AGENTS.md`](pm/AGENTS.md) | pinning, PM-owned environments, plugin quarantine |
| `hermes_platform/` | [`hermes_platform/AGENTS.md`](hermes_platform/AGENTS.md) | host facts and resolvers |

## Source authority

`CODING_STANDARDS.md` owns general coding policy. A branch guide owns its subsystem contract. `SECURITY.md`, `CONTRIBUTING.md`, `plugin-catalog/README.md`, scripts, configuration, and the filesystem own the facts named by those files. Keep one authoritative copy of each rule. Long-form rationale lives in `website/docs/developer-guide/`; the checked-in contributor workflow is [`skills/autonomous-ai-agents/hermes-agent/SKILL.md`](skills/autonomous-ai-agents/hermes-agent/SKILL.md) with `references/contributor-guide.md`.
