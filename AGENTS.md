# Hermes Agent — Compact AGENTS Guide

This file is intentionally compact because Hermes loads AGENTS instructions into the project context. The detailed upstream developer guide is preserved at `docs/agents-developer-guide-full.md`; read it when detailed architecture, CLI, gateway, plugin, slash-command, provider, Kanban, TUI/Dashboard, or contribution guidance is needed.

## Development environment

```bash
# Prefer the repo venv when available.
source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true
```

On this checkout, `/usr/bin/python3` may be Python 3.9 and cannot import newer type syntax in Hermes source. Prefer `venv/bin/python3.11` or the activated venv for Hermes tests and prompt-builder checks.

## Core rules

- Inspect first, then make the smallest safe edit, then verify with evidence.
- Do not run destructive git operations, `sudo`, release/upload/signing actions, credential-affecting commands, or external publication without explicit approval.
- Never print secrets from `~/.hermes/.env`, config files, profiles, OAuth/token stores, signing assets, or credential pools; summarize as `[REDACTED]`.
- Use `get_hermes_home()` / profile-aware helpers for Hermes paths in source code; do not hardcode `~/.hermes` except in docs/examples that explicitly describe the default path.
- Do not change tool schemas, system-prompt assembly, memory, gateway, provider routing, prompt/context loading, plugin/provider surfaces, or security hooks without targeted tests.
- Keep HASOS guidance compact in runtime prompts: load the local runtime-ready slice, not raw packs, long templates, or full skill bundles.
- Treat v0.13 Tenacity features (Kanban, `/goal`, gateway auto-resume, Checkpoints v2, post-write lint, no-agent cron, provider plugins, platform allowlists, redaction defaults) as upstream capabilities to preserve unless a targeted test proves a local integration conflict.

## Repository map

```text
hermes-agent/
├── run_agent.py          # AIAgent conversation loop and runtime/plugin integration
├── model_tools.py        # Tool orchestration, discovery, dispatch, pre/post tool hooks
├── toolsets.py           # Toolset definitions and exposed tool membership
├── cli.py                # Interactive CLI orchestrator
├── hermes_state.py       # SQLite session store
├── hermes_constants.py   # profile-aware Hermes paths
├── agent/                # Provider adapters, prompt builder, memory, compression, guards
├── hermes_cli/           # CLI subcommands, config, setup, plugins, dashboard, Kanban
├── providers/            # ProviderProfile registry and provider plugin support
├── plugins/              # General plugins, model-providers, platforms, memory, dashboard
├── tools/                # Tool implementations and registry
├── gateway/              # Messaging gateway and platform adapters
├── cron/                 # Scheduler and no_agent/script watchdog support
├── skills/               # Built-in skills bundled with the repo
├── optional-skills/      # Heavier/niche skills not active by default
├── ui-tui/               # Ink (React) terminal UI — `hermes --tui`
│   └── src/              # entry.tsx, app.tsx, gatewayClient.ts + app/components/hooks/lib
├── tui_gateway/          # Python JSON-RPC backend for the TUI
├── acp_adapter/          # ACP server (VS Code / Zed / JetBrains integration)
├── scripts/              # run_tests.sh, release.py, auxiliary scripts
├── website/              # Docusaurus docs site
└── tests/                # Pytest suite
```

## Routing reminders

- For release, upload, review, signing, build, privacy, App Store, or wiki standards work, consult the iOS-Wiki `standards-invocation-map` and required substandards first.
- For high-impact review/execution (release, submission, signing, credentials, reset/revert/force-push, public sharing), route through the strongest available model / reviewer path before irreversible steps.
- For safe automated single-user path work, execute through completion when the path is safe; stop only for real blockers or approval-required risks.
- For long-running/background/subagent/external CLI/gateway/cron/MCP/plugin/release/security-sensitive workflows, use Hermes Harness rules: preflight, prompt-accepted/progress evidence, lifecycle state, bounded recovery, and verified closeout.
- For ordinary low-risk tasks, do not over-apply the full harness; use read-only inventory → minimal execution → verification.
- For external agent/design/harness sources, distill clean-room concepts only unless license/provenance explicitly allows reuse. Do not vendor code, scripts, prompts, assets, style recipes, or unknown ZIPs.

## High-value files

- `run_agent.py` — AIAgent conversation loop, plugin lifecycle hooks, HASOS runtime guidance integration.
- `agent/prompt_builder.py` — system prompt, memory/skills/project-context assembly, HASOS runtime guidance.
- `model_tools.py`, `toolsets.py`, `tools/registry.py` — tool discovery, dispatch, toolset membership, tool hook/gate surfaces.
- `hermes_cli/config.py` — default config and migrations.
- `providers/`, `plugins/model-providers/` — v0.13 provider plugin surface.
- `gateway/` — messaging gateway, restart resume, allowlists, platform adapters.
- `cron/` — cron and no_agent watchdog mode.
- `hermes_cli/kanban.py`, `plugins/kanban/` — durable multi-agent Kanban.
- `tests/` — pytest suite; prefer targeted tests first.
- `docs/agents-developer-guide-full.md` — preserved detailed upstream AGENTS developer guide.

## TUI in the Dashboard (`hermes dashboard` → `/chat`)

The dashboard embeds the real `hermes --tui` — not a rewrite. See `hermes_cli/pty_bridge.py` and the `/api/pty` endpoint in `hermes_cli/web_server.py`.

- Browser loads `web/src/pages/ChatPage.tsx`, which mounts xterm.js with the WebGL renderer, fit addon, and unicode11 addon.
- `/api/pty?token=…` upgrades to a WebSocket; auth uses the same ephemeral `_SESSION_TOKEN` as REST via query param.
- The server spawns the TUI through `ptyprocess`; frames are raw PTY bytes each direction; resize is applied on the server.
- Do not re-implement the primary chat transcript or composer in React. Extend Ink so dashboard improvements inherit the same behavior.
- Structured React UI around the TUI is allowed for supporting views such as sidebars, inspectors, summaries, and status panels.

## Standard verification

For prompt/context/runtime changes:

```bash
venv/bin/python3.11 -m py_compile agent/prompt_builder.py run_agent.py model_tools.py toolsets.py
venv/bin/python3.11 -m pytest tests/agent/test_prompt_builder.py tests/run_agent/test_run_agent.py -q -o addopts=
```

For broader source changes, expand to the affected module tests or `scripts/run_tests.sh` before any commit or release.

## Closeout expectation

Report what changed, evidence/tests run, unverified residual risk, and whether wiki/skill updates were needed. If a reusable workflow changed, update the relevant wiki/skill rather than relying on chat memory.
