# Live Documentation Sources

Structured lookup table for verifying Hermes Agent capabilities when uncertain.

When you're unsure whether a feature exists, how it's configured, or how something works, use this table to find the canonical source of truth — from most authoritative (source code) to quickest (CLI --help).

## Feature Category Reference

| Category | Source Code (ground truth) | CLI Verification | Docs URL |
|----------|---------------------------|------------------|----------|
| **Core agent loop** | `run_agent.py` — `AIAgent.run_conversation()` | `hermes chat -q "test"` | https://hermes-agent.nousresearch.com/docs/guide/agent-loop |
| **CLI commands** | `cli.py` — `HermesCLI.process_command()` + `hermes_cli/commands.py` — `COMMAND_REGISTRY` | `hermes --help` → `hermes <cmd> --help` | https://hermes-agent.nousresearch.com/docs/cli/commands |
| **Tools** | `tools/registry.py` — tool auto-discovery; `tools/*.py` — individual tool impls | `hermes tools list` | https://hermes-agent.nousresearch.com/docs/guide/tools |
| **MCP servers** | `mcp_serve.py` + `tools/mcp/` | `hermes mcp list` (or check `config.yaml` → `mcp_servers`) | https://hermes-agent.nousresearch.com/docs/guide/mcp |
| **Cron/scheduling** | `cron/scheduler.py` — `run_job()`, `cron/jobs.py` | `cronjob list` | https://hermes-agent.nousresearch.com/docs/guide/cron |
| **Gateway (platforms)** | `gateway/run.py` — `Gateway.run()`, `gateway/platforms/*.py` — per-platform adapters | `hermes gateway status` | https://hermes-agent.nousresearch.com/docs/gateway/intro |
| **Memory** | `tools/memory_tool.py` — memory tool handler; `plugins/memory/` — memory backends | `hermes doctor` (checks memory subsystem) | https://hermes-agent.nousresearch.com/docs/guide/memory |
| **Skills** | `hermes_cli/skills.py` — skill loader; `skills/*/SKILL.md` — bundled skills | `skills_list` | https://hermes-agent.nousresearch.com/docs/guide/skills |
| **Config** | `hermes_constants.py` — `get_hermes_home()`, path resolution; config YAML at `~/.hermes/config.yaml` | `hermes config list` or cat `~/.hermes/config.yaml` | https://hermes-agent.nousresearch.com/docs/guide/configuration |
| **Provider/Models** | `plugins/model-providers/` — provider adapters; `agent/model_metadata.py` — model metadata | `/model` (in-chat model picker), `hermes model` | https://hermes-agent.nousresearch.com/docs/guide/providers |
| **Plugins** | `plugins/` — plugin index; each subdir has its own `README.md` or `plugin.py` | `hermes plugins list` | https://hermes-agent.nousresearch.com/docs/plugins/intro |
| **Auth/Credentials** | `agent/credential_pool.py` — credential pool rotation; `agent/auth.py` — OAuth flows | Check `~/.hermes/auth.json` and `~/.hermes/.env` | https://hermes-agent.nousresearch.com/docs/guide/authentication |
| **Profiles** | `hermes_constants.py` — profile-aware path resolution | `hermes -p <name> chat` | https://hermes-agent.nousresearch.com/docs/guide/profiles |
| **Kanban boards** | `hermes_cli/kanban_db.py` — SQLite-backed board DB | `hermes kanban boards list` | https://hermes-agent.nousresearch.com/docs/guide/kanban |
| **Agent/Subagent** | `run_agent.py` — main agent loop; `tools/delegate_task.py` — subagent spawning | `hermes chat` (main), `delegate_task` tool (subagent) | https://hermes-agent.nousresearch.com/docs/guide/subagents |
| **Setup wizard** | `hermes_cli/setup_wizard.py` — interactive config wizard | `hermes setup` | https://hermes-agent.nousresearch.com/docs/guide/setup |
| **Debug/Health** | `hermes_cli/doctor.py` — `hermes doctor` | `hermes doctor` | https://hermes-agent.nousresearch.com/docs/guide/troubleshooting |
| **Plugins: image_gen** | `plugins/image_gen/` — image generation providers | `hermes plugins list` | (plugin-specific docs) |
| **Plugins: observability** | `plugins/observability/` — metrics, traces, logging | `hermes plugins list` | (plugin-specific docs) |
| **TUI** | `ui-tui/src/` (TypeScript/React) + `tui_gateway/` (Python backend) | `hermes --tui` | https://hermes-agent.nousresearch.com/docs/guide/tui |

## Three-Tier Fallback

When uncertain about a capability:

1. **Docs first** — check the Docs URL column above. If the docs are clear, use that.
2. **Source code as ground truth** — Hermes is open-source. Read the actual source at `~/.hermes/hermes-agent/<path>`. The source code never lies.
3. **CLI verification** — run the CLI verification command to confirm behavior at runtime.

## Quick Diagnostic Commands

```bash
hermes doctor          # Full health check
hermes config list     # Show current config
hermes tools list      # Show available tools
hermes plugins list    # Show loaded plugins
hermes --help          # CLI reference
hermes <cmd> --help    # Per-command help
cat ~/.hermes/config.yaml  # Raw config
ls ~/.hermes/hermes-agent/ # Source tree
```
