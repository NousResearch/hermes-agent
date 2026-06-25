# CHANGELOG — integrated branch

This file documents the feature PRs that compose the `integrated` distribution of hermes-agent
(fork at github.com/davidgut1982/hermes-agent, base upstream: NousResearch/hermes-agent).

Last updated: 2026-06-04

---

## Features in integrated

### Multi-Gateway Agent (MGA) support

- **feat/mga-1-agent-profile** — Add `AgentProfile` + ContextVar for per-agent paths; enables profile-aware delegation where different agents can have distinct personalities, toolsets, and model configs.
- **feat/mga-2-session-agent-id** — Thread `agent_id` through session identity and DB schema; makes session storage agent-scoped so parallel agents don't share session history.
- **feat/mga-3-gateway-routing** — Route inbound messages via routes table + select_agent hook; the gateway can now dispatch messages to the correct agent profile based on message source/channel.
- **feat/mga-4-cli-subcommand** — Add `hermes agent` subcommand for multi-agent management; CLI tooling for registering and managing agent registry entries.
- **feat/mga-5-cron-api-propagation** — Propagate `agent_id` through scheduled jobs and deliveries; cron jobs now carry the correct agent context so they use the right profile and config.
- **feat/mga-6-docs-fixes** — Wire api_server adapter to MGA routing + cron/delivery agent_id fixes + namespace prefix guard; completes the MGA integration including a guard that prevents tool namespace collisions across agents.

### Dashboard

- **feat/dashboard-agent-profiles-view** — Agent profiles view in the dashboard; visual inspection of which agent profiles are loaded and their configuration.

### Model routing

- **feat/smart-model-routing** — Smart model routing: cheap model for simple queries, capable model for complex ones; reduces cost by ~10x for status checks, lists, and simple questions.
- **feat/skill-model-routing** — Skills hub uses cheap model for skill selection; skill routing decisions don't need a frontier model.
- **feat/model-switch-improvements** — Model switch improvements; cleaner handling of in-session model switching.
- **feat/intent-fast-path-weather** — Intent fast-path for weather queries; weather delegation goes directly to the weather MCP without a full LLM routing pass.

### Tool search

- **feat/tool-search-hybrid-rerank** — Optional embedding reranker for progressive tool disclosure; hybrid BM25 + semantic search for MCP tool selection, dramatically reduces tool hallucination.

### Providers / reliability

- **feat/deepseek-v4-flash-cache-control** — DeepSeek V4 Flash cache control support; enables prompt caching for the DeepSeek flash tier model.
- **fix/fallback-provider-model-key** — Fix fallback provider model key resolution; ensures the correct API key is used when falling back from local to cloud providers.
- **fix/mcp-toolname-prefix-normalisation** — MCP tool name prefix normalization; strips server-name prefixes so tools resolve correctly regardless of how MCP servers name them.
- **fix/delegate-tool-error-classification** — Delegate tool error classification; errors from delegate calls are properly classified (tool error vs agent error) so retry logic works correctly.
- **fix/live-toolcall-fragment-scrub** — Live tool call fragment scrubbing; removes in-flight tool call JSON from the live display to prevent garbled output.
- **fix/fallback-provider-model-key** — Fallback provider model key fix; correct key lookup in multi-provider config.
- **feat/namespace-prefix-guard** — Namespace prefix guard; prevents tool name collisions when multiple MCP servers expose tools with the same base name.

### Infrastructure (this session — 2026-06-04)

- **Stable/dev sandbox split** — Structural separation: stable install at `/opt/hermes/home/.hermes/hermes-agent` (always on `integrated`, runs live gateway), dev sandbox at `/opt/hermes/dev/hermes-agent` (all hacking, `hermesdev` command, isolated `HERMES_HOME`). Prevents the 2026-06-04 outage class where the deploy tree was left on a feature branch.
- **Branch integrity guard** — `hermes-branch-guard.timer` (cheap check every 15 min) + `--smoke` mode on gateway restart; alerts if stable tree drifts from `integrated`.

---

## Applying this changelog to the stable install

The normal upgrade path:
```bash
# In stable tree (DO NOT checkout branches):
git pull origin integrated
uv pip install -e .
# Then: systemctl restart hermes-gateway  (with authorization)
```

## Deferred (follow-up cards)

- Plugin escape-hatch audit: move core patches into `HERMES_HOME/plugins/`
- Rebase cadence automation: weekly `upgrade-hermes.sh` schedule
- Dashboard venv alignment: repoint `hermes-dashboard.service` at editable install
- Legacy venv cleanup: ~1.3 GB in `/opt/hermes/venv-combined5/6`, `venv-main`, `build-combined6`
