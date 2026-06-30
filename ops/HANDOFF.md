# Hermes Deployment & Migration — Handoff / Tracking

> Living tracking doc for the Spacebot→Hermes migration + the deployment.
> **No secrets, IPs, or identifiers here on purpose** — those live in `ops/INFRA.local.md` (gitignored, local-only). This file references it where specifics are needed.

## TL;DR
- Spacebot's upstream maintainer is abandoning it → the prod deployment was migrated to **Hermes Agent** (NousResearch). Prod is LIVE on Hermes with Microsoft Teams working.
- Spacebot decommissioned (backed up). Upstream PRs left clean.
- **Next big work:** a **SurrealDB memory provider** for Hermes (out-of-tree plugin).

## Why migrate
Spacebot is better-designed (Rust, co-equal multi-agent, in-box SurrealDB vector memory) but **abandoned**; Hermes is future-proof — actively maintained (MIT, fast cadence), broad channels incl. Teams, pluggable everything. For a prod dependency, maintained + extensible beats elegant + abandoned. Note: Hermes's "self-improving" framing is mostly scaffolding (frozen base model + curated prompt + FTS keyword search + LLM-authored skills); **real semantic memory = the pluggable provider slot** → that is where the SurrealDB work goes.

## Prod deployment
- Proxmox LXC (Debian 13), reached over a **NetBird overlay**; web admin dashboard on the overlay (basic auth). LLM = OpenRouter. Gateway = systemd **user** service `hermes-gateway`. Hermes is 0.x/fast-moving → **pin the version**.
- Host IP / SSH / dashboard URL / credential locations → **`ops/INFRA.local.md`**.

## Teams — LIVE on Hermes
- New Azure bot created via the **Teams CLI** (`@microsoft/teams.cli` `teams app create`) — it uses the **Teams Developer Portal / Bot Framework registration path, with NO Azure Bot Service resource / billing / F0 tier** (that Azure resource is *optional* for a Teams bot; the mandatory parts are the Entra app + a bot registration + endpoint + Teams channel).
- Enabling Teams required: (1) toggle the platform ON in the dashboard Messaging tab, (2) `pip install microsoft-teams-apps aiohttp` into the Hermes venv (lazy dep; the gateway logs the exact command). Webhook path `/api/messages`. Allowlist locked to the operator; allow-all off.
- Public chain: Azure bot → public endpoint → **cloudflared** tunnel → `localhost:3978` (Hermes webhook). Verified live (real Teams message → reply).
- Teams app: CLI-created; manifest enriched (real descriptions, accentColor) + **SHODAN-style icons** (color 192 + outline 32).
- Bot / tenant / endpoint / allowlist identifiers → **`ops/INFRA.local.md`**.

## Spacebot decommission + upstream
- `spacebot.service` stopped + disabled; data backed up on the LXC (paths in `ops/INFRA.local.md`). Binary + data dir still on disk (full `rm` pending).
- Upstream PRs (spacedriveapp/spacebot), clean/pushed: **#605** (DM `"*"` wildcard), **#607** (Teams adapter backend + `broadcast` serviceUrl routing-key fix), **#608** (Teams Channels UI + app-package generator), **#609** (`find_by_name` case-insensitive channel-id match — generic bug exposed by Teams' mixed-case ids).

## Dev setup (for the SurrealDB provider)
- Hermes **forked → this fork**; cloned locally (path in `ops/INFRA.local.md`). **MemoryProvider ABC:** `agent/memory_provider.py` (+ `hermes_cli/memory_providers.py`).
- **Spacebot work archived** — designs, deep research, plans, and the production SurrealDB impl to port — at **`ops/archive.local/`** (gitignored, local-only). See its `README.local.md` for the relevance map (`surreal_store.rs`, `design.md`, `research-llm-memory-feasibility.md`, the Rust spike). Preserved before Spacebot is deleted.

## NEXT — SurrealDB memory provider
- **Out-of-tree plugin** — Hermes `AGENTS.md` has a "no new in-tree memory providers" policy. Ship as a standalone repo installed into `~/.hermes/plugins/` OR a pip package exposing the `hermes_agent.plugins` entry-point. **Not a fork edit.**
- Implement the **`MemoryProvider` ABC**: required = `name`, `is_available` (NO network calls), `initialize`, `get_tool_schemas`; key hooks = **`prefetch(query)`** (recall before each turn — fast) + **`sync_turn(user, assistant)`** (persist — non-blocking, daemon thread). Optional: `system_prompt_block`, `on_memory_write`, config-schema hooks, `shutdown`, `backup_paths`.
- **Provider owns embeddings + vector search** — Hermes hands raw text; SurrealDB native vector search (HNSW/MTREE + KNN) maps cleanly. **Reuse Spacebot's SurrealDB code** (schema, vector queries, embedding calls). Model on the **`plugins/memory/holographic`** provider (local DB + self-managed retrieval) + **`mem0`** (config/tool-schema shape). ~400–600 lines.
- External providers **AUGMENT** the built-in SQLite/FTS5 memory (one external active via `memory.provider`).

## Gotchas
- The **dev VM also runs the operator's live `tracecat` SOAR** (Temporal) → **throttle heavy builds** (`nice -n 19 ionice -c3`, `systemd-run CPUQuota=…`) or build elsewhere; never `docker system prune`.
- **Teams client cache:** swapping/re-uploading a Teams app → the desktop client shows the OLD app until a full client restart; admin-portal delete ≠ client uninstall; one bot App ID ↔ one Teams app.
- **Per-user durable memory isolation** is undocumented in Hermes (sessions isolate per user; durable memory/profile is owner-centric) — validate before exposing to other end-users.
- **An LLM does not know its own internals** — Hermes's agent confabulated its memory file paths/format/limits; verify against code/config, never the agent's self-description.
