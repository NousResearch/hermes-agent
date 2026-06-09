# Hermes routing matrix and spend policy (M3)

> Read-only artifact. Source: `auto-coder` `config.yaml` + M2b receipt `20260609-171951`.  
> No `config.yaml` edits in this milestone.

**Machine-readable:** `routing-matrix.yaml`  
**Smoke (no network):** `scripts/routing_smoke.py --no-network`

## Runtime truth summary

| Claim | Tag | Evidence |
|-------|-----|----------|
| Default interactive model `grok-composer-2.5-fast` / `xai-oauth` | **Config-only** | M2b G1 |
| Smart routing lanes (cheap / routine / hard) configured | **Config-only** | M2b `smart_model_routing_lanes` |
| Smart routing **classifier importable** from installed `hermes-agent` main tree | **Verified false** | No `hermes_cli/smart_routing.py` on `main` checkout; module exists only in local `.worktrees/*` |
| Gateway owner `auto-coder` + unit running | **Verified** | M2b G6 (`hermes-gateway-auto-coder.service`, PID active) |
| Delegation child defaults | **Config-only** | `delegation:` block in config |

**Blocker for live smart lanes:** Config has `smart_model_routing.enabled: true` but the **running** gateway/CLI checkout (`/home/khall/.hermes/hermes-agent`) does not ship `hermes_cli/smart_routing.py` on `main`. Until that module is merged/reinstalled, turns fail closed to the primary model per `build_turn_route` semantics.

## Lane matrix

| Lane | Provider / model | Cost tier | Toolsets (intent) | Network | Workdir | Timeout / budget | Fallback | Smoke (no-network) | Escalation |
|------|------------------|-----------|-------------------|---------|---------|------------------|----------|-------------------|------------|
| interactive_default | xai-oauth / grok-composer-2.5-fast | high (frontier) | CLI platform toolsets (browser, delegation, terminal, web, …) | yes | session cwd | agent.max_turns 240 | fallback_providers [] | config parse | user approval for destructive |
| smart_cheap | openai-codex / gpt-5.4-mini | low | subset of interactive | yes* | same | per-turn | primary model | `classify_turn("ping")` → cheap* | *requires runtime module |
| smart_routine_coding | openai-codex / gpt-5.3-codex | medium | coding toolsets | yes* | worktrees | per-turn | primary | `classify_turn("add tests for foo")` | *requires runtime module |
| smart_hard | openai-codex / gpt-5.5 | high | full delegation | yes* | same | reasoning xhigh | primary | `classify_turn("architecture security review")` → hard* | *requires runtime module |
| delegation_worker | openai-codex / gpt-5.5 | high | default terminal+file; inherit MCP | yes | isolated | child_timeout 600s; max_iter 240 | parent model | preflight credential shape | block lane if worker auth missing |
| reviewer | openai-codex or opus (manual) | medium | Read, Bash only | optional | PR worktree | max_turns low | — | diff-only review prompt | no auto-merge |
| setup_audit | n/a (local script) | none | subprocess hermes CLI | local | profile home | per-probe timeout | skip gate | `setup_gate_ladder_audit.py` | approval per live probe |
| docs_watcher | n/a (planned M6) | none | curl/hash only | fetch public docs | n/a | cron gated | — | hash unchanged → silent | cron create needs approval |
| gateway_owner | xai-oauth (gateway profile) | always-on | messaging + MCP servers | yes | auto-coder | gateway_timeout 1800 | — | `hermes gateway status` | **no restart w/o approval** |
| cron_steward | varies per job | low–medium | job toolsets | yes for watchers | profile scripts | schedule-bound | no recursive cron | `hermes cron list` | job edit needs approval |

## Spend and runaway policy

1. **Do not raise** `delegation.child_timeout_seconds` to rescue broad workers; fix scope and receipts first (roadmap pitfall #5).
2. **Cap parallelism:** `delegation.max_concurrent_children: 3`, `max_spawn_depth: 3`.
3. **No auto-approve subagents:** `delegation.subagent_auto_approve: false`.
4. **Smart routing:** Prefer no-network `routing_smoke.py` before any live Codex lane ping.
5. **Cron:** Classify jobs with scripts + skills as mutation-capable; `no_agent` script jobs still need delivery review.
6. **Gateway:** Single owner (`auto-coder`); service definition drift → repair plan only (M5).

## Verification commands (M3 done)

```bash
# No config mutation
git -C /home/khall/.hermes/.worktrees/profile-list-fast-alias-scan diff --name-only | grep -E 'config\\.yaml|\\.env' && echo FAIL || echo OK

# Classifier smoke (uses smart_routing worktree if main lacks module)
HERMES_HOME=/home/khall/.hermes/profiles/auto-coder \
/home/khall/.hermes/hermes-agent/venv/bin/python \
  /home/khall/.hermes/.worktrees/profile-list-fast-alias-scan/scripts/routing_smoke.py \
  --no-network --hermes-home /home/khall/.hermes/profiles/auto-coder
```

## Next approval gates

- **Live provider smoke** per lane (spend) — separate operator approval
- **Merge smart_routing** into authoritative checkout before treating smart_* lanes as Verified
- **M4** worker registry proposal (parallel read-only)