# Worker profile registry (M4 proposal)

> **Proposal only** — generated before any profile/config edit.  
> Evidence: live `hermes profile list`, per-profile `config.yaml` model/provider lines, `.env` presence (not contents), M2b receipt `20260609-171951`, M3 `routing-matrix.yaml`.

> **Runtime (2026-06-09):** Hermes **dynamic workflows** — see `worker-registry-dwf-lanes.md` and `worker-registry.yaml`. **Kanban worker dispatch is off**; `kanban-*` rows are inventory only.

## Operator policy (2026-06-09)

**Coding tasks:** use **Grok Composer 2.5 Fast** (`xai-oauth` / `grok-composer-2.5-fast`) for all coding work — interactive sessions, delegation workers, and Kanban implementation lanes unless you explicitly choose otherwise.

- Treat `smart_model_routing` Codex lanes (cheap / routine / hard) as **off by policy** even if still present in `config.yaml`.
- Kanban worker models in registry are **historical config**; proposed dispatch default for coding is **primary Grok**, not openai-codex lanes, until you approve a registry/config realignment.

Live config edits to disable smart routing or retarget Kanban profiles are **approval-gated** — say if you want that applied to `auto-coder` and/or Kanban profiles.

## Registry

| profile_id | archetype | owner_role | model | provider | .env | gateway | cron_host | mutation_allowed | workdirs | toolsets (intent) | credential_shape | last_smoke | rollback / disable | approval_gates |
|------------|-----------|------------|-------|----------|------|---------|-----------|------------------|----------|-------------------|-------------------|------------|-------------------|------------------|
| auto-coder | operator+gateway | primary interactive + gateway owner | grok-composer-2.5-fast | xai-oauth | present | **running** | **yes** (3 jobs) | worktrees w/ approval for push | `/mnt/c/Dev`, `~/.hermes/hermes-agent` | full CLI platform | xAI OAuth + many keys set (M2b names only) | M2b G6 pass | `hermes gateway stop`; disable unit | push, gateway_restart, cron_edit, profile_edit |
| gateway-minimal | gateway standby | emergency minimal gateway | gpt-5.5 | openai-codex | present | stopped | no | gateway only w/ approval | profile home | messaging minimal | openai-codex shape | not run | keep stopped | gateway_transfer |
| kanban-impl | kanban-worker | implementation | gpt-5.5 | openai-codex | **absent** | stopped | no | worktree edits only | card worktree | terminal,file,kanban | **BLOCK: no .env** | not run | unassign worker | credential_repair, spawn |
| kanban-triage | kanban-worker | triage | gpt-5.4-mini | openai-codex | **absent** | stopped | no | read-mostly | board | kanban,web | **BLOCK** | not run | unassign | credential_repair |
| kanban-review | kanban-worker+reviewer | review | gpt-5.4-mini | openai-codex | present | stopped | no | read-only review | PR paths | file,terminal | codex/oauth shape | not run | unassign | spawn after smoke |
| kanban-pr | kanban-worker | PR author | gpt-5.4-mini | openai-codex | **absent** | stopped | no | worktree+PR w/ approval | worktrees | terminal,file | **BLOCK** | not run | unassign | credential_repair |
| kanban-merge-steward | kanban-worker | merge steward | gpt-5.4-mini | openai-codex | **absent** | stopped | no | merge gated | repo | terminal,kanban | **BLOCK** | not run | unassign | credential_repair |
| kanban-ops-steward | kanban-worker | ops steward | gpt-5.4-mini | openai-codex | **absent** | stopped | no | read-only ops | infra docs | terminal,file | **BLOCK** | not run | unassign | credential_repair |
| builder | coder | build slices | openai/gpt-4o-mini | openrouter | present | stopped | no | worktree | `/mnt/c/Dev` | terminal,file | OPENROUTER set | not run | `hermes profile use auto-coder` | push |
| default | operator legacy | fallback root profile | qwen3.7-max | multi | present | stopped | no | discouraged | `~/.hermes` | broad | mixed | not run | migrate to named profiles | profile_edit |
| hoc-canary | safe-research | canary | gpt-4o-mini | openai | unknown | stopped | no | none | read-only | web | openai | not run | delete profile | profile_delete |
| lcm-test | coder | test | gpt-5.4 | — | unknown | stopped | no | test worktrees | local | terminal | unknown | not run | delete | profile_delete |
| mcplab | safe-research | MCP lab | — | — | unknown | stopped | no | none | local | mcp | unknown | not run | delete | profile_delete |
| worker-minimal | kanban-worker | minimal worker | — | — | unknown | stopped | no | minimal | assigned | kanban-worker toolset | unknown | not run | unassign | verify model first |

## Preflight commands (by archetype)

**kanban-worker (before spawn):**
```bash
PROFILE=kanban-impl
test -f "$HOME/.hermes/profiles/$PROFILE/.env" || echo "BLOCK: missing .env for $PROFILE"
hermes -p "$PROFILE" profile show
# Optional spend-free: read config model.provider only
```

**gateway (before restart):**
```bash
hermes gateway status
# Confirm no other profile gateway running; owner must match proposal
```

**operator (before setup mutation):**
```bash
python scripts/setup_gate_ladder_audit.py --hermes-home ~/.hermes/profiles/auto-coder \
  --pythonpath <M1-worktree> --source-repo ~/.hermes/hermes-agent
```

**reviewer:**
```bash
gh auth status
hermes -p kanban-review profile show
```

## Blockers (proposal — do not dispatch until cleared)

1. **Five Kanban profiles lack `.env`** while using `openai-codex` — high crash-loop risk at spawn.
2. **Single gateway enforced** — only `auto-coder` may run; `gateway-minimal` must remain stopped.
3. **Smart routing** — config enabled but runtime module not on main (M3); workers inherit same limitation until merge.

## Receipt schema (per dispatch)

```json
{
  "profile_id": "kanban-impl",
  "archetype": "kanban-worker",
  "preflight": {"env_present": false, "gateway_running": false},
  "verdict": "blocked",
  "approval_required": ["credential_repair"]
}
```

## Verification (M4 done)

```bash
PYTHONPATH=/home/khall/.hermes/.worktrees/profile-list-fast-alias-scan \
  timeout 5s hermes profile list
# Must list same profile ids as registry table
git -C /home/khall/.hermes/.worktrees/profile-list-fast-alias-scan diff --name-only | grep config.yaml || echo OK_no_config_edits
```

## Next steps (operator approval)

1. Credential repair plan for Kanban workers (backup + copy shape, never paste secrets into chat)
2. Align Kanban board worker assignments with registry blockers
3. M5 gateway unit refresh proposal (separate doc)