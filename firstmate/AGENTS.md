# firstmate Fleet — AGENTS.md

## Layer 1: The Liaison (Pi Coding Agent)

**Identity:** `firstmate` profile  
**Runtime:** Docker container `firstmate-liaison`  
**GUI:** Obsidian Local Sidekick plugin on port `3000`

Pi runs the primary interactive session. It connects to the Obsidian vault
(configured via `OBSIDIAN_VAULT_PATH`) and exposes a telemetry/control GUI via
the Local Sidekick plugin.

### Spawn command for Layer 2

```bash
hermes --profile orchestrator \
       --worktree "$HERMES_HOME/.." \
       --load-profiles-from "$HERMES_HOME/repo/profiles" \
       "spawn orchestrator for task: ${TASK_DESCRIPTION}"
```

This explicitly loads the existing specialist profiles from the synced vault
so the orchestrator can route to `coder`, `analyst`, `researcher`, `creative`,
`vision`, `long-context`, `scout`, `deep-thinker`, `thinker`, `messaging-bots`,
etc.

## Layer 2: The Orchestrator (Hermes Agent)

**Identity:** `orchestrator` profile  
**Runtime:** Docker container `hermes-orchestrator` (spawned by Pi)  
**Reasoning engine:** LangGraph / LangChain cyclic graph

The orchestrator:
1. Reads the task from the Obsidian vault `input.md`.
2. Parses OS requirements (`mac`, `windows`, `any`).
3. For agnostic tasks, dispatches concurrently to isolated git worktrees on
   both Mac and Windows execution nodes.
4. For OS-specific tasks, routes only to the matching fleet node.
5. Logs state to TencentDB and writes status back to Obsidian.

### Native profile routing

- `code|debug|refactor|test|pytest|git|patch` → `coder`
- `analyze|compare|evaluate|benchmark|metrics` → `analyst`
- `research|investigate|find|search|arxiv|summarize` → `researcher`
- `write|draft|blog|creative|illustration|story` → `creative`
- `architecture|design|plan|orchestrate|coordinate|delegate` → `orchestrator`
- `vision|image|screenshot|diagram|ocr` → `vision`
- `long context|large file|1m context` → `long-context`
- `quick|scout|short|one-liner|lookup|brief` → `scout`
- `deep think|philosophy|complex reason` → `deep-thinker`
- `reasoning|logical proof|math|philosophical|derive|prove|thought experiment` → `thinker`
- `chat|message|bot|telegram|discord` → `messaging-bots`
- unmatched → `orchestrator`

## Layer 3: The Execution Fleet (OpenCode)

**Identity:** `opencode` worker  
**Runtime:** Docker container `opencode-worker`  
**Sandbox:** `/workspace` volume with isolated git worktrees

OpenCode receives sub-tasks from the orchestrator and executes localized
changes inside Docker sandboxes. Each task gets its own worktree to avoid
contaminating the main checkout.

### Worktree dispatch

```bash
# Mac worktree
GIT_WORK_TREE=/workspace/mac-$(uuidgen) \
  git -C "$HERMES_HOME/repo" worktree add "$GIT_WORK_TREE" main

# Windows worktree (mounted from a Windows node via SMB or prepared locally)
GIT_WORK_TREE=/workspace/win-$(uuidgen) \
  git -C "$HERMES_HOME/repo" worktree add "$GIT_WORK_TREE" main
```

For OS-agnostic tasks, both worktrees are created and the sub-agent runs
concurrently. Results are merged by the orchestrator and persisted to the
Obsidian dashboard.

## Database environment injection

Run these once before `docker compose up`:

```bash
# Supabase (Auth + Relational)
export SUPABASE_URL="https://<project>.supabase.co"
export SUPABASE_KEY="<service-role-key>"
export SUPABASE_DB_HOST="db.<project>.supabase.co"
export SUPABASE_DB_PORT="5432"
export SUPABASE_DB_NAME="postgres"
export SUPABASE_DB_USER="postgres"
export SUPABASE_DB_PASSWORD="<db-password>"

# TencentDB (State persistence)
export TENCENTDB_HOST="<tencentdb-host>"
export TENCENTDB_PORT="3306"
export TENCENTDB_USER="<user>"
export TENCENTDB_PASSWORD="<password>"
export TENCENTDB_DATABASE="hermes_state"
```

Persist them in `$HERMES_HOME/repo/nodes/node4/.env`
and the Docker services will load them via `env_file`.

## Plugin synergy

| Plugin | Container | Role |
|--------|-----------|------|
| Graphify | `firstmate-graphify` | Knowledge graph indexing of the vault + repo |
| SkillOpt | `firstmate-skillopt` (built into orchestrator image) | Nightly self-evolution of skills |
| OpenViking | `firstmate-openviking` | Tiered semantic context storage |
| n8n | `firstmate-n8n` | Webhook handler for Docker build statuses |

## n8n webhook route for build status

POST `http://host.docker.internal:5678/webhook/docker-build-status`

Payload:

```json
{
  "image": "hermes-firstmate",
  "tag": "latest",
  "status": "success|failed",
  "log_url": "https://...",
  "node": "mac|windows|linux"
}
```

The workflow stores the event in Supabase and forwards a summary to the
Obsidian dashboard via the Local Sidekick plugin.
