# agentmemory Agent Reference

This is the operational reference for agents using the local `agentmemory`
setup with Hermes and other projects on this machine.

## Mental Model

agentmemory is one local daemon with project-scoped records inside it.

```text
Codex / other agents
  -> MCP + lifecycle hooks
  -> agentmemory REST http://localhost:3111
  -> ~/.agentmemory data store
```

The service is global to the machine. Project scope is metadata derived from the
working directory, usually the git root basename.

For this repo:

```text
/Users/mac-studio/projects/hermes-agent -> hermes-agent
```

That means:

- Global facts and preferences can be searched from any project.
- Project sessions, observations, memories, lessons, slots, and graph nodes are
  tagged with a project key.
- Automatic context injection should prioritize the current project.
- Explicit recall can still search across memory when the query is broad enough.

## Current Expected Runtime

The known-good setup is:

```text
agentmemory binary: /opt/homebrew/bin/agentmemory
agentmemory version: 0.9.27
REST: http://localhost:3111
Streams: ws://localhost:3112
Viewer: http://localhost:3113
iii engine: ~/.agentmemory/bin/iii v0.11.2
local LLM: http://127.0.0.1:8000/v1
model: qwen3-4b-instruct-2507-4bit
embeddings: local
```

Expected enabled flags in `~/.agentmemory/.env`:

```env
AGENTMEMORY_AUTO_COMPRESS=true
AGENTMEMORY_INJECT_CONTEXT=true
CONSOLIDATION_ENABLED=true
GRAPH_EXTRACTION_ENABLED=true
AGENTMEMORY_REFLECT=true
AGENTMEMORY_SLOTS=true
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_MODEL=qwen3-4b-instruct-2507-4bit
EMBEDDING_PROVIDER=local
```

Codex wiring:

```text
~/.codex/config.toml      -> mcp_servers.agentmemory
~/.codex/hooks.json       -> SessionStart/UserPromptSubmit/PreToolUse/PostToolUse/Stop
~/.agents/skills          -> agentmemory skills, recall/remember/handoff/etc.
~/.zshenv                 -> ~/.agentmemory/bin on PATH for non-interactive shells
```

Hook commands must pass the runtime env explicitly:

```text
AGENTMEMORY_URL=http://localhost:3111
AGENTMEMORY_INJECT_CONTEXT=true
```

Do not rely on `~/.agentmemory/.env` alone for hook process env. The daemon reads
that file, but the hook script process only sees its own environment.

## Onboarding Script

Use the repo helper:

```bash
scripts/agentmemory_onboard.py --local-omlx
```

From any project:

```bash
/Users/mac-studio/projects/hermes-agent/scripts/agentmemory_onboard.py --local-omlx --probe
```

Apply fixes and restart the daemon:

```bash
/Users/mac-studio/projects/hermes-agent/scripts/agentmemory_onboard.py \
  --apply --restart --local-omlx --probe
```

Also install/update global skills:

```bash
/Users/mac-studio/projects/hermes-agent/scripts/agentmemory_onboard.py \
  --apply --restart --local-omlx --probe --install-skills
```

What the script checks:

- `agentmemory` binary and version
- REST health
- `agentmemory status` core flags
- pinned `iii` on PATH
- Codex MCP config
- Codex hook env injection
- `AGENTMEMORY_*` feature flags
- local oMLX model endpoint
- slots endpoint
- optional `SessionStart` context injection probe

## New Project Flow

From another repo:

```bash
cd /path/to/project
/Users/mac-studio/projects/hermes-agent/scripts/agentmemory_onboard.py --local-omlx --probe
```

Shortcut from any project after local setup:

```bash
agentmemory-check
```

Hermes also has quick commands for the same check:

```text
/am-check
/agentmemory-check
```

If checks fail:

```bash
/Users/mac-studio/projects/hermes-agent/scripts/agentmemory_onboard.py \
  --apply --restart --local-omlx --probe
```

Start a new agent session from that project directory. The hook should register
the session with the project key derived from that repo.

On a brand-new project, context can be sparse. That is not a failure. The
important signs are:

- checks pass
- a session appears for the project
- future prompts/tool results create observations
- recall finds project-specific facts after memories exist

The probe validates registration first. Empty injected context is acceptable
when the project has no useful memories yet. To require non-empty injected
context while debugging known-memory projects, add `--strict-probe`.

## How Agents Should Use Memory

Use memory automatically when available, especially for:

- project setup and prior decisions
- known local services and ports
- unresolved work from prior sessions
- user preferences
- previous failures and fixes
- file history before editing high-risk files

Recommended tools:

```text
memory_recall          Search prior memories/observations.
memory_smart_search    Hybrid search with compact results.
memory_sessions        See recent sessions and observation counts.
memory_file_history    Check past observations about files.
memory_save            Save durable facts, decisions, workflows.
memory_lesson_save     Save lessons that should strengthen/decay.
memory_graph_query     Inspect knowledge graph entities/edges.
memory_reflect         Synthesize higher-order insights when corpus supports it.
memory_slot_list       List editable pinned slots.
memory_slot_get        Read one slot.
memory_slot_replace    Update a slot.
```

Prefer scoped, durable memory for facts that should survive:

```text
Remember this for this project: <decision or setup fact>
```

Use global memory for machine-wide facts or durable user preferences:

```text
Remember globally: <preference or local machine convention>
```

## Lifecycle Hook Flow

Expected hook behavior:

1. `SessionStart` registers a session and can inject project context.
2. `UserPromptSubmit` records the user prompt.
3. `PreToolUse` can inject file/tool-specific context.
4. `PostToolUse` records tool outputs and can trigger compression.
5. `Stop` summarizes and closes the session.

The context injection probe returns a block like this when matching memories are
available:

```xml
<agentmemory-context project="hermes-agent">
...
</agentmemory-context>
```

If the project is new and has no useful memory yet, the hook can register the
session and return no injected context. That is still healthy unless
`--strict-probe` is being used.

## Troubleshooting

### Basic Health

```bash
agentmemory status
agentmemory doctor --dry-run
```

Healthy output should show:

```text
Health: healthy
Provider: llm
Embeddings: embeddings
GRAPH_EXTRACTION_ENABLED
CONSOLIDATION_ENABLED
AGENTMEMORY_AUTO_COMPRESS
AGENTMEMORY_INJECT_CONTEXT
```

`doctor --dry-run` should say there are no fixes to run.

### Onboarding Script Fails

Run:

```bash
scripts/agentmemory_onboard.py --local-omlx --probe
```

Common failures:

| Failure | Meaning | Fix |
|---|---|---|
| `server health` failed | daemon is not reachable | `scripts/agentmemory_onboard.py --apply --restart --local-omlx` |
| `iii on PATH` failed | pinned iii is not visible to non-interactive shells | ensure `~/.agentmemory/bin` is in `~/.zshenv` |
| `Codex MCP` failed | MCP server is not wired | `agentmemory connect codex` |
| `Codex hooks` failed | hooks missing or lack injection env | run onboarding script with `--apply` |
| `slots endpoint` disabled | `AGENTMEMORY_SLOTS` unset | run onboarding script with `--apply --restart` |
| `local oMLX` failed | local OpenAI-compatible server is down | start `omlx serve` |
| `SessionStart probe` failed | hook did not run, session was not registered, project key mismatched, or strict context was required | check hooks, daemon health, project root, or run without `--strict-probe` for sparse projects |

### `memory_slot_list` Returns 500

This usually means slots are disabled, not that memory is generally broken.

Check direct REST:

```bash
curl -fsS http://localhost:3111/agentmemory/slots
```

If it returns disabled/503, set:

```env
AGENTMEMORY_SLOTS=true
```

Then restart:

```bash
scripts/agentmemory_onboard.py --apply --restart --local-omlx
```

Verify:

```bash
scripts/agentmemory_onboard.py --local-omlx
```

### Context Injection Is Enabled but No Context Appears

Check the hook commands:

```bash
rg -n 'AGENTMEMORY_INJECT_CONTEXT|session-start|pre-tool-use' ~/.codex/hooks.json
```

Every agentmemory hook command should include:

```text
env AGENTMEMORY_URL=http://localhost:3111 AGENTMEMORY_INJECT_CONTEXT=true
```

Then run:

```bash
scripts/agentmemory_onboard.py --local-omlx --probe
```

If that passes with `registered session ...; no context injected`, the hook is
working but the project has no useful context to inject yet. If you expected
context, require it explicitly:

```bash
scripts/agentmemory_onboard.py --local-omlx --probe --strict-probe
```

If strict mode fails because the project has no memories, save one and retry:

```text
Remember this for this project: <important setup fact>
```

### Graph Is Empty

Check:

```bash
agentmemory status
```

If graph shows `0 nodes, 0 edges`, either the project has no compressed memories
yet or graph extraction has not run. Run normal sessions, save useful memories,
and verify:

```bash
scripts/agentmemory_onboard.py --local-omlx
```

For a small known corpus, graph extraction can be triggered through the REST API,
but prefer normal hook/session flow unless repairing setup.

### Reflection Produces No Insights

`AGENTMEMORY_REFLECT=true` enables reflection, but reflection can still return no
insights when the corpus is too small or not clustered enough. That is expected.

Confirm the feature is enabled and slots are on:

```bash
rg -n 'AGENTMEMORY_REFLECT|AGENTMEMORY_SLOTS' ~/.agentmemory/.env
```

Then keep using memory normally. Reflection becomes useful after enough sessions,
observations, and graph clusters exist.

### Project Scope Looks Wrong

Project scope is usually the git root basename. From the project:

```bash
git rev-parse --show-toplevel
basename "$(git rev-parse --show-toplevel)"
```

If two projects share the same basename, set an explicit project name for the
agent process:

```bash
AGENTMEMORY_PROJECT_NAME=my-unique-project codex
```

Use stable slugs, not absolute paths, for project identifiers.

### Repo Got Dirtied by Skill Install

Project-level `skills add` can create `.agents/`, `skills-lock.json`, and skill
directories. For this setup, global skills are preferred.

Check:

```bash
git status --short
```

If only agentmemory skill install artifacts were created in the repo, remove
them and reinstall globally:

```bash
npx -y skills add rohitg00/agentmemory --all -g
```

Do not delete unrelated user changes.

## Quick Commands

```bash
# Full read-only check for this machine's expected local setup
scripts/agentmemory_onboard.py --local-omlx

# Repair known config drift and restart
scripts/agentmemory_onboard.py --apply --restart --local-omlx --probe

# Status from agentmemory itself
agentmemory status
agentmemory doctor --dry-run

# Check slots
curl -fsS http://localhost:3111/agentmemory/slots

# Check local oMLX
curl -fsS http://127.0.0.1:8000/v1/models

# Check Codex hook env
rg -n 'AGENTMEMORY_INJECT_CONTEXT|agentmemory/plugin/scripts' ~/.codex/hooks.json
```
