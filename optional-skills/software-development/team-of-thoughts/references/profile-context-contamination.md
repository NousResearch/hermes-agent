# Profile Context Contamination

> How profile `environment_hint` / `terminal.cwd` settings cause debate agents to analyze the wrong project, and how to prevent it.

## The problem

Some Hermes profiles (especially `hephaestus` and `talaria`) carry an `environment_hint` in their `config.yaml` that tells the agent which project it works in. For example:

```yaml
# ~/.hermes/profiles/hephaestus/config.yaml
agent:
  environment_hint: |
    DELEGATION RULE — ABSOLUTE AND NON-NEGOTIABLE:
    You work in /path/to/project.
```

When the kanban dispatcher spawns a subagent in this profile, the `environment_hint` injects the wrong project path into the agent's system prompt. The agent reads files from the wrong project. The `clio` profile typically has no such hint and works correctly.

## Symptoms

| Symptom | Looks like | Real issue |
|---------|-----------|------------|
| Wrong stack mentioned | "Reviewed Hono + tRPC + Drizzle" | Target project uses Go + sqlx + PostgreSQL |
| Wrong repo output | Execution cards with wrong `workspace: dir @` path | Agent thinks that is the current project |
| "No X code exists" | "No <project> code exists in the repo" | Agent is looking at the wrong repo — the code actually exists |
| Wrong framework bugs | "Add toast notifications for tRPC" | Target project uses REST/axios, not tRPC |

## Detection

Before dispatching debate agents, check which profiles have environment hints:

```bash
grep -l "environment_hint" ~/.hermes/profiles/*/config.yaml
```

For each profile found, check what project the hint points to:

```bash
grep -A5 "environment_hint" ~/.hermes/profiles/hephaestus/config.yaml
grep -A5 "environment_hint" ~/.hermes/profiles/talaria/config.yaml
```

If the hint points to a different project than the one being debated, you need the override.

## Fix

Override the profile's `environment_hint` by embedding explicit project instructions in the child card body:

```markdown
**IMPORTANT — WORK IN THIS PROJECT:** /home/user/projects/correct-project
**DO NOT** work in /other/project — that is a different project.

**ACTIONS:**
1. ... (all actions scoped to the correct project)
```

Place this at the very top of the card `--body`, before any role-specific instructions. The instruction overrides whatever `environment_hint` the profile injected.

## Which profiles are affected

| Profile | Usually affected? | Why |
|---------|-----------------|-----|
| `clio` | Rarely | Research profile, no delegation-rule hints; works on whatever the card tells it to |
| `hephaestus` | Often | Orchestrator profile with project-specific delegation rules in `environment_hint` |
| `talaria` | Often | Executor profile with project-specific delegation rules in `environment_hint` |

## Permanent fix

For dedicated single-project setups, update the profile's `environment_hint` to match the correct project:

```bash
hermes config set --profile hephaestus agent.environment_hint "DELEGATION RULE ..."
hermes config set --profile talaria agent.environment_hint "... same ..."
```

For multi-project setups, the card-body override approach is the only option.
