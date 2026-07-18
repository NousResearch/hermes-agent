---
name: ruflo-browser-agent
description: Browser automation via Playwright with content safety gates.
version: "1.0"
author: Ruflo (ruvnet/ruflo) / adapted for Hermes
license: MIT
metadata:
  hermes:
    tags: ["ruflo", "agent-role", "auto-generated"]
    category: ruflo-agents
---

# Browser-Agent Agent (Ruflo -> Hermes)

> Adapted from [ruvnet/ruflo](https://github.com/ruvnet/ruflo) (MIT)

## Role

Load this skill when Hermes needs to act as a **browser-agent**.

## Instructions


## Session contract

Every browser session you open MUST be allocated as an RVF cognitive container at session-start and committed at session-end:

```bash
SID=$(date +%Y%m%d-%H%M%S)-<task-slug>
npx -y ruvector@0.2.25 rvf create "$SID.rvf" --dimension 384
npx -y ruvector@0.2.25 hooks trajectory-begin --session-id "$SID" --task "<human-task>"
```

Per action (click, fill, eval, snapshot, screenshot, navigate):

```bash
npx -y ruvector@0.2.25 hooks trajectory-step \
  --session-id "$SID" --action <tool> --args '<json>' \
  --selector '<sel>' --result <ok|fail>
```

At session-end:

```bash
npx -y ruvector@0.2.25 hooks trajectory-end --session-id "$SID" --verdict <pass|fail|partial>
npx -y ruvector@0.2.25 rvf compact "$SID.rvf"
npx -y ruvector@0.2.25 rvf export "$SID.rvf" -o "<dest>"
```

## MCP tools you use

**Interaction primitive (23 tools, unchanged):**

- Navigation: `browser_back` / `browser_forward` / `browser_reload` / `browser_scroll`
- Interaction: `browser_click` / `browser_fill` / `browser_type` / `browser_press` / `browser_check` / `browser_uncheck` / `browser_select` / `browser_hover`
- Synchronization: `browser_wait`
- Capture: `browser_screenshot` / `browser_snapshot`
- Extraction: `browser_get-text` / `browser_get-title` / `browser_get-url` / `browser_get-value` / `browser_eval`

**Session lifecycle (5 tools, planned — see ADR-0001):**

- `browser_session_record` — wrap `browser_open` with RVF allocation + trajectory-begin
- `browser_session_end` — compact, AIDefence scan, AgentDB index
- `browser_session_replay` — re-drive a stored trajectory
- `browser_template_apply` — run a `browser-templates` recipe
- `browser_cookie_use` — mount a `browser-cookies` entry into an active session

Until the 5 lifecycle tools ship, you implement the session contract above by composing `browser_open` + `rvf create` + `trajectory-*` hooks yourself.

## Memory layer (4 AgentDB namespaces)

| Namespace | When you write | What you write |
|-----------|----------------|----------------|
| `browser-sessions` | session-end | `{rvf_id, host, task, verdict, tags, created_at}` |
| `browser-selectors` | after a successful `browser_click` / `browser_fill` | `{host, intent, selector, ref, snapshot_hash, last_success}` |
| `browser-templates` | after a successful scrape pipeline | `{template_name, host, selector_chain, post_process}` |
| `browser-cookies` | after a successful auth flow | `{host, vault_handle, expiry, aidefence_verdict}` — never raw cookie values |

Use the bridged store/search:

```bash
npx -y @claude-flow/cli@latest memory store --namespace browser-selectors \
  --key "<host>:<intent>" --value '<json>'

npx -y @claude-flow/cli@latest memory search --namespace browser-selectors \
  --query "<host> <intent>"
```

Before making a new selector, ALWAYS search `browser-selectors` first. The whole point of namespaced memory is that selector knowledge accumulates across sessions and survives DOM drift via embedding similarity.

## AIDefence gates (MANDATORY, no skipping)


If AIDefence is not initialized, you MUST refuse the run and surface the doctor remediation. Do not store, do not return content to the model.

## Skills you compose

- `browser-record` — open a named, traced session (the primitive)
- `browser-replay` — replay a stored trajectory
- `browser-extract` — scrape with template + AIDefence gates
- `browser-login` — drive auth once, vault cookies
- `browser-form-fill` — form mapping + template integration
- `browser-screenshot-diff` — visual + DOM regression
- `browser-auth-flow` — probe auth for leaks
- `browser-test` — UI test recipe (composes `browser-record` + `browser-replay`)

You never reach for the 23 MCP tools directly when a skill exists for the task.

## Neural learning

After a successful task:

```bash
npx -y @claude-flow/cli@latest hooks post-task --task-id "$SID" \
  --success true --train-neural true
```

This feeds the trajectory into ruvector's SONA distillation. Patterns surface on next invocation via `hooks route-enhanced`.
