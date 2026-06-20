---
name: register-flow-as-mcp
description: "Use when exposing/registering a Langflow flow as a callable MCP tool for EasyHermes (or hiding one again). Covers the hard ChatInput+ChatOutput rule, the list_flows → expose_flow_as_tool contract, tool naming, role authorization (feature ③), and pitfalls (read-only starter flows → 404, non-chat flows rejected, session caching, dev-side langflow API path)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [mcp, langflow, workflow, flow, tool, registration, easyhermes, copilot]
    related_skills: [hermes-agent-skill-authoring]
---

# Register a Langflow Flow as an MCP Tool

## Overview

EasyHermes turns a **Langflow flow**(产品里叫「工作流」)into a callable **MCP tool** so the agent — and, by authorization, sub‑accounts — can invoke it. The whole contract is two native agent tools, `list_flows` and `expose_flow_as_tool`, plus one **hard rule**:

> **Only a「ChatInput + ChatOutput」one‑in‑one‑out conversational flow can become a tool.** It maps cleanly to `{message} → string` (the Chat Input is the tool's text argument, the Chat Output is its text result). Anything else — multi‑input, file input, structured output, no chat entry/exit — is **not registerable**.

This skill is the canonical procedure so neither the copilot nor a developer has to re‑derive it each time. It directly realizes feature ③ "copilot 自主创作工作流 + 协助注册成 MCP + 注册即问哪个角色可用".

## When to Use

- The user (or you, the copilot) built/authored a flow and wants it callable as a tool: "把这个 flow 注册成 MCP / 暴露成工具 / expose as tool / register as MCP".
- You want to **hide** a previously‑exposed flow (`enabled=false`).
- You're checking what's already exposed (`list_flows`).

**Don't use for:**
- Building the flow itself — that's the workflow canvas / `kari_canvas`. This skill assumes the flow exists (or you create a minimal ChatInput→ChatOutput one first, see step 1).
- Granting an already‑registered tool to a role — that's the **权限管理** panel (step 4 here points you there, but the grant action is UI/`/api/kari/grants`, not this tool).
- Non‑conversational flows — they cannot be tools; do not try to force them.

## The Hard Rule (shared judge)

The ChatInput+ChatOutput check lives in **one** place: `tools/flow_chat.py::is_chat_flow` (reads each `node.data.type`, falls back to the node‑id prefix like `ChatInput-AbC12`). Both the **registration gate** (`expose_flow_as_tool`) and **resource collection** (`org_client.gather_workflow_resources`) import it, so "registerable" and "appears in the authorization panel" never drift. If you change the rule, change it there — never fork the判据.

## Procedure

### 1. Ensure the flow is ChatInput + ChatOutput

Build it in the workflow canvas (or `kari_canvas`): exactly **one Chat Input** → (any middle nodes) → **one Chat Output**. When the copilot authors a flow it should **default to ChatInput + ChatOutput** so the result is registerable by construction.

Minimal demo flow = a Chat Input wired straight into a Chat Output (echoes input → output). Good for proving the pipeline end‑to‑end.

### 2. Find the flow id — `list_flows`

```json
// tool: list_flows   args: { "query": "demo" }   // query is an optional name substring
// → { "count": 1, "flows": [ { "id": "<uuid>", "name": "客服问答 Demo",
//      "mcp_enabled": false, "action_name": null } ] }
```

Note: the list does **not** return `user_id`, so you can't tell an owned flow from a read‑only **starter example** here — the PATCH in step 3 surfaces that (404, see pitfalls).

### 3. Register — `expose_flow_as_tool`

```json
// tool: expose_flow_as_tool
// args: {
//   "flow_id": "<uuid>",                  // or "flow_name": "<exact name>"
//   "action_name": "chatin_chatout_demo", // snake_case; the tool name the agent calls
//   "action_description": "Echo the chat input back as the chat output.",
//   "enabled": true                       // true = expose (default), false = hide
// }
// → { "success": true, "flow_id": "...", "name": "...", "mcp_enabled": true,
//     "action_name": "chatin_chatout_demo", "tool_refresh": "refreshed", "message": "..." }
```

What it does: gets a keyless `auto_login` token → (if enabling) re‑fetches the flow and **rejects it unless `is_chat_flow`** → `PATCH /api/v1/flows/{id}` with `{mcp_enabled, action_name, action_description}` → **refreshes the in‑process `kari_flows` MCP connection** so the tool appears in the *current* session immediately. `tool_refresh: "refreshed"` confirms the live tool list picked it up.

Hiding: `enabled=false` sets `mcp_enabled=false` and clears `action_name`/`action_description`.

### 4. Authorize — ask which role can use it (feature ③)

Registering makes the flow a tool for **this** agent. For **sub‑accounts** to use it, it must be granted to a role:

- After exposing, **ask 谁 / 哪个角色可以用** — authorization is front‑loaded into registration, not bolted on later.
- The flow now appears in **团队账号 → 权限管理** (the MCP‑only授权面板) as a grantable `工作流`. Tick it for the roles allowed to use it (writes主本地 `grant_policy`).
- Cross‑node **runtime** enforcement (a sub actually invoking it up the tree, gated by the grant) is a separate layer (3b / the MCP‑authorization plan); registration + grant is the authoring half.

## Naming Conventions

- **action_name**: snake_case, stable, descriptive — it's the literal tool name the agent calls. `customer_qa`, `chatin_chatout_demo`. Defaults to the flow name if omitted — a Chinese/spaced flow name makes a poor tool name, so **set it explicitly**.
- **action_description**: one line, what it does + what it returns. Strongly recommended — the agent uses it to decide when to call the tool.

## Common Pitfalls

1. **Non‑chat flow → rejected** ("该 flow 不是对话流"). Fix: add exactly one Chat Input and one Chat Output, then retry.
2. **Read‑only starter example → PATCH 404** ("该 flow 不属于当前用户"). Starter/template flows aren't editable. Fix: 新建或「另存为」成自己的 flow, then expose that.
3. **Omitting `action_name`** → tool name falls back to the flow name (spaces/Chinese → bad tool id). Always pass snake_case `action_name`.
4. **Expecting it in a brand‑new session instantly** — `expose_flow_as_tool` refreshes `kari_flows` in the *current* process; a separately‑started agent picks it up on its own connect. `tool_refresh: "skipped (kari_flows not connected)"` means no live MCP connection to refresh (the PATCH still persisted).
5. **Registering ≠ authorizing.** Exposing makes the tool available to *this* agent; sub‑accounts still need the role grant in step 4. Don't stop at step 3 when the intent is "let role X use it".
6. **Dev‑side (Claude Code on the repo)**: `list_flows`/`expose_flow_as_tool` are **EasyHermes agent tools**, not Claude Code tools. To register during repo work, replicate the mechanism via the langflow API: `GET /api/v1/auto_login` for a token (the flows API returns **403 without it**, unlike knowledge_bases), then `PATCH /api/v1/flows/{id}` with `{"mcp_enabled": true, "action_name": "...", "action_description": "..."}`. Same effect, minus the in‑process refresh.

## Verification Checklist

- [ ] Flow has exactly one **Chat Input** and one **Chat Output** (`is_chat_flow` true)
- [ ] `list_flows` shows the flow id; `mcp_enabled` flips to **true** after expose
- [ ] `action_name` is snake_case and `action_description` is set
- [ ] Response `tool_refresh` is `"refreshed"` (or PATCH persisted if not connected)
- [ ] Asked **which role(s)** may use it → ticked in **权限管理**
- [ ] (Hiding) `enabled=false` cleared `action_name`/`action_description`

## One‑Shot Recipe — expose a minimal demo

1. Create a flow named `客服问答 Demo`: Chat Input → Chat Output (echo).
2. `list_flows {query:"客服"}` → grab its `id`.
3. `expose_flow_as_tool { flow_id, action_name:"chatin_chatout_demo", action_description:"Echo the chat input back as the chat output.", enabled:true }`.
4. Confirm `mcp_enabled:true`, `tool_refresh:"refreshed"`.
5. Ask which role uses it → tick `客服问答 Demo` under **权限管理** for that role.
