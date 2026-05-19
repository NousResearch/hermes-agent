# Plan 002: Hermes Self-Organization

**Status:** Not started  
**Date:** 2026-05-18  
**Branch:** feat/002-self-organization (to be created)  
**Depends on:** Plan 001-0 (HermesIdentity dataclass must exist before Phase B)  

---

## Context

Hermes currently stores all state in a flat `~/.hermes/` directory with no user namespacing, no runtime isolation between sessions or subagents, no credential scoping, and no MCP connection discipline. Skills, plans, memory, and agent outputs are mixed together and not structurally separable.

This plan canonicalizes the directory layout, makes the runtime safe for multi-user and multi-agent execution, introduces a centrally-managed MCP connection pool (shared process, per-user credentials injected per session), and makes every subtree independently cloud-routable via env vars — without breaking any existing local workflow.

## Key Design Insights

1. **Pooled MCP proxy, not per-session processes.** A central `MCPGateway` process runs each MCP server once. Per-session credential injection happens via a thin proxy layer that stamps headers/env on each tool call — not by spawning new MCP processes. Same pattern as a database connection pool with per-query auth context.

2. **`runtime/sessions/{id}/` is the isolation unit.** Every session (parent agent + all its subagents) gets an ephemeral sandbox. Subagent workspaces are nested beneath, invisible to the parent. Nothing in `runtime/` is ever synced or persisted — it's `/tmp`.

3. **Credential refs, never values.** `users/{id}/credentials/*.ref` files contain only URIs pointing to Keychain/Secrets Manager. A `CredentialResolver` fetches values at session start, injects them as ephemeral env vars into the session sandbox, and shreds them on close. Agents only ever see env vars.

4. **Every path segment is a service boundary.** `users/{id}/skills/` → Skills Service, `sessions/` → Neon PostgreSQL, `users/{id}/memory/` → Atlas. Cloud routing requires only setting env vars; zero code changes.

5. **Write always goes to personal scope; reads walk up the chain.** personal → team → global. This is the CSS specificity model. Agents cannot write to team or global scope directly.

---

## Phase Index

| Phase | Title | Effort (1–5) | Risk (1–5) | Priority | Dependencies | Status |
|-------|-------|-------------|-----------|----------|--------------|--------|
| 002-A | Directory Layout Migration | 2 | 1 | P0 | None | Not started |
| 002-B | Runtime Session Isolation | 3 | 2 | P0 | 002-A, Plan 001-0 | Not started |
| 002-C | MCP Connection Pool (MCPGateway) | 4 | 3 | P0 | 002-B | Not started |
| 002-D | Cloud Env-Var Routing | 2 | 1 | P1 | 002-A | Not started |

## Execution Sequence

```
002-A ──► 002-B ──► 002-C
002-A ──► 002-D      (D can run in parallel with B/C)
```

---

## Phase Details

### Phase 002-A — Directory Layout Migration

**What:** Create the canonical `~/.hermes/users/{id}/` tree and migrate existing flat content into it. Additive only — no existing paths are removed until Phase D. Updates `hermes_constants.py` to expose the new root paths.

**Files to modify:**
- `hermes_constants.py` — add `get_user_home(user_id)`, `get_plans_root(user_id)`, `get_skills_root(user_id)`, `get_memory_root(user_id)`, `get_artifacts_root(user_id)`, `get_credentials_root(user_id)`
- `agent/memory.py` (or wherever MEMORY.md/USER.md are read/written) — update path resolution to use `get_memory_root()`
- Migration script: `scripts/migrate_002a.sh` — creates directory tree, moves `memories/` → `users/{id}/memory/`, begins writing new plans to `users/{id}/plans/active/`

**Acceptance criteria:**
- [ ] `~/.hermes/users/blake/` directory tree exists with all subdirs
- [ ] `get_user_home("blake")` returns `~/.hermes/users/blake/`
- [ ] MEMORY.md and USER.md resolve from `users/blake/memory/`
- [ ] `pytest tests/ -v` — zero regressions (all existing tests pass)
- [ ] `hermes` CLI starts and runs a turn without errors

### Phase 002-B — Runtime Session Isolation

**What:** Every conversation session gets an isolated ephemeral sandbox at `runtime/sessions/{session-id}/`. Subagents spawned via `delegate_task` write to a sub-directory of that sandbox. Identity, toolset, and workspace are propagated from parent to child — with the child only getting a strict subset of the parent's toolset. Sandbox is destroyed (outputs promoted to `artifacts/`) at session end.

**Files to modify:**
- `run_agent.py` — `AIAgent.__init__` gains `session_runtime_dir` param; on construction creates `runtime/sessions/{id}/workspace/`
- `model_tools.py` / `tools/delegate_task.py` — subagent workspace forced to `runtime/sessions/{id}/subagents/{sub-id}/workspace/`; toolset intersection enforced
- `hermes_state.py` or session lifecycle hook — on session close, move `workspace/outputs/` → `users/{id}/artifacts/sessions/{date}-{id}/outputs/`; then `rm -rf runtime/sessions/{id}/`
- New file: `agent/session_runtime.py` — `SessionRuntime` class managing workspace creation, output promotion, and cleanup

**Acceptance criteria:**
- [ ] Each new session creates `runtime/sessions/{session-id}/workspace/` on init
- [ ] A spawned subagent writes only to `runtime/sessions/{id}/subagents/{sub-id}/workspace/`
- [ ] Subagent cannot receive a toolset wider than its parent's (intersection enforced, extras silently dropped)
- [ ] On session end, `workspace/outputs/` is moved to `artifacts/sessions/`; `runtime/sessions/{id}/` is deleted
- [ ] `runtime/` is empty after all sessions close
- [ ] `pytest tests/ -v` — zero regressions

### Phase 002-C — MCP Connection Pool (MCPGateway)

**What:** Replace per-session MCP subprocess spawning with a single `MCPGateway` process that runs each MCP server once and multiplexes calls from all sessions. Per-user credentials are injected at the **call level** (headers or env overlay), not at process startup. The gateway enforces scope: a session can only call tools for which it has a valid credential. This is the database connection pool analogy — one pool, per-query auth context.

The key insight: instead of `N users × M servers = N×M processes`, we get `M servers = M processes` with a credential-stamping proxy in front.

**Files to create/modify:**
- New file: `agent/mcp_gateway.py` — `MCPGateway` singleton:
  - Maintains one MCP subprocess per server definition (from `system/mcp-servers.json`)
  - Exposes `call_tool(session_identity, server_name, tool_name, args)` 
  - Resolves the calling session's credential for `server_name` via `CredentialResolver`
  - Injects credential as a per-call env overlay or Authorization header
  - Enforces scope: if session has no credential for a server, raises `MCPAccessDenied`
- New file: `agent/credential_resolver.py` — `CredentialResolver`:
  - Reads `users/{id}/credentials/mcp-servers.json` → `token.ref` files
  - Resolves `keychain://hermes/{user-id}/{service}` via macOS `security` CLI (local)
  - Resolves `secrets-manager://...` via `boto3` (cloud)
  - Caches resolved values in memory for the session lifetime; never writes to disk
- Modified: `tools/` MCP tool implementations — route through `MCPGateway.call_tool()` instead of direct subprocess calls
- Modified: `hermes_constants.py` — add `get_mcp_servers_config()` returning path to `system/mcp-servers.json`

**Acceptance criteria:**
- [ ] Starting two concurrent sessions does NOT start duplicate MCP server processes (verified via `pgrep`)
- [ ] Session A calling a Gmail tool uses Session A's Gmail token; Session B's call uses Session B's token
- [ ] A session with no credential for a server receives `MCPAccessDenied`, not a generic error
- [ ] Credential values are never written to any file during a session (only in-memory cache in `CredentialResolver`)
- [ ] MCP server crashes are caught and surfaced as graceful errors, not agent hangs
- [ ] `pytest tests/test_mcp_gateway.py -v` — all pass
- [ ] `pytest tests/ -v` — zero regressions

### Phase 002-D — Cloud Env-Var Routing

**What:** Make every path-returning function in `hermes_constants.py` check an env var override before returning the local default. This is the sole change needed to point any subtree at a cloud service — no code changes in callers. Document the full env var contract and add a validation check on startup.

**Files to modify:**
- `hermes_constants.py` — each path function checks its env var:
  - `HERMES_USERS_ROOT` → overrides `~/.hermes/users/`
  - `HERMES_SYSTEM_SKILLS_ROOT` → overrides `~/.hermes/system/skills/`
  - `HERMES_SESSIONS_ROOT` → overrides `~/.hermes/sessions/` (PostgreSQL URI accepted)
  - `HERMES_MEMORY_BACKEND` → overrides local MEMORY.md path (Atlas MCP URI accepted)
  - `HERMES_CREDENTIALS_BACKEND` → overrides `keychain://` (Secrets Manager URI accepted)
  - `HERMES_RUNTIME_ROOT` → overrides `~/.hermes/runtime/`
  - `HERMES_AUDIT_SINK` → overrides `~/.hermes/logs/` (S3 URI accepted)
- New file: `scripts/validate_env.py` — checks env vars at startup, warns if paths don't exist, fails fast on malformed URIs
- Update `README.md` / `docs/` — document the env var contract

**Acceptance criteria:**
- [ ] Setting `HERMES_USERS_ROOT=/tmp/test-users/` causes all user path resolution to use that root
- [ ] `validate_env.py` prints a warning (not error) if a var is set but the path doesn't exist
- [ ] `validate_env.py` exits non-zero if a var is set with a malformed URI
- [ ] No path hardcoded in any tool file — all routed through `hermes_constants.py`
- [ ] `pytest tests/test_constants.py -v` — all pass
- [ ] `pytest tests/ -v` — zero regressions

---

## Budget Estimate

| Component | Local dev | Cloud (prod) |
|-----------|-----------|-------------|
| MCPGateway process (per Hermes instance) | Free (subprocess) | ~$0 (sidecar in Fargate task) |
| CredentialResolver (macOS Keychain) | Free | N/A |
| CredentialResolver (Secrets Manager) | N/A | ~$0.05/10k calls |
| `runtime/` ephemeral storage | Free (local `/tmp`) | ~$0.30/GB-month (tmpfs on EFS) |
| `users/{id}/skills/` → S3 | N/A (Phase D env var) | ~$0.50/month |
| **Total delta from current** | **$0** | **<$5/mo** |

---

## Open Questions (must resolve before Phase C begins)

- **Q-C.1:** Should `MCPGateway` run as a separate sidecar process or as a thread within the Hermes process? (Recommended: same-process thread pool for local; separate sidecar on Fargate for cloud isolation)
- **Q-C.2:** How should credential rotation be handled mid-session? (Recommended: `CredentialResolver` has a `refresh(session_id)` method; called on `MCPAccessDenied` once before propagating the error)
- **Q-C.3:** For MCP servers that don't support per-call auth injection (e.g. stdio servers without header support), what's the fallback? (Recommended: spawn a dedicated process for that session only — fall back to current behavior for non-parameterizable servers)
