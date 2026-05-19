# Status — Plan 002: Hermes Self-Organization

**Status:** READY TO IMPLEMENT (phase files complete)  
**Last updated:** 2026-05-18  
**Blocked by:** Phase B requires Plan 001-0 (HermesIdentity dataclass); Phase A and D have no blockers  
**Blocks:** Plan 001-A (Scoped Skills), Plan 001-B (Scoped Memory) — both need user path roots from Phase A

## Phase Progress

| Phase | Title | Status | Notes |
|-------|-------|--------|-------|
| 002-A | Directory Layout Migration | Not started | No blockers — safe to start |
| 002-B | Runtime Session Isolation | Not started | Blocked by 002-A + Plan 001-0 |
| 002-C | MCP Connection Pool (MCPGateway) | Not started | Blocked by 002-B |
| 002-D | Cloud Env-Var Routing | Not started | Blocked by 002-A only; can run parallel with B/C |

## Open Questions (Blake must resolve before Phase C begins)

- **Q-C.1:** MCPGateway same-process thread pool vs. separate sidecar? (Recommended: same-process for local, sidecar on Fargate for cloud)
- **Q-C.2:** Credential rotation mid-session? (Recommended: `CredentialResolver.refresh()` on first 401 before propagating error)
- **Q-C.3:** MCP servers that don't support per-call auth injection (stdio without header support)? (Recommended: spawn dedicated process for that session only — fallback to current behavior)

## Resumption Context

- **Next phase:** 002-A — can start immediately, no blockers
- **Phase file locations:** `plans/002-hermes-self-organization/phases/phase-002-a.md` through `phase-002-d.md`
- **Spec:** `~/.hermes/STRUCTURE.md` (v2.0) — read before any phase
- **Key files to touch in Phase A:**
  - `hermes_constants.py` — add 8 path functions after `get_skills_dir()` (line ~286)
  - `tools/memory_tool.py` — update `get_memory_dir()` (line ~56) to check new location first
  - Create `scripts/migrate_002a.sh`
- **Key files to touch in Phase B:**
  - `run_agent.py` — `AIAgent.__init__` at line ~1889 (after `self.session_id` is set)
  - `tools/delegate_task.py` — subagent workspace injection
  - `tools/terminal_tool.py` — default workdir when not explicitly set
  - Create `agent/session_runtime.py`
- **Key files to touch in Phase C:**
  - Create `agent/mcp_gateway.py` and `agent/credential_resolver.py`
  - Modify `tools/mcp_tool.py` — add `call_tool_with_credential()` wrapper
  - Note: `mcp_tool.py` already pools MCP processes via `_servers` dict + background event loop — do NOT rewrite, just add the credential injection layer on top
- **Key files to touch in Phase D:**
  - `hermes_constants.py` — add env var checks to the 8 functions from Phase A
  - Create `scripts/validate_env.py` and `docs/env-vars.md`
  - `run_agent.py` — call `validate_hermes_env()` early in `AIAgent.__init__`
- **No adaptations yet**
