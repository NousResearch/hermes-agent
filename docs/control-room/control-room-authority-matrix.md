# Control Room V1 — Authority Matrix & Scope Semantics (Phase 0)

**Plan refs:** CR-003 (authority matrix), CR-004 (dashboard spike), CR-006 (scope semantics).
**Date:** 12 August 2026. **Worktrees:** kenseiagent-control-room-20260812 @ 66f18cc245, kensei-dashboard-control-room-20260812 @ e1de73a.

---

## 1. Scope semantics (CR-006)

- **Default scope: current profile.** Every Control Room surface reads/writes the
  profile it is running under. CLI/TUI/Desktop are inherently single-profile
  (the process owns a profile).
- **Explicit all-profile view only in Dashboard/Desktop, and only where the BFF
  has configured access.** Dashboard reads `GATEWAY_PROFILES` from
  `backend/settings.py` — aggregation is allowed only over that explicit list,
  and every aggregated row carries a visible `profile` label. A profile name is
  never hidden on a multi-profile row.
- A snapshot generated for profile X must never contain rows owned by profile Y
  unless the surface is in an explicitly configured all-profile mode AND the row
  is labeled. This is enforced by contract invariant `validate_invariants()` and
  covered by tests `TestNoCrossProfileLeakage`.

## 2. Authority matrix (CR-003)

Each V1 action maps to ONE authoritative backend. Control Room never writes
directly to SQLite, never calls `hermes_peer.plugin._manager`, and never
implements a second approval/state machine. Where the authoritative path does
not exist end-to-end, the action is **unavailable** — rendered as unavailable,
never as a dead control.

| Action | Authoritative source (verified) | Renderable? | Notes |
|---|---|---|---|
| Snapshot read (home/sections) | gateway session registry, process/delegation registries, Kanban read API, Hermes Peer public API, system health provider | YES (Phase 1) | All providers compose into `ControlRoomSnapshot`; unavailable providers yield typed unavailable rows |
| Approval allow/deny | existing `approval.respond` gateway RPC (`tui_gateway/methods_prompt.py`) | CLI/TUI/Desktop YES via gateway; **Dashboard NO** | Dashboard BFF `pending_approvals()` hard-returns `{available: false}` (backend/system_health.py:179). No live approval path exists from the dashboard. V1 dashboard must render approvals unavailable. |
| Peer message send / inbox release-refuse / request lifecycle | Hermes Peer public API (`hermes_peer/commands.py`, `hermes_peer/tools.py`: `peer_send_message`, `peer_read_inbox`, `peer_request_*`) | CLI/TUI/Desktop YES when plugin loaded; **Dashboard NO** | BFF has zero peer integration (no imports, no RPC client, no CLI shell-out — verified CR-004). V1 dashboard messages view must render unavailable. |
| Process kill | `process.kill` / `process.list` session-scoped gateway RPC (`tui_gateway/methods_tools.py`) | CLI/TUI/Desktop YES via gateway | Preserves session ownership checks. Dashboard: no path — unavailable. |
| Subagent interrupt / steer | `subagent.interrupt`, `subagent.steer` (`tui_gateway/methods_session.py`) | CLI/TUI/Desktop YES via gateway | Dashboard: no path — unavailable. |
| Delegation pause | `delegation.pause` / `delegation.status` (`tui_gateway/methods_session.py`) | CLI/TUI/Desktop YES via gateway | Dashboard: no path — unavailable. |
| Kanban create/comment/moves | existing Kanban plugin validated local write APIs with the Kanban write lock | All surfaces YES via BFF `kanban_local_writes.py` (dashboard) / plugin (core) | Never raw SQLite from a renderer. Retry/redispatch only through the existing Kanban task state machine — no generic "retry" button. |
| New Task / New Message / New Agent Run | existing background/delegation route (New Agent Run), Kanban create (New Task), peer send (New Message) | YES where the underlying capability exists | New Agent Run always shows target/profile/scope before confirmation (CR-206). |
| System restart/update | existing confirmation/approval gates + action polling | Through existing route only | Control Room never invokes restart/update outside the existing approval-gated route. |
| Cross-profile target mutation | — | NEVER | Any action whose target profile differs from the snapshot scope is rejected (`CROSS_PROFILE`). |

## 3. CR-004 spike findings — dashboard-to-profile communication

### Question
Can the dashboard BFF reach live peer messages and approvals for a profile?

### Evidence (from kensei-dashboard worktree @ e1de73a)
- `backend/app.py` exposes `/api/fleet`, `/api/attention`, `/api/system/approvals`
  (+ `POST /api/system/approvals/{id}/respond`), `/api/gateway-health`.
- `backend/system_health.py:179` `pending_approvals()` is a hard stub:
  `return {"available": False, "items": []}` — comment: "approvals are managed
  via MCP protocol only".
- `backend/system_health.py:184` `respond_approval()` shells out to a
  `hermes permissions` CLI subprocess that does not exist in the product
  surface (hermes has no such subcommand).
- `grep -r "hermes_peer\|peer_message\|peer_inbox" backend/` → **no matches**.
  The BFF has no peer imports, no gateway RPC client, no peer CLI shell-out.
- The BFF's only live profile data sources are: `gateway_state.json` + systemd
  status (`profiles.py`), per-profile `state.db` session counts, Kanban read
  models (`kanban_local.py`), and system health probes.

### Conclusion
**No live path exists today from the dashboard to peer messages or approvals.**
This is exactly the risk the plan flagged (§2.2, §6.3). Consequences for V1:

1. Dashboard Control Room shows `approvals: unavailable` and
   `peer_messages: unavailable` in `capabilities` — and renders those sections
   as read-only/unavailable, never with working-looking controls.
2. A real authority path is a Phase 1+ prerequisite for enabling those controls:
   either (a) BFF → gateway RPC client over the local transport, or (b) BFF →
   Hermes Peer public API via a documented adapter. Both are new, deliberate
   work; neither is silently assumed.
3. Kanban reads/writes (`kanban_local.py`, `kanban_local_writes.py`) ARE real
   today and are the first dashboard-actionable surface.

## 4. Phase 0 gate position

Per plan §Gate: "no UI work begins until a real data/action path exists for
every V1 action advertised." The contract now defines typed
`unavailable` semantics, so an unavailable source is *visible as unavailable*
rather than blocking all UI. The Phase 0 gate is therefore:

- Read paths: real providers exist for sessions/processes/delegations/Kanban/
  system; peer provider exists when the plugin is loaded (CLI/TUI/Desktop).
- Action paths: approval.respond, process.kill, subagent.*, delegation.*,
  Kanban writes, peer public API — all exist on the core side.
- Dashboard action paths: Kanban only. Approvals + peer messages on dashboard
  are explicitly unavailable until a real adapter is built.

This satisfies the plan's intent: every V1 action advertised has either a real
path or a typed unavailable state. No dead controls.
