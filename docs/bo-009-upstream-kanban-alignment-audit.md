# BO-009 — Upstream Kanban alignment audit

Authority: Kanban `BO-009` / `t_4a014f93`  
Parent: `BO-002` / `t_9139a9a9`  
Repo: `chriskim12/hermes-agent`  
Boundary: review=Chris, push/PR=allowed, merge=separate approval, gateway restart/reload=forbidden, secrets/env/BWS mutation=forbidden.

## Verdict

BO-002's Kanban-native operating model is directionally aligned with upstream Hermes Kanban, with one important correction:

- **Project namespaces (`BO`, `DC`, `WS`, `RS`) are governance/public-id namespaces, not hard isolation boundaries.**
- Upstream Kanban's intended hard isolation boundary is **board**: separate DB, workspaces, logs, dispatcher context, and `HERMES_KANBAN_BOARD`-scoped worker tools.

Therefore the target state should be:

```text
Board = project/domain hard isolation boundary
Namespace/public id = human/governance allocation authority inside or across boards
Tenant = soft filter/fleet marker inside a board
Projection/review surfaces = never authority
```

## Evidence checked

### Upstream documentation

Fetched live upstream docs from:

- https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban

Relevant upstream anchors:

- Kanban is a durable task board / queue / state machine, not `delegate_task` RPC.
- Workers spawned by the dispatcher should use `kanban_*` tools rather than shelling out to `hermes kanban ...`.
- Raw task `done` / completed run is worker evidence, not final project closeout by itself.
- `task_runs.summary` and `task_runs.metadata` are the native home for structured handoff evidence.
- Dashboard/review queues are operations surfaces, not closeout authority.
- Boards are the hard isolation boundary: separate SQLite DB, workspaces, logs, dispatcher board context, and `HERMES_KANBAN_BOARD`.
- Tenants are soft filters/namespaces inside a board.

### Local implementation snapshot

Local main at audit start:

- HEAD: `23dc7f890f62d41f0fe4777f4c3e21b0b60aebea`
- Installed CLI exposes Kanban lifecycle commands, runs, context, notifications, dashboard plugin, etc.
- Installed/source CLI did **not** expose `hermes kanban boards ...` or `hermes kanban --board ...` at audit time.
- Local dashboard Kanban plugin exists under `plugins/kanban/dashboard/` and reads/writes the same `kanban_db` code path.
- Local dashboard plugin supports columns, tenant filter, assignees, task links/progress, comments, events, runs, bulk actions, WebSocket event updates.

## Alignment map

| BO-002 principle | Upstream alignment | Notes |
|---|---:|---|
| New work defaults to Kanban authority | Aligned | Upstream Kanban is durable work queue/state machine. |
| Linear is legacy/reference/projection for new Kanban work | Aligned | Upstream projection guidance says external ledgers must remain concise/idempotent and non-authoritative. |
| `worker_done -> review_ready -> closed` governance phases | Aligned | Upstream docs explicitly separate worker completion from final project closeout. |
| PR/checks as evidence/review surface, not authority | Aligned | Fits upstream review queue/projection boundary. |
| Projection authority claims fail-closed | Aligned | Prevents dashboard/wiki/Discord/Linear views from becoming stale competing SSOTs. |
| BO/DC/WS/RS as current namespace split | Partially aligned | Good as public-id/governance namespace; not enough for upstream hard isolation. |
| Hermes direct control-plane closeout | Acceptable for migration/control-plane work | Dispatcher-spawned worker path should be piloted for normal execution. |

## Project board migration stance

Chris approved project-by-project boards if upstream intent supports them. It does: upstream docs describe one board per project/repo/domain as the hard isolation model.

However, local code/CLI currently lacks the upstream `boards` command surface. That makes immediate board migration unsafe. The correct sequence is:

1. Keep current default-board namespace operation as the compatibility baseline.
2. Add or sync upstream board support locally.
3. Create a migration plan for project boards:
   - `brain-os` for BO
   - `dailychingu` for DC
   - `whystarve` for WS
   - `risu` for RS
4. Preserve historical public ids and links as references.
5. Do not cross-link tasks across boards; use explicit text references for cross-project refs.
6. Run dispatcher/dashboard/worker smoke per board before declaring board split active.

## Dashboard / readability audit

Chris noted the existing Hermes-native Kanban dashboard is hard to read. The native dashboard should remain the baseline because it writes through `kanban_db` and tails `task_events`, but readability can be improved without creating another SSOT.

Recommended hierarchy:

1. **Improve native dashboard plugin first**
   - Add/verify clearer review queues: active, blocked, stale/reclaim, failed, worker_done, review_ready, closed.
   - Make parent progress and public IDs more prominent.
   - Add compact/dense mode for high-card projects.
   - Add project/board selector once board support exists.
   - Keep REST writes routed through `kanban_db` only.

2. **Add read-only projection views only if they are clearly labeled**
   - Discord digest / thread summaries.
   - CLI `stats/watch/runs/context` summaries.
   - A lightweight read-only review page is acceptable only if it reads from Kanban and cannot mutate task state.

3. **Avoid competing dashboards**
   - Anything with independent state, stale cached status, or closeout authority conflicts with the SSOT model.

Security boundary remains unchanged: localhost/Tailscale/SSH tunnel only unless a separate auth/reverse-proxy design is approved.

## Dispatcher-native worker pilot

BO-002/BO-008 were valid Hermes-direct control-plane tasks, but upstream-native execution should be proven with a small dispatcher-spawned worker pilot.

Pilot should verify:

- Task created/assigned in Kanban.
- Dispatcher sets `HERMES_KANBAN_TASK` and, when board support exists, `HERMES_KANBAN_BOARD`.
- Worker starts by calling `kanban_show()`.
- Worker emits `kanban_heartbeat()` during execution if long-running.
- Worker finishes with `kanban_complete(summary=..., metadata={...})`.
- `task_runs` contains structured handoff metadata.
- Gateway notification/subscription routes terminal event back to the correct thread when enabled.

Do not restart/reload gateway for this pilot without separate approval.

## Recommended follow-up cards

### BO-010 — Multi-board support local/upstream gap

Goal: bring local implementation into alignment with upstream board semantics or prove why the feature is unavailable in this fork/version.

Done when:

- `hermes kanban boards list/create/switch/show` and `--board <slug>` support are verified or implemented.
- Board DB/workspace/log path isolation is tested.
- `HERMES_KANBAN_BOARD` is passed to dispatcher-spawned workers.
- Cross-board linking is rejected.
- Existing default-board compatibility remains intact.

### BO-011 — Dashboard readability and review-queue UX

Goal: improve the existing Hermes Kanban dashboard readability without creating a competing SSOT.

Done when:

- Native dashboard shows clearer review/operations queues or filters.
- BO/DC/WS/RS or future boards are easy to distinguish.
- public id, review phase, parent progress, and blockers are prominent.
- All writes still go through `kanban_db`; any extra view is read-only/projection-labeled.

### BO-012 — Dispatcher-native worker pilot

Goal: run a small safe Kanban task through the upstream-intended worker path.

Done when:

- Dispatcher-spawned worker uses `kanban_*` tools.
- Structured run metadata lands in `task_runs`.
- Completion/blocked/crashed event behavior is observable.
- No gateway restart/reload is performed without explicit approval.

## BO-009 closeout stance

BO-009 itself should close after this audit lands with evidence. It should not directly migrate production projects to new boards because local board support is currently missing from the repo/CLI surface and would require a dedicated implementation card.
