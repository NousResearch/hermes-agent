# Control Room — Shared Fixture Matrix (CR-601)

Every scenario the Control Room surfaces must be exercised somewhere in the
test suite. This matrix maps each scenario to its fixture and the test(s)
that cover it. The rule: **no scenario ships untested**; a scenario that
cannot be tested in this environment is listed with its blocker, never
silently dropped.

Legend: `unit` = deterministic unit test with fakes; `e2e` = live-session test;
`env-gated` = test exists but requires infrastructure absent in this workspace.

| # | Scenario | Fixture | Covered by | Status |
|---|----------|---------|-----------|--------|
| 1 | No gateway running | `fleet()` returns empty | `test_service.py::test_empty_fleet_snapshot` (KenseiAgent) | ✅ unit |
| 2 | No peer plugin installed | `capabilities.peer_messages=false`, no `hermes_peer` import | `test_service.py::test_peer_unavailable`, `test_control_room.py::test_peer_and_approvals_unavailable` (Dashboard BFF) | ✅ unit |
| 3 | Idle system (nothing needs you) | attention list empty, severity ok | `test_service.py::test_idle_snapshot_zero_counts` | ✅ unit |
| 4 | Active agent (gateway up, no attention) | fleet row `active=true` | `test_service.py::test_active_agents_count` | ✅ unit |
| 5 | Active subagent (session running) | subagent run in session store | `test_service.py::test_subagent_runs_in_snapshot` | ✅ unit |
| 6 | Held peer message | peer inbox non-empty | `test_actions.py::test_peer_send_flow`, `test_flows.py::TestNewMessage` (message creation) — *receipt/request lifecycle requires live peer* | ⚠️ unit covers creation; e2e env-gated (CR-603) |
| 7 | Blocked kanban task | task status `blocked` | `test_service.py::test_blocked_task_attention`, `test_actions.py::test_kanban_block_move` | ✅ unit |
| 8 | Pending approval | approval row exists | `test_actions.py::test_approval_respond_executor` (via `hermes_peer.tools.respond_to_approval`) | ✅ unit |
| 9 | Stale row / stale action | `expected_revision` mismatch | `test_actions.py::test_stale_revision_rejected` | ✅ unit |
| 10 | Multi-profile dashboard | `profile=` param differs from default | `test_control_room.py::test_version_and_profile` (Dashboard BFF) | ✅ unit |

## Environments

| Environment | Where it runs | What it proves |
|-------------|---------------|----------------|
| KenseiAgent Python suite | worktree `kenseiagent-control-room-20260812`, `tests/control_room` + `tests/tui_gateway` | contract, service, router, executors, flows, gateway RPCs |
| KenseiAgent TUI vitest | worktree `ui-tui`, `src/__tests__/controlRoomOverlay.test.tsx` | overlay open/close, badge, RPC consumption |
| Dashboard BFF pytest | dashboard worktree `backend/tests/test_control_room.py` | BFF contract shape, attention ordering, unavailable caps, read-only |
| Dashboard web vitest | dashboard worktree `web/src/pages/ControlRoom.test.tsx` | home tiles, needs-you count, deep-links |

## Gaps (honest)

- **CR-603 (live peer lifecycle)**: requires two real Hermes sessions plus the
  editable Peer plugin. The Peer plugin suite runs (756 pass / 7 env-gated
  e2e fail / 25 skip) but a live two-session message-receipt run was not
  executed in this workspace — no gateway sessions were running during the
  integration window. Listed as env-gated, not complete.
- **Desktop vitest (Phase 4)**: dashboard web vitest runs (pnpm), but the
  KenseiAgent `apps/desktop` vitest suite cannot load its vite config in this
  environment (missing `@vitejs/plugin-react`, `@tailwindcss/vite` from the
  hoisted partial node_modules; canonical identical). Keybind-contract tests
  were verified via a minimal config (3/3 pass) and the rest are authored
  source. Documented in the Phase 4 commit; needs a full workspace install to
  execute.
