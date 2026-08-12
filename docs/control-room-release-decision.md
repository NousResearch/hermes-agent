# Control Room V1 — Release / Activation Decision Packet (CR-608)

Date: 2026-08-12
Author: KENSEI (self-review against actual test output)
Status: **DECIDED — OPTION B (HOLD)** — not activated until the two env-gated gaps are closed. Sahil selected B on 2026-08-12.

---

## 0. Decision record

- **2026-08-12** — Sahil selected **Option B: HOLD**. No merge to canonical, no
  activation, until CR-603 live peer E2E AND desktop vitest both pass.
- Worktree branches remain intact for resumption; canonicals untouched.

## 1. What shipped

Control Room V1: a unified keyboard-first attention surface across four
surfaces, driven by one shared contract.

| Surface | Deliverable | Keybinding |
|---------|-------------|------------|
| CLI | `/control` command + status segment + Ctrl+P | Ctrl+P |
| Ink TUI | Full-screen overlay + attention badge | Ctrl+P / Esc |
| Native Desktop | Command Center renamed → Control Room, attention-first home, status badge | Ctrl/Cmd+P |
| Kensei Dashboard | `/control-room` route, HealthStrip badge, BFF endpoint | Ctrl+P |

## 2. Commits

KenseiAgent worktree `feat/control-room-v1-20260812` (base 66f18cc245):
- `8b6c26f7f8` Phase 0 — contract, TS mirror, authority matrix
- `f6599d6c8f` Phase 1 — snapshot service, /control backbone
- `e0bc87c334` Phase 2 — action router + executors + flows
- `70bd4f40aa` Phase 3 CLI — status segment, Ctrl+P
- `e5c2264b92` Phase 3 TUI — Ink overlay, badge, gateway RPCs
- `05dc4c16b6` Phase 4 Desktop — rename, Ctrl+P, home, status badge
- `cee98adc9e` Phase 6 docs — matrix, security, perf, a11y, guide, demo

Dashboard worktree (base e1de73a):
- `9a1870b` Phase 5 — BFF contract, Ctrl+P view, HealthStrip badge, tests

## 3. Test evidence (actual output, not claims)

| Suite | Result |
|-------|--------|
| control_room + tui_gateway Python | **517 passed, 1 skipped** |
| CLI status bar | **20 passed** |
| TUI vitest | **1547 passed, 12 failed** — 12 are pre-existing baseline (scrollBox/virtualHistory/MoA), identical on canonical |
| Dashboard BFF pytest | **7 passed** |
| Dashboard web vitest | **4 passed** |
| Desktop keybind tests | **3 passed** (via minimal config; full desktop vitest env-blocked) |

TS mirror gate: `python -m control_room.export_ts_schema --check` — pass.

## 4. Honest gaps (NOT hidden)

1. **CR-603 live peer lifecycle E2E — deferred (env-gated).** Needs two real
   Hermes gateways + editable Peer plugin; no live sessions existed during the
   integration window. Unit coverage exists for message creation and inbox
   actions. This is the one open QA item.
2. **Desktop full vitest — env-blocked.** The hoisted node_modules lacks
   `@vitejs/plugin-react` / `@tailwindcss/vite` / `@tabler/icons-react` /
   `@tanstack/react-query`; canonical is identical; npm install fails on
   phantom version resolution. Keybind contract verified 3/3; remaining
   desktop tests are authored source, not executed. Requires a working
   workspace install (pnpm resolved fine for the dashboard — a similar scoped
   install should work for the monorepo).
3. **Dashboard BFF capabilities** `approvals`, `peer_messages`,
   `delegation_control` are hardcoded unavailable in v1 — correct per contract
   (no live path), but they become real only when the BFF gains a gateway RPC
   client. Kanban + process + system rows are real.

## 5. Security verdict

No critical/high findings. All mutations route through existing authoritative
APIs; zero raw SQL in `control_room/`; confirmation contract is enforced in
the router (not the UI); cross-profile guard; stale-revision re-fetch;
duplicate-submit guard. See `docs/control-room-security-review.md`.

## 6. Activation decision — DECIDED

- **B. HOLD (SELECTED 2026-08-12)** — no merge, no activation, until:
  1. CR-603 live two-session peer E2E passes, and
  2. Desktop vitest suite passes after a working workspace install.
  Worktrees stay as-is for resumption; canonicals remain untouched.

## 7. Follow-up tasks to close the HOLD (required before activation)

1. **CR-603 live peer E2E** — two real Hermes gateways + editable Peer plugin:
   message receipt, inbox action, request lifecycle visible through Control
   Room. Needs a maintenance window with live gateways.
2. **Desktop workspace install repair** — pnpm (or npm with fixed resolution)
   install in the KenseiAgent monorepo so the authored desktop vitest suite
   actually runs; then execute and green it.
3. **Post-activation hardening** — wire BFF gateway RPC client → real
   approvals/peer capabilities (currently typed unavailable by design).
4. Update the operator guide if any activation-step differences emerge.

---
KENSEI sign-off on claims: verified against terminal output above; gaps are
explicit and not buried. Awaiting Sahil's selection.
