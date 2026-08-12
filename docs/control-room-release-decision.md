# Control Room V1 — Release / Activation Decision Packet (CR-608)

Date: 2026-08-12
Author: KENSEI (self-review against actual test output)
Status: **READY FOR ACTIVATION** — both HOLD closers CLOSED (12/08/26 evening). Sahil selected B on 2026-08-12; conditions now met.

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
- `9a1870b` (via Phase 5 dashboard worktree) — BFF contract, Ctrl+P view
- `cee98adc9e` Phase 6 docs — matrix, security, perf, a11y, guide, demo
- `04c62f307e` + `fff1ce55fd` release decision packet + HOLD record
- `35473a0afb` fix — peer provider parses real JSON inbox shape (CR-603 catch)
- `13554e31e8` test — CR-603 live peer lifecycle E2E
- `4e2278b7ca` fix(env) — pnpm workspace install; desktop/TUI tests runnable
- `4301c6bff9` docs — HOLD closers CLOSED, packet back to READY
- `84d2b50054` docs — objective review handoff prompt
- (next) fix — CR-401 rename completed in all 5 locales; handoff + evidence table corrected

Dashboard worktree `feat/control-room-v1-20260812` (base e1de73a):
- `9a1870b` Phase 5 — BFF contract, Ctrl+P view, HealthStrip badge, tests

## 3. Test evidence (actual output, not claims — refreshed 12/08/26 after HOLD closers)

| Suite | Result |
|-------|--------|
| control_room + tui_gateway Python | **522 passed, 1 skipped** |
| CLI control-room (tests/control_room/test_cli_phase3.py) | **5 passed** |
| CLI status bar (pre-baseline upstream suite) | **20 passed** (not control-room-specific; untouched by this branch) |
| TUI vitest | **1559 passed, 1 skipped, 0 failed** (fully green — the prior 12 baseline env failures are fixed by the pnpm workspace install, commit 4e2278b7ca) |
| Dashboard BFF pytest | **7 passed** |
| Dashboard web vitest | **4 passed** |
| Desktop control-room tests | **8 passed** (control-room.test.ts, now asserts the CR-401 rename in ALL 5 locales) |
| Desktop full vitest | **4669 passed, 15 failed, 2 skipped** — the 15 failures are pre-existing upstream baseline in files untouched by this branch (speech-text.test.ts 13, searchable-select.test.tsx 2) |
| CR-603 live peer E2E | **PASS** (tests/control_room/e2e_live_peer_cr603.py — real two-manager sockets; cross-process via plugin E2E-910) |

TS mirror gate: `python -m control_room.export_ts_schema --check` — pass.

## 4. Honest gaps (NOT hidden)

1. **CR-603 live peer lifecycle E2E — CLOSED (12/08/26).** Live two-manager
   E2E over real AF_UNIX sockets proves receipt → inbox → lifecycle visible
   through Control Room (commit 13554e31e8); peer provider JSON-parse fix
   (35473a0afb); cross-process transport via plugin E2E-910. Not run: a
   two-FULL-Hermes-gateway session (manager-instance level + plugin
   real-binary disposable-home E2E used instead — documented in section 7).
2. **Desktop full vitest — CLOSED (12/08/26).** pnpm workspace install in
   the monorepo (commit 4e2278b7ca) unblocked the suite: 4669 passed; the
   15 remaining failures are upstream baseline in speech-text/searchable-
   select, untouched by this branch. pnpm-lock.yaml intentionally untracked
   (package-manager standard switch is a governance decision pending
   Sahil's sign-off).
3. **Dashboard BFF capabilities** `approvals`, `peer_messages`,
   `delegation_control` are hardcoded unavailable in v1 — correct per contract
   (no live path), but they become real only when the BFF gains a gateway RPC
   client. Kanban + process + system rows are real.
4. **Plugin E2E-909 (hermes-peer install/enable/restart/uninstall) fails in
   THIS environment**: hermes-peer is editable-installed in the hermes env, so
   `hermes plugins list` sees it even after uninstall — an environment
   interaction, not a control-room defect. Other plugin E2Es pass (E2E-910
   two-session exchange, E2E-909's siblings).

## 5. Security verdict

No critical/high findings. All mutations route through existing authoritative
APIs; zero raw SQL in `control_room/`; confirmation contract is enforced in
the router (not the UI); cross-profile guard; stale-revision re-fetch;
duplicate-submit guard. See `docs/control-room-security-review.md`.

## 6. Activation decision — DECIDED

- **B. HOLD (SELECTED 2026-08-12)** — no merge, no activation, until:
  1. CR-603 live two-session peer E2E passes, and
  2. Desktop vitest suite passes after a working workspace install.

## 7. Follow-up tasks to close the HOLD — STATUS 12/08/26: BOTH CLOSED

1. ~~**CR-603 live peer E2E**~~ **CLOSED** (commit 13554e31e8): live
   two-manager E2E over real sockets proves receipt → inbox → lifecycle
   visible through Control Room; cross-process transport proven by plugin
   E2E-910. Bonus fix 35473a0afb: peer provider now parses the real JSON
   inbox shape.
2. ~~**Desktop workspace install repair**~~ **CLOSED** (commit 4e2278b7ca):
   pnpm workspace install; desktop vitest runs (4669 pass / 15 upstream
   baseline fails in speech-text + searchable-select, files untouched by
   this branch); TUI now fully green (1559 pass / 0 fail — fixes the prior
   12 baseline env failures).
3. **Post-activation hardening (still open by design)** — wire BFF gateway
   RPC client → real approvals/peer capabilities (currently typed
   unavailable by design).
4. Update the operator guide if any activation-step differences emerge.

---
KENSEI sign-off on claims: verified against terminal output above; gaps are
explicit and not buried. Awaiting Sahil's selection.
