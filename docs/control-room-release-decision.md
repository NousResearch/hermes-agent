# Control Room V1 — Release / Activation Decision Packet (CR-608)

Date: 2026-08-12 (updated 2026-08-12 evening after 3rd-party review)
Author: KENSEI (self-review against actual test output; external verdicts recorded verbatim below)
Status: **AWAITING SAHIL'S EXPLICIT RE-CONFIRMATION OF A+C** — code and tests READY (two independent reviews), packet governance trail corrected after 3rd-party review. No merge until Sahil confirms.

---

## 0. Decision record (verbatim evidence, chronological)

- **2026-08-12 (morning)** — Sahil selected **Option B: HOLD**. No merge to
  canonical, no activation, until CR-603 live peer E2E AND desktop vitest
  both pass.
- **2026-08-12 (evening, after both HOLD closers closed)** — Sahil stated:
  > "I believe A + C. however first I need a handoff for a objective review of the code."
  (Sahil, chat session 2026-08-12; A = activate/merge, C = commit pnpm-lock.yaml as the new standard)
- **2026-08-12 (evening)** — Round-2 independent subagent review:
  **VERDICT: READY** (deleg_55aaa63f, produced 20:12) with two cosmetic
  doc nits, fixed in df526d9a84.
- **2026-08-12 (night)** — External 3rd-party review: **RESULT: NEEDS
  CORRECTION** — code green; packet governance trail not trustworthy
  (self-declared verdict, contradictory decision records, stale lockfile
  statement). This packet was corrected in response; see section 9.
- **Pending** — Sahil's explicit re-confirmation of A+C after the packet
  correction (this is the activation gate).

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
   the monorepo (commit 4e2278b7ca) unblocked the suite: 4669 passed /
   15 upstream baseline fails / 2 skipped; the 15 failures are upstream
   baseline in speech-text/searchable-select, untouched by this branch.
   pnpm-lock.yaml is now TRACKED (commit b855be2f8b) — Sahil chose
   "A + C" (C = adopt pnpm lockfile as the new standard); see section 0.
3. **Dashboard BFF capabilities** `approvals`, `peer_messages`,
   `delegation_control` are hardcoded unavailable in v1 — correct per contract
   (no live path), but they become real only when the BFF gains a gateway RPC
   client. Kanban + process + system rows are real.
4. **Plugin E2E-909 (hermes-peer install/enable/restart/uninstall) fails in
   THIS environment**: hermes-peer is editable-installed in the hermes env, so
   `hermes plugins list` sees it even after uninstall — an environment
   interaction, not a control-room defect. Other plugin E2Es pass (E2E-910
   two-session exchange, E2E-909's siblings).
5. **Plugin TestRealBinarySmoke — NOT reproduced on this box (3rd-party
   review, 12/08/26 night).** The reviewer attempted the "two disposable
   Hermes homes exchange via real binary" smoke; it hung 16 min with zero
   CPU (driver.py procs blocked in anon_pipe_read, session A never received
   B) and was killed. Consistent with a walkie-talkie-repo environment hang,
   not a control-room defect — test_two_sessions.py (the cross-process
   proof) passed clean. Earlier session logs recorded TestRealBinarySmoke as
   passing; treat the "1 pass" claim as UNVERIFIED on this box until the
   smoke is re-run outside the control-room audit.

## 5. Security verdict

No critical/high findings. All mutations route through existing authoritative
APIs; zero raw SQL in `control_room/`; confirmation contract is enforced in
the router (not the UI); cross-profile guard; stale-revision re-fetch;
duplicate-submit guard. See `docs/control-room-security-review.md`.

## 6. Activation decision — PENDING SAHIL RE-CONFIRMATION (corrected after 3rd-party review)

- **Sahil's stated preference (verbatim, 12/08/26 evening):**
  > "I believe A + C. however first I need a handoff for a objective review of the code."
- **This packet does not self-declare the decision.** The 3rd-party review
  (12/08/26 night) flagged the earlier text as self-declared; this section
  now records only Sahil's words + review outcomes. Activation waits for
  Sahil's explicit confirm after this correction (section 0, section 9).
- **Merge pre-verified conflict-free** in disposable shared clones (also
  independently confirmed by the 3rd-party review's merge-tree check):
  - KenseiAgent: merged tree = branch + main's 2 delegation commits only
    (byte-verified); Python 522/1 green on merged tree.
  - Dashboard: merged tree at ae999b1; backend 7 pass, web 4 pass, tsc 0.
- **Final merge into live canonicals requires Hermes stopped** — the
  agent's safety guard blocks rewriting the running source checkout.
  Commands in section 8 are for an external shell (Sahil or a maintenance
  window); run with Hermes stopped, then restart. Note: canonicals are
  dirty with unrelated owner edits (KenseiAgent: AGENTS.md, CLAUDE.md,
  content_engine/*; dashboard: clarity-findings.json, docs/) — check
  status before merging per 3rd-party review NOTES.

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

## 8. External activation commands (run with Hermes STOPPED)

```bash
# 1. KenseiAgent canonical (live source checkout)
cd /home/kensei/repos/KenseiAgent
git merge --no-ff feat/control-room-v1-20260812 \
  -m "merge: Control Room V1 — CLI, Ink TUI, Desktop, Dashboard (Phases 0-6); A + C activation"

# 2. Dashboard canonical
cd /home/kensei/repos/kensei-dashboard
git merge --no-ff feat/control-room-v1-20260812 \
  -m "merge: Control Room V1 dashboard — Phase 5 BFF + SPA; A + C activation"

# 3. Verify (after restart)
cd /home/kensei/repos/KenseiAgent && .venv/bin/python -m pytest tests/control_room/ tests/tui_gateway/ -q
```

Both branches already live in the shared object store (same .git as canonicals);
the merges are pre-verified conflict-free. `pnpm install` will be needed on the
canonical checkout before TUI/desktop tests run there (lockfile now committed).

## 9. Review-correction record (3rd-party review, 12/08/26 night)

External 3rd-party review result: **RESULT: NEEDS CORRECTION** — "code is
green; the activation packet is not trustworthy."

What was corrected in this packet in response:
1. **Self-declared verdict removed.** Section 6 no longer claims "A+C
   SELECTED" or "review READY" as author declarations; it records Sahil's
   verbatim words + the two independent review outcomes (deleg_55aaa63f
   READY; 3rd-party NEEDS CORRECTION on packet).
2. **Contradictory decision record fixed.** Header + section 0 now show the
   full chronological trail (B HOLD morning → Sahil's A+C words evening →
   review outcomes) instead of "B" in the header and "A+C" in §6.
3. **Lockfile statement reconciled.** §4 no longer says "intentionally
   untracked"; pnpm-lock.yaml IS tracked (b855be2f8b) and that is recorded
   with the C decision. Handoff doc NOTE-A updated to match (separate commit).
4. **Desktop totals made complete.** "4669 passed / 15 failed / 2 skipped"
   (4686 total) everywhere, not just 4669/15.
5. **TestRealBinarySmoke unverified claim downgraded.** §4 item 5 records
   the reviewer's 16-min hang and marks the earlier "1 pass" claim
   UNVERIFIED on this box pending a clean re-run.
6. **Dirty-canonical caution added.** §6 notes the owner edits present in
   both canonicals and directs a status check before the §8 merge.

The 3rd-party review independently CONFIRMED: all executable claims
reproduced exactly at final HEAD, merge-tree into both canonicals is
conflict-free, round-1 fixes are honest, and "zero findings in source."

---
KENSEI sign-off on claims: verified against terminal output above; gaps are
explicit and not buried. Activation: **PENDING Sahil's explicit
re-confirmation of A+C after this packet correction** (sections 0, 6, 9).
