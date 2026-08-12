# KENSEI Control Room V1 — SHARE-READY CHAT HANDOFF PROMPT (12/08/26)

> Copy everything below the line into a FRESH chat with an independent
> auditor/reviewer agent. Self-contained: no prior context needed.
> Producing baseline: code final `c4e07dd3bf` (KenseiAgent worktree, after
> round-1 review corrections) and `9a1870b` (Dashboard worktree). Branch
> tips add docs only — see HEAD note.

---

You are an independent auditor reviewing the Kensei Control Room V1
deliverable. Assume every claim below is WRONG until you have reproduced
it yourself. Never trust the repo's own docs — verify against source and
live execution. Do NOT modify source; read-only audit. A failed command
IS the finding. No fabricated numbers.

There are TWO worktrees. Review each.

## Repository state (verify FIRST)

Worktree 1 — KenseiAgent core (CLI, TUI, Desktop, Python control_room):

- Repo: /home/kensei/worktrees/kenseiagent-control-room-20260812
- Branch: feat/control-room-v1-20260812
- HEAD: ANY descendant of `c4e07dd3bf` whose diff `c4e07dd3bf..HEAD`
  touches ONLY docs/ — if HEAD touches code, that is a finding
- Code final: `c4e07dd3bf` (includes round-1 corrections: CR-401 rename
  completed in all 5 locales, handoff/evidence-table fixes)
- Code baseline: `66f18cc24517fdcc52bd0e0f6ffd5cc2f556522c`
  (diff baseline..HEAD must be limited to control_room/, hermes_cli/,
  cli.py, tui_gateway/, ui-tui/, apps/desktop/, tests/, scripts/,
  docs/, pnpm-workspace.yaml, ui-tui/package.json — anything outside
  that set is a finding)

```bash
cd /home/kensei/worktrees/kenseiagent-control-room-20260812
git branch --show-current              # expect: feat/control-room-v1-20260812
git rev-parse HEAD
git merge-base --is-ancestor c4e07dd3bf HEAD && echo "code-final ancestry OK"
git diff --name-only c4e07dd3bf..HEAD | cat   # expect: docs/ + pnpm-lock.yaml (see NOTE-A)
git diff --check 66f18cc245..HEAD      # expect: exit 0 (no whitespace errors)
git status --short                     # expect: clean (pnpm-lock.yaml TRACKED; see NOTE-A)
```

Worktree 2 — Kensei Dashboard (BFF + React SPA):

- Repo: /home/kensei/worktrees/kensei-dashboard-control-room-20260812
- Branch: feat/control-room-v1-20260812
- HEAD: `9a1870ba0fdbdc71b6750993427142257433e2f3` (Phase 5 dashboard;
  no later commits expected)
- Code baseline: `e1de73a782ff58a847c23dd09d70d958172d827c`

```bash
cd /home/kensei/worktrees/kensei-dashboard-control-room-20260812
git branch --show-current              # expect: feat/control-room-v1-20260812
git rev-parse HEAD                     # expect: 9a1870b...
git diff --check e1de73a..HEAD         # expect: exit 0
git status --short                     # expect: clean
```

NOTE-A: pnpm-lock.yaml is TRACKED in worktree 1 (commit b855be2f8b) —
Sahil chose "A + C" (C = adopt the pnpm lockfile as the new
package-manager standard) on 12/08/26 evening; the 3rd-party review
flagged the earlier "untracked" statement as stale. Its presence is
expected and NOT a defect.

If ANY state check fails, stop and report — do not continue.

## What was corrected (round-1 findings, commit c4e07dd3bf)

1. CR-401 rename was English-only — completed in ar/ja/zh/zh-hant
   (labels + nav.controlRoom keys); control-room.test.ts now asserts the
   rename label in EVERY locale, so an en-only rename fails CI.
2. Handoff claim-A parenthetical double-counted (99 vs actual 94) — fixed.
3. Release-decision section 3 evidence table was stale (517/1547-12/3)
   vs live (522/1559-0/8/4669-15 + CR-603 PASS) — refreshed.
4. speech-text citation corrected to 51c68d4ab1.

All other round-1 findings were VERIFIED (state clean, scope clean,
no raw SQL, no dead controls, typed-unavailable everywhere claimed).

## Claims to verify

### A. Python control_room + tui_gateway suites (reproduce ~1 min)

```bash
cd /home/kensei/worktrees/kenseiagent-control-room-20260812
/home/kensei/repos/KenseiAgent/.venv/bin/python -m pytest \
  tests/control_room/ tests/tui_gateway/ -q
```

- EXPECTED: 522 passed, 1 skipped (run the FULL two dirs; the
  control_room count alone is 94 — that includes the 5 live-shape
  tests in test_peer_live_shape.py)
- Evidence: control_room/contract.py, service.py, actions.py,
  executors.py, flows.py; tests/control_room/test_peer_live_shape.py

### B. TS mirror determinism gate

```bash
cd /home/kensei/worktrees/kenseiagent-control-room-20260812
/home/kensei/repos/KenseiAgent/.venv/bin/python -m control_room.export_ts_schema --check
```

- EXPECTED: "OK: ...control_room/schema/control-room-v1.ts is up to date"
  (exit 0). The TS types are GENERATED from the Python pydantic DTOs;
  the checked-in mirror must never drift.

### C. CR-603 live peer lifecycle E2E (reproduce ~30s)

```bash
cd /home/kensei/worktrees/kenseiagent-control-room-20260812
/home/kensei/repos/KenseiAgent/.venv/bin/python \
  tests/control_room/e2e_live_peer_cr603.py
```

- EXPECTED: final line "CR603 LIVE PEER E2E (manager path): PASS",
  exit 0. It spawns two REAL PeerSessionManager instances (the Hermes
  Peer plugin manager) over real AF_UNIX sockets + shared runtime root:
  alpha->bravo send returns held, bravo's real MessageStore persists,
  Control Room reads the real inbox via peer_read_inbox, provider
  parses the real JSON, router releases through the real tool, held ->
  queued and the host receives the peer-wrapped message.
- Cross-process transport is independently proven by the peer plugin's
  own E2E: /home/kensei/repos/hermes-walkie-talkie
  `tests/e2e/test_two_sessions.py` (5 pass) and
  `tests/e2e/test_install_and_binary.py::TestRealBinarySmoke` (1 pass,
  two disposable Hermes homes exchange through the real binary).

### D. TUI vitest suite (reproduce ~1 min)

```bash
cd /home/kensei/worktrees/kenseiagent-control-room-20260812/ui-tui
npx vitest run
```

- EXPECTED: 141 files passed, 1559 tests passed, 1 skipped, 0 failed.
  This includes controlRoomOverlay.test.tsx (10 tests) and
  controlRoomBadge coverage. NOTE: the prior "12 baseline failures"
  no longer exist — the pnpm workspace install fixed them (they were
  chalk-instance / env issues, see commit 4e2278b7ca).

### E. Desktop control-room tests (reproduce ~1 min)

```bash
cd /home/kensei/worktrees/kenseiagent-control-room-20260812/apps/desktop
npx vitest run src/lib/keybinds/control-room.test.ts
```

- EXPECTED: 1 file passed, 8 tests passed. Covers CR-403 keybind
  defaults (mod+p = nav.controlRoom, palette = mod+k only), CR-401
  i18n rename across all 5 locales (ar/en/ja/zh/zh-hant — each locale
  asserts a translated nav.controlRoom label AND a section label that
  is not the old "Command Center" default; the zh-hant->zhHant
  export-name mapping is handled), CR-402 home section.
- The full desktop suite runs 4669 passed / 15 failed. The 15 failures
  are PRE-EXISTING UPSTREAM BASELINE in files untouched by this branch:
  `src/lib/speech-text.test.ts` (13, sanitizeTextForSpeech regex drift
  from upstream) and
  `src/app/settings/searchable-select.test.tsx` (2). Verify by
  `git log --oneline -1 -- apps/desktop/src/lib/speech-text.ts` — the
  last change is 51c68d4ab1 (upstream "Add Hermes desktop app"), not
  this branch. If you find any failure in a control-room-touched file,
  that IS a finding.

### F. Dashboard BFF + SPA (reproduce ~1 min)

```bash
cd /home/kensei/worktrees/kensei-dashboard-control-room-20260812
/home/kensei/repos/KenseiAgent/.venv/bin/python -m pytest backend/tests/ -q
cd web && npx vitest run && npx tsc -b
```

- EXPECTED: backend 7 passed; web 4 passed; tsc -b exit 0.
- Covers CR-501 (Ctrl+P opens Control Room, Cmd+K preserved),
  CR-503/505 (BFF /api/control-room contract-shaped snapshot with
  typed unavailable for approvals/peer/delegation), CR-502 (HealthStrip
  attention badge), CR-504 (profile scope), CR-506 (deep-links),
  CR-507 (tests).

### G. Adversarial safety checks (do not skip)

```bash
cd /home/kensei/worktrees/kenseiagent-control-room-20260812
/home/kensei/repos/KenseiAgent/.venv/bin/python -m pytest \
  tests/control_room/test_actions.py tests/control_room/test_peer_live_shape.py -q
```

- EXPECTED: all pass. These pin: confirmation contract enforced in the
  router (not the UI), stale-revision guard (expected_revision
  mismatch rejected), cross-profile guard, duplicate-submit protection,
  unknown-action rejection, missing-runner -> typed unavailable (no
  fake success), peer provider JSON-shape parsing (held/queued states,
  malformed output degrades empty, never crashes).
- Verify NO raw SQLite writes in control_room/. Use an EXACT grep that
  cannot false-positive on the `updated_at` field:

```bash
cd /home/kensei/worktrees/kenseiagent-control-room-20260812
grep -rn "sqlite3\.\|cursor()\|\.execute(" control_room/ || echo "NO RAW SQL: PASS"
grep -rn "kanban_db\.\(create_task\|block_task\|unblock_task\|complete_task\)" control_room/executors.py
```

- EXPECTED (first): `NO RAW SQL: PASS` (zero matches).
  EXPECTED (second): the four call sites listed — every kanban mutation
  goes through `kanban_db.connect_closing()` (the established locked
  write route), never direct SQL. Any `.execute(` or `cursor()` in
  control_room/ is a finding.

## Honest limitations (documented, not hidden — verify they exist in
docs/control-room-release-decision.md section 4 (honest gaps) and
section 7 (follow-up tasks))

1. Post-activation hardening open by design: dashboard BFF has NO
   gateway RPC client; approvals/peer/delegation capabilities render as
   typed `unavailable` in V1. Verify control_room.py returns
   `{available: False}` for those three capabilities.
2. Plugin E2E-909 (hermes-peer install/enable/restart/uninstall) fails
   in THIS environment: hermes-peer is editable-installed in the hermes
   env, so `hermes plugins list` sees it even after uninstall. This is
   an environment interaction, not a control-room defect. Verify the
   other plugin E2Es pass (see C).
3. pnpm-lock.yaml tracked as the new standard (b855be2f8b) — see NOTE-A.
4. Live two-FULL-Hermes-gateway E2E (as opposed to two manager
   instances) was not run; CR-603 is proven at the manager-instance
   level plus the plugin's own real-binary disposable-home E2E.

## Time-box guidance

| Lane | Time | Action |
|---|---|---|
| State verification (both worktrees) | ~3 min | MUST run |
| A Python suites | ~1 min | MUST run |
| B TS mirror gate | ~20s | MUST run |
| C CR-603 live E2E | ~30s | MUST run |
| D TUI vitest | ~1 min | MUST run |
| E Desktop control-room tests | ~1 min | MUST run |
| F Dashboard BFF+SPA | ~2 min | MUST run |
| G Adversarial safety | ~1 min | MUST run |
| Full desktop suite (4669 tests) | ~7 min | RUN if time allows — else verify only the 15 known baseline failures match speech-text/searchable-select |
| Plugin walkie-talkie full suite | ~15 min | SKIP — verify C's two named E2E files only |

## Deliverable format

```
VERDICT: READY | NEEDS CORRECTION | NOT READY   (one line)

VERIFIED:
  - [claim] reproduced: command -> output summary

REFUTED / QUESTIONABLE:
  - [claim] expected X, got Y (command, output)

GAPS:
  - [claim] not verified because <reason> (time/env/credentials)

NOTES:
  - Anything the owner should know that the claims didn't cover.
```

The reviewer owns the READY verdict. The authoring agent must not
self-declare readiness.
