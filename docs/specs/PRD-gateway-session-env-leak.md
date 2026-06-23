# PRD — Gateway session-state process-global `os.environ` leak (the v3-latch bug class, generalized)

**Status:** v0.2 — Opus Pass-1 review complete; 3 BLOCKERS folded. The fix REFRAMED by the review: the root cause is **`_set_session_env` never binds the `HERMES_SESSION_ID` contextvar at all** (it passes `session_key` but omits `session_id`), so on a normal cached-agent gateway turn the SESSION_ID contextvar is `""` and only `os.environ` carries a (clobbered) value. The correct fix is to **bind `session_id` into the per-turn contextvar** (then the os.environ writes become droppable), not just to gate the writes. See Ground-Truth §0 + Resolved Decisions.
**Author:** Apollo
**Date:** 2026-06-22
**Related:** `PRD-send-message-origin-leak-v3-cron-session-latch.md` (v3 fixed ONE instance of this class — `HERMES_CRON_SESSION`). This PRD addresses the **remaining instances** + the **root-cause class** discovered during v3 closeout.

---

## 0. Ground-Truth (measured against the live tree, 2026-06-22 — read BEFORE the design)

The discovery that motivated this PRD: from a live interactive `#fix-issues` turn, `env | grep` in a `terminal` subprocess showed `HERMES_CRON_SESSION=1` **and** `HERMES_SESSION_ID/CHAT_ID/KEY` belonging to a *different concurrently-active session* (`ee660b`, running an every-2-min `no_agent` cron sweeper) — not my turn's session. The gateway is a **single process** that runs concurrent sessions (asyncio tasks) + an in-process cron tick; any code that writes session state to **process-global `os.environ`** clobbers it for every other concurrent session.

### 🔴 PHASE-0 / OPUS-PASS-1 GROUND-TRUTH CORRECTION (the reframe — 3 blockers)

The original v0.1 design assumed the `HERMES_SESSION_ID` *contextvar* carried the live id during a gateway turn (it doesn't), and would have **regressed** kanban stamping to empty. Corrected, code-verified facts:

1. **`_set_session_env` (`gateway/run.py:12825`) binds `session_key` but OMITS `session_id`.** So `set_session_vars` sets `_SESSION_ID` to `""` (the default), and `get_session_env("HERMES_SESSION_ID")` returns `""` (no os.environ fallback, because `""` ≠ `_UNSET`). The **only** writers of the real id are `set_current_session_id`, called ONLY at fresh-agent construction (`agent/agent_init.py:1087`) and on compression split (`agent/conversation_compression.py:984`) — **never on a cached-agent reuse turn** (the common path). ⇒ On a normal gateway turn the SESSION_ID contextvar is empty; only `os.environ["HERMES_SESSION_ID"]` (the clobbered global) carries a value. **This is the actual root cause** — and it means the fix is *bind the contextvar per turn* (then drop the os.environ writes), not just *gate the writes*.
   - Contrast: `session_key` IS passed by `_set_session_env`, so `_SESSION_KEY` is correctly bound every turn → D-2 (drop the SESSION_KEY os.environ write) was sound as written.
2. **The gateway-detection predicate `_HERMES_GATEWAY=1` LEAKS into kanban worker subprocesses.** Gateway sets it (`gateway/run.py:1217`); the restart watcher pops it (`:5107/:5137`) but the **kanban worker spawn does NOT** (`hermes_cli/kanban_db.py:6779` `env = dict(os.environ)`, no pop). So a worker is misclassified "gateway" → a predicate that suppresses the worker's os.environ write would break I3 (the worker's own-session stamping relies on that write). ⇒ Either pop `_HERMES_GATEWAY` in the worker spawn, OR don't rely on a write-suppression predicate for the worker path at all (the contextvar-binding reframe makes this moot — see D-1').
3. **A `get_session_env`-only isolation test is a FAKE-GREEN** — it passes on the *current buggy* code because `get_session_env` is contextvar-first and each task reads its own (empty/own) value; the bug lives in `os.environ`, which such a test never inspects. The real test must assert the **`os.environ` snapshot is not cross-contaminated** under two concurrent contexts, AND (post-fix) that `get_session_env("HERMES_SESSION_ID")` returns the live id (not `""`). Must be RED on current `main`.



**The session-context module was ALREADY migrated to contextvars for exactly this reason** (`gateway/session_context.py` docstring literally describes the bug). `set_session_vars` (the main per-turn binder) is **contextvar-only**. But residual `os.environ` writes remain.

### A. Live process-global `os.environ["HERMES_SESSION_*"]` WRITERS that run in the gateway process

| Site | Var | When | Notes |
|---|---|---|---|
| `gateway/run.py:15287` | `HERMES_SESSION_KEY` | every turn | comment: "Keep os.environ as fallback for CLI/cron" |
| `gateway/session_context.py:109` (`set_current_session_id`) | `HERMES_SESSION_ID` | unconditional, on every call | called by ↓ |
| `agent/agent_init.py:1089` | `HERMES_SESSION_ID` (via `set_current_session_id`) | agent init | in-gateway during agent construction |
| `agent/conversation_compression.py:986` | `HERMES_SESSION_ID` (via `set_current_session_id`) | on compaction split | in-gateway |

`set_session_vars` itself is clean (contextvar-only). `acp_adapter/server.py:1503,1525` and the `tui_gateway`/`cli`/`hermes_cli` writers run in **separate process models** (ACP server, TUI worker, CLI) — single-session-per-process, so os.environ is correct there and OUT OF SCOPE (see Non-Goals).

### B. The READERS — and why the live blast radius is SMALL (but not zero)

The consequential readers were **already migrated to contextvar-first**, which is why v3 + the live e2e showed *no actual misroute*:

| Reader | Var | Resolution | Risk |
|---|---|---|---|
| `tools/approval.py` `get_current_session_key` | `SESSION_KEY` | approval-contextvar → session-contextvar → os.environ | contextvar-first ✓ |
| `tools/approval.py:159` `_get_session_platform` | `SESSION_PLATFORM` | contextvar (`get_session_env`) → os.environ | contextvar-first ✓ |
| `tools/terminal_tool.py:208` | `SESSION_KEY` | `get_session_env` → os.environ | contextvar-first ✓ |
| `tools/send_message_tool.py` (v1/v2/v3) | platform/chat/cron | `get_session_env`/send-origin | contextvar-first ✓ |
| `tools/kanban_tools.py:124` `_stamp_worker_session_metadata` | `SESSION_ID` | **raw `os.environ.get`** | gated on `HERMES_KANBAN_TASK==task_id` → only true in **dispatcher-spawned worker subprocesses** (separate process, os.environ correct there). In the gateway this early-returns. **Low risk in practice.** |
| `tools/kanban_tools.py:744` (create-task) | `SESSION_ID` | `args.get("session_id") or` **raw `os.environ.get`** | stamps `worker_session_id` on a created task; on the **gateway** path (not a worker subprocess) this could stamp a *clobbered* session id. **The one real wrong-data risk.** |

**Honest severity:** because the dangerous readers (approval gating, send routing) are contextvar-first, the v3-class bleed does **not** currently cause a security misroute or a wrong approval decision. The measured real impacts are: (1) **diagnostic confusion** — the `terminal env|grep` probe lies (already documented in the misrouting skill); (2) **`kanban_tools.py:744` can stamp a wrong `worker_session_id`** on a task created from the gateway while a concurrent cron/session clobbered `HERMES_SESSION_ID` (data-attribution bug, not security). This honesty matters for scoping: this is a **correctness + hygiene + future-proofing** fix, not an active-incident security fix.

---

## 1. Summary & Goal

**Goal:** Eliminate process-global `os.environ` writes of per-session state in the **gateway process**, so concurrent sessions (and the in-process cron tick) can never clobber each other's session identity — closing the bug *class* that v3 fixed one instance of. Make the raw `os.environ` honest (so the diagnostic stops lying), and fix the one direct-reader (`kanban_tools`) that can act on a clobbered value.

**Why now:** v3 fixed `HERMES_CRON_SESSION`; the closeout re-verification surfaced that the *same class* persists for `HERMES_SESSION_ID`/`HERMES_SESSION_KEY`. Fix the class while the context is warm and the pattern (v3) is fresh.

This PRD is structured as **north-star (full class migration)** + **v0.1 cut (the two highest-value, lowest-risk fixes the user approved: #2 gateway-aware session-id/key writes, #3 kanban direct-reader)**.

---

## 2. Non-Goals

- **NOT** touching the **CLI / TUI-worker / ACP-server / oneshot** os.environ writes (`hermes_cli/*`, `tui_gateway/*`, `cli.py`, `acp_adapter/server.py`, `hermes_cli/oneshot.py`). Those are **single-session-per-process** entrypoints where os.environ is the correct, intended mechanism — migrating them is pure risk with no benefit. (Exception: if the audit finds one of these is *also* imported/run inside the gateway process, it gets pulled in — Phase 0 confirms none are.)
- **NOT** changing boot-time gateway config writes (`HERMES_QUIET`, `HERMES_EXEC_ASK`, `HERMES_MAX_ITERATIONS`, `HERMES_TIMEZONE`, media/display settings in `gateway/run.py:1264-1469`). Those are set **once at gateway startup** and are process-wide *by design* — not per-session, no clobber.
- **NOT** removing the `os.environ` **fallback reads** in `get_session_env`/`get_current_session_key`. The contextvar→os.environ→default chain stays (it's what makes CLI/cron/tests work); we only stop the gateway from *writing* the global.
- **NOT** a new env var or config surface.
- **NOT** re-opening v3 (its `HERMES_CRON_SESSION` fix stays; this is additive).

---

## 3. Constitution / Invariants

- **I1 (the core fix): no gateway code path writes per-session state to process-global `os.environ`.** The set of vars: `HERMES_SESSION_ID`, `HERMES_SESSION_KEY` (and, already done in v3, `HERMES_CRON_SESSION`).
  - *Why it matters:* process-global writes in a multi-session process are the entire bug class.
  - *Closeout proof:* `grep` shows the gateway-reachable writers (`gateway/run.py:15287`, `set_current_session_id`'s os.environ line, the two callers) no longer write os.environ when running in the gateway; a test that simulates two concurrent sessions in one process and asserts neither sees the other's `HERMES_SESSION_ID` in `get_session_env`.

- **I2 (CLI/cron/test back-compat preserved): single-process entrypoints (CLI, cron standalone, ACP, TUI worker) still get session state via os.environ where they rely on it.** The contextvar carries the value in-gateway; the os.environ fallback-read still resolves for non-gateway processes.
  - *Why it matters:* `set_current_session_id` exists *because* the CLI rotates sessions in-process and tools read `get_session_env("HERMES_SESSION_ID")` with an os.environ fallback. Breaking that regresses `/new`/`/resume`/`/branch` + compression-split session tracking on the CLI.
  - *Closeout proof:* the CLI session-rotation path still updates the resolvable session id (via contextvar in-process OR os.environ when not in a gateway); existing CLI/session tests stay green.

- **I3 (kanban worker attribution unchanged): a dispatcher-spawned kanban worker (separate process, `HERMES_KANBAN_TASK` set) still stamps its OWN `worker_session_id` correctly.**
  - *Why it matters:* the kanban fix (#3) must not break the legitimate worker-subprocess path (where os.environ IS correct).
  - *Closeout proof:* a test with `HERMES_KANBAN_TASK` set + a session id present asserts the worker stamps its own id; a gateway-path test (no `HERMES_KANBAN_TASK`, concurrent clobber) asserts it reads the *contextvar* session id, not the clobbered global.

- **I4 (no behavior change to v3 / send routing / approval gating): the contextvar-first readers keep resolving identically.** This PRD removes *writes*, not the read chain.
  - *Closeout proof:* v3 tests + send_message_origin + cron_approval suites stay green; a live bare `send_message` still routes to the current channel.

---

## 4. Resolved Decisions

- **D-1' (REFRAMED by Opus Pass-1 — supersedes the original D-1) — bind `session_id` into the per-turn contextvar, THEN drop the gateway os.environ writes.** The root cause is that `_set_session_env` (`gateway/run.py:12825`) never binds `session_id`. Fix:
  1. **`gateway/run.py:12825` — pass `session_id=context.session_id` (or the session-entry id) into `set_session_vars`.** This binds `_SESSION_ID` per-turn for EVERY turn (cached or fresh), exactly as `session_key` already is. Now `get_session_env("HERMES_SESSION_ID")` returns the live id on the normal path — which is the precondition that makes D-3 (kanban) work at all.
  2. **`set_current_session_id` (`gateway/session_context.py:109`) — keep the contextvar write; make the `os.environ` write gateway-aware** (write os.environ only when NOT `_HERMES_GATEWAY`). With the contextvar now bound per-turn, the os.environ write is only needed by single-process entrypoints (CLI rotation, cron-standalone).
  - This is strictly better than "just gate the writes": it fixes the *missing contextvar bind* (the real bug) so every contextvar-first reader gets the right id, and reduces os.environ to a CLI/cron-only fallback.
  - *Rejected — only gate the writes (original D-1):* leaves the SESSION_ID contextvar empty on cached turns, so D-3 would stamp `""`. Falsified by the review.
- **D-2 — `gateway/run.py:15287` (`HERMES_SESSION_KEY` per-turn write) is DROPPED in-gateway (confirmed sound).** `_set_session_env` already binds `session_key` into the contextvar every turn, and the only gateway-reachable readers (`get_current_session_key` `approval.py:148`, `terminal_tool.py:206`) are contextvar-first. So the per-turn os.environ write is pure pollution → drop it. (The CLI/cron paths set their own `HERMES_SESSION_KEY` via their own entrypoints; not affected.)
- **D-3 — kanban `:744` + `:124` (#3): read the contextvar via `get_session_env`, not raw os.environ.** `args.get("session_id") or os.environ.get("HERMES_SESSION_ID")` → `args.get("session_id") or get_session_env("HERMES_SESSION_ID")`. **Now valid because D-1' binds the contextvar per turn.** The worker-subprocess path (I3) still resolves: workers run in a separate process with no bound contextvar → `get_session_env` falls through to that process's correct os.environ `HERMES_SESSION_ID`.
  - 🔴 **B2 mitigation (required):** the kanban worker spawn (`hermes_cli/kanban_db.py:6779`) inherits `_HERMES_GATEWAY=1` un-popped. To keep the worker correctly classified single-session (so D-1''s gated os.environ write fires *in the worker*, populating the fallback I3 depends on), **pop `_HERMES_GATEWAY` from the worker spawn env** (mirroring `gateway/run.py:5107/5137`). Without this, a worker would skip its os.environ write AND have no contextvar → stamp nothing.
- **D-4 — north-star (#4) is its own roadmap, not v0.1.** Unchanged. v0.1 = D-1' + D-2 + D-3 + B2 pop. Broader audit ships per-trigger.

---

## 5. Architecture / Design

### The pattern (mirrors v3 exactly)
v3's fix: a process-global flag → task-isolated contextvar, set/cleared per scope, readers use a context-aware helper. This PRD applies the same shape to the residual session-id/key writes:
- **Writes:** gateway-aware — contextvar always; os.environ only in single-session-per-process contexts.
- **Reads:** already contextvar-first (no change), plus the one kanban raw-reader migrated to `get_session_env`.

### Edits (v0.1 cut)
1. **`gateway/session_context.py`** — `set_current_session_id`: gate the `os.environ["HERMES_SESSION_ID"]` write behind a "not in concurrent gateway" check; always set the contextvar. Add a small helper `_is_concurrent_gateway()` (or reuse the existing signal) — Phase-0 picks the exact predicate.
2. **`gateway/run.py:15287`** — drop or gate the per-turn `os.environ["HERMES_SESSION_KEY"]` write (D-2; the contextvar is already set by `_set_session_env`).
3. **`tools/kanban_tools.py:744` + `:124`** — `os.environ.get("HERMES_SESSION_ID")` → `get_session_env("HERMES_SESSION_ID")` (contextvar-first, os.environ fallback preserves the worker-subprocess path). *Valid only because edit #0 binds the contextvar per turn.*
0. **`gateway/run.py:12825` (`_set_session_env`) — pass `session_id=` into `set_session_vars`** (the root-cause edit; do this FIRST). Plus **`hermes_cli/kanban_db.py:6779` — pop `_HERMES_GATEWAY` from the worker spawn env** (B2).
4. **Tests** — concurrent-session **os.environ** isolation test (must be RED on current main); kanban gateway-vs-worker test; CLI/cron back-compat assertion.

### North-star (#4) — full class migration roadmap
| Version | What ships | Trigger | Maps to |
|---|---|---|---|
| **v0.1** (this build) | gateway `HERMES_SESSION_ID`/`KEY` writes made gateway-aware + kanban direct-reader migrated | now (approved) | §6 Phases 1-3 |
| v0.2 | audit + migrate any remaining gateway-reachable per-session os.environ writer found by an enforcement grep test | if Phase-0/closeout grep finds a writer beyond the 4 mapped | §6 Phase 4 (audit) |
| v0.3 | a CI lint/test that FAILS if a new `os.environ["HERMES_SESSION_*"] =` write is added to a gateway-reachable module | if a regression reintroduces the class | future |

---

## 6. Implementation Phases

- **Phase 0 — ground-truth (DONE in Opus Pass-1; see §0 correction).** Confirmed: `_set_session_env` omits `session_id`; `_HERMES_GATEWAY=1` is the gateway signal but leaks into kanban workers (un-popped); a `get_session_env`-only isolation test is a fake-green. Design reframed (D-1').

- **Phase 1 — bind `session_id` per turn + drop/gate the writes (the root-cause cut, D-1' + D-2).**
  - **Edit 1a:** `_set_session_env` (`gateway/run.py:12825`) passes `session_id=<session-entry id>` into `set_session_vars`. Also audit the other in-gateway `set_session_vars` callers (`gateway/platforms/api_server.py:3521,3791`, and cron `run_job` `cron/scheduler.py:1622`) and pass `session_id` where the session id is available (RC-5).
  - **Edit 1b:** `set_current_session_id` keeps the contextvar write; gates the os.environ write behind `not _HERMES_GATEWAY`.
  - **Edit 1c:** drop the `gateway/run.py:15287` `HERMES_SESSION_KEY` os.environ write.
  - *Unit/script check:* in a bound gateway turn context, `get_session_env("HERMES_SESSION_ID")` returns the live id (NOT `""`); `os.environ["HERMES_SESSION_ID"]` is NOT written by the gateway path. In a non-gateway (CLI, `_HERMES_GATEWAY` unset) context, `set_current_session_id` DOES write os.environ.
  - *E2E/integration (REQUIRED — concurrency, the RED-on-main test):* two concurrent contexts each bind their own session via the real `set_session_vars`(`session_id=`)/`set_current_session_id`; **assert the raw `os.environ["HERMES_SESSION_ID"]` is NOT cross-contaminated** (the real bug surface — a `get_session_env`-only assert is a fake-green per §0.3) AND each context's `get_session_env` returns its own id. Must FAIL on current `main` (where the unconditional os.environ write clobbers).
  - *Negative/adversarial:* in-gateway cron tick concurrent with an interactive turn — neither sees the other's `HERMES_SESSION_ID` (contextvar or os.environ).
  - *Verify with:* `pytest tests/gateway/test_session_id_isolation.py -o 'addopts=' -q` → pass; confirm RED on stash of the fix.

- **Phase 2 — kanban worker `_HERMES_GATEWAY` pop (B2) + verify back-compat.**
  - **Edit 2a:** `hermes_cli/kanban_db.py:6779` worker spawn pops `_HERMES_GATEWAY` from the child env (so the worker is single-session-classified and its gated os.environ write fires).
  - *Unit/script check:* the worker spawn env dict does not contain `_HERMES_GATEWAY`.
  - *Negative/adversarial:* a worker process (`_HERMES_GATEWAY` absent, `HERMES_KANBAN_TASK` set, session id in os.environ, no contextvar) → `set_current_session_id` writes os.environ → `get_session_env("HERMES_SESSION_ID")` resolves the worker's own id.
  - *Verify with:* `pytest tests/tools/test_kanban*.py -o 'addopts=' -q` → pass.

- **Phase 3 — kanban direct-reader migration (#3, D-3).** `:744` + `:124` → `get_session_env`.
  - *Unit/script check:* `_create_task` with no `args["session_id"]` + a bound contextvar `HERMES_SESSION_ID` stamps the contextvar value (now non-empty thanks to Phase-1); with neither, stamps None.
  - *E2E/integration (REQUIRED — the wrong-data path):* task created in bound session A while session B's id sits in os.environ (clobbered); assert the created task stamps A's id (contextvar), not B's. Must FAIL on current main (where it reads raw os.environ = B).
  - *Negative/adversarial (I3):* worker subprocess (no contextvar, id in os.environ via Phase-2's write) still stamps its own id via the os.environ fallback.
  - *Verify with:* `pytest tests/tools/test_kanban_session_attribution.py -o 'addopts=' -q` → pass.

- **Phase 4 — north-star audit (v0.2, deferred unless the enforcement test finds more).** A `test_no_gateway_session_env_writes` grep-test that asserts no gateway-reachable module writes `os.environ["HERMES_SESSION_*"]` outside the sanctioned single-process entrypoints (allowlist CLI/TUI/ACP/oneshot + the `except`-fallback lines in `agent_init.py:1089`/`conversation_compression.py:986`).
  - *Verify with:* the grep-test passes (current violators fixed) and fails if a new one appears.

---

## 7. Security, Privacy, Ops, Observability

- **Security posture (honest):** this is **not** an active security-incident fix — the dangerous readers (approval gating, send routing) are already contextvar-first, so the bleed does not currently cause a misroute or wrong approval. It IS a correctness/hygiene fix that (a) closes a latent footgun (any *future* raw-os.environ reader would inherit the clobber), (b) fixes a real data-attribution bug (`kanban:744` wrong `worker_session_id`), (c) makes the diagnostic honest.
- **Ops/rollback:** code change to the live editable install; deploy = fork PR merge + **gateway restart** (privileged §7 — PAUSE for Ace's go). Rollback = revert + restart.
- **Observability:** after fix, `terminal env|grep HERMES_SESSION_*` reflects only the true (or empty) per-process state; the misrouting skill's "cross-check execute_code" caveat can note the diagnostic is reliable again for non-clobbered vars.

---

## 8. Risks & Mitigations

- **R1 — gateway-detection predicate is wrong → CLI session rotation regresses (I2) OR gateway still writes os.environ.** *Mitigation:* Phase-0 confirms the predicate empirically (set in gateway, unset in CLI/cron) before coding; Phase-1 has explicit both-directions tests (gateway: os.environ untouched; CLI: os.environ updated).
- **R2 — a raw-os.environ reader I didn't find acts on the dropped SESSION_KEY write.** *Mitigation:* Phase-0 greps every reader; the only gateway-reachable one (`terminal_tool:208`) is contextvar-first. The grep is the gate, not my memory.
- **R3 — kanban worker-subprocess path breaks (I3).** *Mitigation:* `get_session_env` keeps the os.environ fallback, so the worker (no contextvar) resolves identically; Phase-3 negative test proves it.
- **R4 — scope creep into the CLI/TUI/ACP writers.** *Mitigation:* hard Non-Goal; those are single-session-per-process by design. Phase-4 audit allowlists them explicitly.
- **R5 — over-claiming severity.** *Mitigation:* §7 + Ground-Truth state plainly this is hygiene/correctness, not an active security fix. Don't sell it as a vuln patch.

---

## 9. Open Questions

- **OQ1 — RESOLVED (Opus Pass-1):** predicate is `_HERMES_GATEWAY` (set `gateway/run.py:1217`, unset in CLI/cron-standalone) — BUT it leaks into kanban worker subprocesses un-popped, so Phase-2 must pop it. The contextvar-binding reframe (D-1') makes the gateway *reads* correct regardless; the predicate only governs the CLI/cron os.environ-fallback write.
- **OQ2 — RESOLVED (Opus Pass-1):** `gateway/run.py:15287` SESSION_KEY write can be **dropped** in-gateway — `_set_session_env` already binds the `session_key` contextvar and the only gateway-reachable readers are contextvar-first.

---

## 10. Acceptance Criteria

- [ ] **AC0 (root cause, RC by Pass-1):** On a normal cached-agent gateway turn, `get_session_env("HERMES_SESSION_ID")` returns the live session id (NOT `""`), because `_set_session_env` now binds `session_id`. Evidence: `tests/gateway/test_session_id_isolation.py::test_session_id_bound_on_turn` (RED on current main where it returns `""`).
- [ ] **AC1 (I1 — the REAL isolation test, not the fake-green):** Two concurrent in-process contexts each binding their own session leave `os.environ["HERMES_SESSION_ID"]` **NOT cross-contaminated** AND each context's `get_session_env` returns its own id. Evidence: `::test_concurrent_sessions_os_environ_not_contaminated` — must be RED on current main (the unconditional os.environ write clobbers); a `get_session_env`-only assertion is explicitly insufficient (§0.3).
- [ ] **AC2 (I1):** `grep` shows no gateway-reachable per-turn `os.environ["HERMES_SESSION_ID"|"HERMES_SESSION_KEY"] =` write (the SESSION_KEY write dropped; the SESSION_ID write gated behind `not _HERMES_GATEWAY`). Evidence: grep + Phase-4 enforcement test.
- [ ] **AC3 (I2):** CLI + cron-standalone (and the popped kanban worker) session rotation still resolves the active session id via the gated os.environ write. Evidence: existing CLI/session tests green + a non-gateway (`_HERMES_GATEWAY` unset) `set_current_session_id` test asserting os.environ IS updated.
- [ ] **AC4 (I3 — #3):** A task created from the gateway under concurrency stamps the correct (contextvar) `worker_session_id`, not a clobbered global (RED on main); a kanban worker subprocess (B2-popped, no contextvar) still stamps its own via os.environ. Evidence: `tests/tools/test_kanban_session_attribution.py` (both arms).
- [ ] **AC4b (B2):** the kanban worker spawn env does not contain `_HERMES_GATEWAY`. Evidence: `::test_worker_spawn_drops_gateway_flag`.
- [ ] **AC5 (I4):** v3 + send_message_origin + cron_approval suites stay green; a live bare `send_message` routes to the current channel post-deploy. Evidence: pytest + staged live send.
- [ ] **AC6 (honest scope):** §7 and Ground-Truth state the severity accurately (hygiene/correctness + one data-attribution bug, not an active security misroute). Evidence: inspection.

---

## Appendix — the v3 precedent
v3 (`HERMES_CRON_SESSION`) is the proven template: process-global flag → `_VAR_MAP` ContextVar + `is_cron_session()` context-aware reader + set/clear per scope + delegate-boundary rebind. This PRD reuses that exact shape for the residual `HERMES_SESSION_ID`/`KEY` writes. The key difference: the session-id/key **readers are already contextvar-first**, so this is mostly *stopping the writes* + one reader migration — smaller than v3.
