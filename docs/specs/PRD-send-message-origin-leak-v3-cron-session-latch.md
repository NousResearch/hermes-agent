# PRD — send_message origin-leak v3: the `HERMES_CRON_SESSION` process-global latch

**Status:** v0.2 — Opus Pass-1 review complete; all findings folded (B1 security-regression blocker → Phase 5 fix; B2 fake-green → AC2b real proof; RC-3 registration gate → AC9; RC-4 ordering constraint). Ready for build.
**Author:** Apollo
**Date:** 2026-06-22
**Supersedes-as-dominant-cause:** v1 (PR #71, main-turn bare target) and v2 (PR #83, subagent context-loss) — both real but **secondary**. This v3 is the dominant, proven cause of the recurring leak.
**Related:** `docs/specs/PRD-send-message-origin-leak.md` (v1), `docs/specs/PRD-send-message-origin-leak-v2.md` (v2); skill `agent-message-misrouting-diagnosis`.

---

## 1. Summary & Goal

**Symptom:** An interactive (live, user-facing) agent turn's `send_message` with a bare/blank `target` silently routes to the Discord/Telegram **home channel** instead of the channel the user is in — intermittently, surviving restarts, and on **main turns** (not just subagents). It recurred after both v1 and v2 shipped and was caught **live, in Apollo's own main turn**, on the v2 gateway (PID 95240, booted 03:28 PT 2026-06-22).

**Root cause (PROVEN, not theorized):**
1. The gateway runs the **cron scheduler in-process** — `gateway/run.py:17161-17186` ticks `cron.scheduler.tick()` on a background thread inside the same process that serves live messaging turns.
2. The first cron job to run executes `cron/scheduler.py:1586`:
   ```python
   # This env var is process-wide and persists for the lifetime of the
   # scheduler process — every job this process runs is a cron job.
   os.environ["HERMES_CRON_SESSION"] = "1"
   ```
   It sets the flag in **process-global `os.environ`**, on the explicit (and, in this topology, **false**) assumption that "the scheduler process only runs cron jobs."
3. The flag is **never cleared** — `run_job`'s `finally` restores `TERMINAL_CWD` and the per-job ContextVars, but not `HERMES_CRON_SESSION` (only test code pops it).
4. After the first cron tick of a boot, **every** interactive turn in that gateway process sees `HERMES_CRON_SESSION=1`. `send_message`'s `_has_messaging_origin()` (tools/send_message_tool.py:176) checks the cron-guard **first** → returns `False` → `_resolve_send_target` returns `NOT_IN_TURN` → home fallback. The v2 tripwire `_warn_unexpected_home_fallback` shares the same cron-first early-return, so it stays **silent**.

**Proof:** A `terminal` subprocess (inherits the gateway's real `os.environ`) returns `HERMES_CRON_SESSION=1` on the live gateway right now; a bare `send_message` from Apollo's live main turn returned `"Sent to discord home channel (chat_id: 1502228850338435153)"` while the user was in `#fix-issues` (`1517844402897424504`). The flag is latched and never cleared in production code.

**Why it explains everything the subagent theory could not:** main-turn leaks (not just subagents); intermittency (only after the first cron tick); restart-survival (re-arms on the next tick); and the "RC#2 main-turn seam" that looked real in v2 but was unprovable — it WAS real, just a stuck flag, not a code seam.

**Goal:** Make `HERMES_CRON_SESSION` **task/thread-isolated** so a cron job's flag can never bleed into a concurrent interactive turn in the same process — mirroring the ContextVar migration the same function already performs for the cron *delivery* vars one line below. Plus close the tripwire's blind spot so any future leak self-diagnoses.

---

## 2. Non-Goals

- **NOT** re-opening or reverting v1/v2 — both fixes stay; they close real (secondary) seams. v3 is additive.
- **NOT** changing cron approval semantics (`approvals.cron_mode`), delivery routing, or any cron job's behavior. A cron job must still be recognized as cron for the entire duration of its own run.
- **NOT** moving the cron scheduler out of the gateway process (a much larger architectural change; the in-process tick is intentional). v3 fixes the flag's *scope*, not the topology.
- **NOT** disabling `send_message` or adding new config surface. No new `HERMES_*` env var (the fix *removes* a process-global one).

---

## 3. Constitution / Invariants

- **Invariant I1 (the fix): a cron job's cron-session marker is visible to that job's own execution lineage ONLY, never to a concurrent interactive gateway turn in the same process.**
  - *Why it matters:* the whole leak. Process-global state in a process that serves both cron and live turns is the defect.
  - *Closeout proof:* a test that sets the cron marker in a simulated cron lineage and asserts a *separate* (interactive) context reads it as **absent** → its `send_message` resolves to ORIGIN, not home. Plus: `grep` shows no production code writes `os.environ["HERMES_CRON_SESSION"]`.

- **Invariant I2 (no regression of cron approval gating — MAIN turn AND subagents): inside a cron job's run AND inside any subagent that cron job spawns, every approval/safety consumer that branches on `HERMES_CRON_SESSION` continues to see it as truthy.**
  - *Why it matters:* `HERMES_CRON_SESSION` gates dangerous-command / code-exec auto-approve-or-deny in `tools/approval.py` (lines 147, 1187, 1424, 1713) — OWASP LLM06 excessive-agency boundary. Cron default is **deny**. A fix that makes a cron job (or its child) *stop* being recognized as cron would either block jobs forever (pending approval, no listener) or **strip their `cron_mode` protection → auto-approve dangerous commands**. 🔴 **Opus Pass-1 BLOCKER (B1):** the naive "ContextVar-only, no os.environ" change does exactly this for **cron-spawned subagents** — they run in a bare `ThreadPoolExecutor` worker that does **not** inherit contextvars (`tools/delegate_tool.py:1312` comment: *"contextvars do NOT cross the ThreadPoolExecutor boundary"*; no `copy_context()` in the file). **Today** the process-global `os.environ` flag is visible in that child thread, so cron subagents are correctly denied; removing the global without a child-boundary rebind flips them deny→auto-approve. The fix MUST capture-and-rebind the cron marker at the delegate boundary, exactly as v2's `_bind_child_send_origin` does for send-origin (`tools/delegate_tool.py:1490-1513,1712`).
  - *Closeout proof:* (a) a test running the 4 approval readers inside the cron **main** lineage → cron=True; **(b) a test that drives `run_job` → `delegate_task`/`_run_single_child` under `cron_mode=deny` and asserts a dangerous command AND `execute_code` inside the spawned CHILD are still BLOCKED** (the load-bearing B1 proof — a main-thread os.environ-set test is INSUFFICIENT and must NOT be cited for this); (c) existing `tests/tools/test_cron_approval_mode.py` stays green.

- **Invariant I3 (cron delivery unchanged): cron output still delivers to `job["origin"]` via `_resolve_origin(job)` + `HERMES_CRON_AUTO_DELIVER_*`.**
  - *Closeout proof:* cron delivery tests stay green; a smoke cron job still posts to its configured destination.

- **Invariant I4 (worker-thread reuse safety): the persistent cron `ThreadPoolExecutor` reuses worker threads; the cron marker must be reset at job end so a later job (or any work) on the same worker thread does not inherit a stale marker.**
  - *Why it matters:* ContextVars set via `.set()` persist in a thread's context across tasks unless reset; a reused worker thread would otherwise carry a previous job's marker.
  - *Closeout proof:* a test that runs job A (sets marker), completes it, then asserts the marker is cleared on the same lineage before job B; the reset uses the `var.reset(token)` token saved at set time.

- **Invariant I5 (test back-compat): tests and any separate-process scheduler that set `os.environ["HERMES_CRON_SESSION"]` still behave correctly.**
  - *Why it matters:* `tests/tools/test_cron_approval_mode.py`, `test_send_message_origin.py`, `test_subagent_send_origin.py`, `test_hardline_blocklist.py` set the flag via `os.environ`/`monkeypatch`. The context-aware read must fall back to `os.environ` when the ContextVar is unset.
  - *Closeout proof:* those suites stay green unmodified.

---

## 4. Resolved Decisions

- **D-1 — ContextVar migration, mirroring the existing delivery-var pattern (chosen).** The fix adds `HERMES_CRON_SESSION` to `gateway/session_context.py`'s `_VAR_MAP` as a ContextVar, sets it in `run_job` (worker-thread lineage), clears it in the `finally`, and switches the readers to a context-aware read. This is the *exact* pattern the code already applies to `HERMES_CRON_AUTO_DELIVER_*` three lines below the bug, with the comment *"Use ContextVars for per-job session/delivery state so parallel jobs don't clobber each other's targets (os.environ is process-global)."* The author recognized the bug class for the delivery vars and left this one flag behind.
  - *Rejected — set/clear in `os.environ` finally:* unsafe under parallel cron jobs (job B's `finally` would clear job A's flag mid-flight, flipping A's approval gating) AND still process-global, so it would still leak into interactive turns *during* any cron job. The code comment itself says os.environ is the wrong tool here.
  - *Rejected — reference-count the os.environ flag:* still process-global → still leaks into interactive turns while any cron job runs. Only fixes the permanent-latch, not the concurrent-leak. Partial.
- **D-2 — contextvar-only set in `run_job` (do NOT also write os.environ) — PAIRED with a delegate-boundary rebind (B1 fix).** `run_job` sets *only* the ContextVar (writing os.environ would re-introduce the interactive-turn leak). **But because cron jobs can spawn subagents through a bare `ThreadPoolExecutor` that does not inherit contextvars, the cron marker MUST also be captured at spawn and re-set inside the child-run wrapper** — exactly as v2 does for send-origin. Without this rebind, D-2 is a security regression (cron subagents lose deny-gating). Tests that set os.environ still work via the read's fallback (I5).
- **D-3 — single context-aware helper for all readers.** Add one helper (e.g. `gateway/session_context.is_cron_session()` → `is_truthy_value(get_session_env("HERMES_CRON_SESSION", ""))`) and replace all 6 `env_var_enabled("HERMES_CRON_SESSION")` sites. `get_session_env` already does ContextVar→os.environ→default, giving I1 isolation and I5 back-compat for free. Grep-gate that no production reader uses the raw `env_var_enabled("HERMES_CRON_SESSION")` afterward.
- **D-4 — tripwire blind-spot.** With I1, an interactive turn no longer reads cron=true, so `_warn_unexpected_home_fallback` will correctly fire on any *remaining* leak path (it no longer early-returns for interactive turns). No further tripwire change is strictly required, but the spec keeps the tripwire and adds a test that it FIRES for an interactive turn that falls to home (proving the blind spot is gone).

---

## 5. Architecture / Design

### Control flow today (buggy)
```
gateway process (PID 95240)
├── asyncio loop: serves interactive turns ──────────────┐
│     send_message(bare) → _has_messaging_origin()        │  reads os.environ
│       step 1: env_var_enabled("HERMES_CRON_SESSION") ───┼──► "1"  → False → HOME ❌
└── background thread: cron_tick() → run_job()            │
      os.environ["HERMES_CRON_SESSION"] = "1"  (never cleared, process-global) ──┘
```

### Control flow after fix
```
gateway process
├── asyncio loop: interactive turn (its own context)
│     send_message(bare) → is_cron_session()
│       get_session_env("HERMES_CRON_SESSION") → ContextVar UNSET → os.environ UNSET → "" → False-for-cron
│       → _get_session_platform() truthy → True → ORIGIN ✅ (current channel)
└── cron worker thread T (its own context)
      run_job(): set ContextVar HERMES_CRON_SESSION="1" (token saved) — BEFORE copy_context
        main agent run + tool calls (copy_context inherits T's context) → is_cron_session() → True ✅
        delegate_task child → bare ThreadPoolExecutor (does NOT inherit T's context!)
          child wrapper: re-set HERMES_CRON_SESSION from captured _is_cron_child → is_cron_session() True ✅ (B1 fix)
          child finally: reset
      finally: ContextVar.reset(token)  → T clean for reuse (I4)
```

### Edits (minimal-diff)
1. **`gateway/session_context.py`**
   - Add `_CRON_SESSION: ContextVar = ContextVar("HERMES_CRON_SESSION", default=_UNSET)` and **register in `_VAR_MAP`** (🔴 **hard gate, Opus B-3:** without the `_VAR_MAP` registration, `get_session_env` *always* falls through to `os.environ` → isolation silently broken AND the main cron agent reads cron=False (loses gating). A Phase-1 thread-isolation assertion must catch a missing registration.)
   - Add `def is_cron_session() -> bool:` → `from utils import is_truthy_value; return is_truthy_value(get_session_env("HERMES_CRON_SESSION", ""), default=False)`.
   - Expose a token-based `set_cron_session()`/`clear_cron_session(token)` pair (like `set_send_origin`/`clear_send_origin`), reusing the `_VAR_MAP[...]` mechanism the delivery vars use.
2. **`cron/scheduler.py`** (`run_job`)
   - Replace `os.environ["HERMES_CRON_SESSION"] = "1"` (line 1586) with a ContextVar `set_cron_session()` that returns a reset token (stored alongside `_ctx_tokens`). 🔴 **Ordering constraint (Opus RC-4):** the `set` MUST stay **before** the per-job `_cron_context = copy_context()` (cron/scheduler.py:~1905) so the main agent's run inherits it. State this inline in the code.
   - In the `finally`, reset the cron-session ContextVar (next to `clear_session_vars(_ctx_tokens)`).
3. **`tools/delegate_tool.py`** (🔴 **B1 fix — the security-load-bearing edit**): give the cron marker the **same capture-and-rebind** treatment v2 gave send-origin. At spawn capture (`~:1326-1340`, where `_blackbox_parent_*` is snapshotted in the parent thread that DOES hold the cron ContextVar), capture `child._is_cron_child = is_cron_session()`. In the child-run wrapper `_run_with_thread_capture` (`:1712-1721`, next to `_bind_child_send_origin`), if `_is_cron_child` is truthy, `set_cron_session()` (token saved) and reset in the `finally`. This restores the deny-gating that the removed process-global os.environ flag used to provide for cron subagents. Applies to the synchronous, batch, AND background spawn paths (they all funnel through `_run_single_child`/`_run_with_thread_capture`).
4. **`tools/approval.py`** (lines 147, 1187, 1424, 1713) and **`tools/send_message_tool.py`** (lines 176, 277): replace `env_var_enabled("HERMES_CRON_SESSION")` with `is_cron_session()` (import from `gateway.session_context`). Same truthiness semantics + ContextVar isolation + os.environ fallback.
5. **No change** to the v1/v2 resolver logic beyond the reader swap; the cron-guard stays *first* (its ordering is correct — a real cron job must still home).

---

## 6. Implementation Phases

- **Phase 1 — ContextVar + helper in `session_context.py`.** Add `_CRON_SESSION` to `_VAR_MAP`, add `is_cron_session()`, add setter/reset (token-based).
  - *Unit/script check:* `set` then `is_cron_session()` True; `reset(token)` then False; with ContextVar unset, `os.environ["HERMES_CRON_SESSION"]="1"` → `is_cron_session()` True (fallback, I5).
  - *E2E/integration:* N/A (pure context primitive; exercised in Phase 3/4).
  - *Negative/adversarial:* a second thread with no set sees `is_cron_session()` False even while the first thread has it set (thread isolation).
  - *Verify with:* `pytest tests/gateway/test_session_context_cron.py -o 'addopts=' -q` → all pass.

- **Phase 2 — migrate readers (approval + send_message).** Swap all 6 `env_var_enabled("HERMES_CRON_SESSION")` → `is_cron_session()`.
  - *Unit/script check:* existing `tests/tools/test_cron_approval_mode.py`, `test_send_message_origin.py`, `test_subagent_send_origin.py`, `test_hardline_blocklist.py` stay green (they set os.environ → fallback covers them).
  - *Negative/adversarial:* `grep -rn 'env_var_enabled("HERMES_CRON_SESSION")' tools/ gateway/ cron/` returns **zero** production hits (only the helper + tests reference the name).
  - *Verify with:* `pytest tests/tools/test_cron_approval_mode.py tests/tools/test_send_message_origin.py tests/tools/test_subagent_send_origin.py tests/tools/test_hardline_blocklist.py -o 'addopts=' -q` → all pass; grep returns 0.

- **Phase 3 — `run_job` set/reset (contextvar-only, no os.environ).** Replace the os.environ write with a ContextVar set (token saved); reset in `finally`.
  - *Unit/script check:* a unit that calls `run_job` setup/teardown (or a thin extract) and asserts the marker is set during and reset after.
  - *E2E/integration (REQUIRED — changes routing-adjacent process state):* drive a real cron job through `cron.scheduler.tick()` (cheap no-op job), then assert from a *separate* context that `is_cron_session()` is False AND `os.environ.get("HERMES_CRON_SESSION")` is None (the process is no longer poisoned). Then a bare interactive-context `_resolve_send_target("discord")` returns ORIGIN, not NOT_IN_TURN.
  - *Negative/adversarial:* run two cron jobs in parallel through the pool; each sees its own marker True during its run; neither's reset clears the other; after both, the marker is gone.
  - *Verify with:* `pytest tests/cron/test_cron_session_isolation.py -o 'addopts=' -q` → all pass.

- **Phase 4 — tripwire blind-spot test + the integration proof.** Assert `_warn_unexpected_home_fallback` (or the resolver) now FIRES/returns home only for a genuine no-origin interactive case, and that an interactive turn after a cron tick resolves to ORIGIN.
  - *Unit/script check:* simulate "cron tick happened" (set then reset the cron marker via the real path), then in a fresh interactive context with `HERMES_SESSION_PLATFORM=discord, HERMES_SESSION_CHAT_ID=<chan>` bound, assert `_resolve_send_target("discord")` → `(ORIGIN, <chan>, …)`. Pre-fix (cron marker latched in os.environ) this returns `NOT_IN_TURN`; the test must be RED on the old code and GREEN on the new (prove the seam is load-bearing).
  - *E2E/integration (REQUIRED):* on the live gateway (gated, post-approval restart), after a cron tick, a bare `send_message` from an interactive turn lands in the **current** channel; `terminal` shows `HERMES_CRON_SESSION` unset in the gateway os.environ.
  - *Negative/adversarial:* a real cron job's bare `send_message` still goes to its `job["origin"]`/home delivery path (cron unchanged).
  - *Verify with:* `pytest tests/tools/test_send_message_origin.py -o 'addopts=' -q` + the live e2e (staged so Ace sees the message land in the right channel).

- **Phase 5 — 🔴 B1 fix: cron-marker capture-and-rebind at the delegate boundary (the security-load-bearing phase).** Capture `child._is_cron_child = is_cron_session()` at spawn (parent thread, holds the cron ContextVar); in `_run_with_thread_capture`, re-`set_cron_session()` when `_is_cron_child` is truthy, reset in `finally`. Covers sync/batch/background spawn paths.
  - *Unit/script check:* a child wrapper invoked with `_is_cron_child=True` → `is_cron_session()` True inside the child callable; reset after; with `_is_cron_child=False` → False inside the child.
  - *E2E/integration (REQUIRED — security boundary):* drive `run_job` (cron, `approvals.cron_mode=deny`) → a real `delegate_task` child that attempts a **dangerous command** AND an **`execute_code`**; assert BOTH are **BLOCKED inside the child** (not auto-approved). This is the B1 proof — it must be RED on the contextvar-only-without-rebind code (child auto-approves) and GREEN with the rebind. A main-thread os.environ-set test does NOT satisfy this.
  - *Negative/adversarial:* a NON-cron (interactive) parent spawning a child → child `is_cron_session()` False (we did not over-bind cron onto interactive subagents); and a grandchild of a cron child also inherits cron=True (transitive, like send-origin's grandchild chaining).
  - *Verify with:* `pytest tests/tools/test_cron_subagent_approval.py -o 'addopts=' -q` → all pass (esp. the deny-preserved child test).

---

## 7. Security, Privacy, Ops, Observability

- **Security (the load-bearing one):** `HERMES_CRON_SESSION` is a **security gate** (approval auto-mode for dangerous commands/code-exec). The migration must keep it truthy for the cron job's own lineage (I2) and absent for interactive turns (I1). The negative tests assert both directions. No secret/PII in any log; the tripwire already logs no message body.
- **Ops/rollback:** pure code change to the live editable install; deploy = fork PR merge + **gateway restart** (privileged per SOUL §7 — PAUSE for Ace's go). Rollback = revert the commit + restart. The bug is intermittent, so post-deploy verification must include a forced cron tick followed by an interactive bare-send landing in the right channel.
- **Observability:** after fix, the v2 tripwire becomes meaningful for interactive turns again — any *future* home-fallback despite a resolvable session logs a one-line diagnostic (no PII).

---

## 8. Risks & Mitigations

- **R1 — ContextVar not visible to a cron reader that runs off the worker-thread lineage. ⚠️ This is the B1 blocker, now fixed in Phase 5.** The MAIN cron agent's tool calls DO inherit the marker (via `tool_executor`'s `copy_context()`/`propagate_context_to_thread` at submit + the per-job `copy_context()` at scheduler.py:~1905). **But a cron-spawned SUBAGENT runs in a bare `ThreadPoolExecutor` that does NOT inherit contextvars** (`delegate_tool.py:1312`) — so the marker does not reach the child without an explicit rebind. *Mitigation:* Phase 5 captures-and-rebinds the marker at the delegate boundary (the same mechanism v2's `_bind_child_send_origin` uses *because* contextvars don't cross that boundary). Phase-5 e2e proves a cron child still gets deny-gated.
- **R2 — a cron-spawned subprocess checks `HERMES_CRON_SESSION` via raw `os.getenv`.** *Mitigation:* the terminal bridge (`tools/environments/local.py`) copies `_VAR_MAP` ContextVars into subprocess env, so adding `HERMES_CRON_SESSION` to `_VAR_MAP` makes it bridge automatically. Grep for raw `os.getenv("HERMES_CRON_SESSION")` / `os.environ[...HERMES_CRON_SESSION...]` readers and migrate any found.
- **R3 — worker-thread reuse leaks the marker to a non-cron task on the same pool thread.** *Mitigation:* I4 reset-in-finally with the saved token; Phase-3 parallel test proves it.
- **R4 — separate-process (`hermes cron`) scheduler relied on os.environ being process-global for some out-of-lineage reader.** *Mitigation:* in a separate process there is no interactive turn to leak to, and the in-lineage readers work via the ContextVar; the `get_session_env` os.environ fallback preserves any test/edge that still sets the env. If a real out-of-lineage prod reader is found, keep an explicit os.environ write *only in the separate-process entrypoint* (not the in-gateway tick) — but Phase-2 grep is expected to show none.
- **R5 — false sense that v3 closes 100%.** *Mitigation:* v1/v2 stay; the tripwire stays as defense-in-depth. Honest posture: v3 closes the dominant cause; the tripwire catches any residual.

---

## 9. Open Questions

- **OQ1:** Is there a separate-process `hermes cron` entrypoint that runs jobs *without* the in-gateway tick, and does anything in that process read the flag off-lineage? (Phase-2 grep + a check of the cron CLI entrypoint answers it. Default assumption: contextvar-only is sufficient; add a scoped os.environ write in the standalone entrypoint only if a real off-lineage reader exists.)
- **OQ2 (resolved — CORRECTED by Opus Pass-1):** Does the ContextVar reach the agent's tool calls in a cron job? For the **main** cron agent, yes — `copy_context()` at scheduler.py:~1905 + tool_executor propagation. For a **cron-spawned subagent**, NO — the bare `ThreadPoolExecutor` does not inherit contextvars; that's the B1 blocker, fixed by the Phase-5 capture-and-rebind. (The original spec's claim "same copy_context seam v2 relies on" was inverted: v2 relies on an *explicit rebind* precisely because the seam does NOT auto-propagate.)
- **OQ3 (resolved — Opus Pass-1):** Does the separate-process `hermes cron run` entrypoint (`hermes_cli/cron.py` → `run_job_now` → `cron/scheduler.py:~2160`) have the same subagent gap? Yes — but there's no interactive turn to leak to in that process; the B1 rebind (Phase 5) fixes the deny-gating there too. Contextvar-only is correct for both entrypoints once Phase 5 lands.

---

## 10. Acceptance Criteria

- [ ] **AC1 (I1):** After a real cron tick in-process, a separate interactive context reads `is_cron_session()` False and a bare `_resolve_send_target("discord")` returns `(ORIGIN, <bound chat>, …)`. Evidence: `pytest tests/cron/test_cron_session_isolation.py::test_interactive_turn_not_poisoned_after_cron_tick` (RED on old code, GREEN on new).
- [ ] **AC2 (I2 — main lineage):** Inside a cron job lineage, all 4 `approval.py` cron branches and both `send_message` cron branches see cron=True. Evidence: `tests/tools/test_cron_approval_mode.py` green + a new `test_cron_readers_see_cron_in_job_lineage`.
- [ ] **AC2b (I2 — SUBAGENT lineage, the B1 proof):** A cron job (`cron_mode=deny`) spawning a `delegate_task` child → a dangerous command AND an `execute_code` inside the **child** are BLOCKED. Evidence: `tests/tools/test_cron_subagent_approval.py::test_cron_child_dangerous_command_blocked` + `::test_cron_child_execute_code_blocked` — RED on contextvar-only-without-rebind, GREEN with Phase-5. (A main-thread os.environ test does NOT satisfy this.)
- [ ] **AC3 (no prod os.environ writer):** `grep -rn 'os.environ\["HERMES_CRON_SESSION"\]\s*=' cron/ gateway/ tools/` returns zero. Evidence: grep output.
- [ ] **AC4 (no prod raw reader):** `grep -rn 'env_var_enabled("HERMES_CRON_SESSION")' tools/ gateway/ cron/` returns zero (helper + tests only). Evidence: grep output.
- [ ] **AC5 (I4):** Parallel cron jobs don't cross-clear; marker gone after both. Evidence: `test_parallel_cron_jobs_isolated_markers`.
- [ ] **AC6 (I5 back-compat):** The four existing suites that set `os.environ["HERMES_CRON_SESSION"]` stay green unmodified. Evidence: their pytest runs.
- [ ] **AC7 (tripwire blind-spot closed):** An interactive turn that genuinely falls to home (no origin) now triggers `_warn_unexpected_home_fallback` (previously suppressed by the latched cron flag). Evidence: `test_tripwire_fires_for_interactive_home_fallback`.
- [ ] **AC8 (live e2e, gated):** On the restarted gateway, after a cron tick, a bare interactive `send_message` lands in the current channel; `terminal` shows `HERMES_CRON_SESSION` unset in gateway os.environ. Evidence: staged message Ace sees + terminal output.
- [ ] **AC9 (registration gate, Opus RC-3):** `is_cron_session()` is thread-isolated — set in one thread, asserted False in another with no set. Catches a missing `_VAR_MAP` registration that would silently re-break isolation. Evidence: `test_is_cron_session_thread_isolated`.

---

## Appendix — evidence log (2026-06-22)

- `gateway/run.py:17161-17186` — in-process cron tick.
- `cron/scheduler.py:1586` — `os.environ["HERMES_CRON_SESSION"] = "1"`, comment assumes scheduler-only process; never cleared in `finally` (only `TERMINAL_CWD` + ContextVars are).
- `cron/scheduler.py:~1588-1620` — the delivery-var ContextVar migration with the "os.environ is process-global" comment (the precedent this fix extends).
- Readers: `tools/approval.py:147,1187,1424,1713` (approval gating), `tools/send_message_tool.py:176,277` (origin resolve + tripwire).
- Live proof: `terminal` → `HERMES_CRON_SESSION=1` in gateway PID 95240; Apollo main-turn bare `send_message` → `"Sent to discord home channel (chat_id: 1502228850338435153)"` while user in `#fix-issues` (`1517844402897424504`); `execute_code` (sanitized subprocess) saw it absent — the discrepancy is the env-inheritance vs sanitized-subprocess difference, with the raw-`os.environ` terminal read authoritative.
