# Evidence pack + build brief — executor context residue (misbinding mech-B / RC3)

Date: 2026-07-10 evening · Author: Apollo · Worktree: ~/.hermes/worktrees/ctx-residue
Branch: fix/executor-ctx-residue (off fork/main @ f61bcb640, WITH #271 merged)
Parent spec: ~/.hermes/plans/2026-07-10_gateway-session-context-misbinding-spec.md
(READ IT FULLY — esp. §4 RC3, §5 B2 invariants. This brief is the RC3 branch it mandated.)

## 0. What is already FIXED vs what is NOT

PR #271 (merged, deployed, running) fixed the TURN-ENTRY binding class:
`_run_agent` recursion paths now bind the correct per-turn context, with a
mismatch WARNING at ~gateway/run.py:19398 comparing the async task's bound
`HERMES_SESSION_KEY` against the turn's source-derived key.

TONIGHT'S FRESH EVIDENCE (post-#271, runtime f61bcb640, gateway pid 72309):
an interactive turn in session `agent:main:discord:thread:1525078186545909801:…`
(triggered by a REAL user message in that thread) executed its terminal-tool
subprocess with the COMPLETE identity of session
`agent:main:discord:thread:1525269534926442529:…`:
  HERMES_SESSION_KEY/CHAT_ID/THREAD_ID=1525269534926442529
  HERMES_SESSION_MESSAGE_ID=1525277291276664964 (a ~16:05 message in the
  FOREIGN thread), USER_NAME=Ace, chat_name of the foreign thread.
Two fresh persistent shells (pids 84668@16:08, 88531@16:12, 99737 parent
80971@16:08:48) all carried it — so this is NOT stale-shell-env; the shells
were freshly spawned BY THIS TURN's tool calls and inherited the foreign
binding live.

CRITICAL: the #271 mismatch WARNING fired ZERO times this boot
(`grep -c 'Agent executor context mismatch' gateway.log` = 0). So the async
task WAS correctly bound at _run_agent entry. The foreign identity enters
BETWEEN the async-task bind and the tool-worker subprocess spawn.

Corroboration: the same turn's PROMPT contained a foreign session's restart
handoff (SGR-3A0BFCE6, an auth.json-lane session) — a delivery misroute
consistent with the same identity confusion.

## 1. The seam to diagnose (RC3 outcome (b) — pool residue / unwrapped submit)

The propagation chain for a gateway turn:
  async task (bound via _set_session_env / #271 choke point)
    → GatewayRunner._run_in_executor_with_context (gateway/run.py ~17240):
      copy_context() at call time → executor thread runs ctx.run(func)
    → the AGENT WORKER THREAD runs the whole conversation loop under that
      snapshot
    → tool dispatch: agent/tool_executor.py:803
      executor.submit(propagate_context_to_thread(_run_tool), …) — copies the
      AGENT WORKER THREAD's context to the tool worker
    → tool worker spawns subprocess; tools/environments/local.py
      _inject_session_context_env bridges ContextVars → child env
      (ContextVar-authoritative when session_context_engaged()).

If the async task was bound correctly but the subprocess saw a foreign
identity, the corruption is inside this chain. Suspects, in order:

S1. A path that runs the conversation loop on a pool thread WITHOUT the
    ctx.run wrapper (unwrapped executor.submit / to_thread / plain Thread) —
    the pool thread then runs under ITS OWN thread-level contextvars, i.e.
    residue from a previous job that SET vars at thread level.
    Grep every executor.submit / run_in_executor / asyncio.to_thread in
    gateway/run.py, agent/, tools/ and classify wrapped-vs-unwrapped.
    KNOWN WRAPPED: gateway _run_in_executor_with_context, tool_executor:803,
    async_delegation 233/425. Find the UNWRAPPED ones.

S2. A code path that calls set_session_vars / _CONTEXTVAR.set() ON a pool
    thread directly (not via ctx.run isolation) — that write persists on the
    thread across jobs (residue SOURCE). Grep set_session_vars call sites +
    any .set( on the session ContextVars outside gateway/session_context.py.
    Candidates: cron scheduler per-job set/reset (PR #88 pattern),
    tui_gateway/slash_worker.py:121 (os.environ write path),
    approval.py set_current_session_key, delegate_tool child binds,
    _bind_child_cron_session.

S3. The FIFO / steer / queued-continuation drain: an event enqueued by
    session A's turn but drained/executed inside session B's task or thread.
    (My turn had queued steer notices; the foreign MESSAGE_ID was ~16:05,
    close to my turn's start.)

S4. Streaming/tool-notification callbacks (notify_on_complete watchers,
    process_registry watcher threads) that call back into agent machinery
    with a stale captured context.

## 2. Mandatory diagnosis-first protocol (spec §4)

1. Build the failing test FIRST, on the surface that broke: a REAL
   tool-shaped call through the worker-thread path asserting on the env dict
   produced by _inject_session_context_env INSIDE the worker (grep spec §4
   B1 — MainThread ContextVar reads are execution-invalid for this gate).
2. Reproduce the residue deterministically: two concurrent fake sessions A/B
   through the real GatewayRunner dispatch path (hermetic temp HERMES_HOME,
   see skill hermes-gateway-test-hermeticity + existing harness patterns in
   tests/gateway/test_session_context_inheritance.py), N interleaved turns,
   assert bidirectionally: B's worker sees B AND A's worker never sees B
   (spec RC2). If S1/S2 is right, the failure will show within a bounded
   number of interleavings on a size-1 executor (force thread reuse:
   max_workers=1 makes residue deterministic).
3. Write the diagnosis note (path → verified file:line → mechanism →
   failing test name) at docs/sync/2026-07-10-ctx-residue-DIAGNOSIS.md
   BEFORE the fix commit. If the repro does NOT fall out of S1–S4, STOP and
   report honestly — do not fix blind.

## 3. Fix contract (spec §5, B2 mode-independent invariants)

- Fix the residue CLASS: every executor job that can touch session
  ContextVars runs under a fresh copied context (ctx.run), and/or
  thread-level writes are reset in a finally (token discipline). Prefer the
  smallest structural choke point (e.g. wrap the executor submit seam or
  make DaemonThreadPoolExecutor context-propagating by default) over
  patching N call sites — but ONLY if the bind-precedes-snapshot invariant
  holds at that choke point.
- Token/nesting semantics per spec B2: inner exit restores OUTER binding,
  never blanket reset.
- Keep the #271 mismatch WARNING; if diagnosis shows the corruption happens
  AFTER that check, ADD a second cheap assert at the tool-executor submit
  seam (log-only, compares worker context key vs the turn's expected key —
  the net that would have caught tonight's case).
- No new env vars, no config knobs, no cache-breaking, no changes to
  _inject_session_context_env's strip semantics (they are correct and
  tested).
- ALSO (small, same PR — the probe surface): tools/code_execution_tool.py
  runs in-process; expose the LIVE agent object to executed code as `agent`
  in the execution namespace WHEN the tool executes in-process for the
  top-level agent (it already has access to the process globals — this adds
  a first-class handle instead of forcing hacks). Guard: only bind if the
  executing context can resolve its owning AIAgent (tool_executor knows it);
  never for sandboxed/subprocess modes. This gives operators a direct probe:
  execute_code 'print(agent.context_compressor._recent_skews)'. Add a test.

## 4. Gates (I certify, you don't self-certify)

- New failing-first tests fail on fork/main@f61bcb640, pass with the fix,
  bidirectional + mutation-checked (revert fix → red).
- Full files green: tests/gateway/test_session_context_inheritance.py,
  tests/tools/test_local_env_session_leak.py, tests/agent/ tool-executor
  suites, plus your new file(s).
- py_compile on every touched file; scripts/run_tests.sh for suites.
- Do NOT commit — leave the worktree dirty; Apollo reviews, commits, lands.
- Write the diagnosis doc + a one-page CHANGES summary at
  docs/sync/2026-07-10-ctx-residue-CHANGES.md.

## 5. Boundaries

- Work ONLY in this worktree. Never touch ~/.hermes/hermes-agent (live
  checkout) or ~/.hermes/runtime. No gateway restarts. No pushes.
- Tests hermetic: temp HERMES_HOME, no live state.db/sessions.json.
- The live gateway is running the code you're reading — flag anything you
  find that's actively dangerous (vs latent) in the CHANGES doc header.
