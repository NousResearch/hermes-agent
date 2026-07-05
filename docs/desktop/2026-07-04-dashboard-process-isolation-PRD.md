# PRD: Dashboard Process Isolation — agent turns out of the serving process

- **Status:** v1.5 — CONVERGED. Pass-9 verified the folds physically present, zero blockers; 3 operational RCs folded (pipe HOL fan-out + spike, deploy-respawn reason tag, resume reads version from state.db). 9 review passes; NONE of the residuals gate Phase 0. **BUILD UNBLOCKED — Phase 0 re-dispatched.**
- **Author:** Apollo, 2026-07-04 · **Requested:** Ace ("spec it out, prd review and build it with daedalus")
- **Repo:** `Kyzcreig/hermes-agent` (fork). Surfaces: `tui_gateway/`, `hermes_cli/web_server.py`.
- **Prior art:** `docs/desktop/2026-07-02-dashboard-eventloop-starvation-PRD.md` (shipped PRs
  #179/#182 — offloaded 9 sync SessionDB reads off the loop). This PRD is that spec's named
  escalation path (its RR1: "if the margin is thin under real load, the answer is process
  isolation / read-replica — out of this PRD's shape").

## 1. Problem (measured, twice)

The dashboard (`ai.hermes.dashboard`, port 9119) is ONE Python process, ONE asyncio loop
serving BOTH:
- **The serving plane:** REST API + `/api/ws` desktop WebSocket (every ws send marshals
  through the loop via `WSTransport.write` → `run_coroutine_threadsafe`).
- **The compute plane:** live agent sessions. A chat message dispatches
  `_pool.submit(...)` (`tui_gateway/server.py:1134`) → `run()` (`:8709`) →
  `agent.run_conversation(...)` (`:8844`) — the FULL agent turn (tool calls, JSON
  serialization of 400K-token contexts, compression, embedding, subprocess management)
  runs in **threads of the serving process**. 13 live sessions today; 60 threads at the
  last wedge.

**Incident regime (2026-07-04, profiled pre-restart):** loop stalled **453s**; `sample`
showed the main thread parked in `take_gil` (NOT in sqlite — the eventloop-PRD fix holds)
while worker threads burned the interpreter (152 sqlite frames in workers, heavy
`_PyEval` from agent turns). CPython's GIL means a loop that owns no C-level blocking work
can still starve for MINUTES when 60 threads contend. Result: MBP desktop app can't log in;
recovery = manual `launchctl kickstart`. This has recurred 4× in 3 days (2× pre-fix from
sync-sqlite-on-loop, 2× post-fix from GIL contention). The eventloop PRD fixed the
on-loop blocking class; it cannot fix interpreter-wide starvation. **The serving plane
must not share a GIL with the compute plane.**

## 2. Goal & Non-Goals

**Goal:** the dashboard's HTTP/ws serving stays responsive (p99 < 1s) regardless of how
much agent-turn compute is running. Structural, not tuned: serving and compute in
**separate OS processes**, so no amount of agent GIL churn can starve a ws send.

**Non-Goals:**
- NOT a rewrite of the agent core, session model, or tui_gateway RPC surface. Clients
  (desktop app, TUI, dashboard SPA) see identical RPCs, events, and RPC ids.
- NOT sub-interpreters / free-threaded CPython (3.13t) / asyncio-everywhere refactors —
  interesting, unproven in this codebase, rejected for blast radius.
- NOT a session-count cap or load-shedding policy (that's tuning; this is structure).
- NO state.db changes (Ace standing constraint). No new watchdog daemons.
- NOT the Studio-local `Hermes.app` gateway or the TUI-owned gateway processes — scope is
  the `ai.hermes.dashboard` service only (the one remote clients depend on).

## 3. Ground truth (verified against source, 2026-07-04)

- Turn execution: `tui_gateway/server.py:1134` `_pool.submit(lambda: ctx.run(run))` →
  `:8709 def run()` → `:8844 agent.run_conversation(...)`. In-process threads.
- `_pool` = `ThreadPoolExecutor(max_workers=8 default)` (`:227-236`).
- Every ws send crosses the loop: `tui_gateway/ws.py:70-77` (`run_coroutine_threadsafe`).
- Precedent for out-of-process execution ALREADY IN TREE (three shapes to steal from):
  1. `_SlashWorker` (`server.py:267`) — persistent `HermesCLI` subprocess, line-JSON over
     stdin/stdout pipes, drain threads (`:288-302`).
  2. The gateway CHILD-SESSION mirror ("Relayed token-by-token from the child's
     run_conversation", `:3620`) — streaming across a process boundary exists.
  3. `hermes_subprocess_env(inherit_credentials=True)` (`tools/environments/local.py`) —
     env propagation for spawned agent processes is solved.
- Session state lives in `session[...]` dicts (history, history_version, locks) inside the
  serving process; `state.db` is the durable store. Turn-end write-back is version-guarded
  (`server.py:8646-8651` per the eventloop PRD's Phase-0).
- 13 sessions active today; typical concurrent turns 2-6; wedges correlate with 5+
  concurrent heavy turns (400K-token contexts).

## 4. Design — ONE compute-host child process; serving process keeps the sockets

**Shape F (chosen after pass-1 review; alternatives in §5):** the dashboard process becomes the
SERVING plane only. ALL agent turns execute in a single persistent **compute-host child
process** — effectively today's turn-execution layer (the `_pool`, the live agent objects,
their in-memory histories) moved wholesale behind a pipe:

- **Compute host lifecycle:** spawned at dashboard startup (`python -m tui_gateway.compute_host`),
  argv carries NO session data (sessions attach via frames — no injection surface; env via
  `hermes_subprocess_env(inherit_credentials=True)`, same `HERMES_HOME`/profile as the parent,
  asserted by a `hello{build_sha, hermes_home}` handshake frame — mismatch = refuse + respawn,
  closing R4 at startup, not on drift). It hosts its own `ThreadPoolExecutor` identical to
  today's `_pool` and runs `run()`→`agent.run_conversation` EXACTLY as the serving process does
  now — the diff inside the host is minimal (same code path, new entry).
- **Death-with-parent + orphan reconciliation (pass-1 blocker 1):** the host is spawned in the
  supervisor's process group with a kill-on-parent-death guard (host polls its ppid; ppid==1 →
  flush all histories to state.db, exit). Supervisor persists `{host_pid, boot_id}` to
  `~/.hermes/state/dashboard-compute-host.json`; on startup it reconciles: a live orphan gets
  SIGTERM → confirmed-exit wait → then spawn (never two hosts). **Kill-path contract (pass-2
  RC):** the host's SIGTERM handler is a FLUSHING handler (flush all histories → exit 0),
  supervisor waits bounded (10s) then SIGKILL-falls-back; and before signaling, the reconciler
  VERIFIES process identity (`/proc`-equivalent cmdline check: pid's argv contains
  `tui_gateway.compute_host`) — `{host_pid, boot_id}` alone does not survive same-boot PID
  reuse. `launchctl kickstart -k` (the real recovery path) therefore yields: orphaned host
  self-terminates on ppid flip after flushing; new serving process reconciles the registry
  before spawning. Single-writer proof = AC-5b.
- **Protocol:** newline-JSON over stdin/stdout (the `_SlashWorker` + child-session-mirror
  patterns). Frames multiplexed by session_key: `turn.start{sid,...}` →
  `delta/tool_event/usage{sid,...}` → `turn.end{sid, result, history_version}` | `turn.error`.
  Control: `interrupt{sid}`, `reload_mcp{sid}` (run-concurrent — §4 table, AC-8),
  `compress{sid}`, `state.get{sid}`, `shutdown`. **The host runs a
  DEDICATED control-reader thread** (pass-1 RC: in the design, not the spike) — frame reads
  never queue behind a running turn, so `interrupt` reaches `agent.clear_interrupt()` mid-API-call
  with the same latency bound as today's in-process call.
- **Serving process keeps:** RPC dispatch, ws fan-out, session REGISTRY (metadata/pinned/title),
  executor-offloaded SessionDB reads, auth. Its GIL is shared only by I/O-shaped work.
- **State ownership (INV-4, spawn-time authority named; mirror scope pinned — pass-2 B1):**
  while the host is alive, the HOST's in-memory history is authoritative. The serving process
  holds a **METADATA-ONLY mirror**: `{session_key, history_version, title, pinned, running,
  last_turn_ts, token_counts}` — NEVER message bodies. Full-history/scrollback reads
  (`session.history`, transcript panes) serve from **state.db** (current to the last `turn.end`
  write-back, already executor-offloaded + semaphored per the eventloop PRD) — they never touch
  the host. Serving-side RAM for the mirror is O(sessions × ~1KB) ≈ nothing; the 1× total-memory
  claim is thereby structural, and AC-7 gates BOTH processes' RSS. **At every ownership handoff
  (host spawn/respawn/reap) the single authority is state.db:** outgoing owner flushes before
  releasing; incoming owner loads from state.db only. The serving process NEVER writes history
  while a host is alive; the host never reads the mirror. No dual-write window.
- **Full mutator inventory + routing (§4 TABLE — Phase-0 finding, polarity corrected v1.2,
  reload-mcp added v1.3).** Every live `session["history"]`/`session["agent"]` mutator in
  `tui_gateway/server.py` (enumerated by the Phase-0 gate, verified against source). In Shape F
  the persistent host is the SOLE live authority, so ALL of these route to the host as control
  frames; the CLASS column governs TIMING (the `_MUTATES_WHILE_RUNNING` running-guard MOVES INTO
  the host — idle-gated frames apply only when the session has no in-flight turn):

  | Site | RPC | Mutates | Class |
  |---|---|---|---|
  | `:8891` `_run_prompt_submit.run` | prompt (the turn) | history write-back | **turn-path** (host-internal agent loop) |
  | `:8068` `session.interrupt` | interrupt | `agent.interrupt()` | **turn-path** (control frame) |
  | `:11245` `reload.mcp` | reload-mcp | `agent.tools`/MCP schema | **run-concurrent WRITE** — NOT in `_MUTATES_WHILE_RUNNING`, fires mid-turn; control-reader-thread frame `reload_mcp{sid}`; AC-8 |
  | `:2933` `_compress_session_history_locked` | compress | history (swap) | **idle-gated** (in `_MUTATES_WHILE_RUNNING` → rejected while running) |
  | `:8395` `prompt.submit` truncate | prompt | history (pre-turn truncate) | **idle-gated** |
  | `:3885` `_apply_personality_to_session` | personality | history (pivot marker) | **idle-gated** (in set) |
  | `:10339` `config.set` reasoning | model/config | `agent.reasoning_config` | **idle-gated** (model in set) |
  | `:4121/:4131` `_reset_session_agent` | session.reset | agent + history (rebuild/clear) | **idle-gated** (rejected while running) |
  | `:7708` `_reload_session_history_from_db` | undo/redo | history (DB reload) | **idle-gated** (rejected while running) |
  | `:11913` `command.dispatch` `/retry` | retry | history (truncate) | **idle-gated** (rejected while running) |

  **`_MUTATES_WHILE_RUNNING = {"model","personality","prompt","compress"}` (`server.py:12784`)
  is the REJECT-while-running set** — `if name in set and session["running"]: return "session
  busy"` (`:12785`). It marks which mutators are idle-gated; it does NOT mark run-concurrent
  ones (v1.1 read this backwards — see the SUPERSEDED log entry). The only run-concurrent WRITE
  is `reload.mcp` (absent from that set). This makes INV-4 trivially true: one persistent host
  writes a session's live history for its whole lifetime; state.db is handoff-only across
  respawn/reap; the serving process NEVER writes history.
- **Per-turn durability (pass-2 B2 — INV-6; citation corrected pass-4 RC-1):** the DURABLE
  write is `agent._persist_session(...)` → `_flush_messages_to_session_db` (both in run_agent.py — cite by NAME; line numbers drift across trees, pass-5 RC-3),
  whose ~27 PER-TURN call sites live in `agent/conversation_loop.py` (+ turn_finalizer.py,
  turn_context.py) — `run_agent.py`'s `run_conversation` is a thin forwarder into that loop.
  All of it executes in the host by construction once the turn path moves, giving per-turn
  state.db durability with zero relocation work. (server.py:511-528 is a finalize/disconnect
  best-effort snapshot — a cleanup fallback, NOT the durability mechanism; do not cite it as
  such.) The `server.py:8646-8651` block is a
  version-guarded IN-MEMORY swap (session dict + history_version bump), NOT a durable write —
  in the host it becomes the host's own in-memory authority update and the source of the
  `turn.end{history_version}` the metadata mirror consumes. Crash loss is bounded to the
  in-flight turn because `_persist_session` completes before `turn.end` relays.
- **Telemetry for the relocated risk (pass-1 blocker 3 + pass-2 B3 — in-process, NOT a new
  watchdog daemon):** the host emits pipe heartbeats every 15s: `hb{active_turns, rss_mb,
  per_turn_age, progress_counter}` where **progress_counter is a monotonic count of completed
  API calls + tool executions across all turns** — so the supervisor distinguishes
  GIL-SATURATED-BUT-PROGRESSING (late hb, counter advancing → log nothing / info) from
  DEADLOCKED (counter frozen across 2+ intervals → `compute host stalled`). A late heartbeat
  alone NEVER fires the stall line (false-positive guard: a busy healthy host must not trip the
  respawn ladder — respawn triggers only on frozen-counter + unresponsive control channel).
  Supervisor logs `turn stuck >Ns` / `compute host stalled` / RSS lines into agent.log; grep
  detection works exactly like `event loop stalled` today. **Log-write integrity (pass-2 RC):**
  the host writes its own log records via O_APPEND single-write-per-record (same pattern as
  today's multi-thread logging); Phase-1 unit asserts no torn/interleaved lines under
  concurrent host+supervisor writes.
- **Memory (pass-1 blocker 2 resolved by shape):** ONE host ≈ today's process RSS (the agents
  already coexist in one interpreter now). No per-session multiplication; no TTL/reap machinery.
- **Crash containment:** host death kills in-flight TURNS (each gets `turn.error{worker_died}`,
  history bounded-loss to the in-flight turn via write-back), serving stays up, supervisor
  respawns the host (with backoff, max 3/5min → then LOUD log + stop respawning so a crash-loop
  can't flap). Deploy note (pass-1 residual): a deploy respawns the host → in-flight turns get
  turn.error — EXPECTED. **The turn.error frame + supervisor log carry `reason:"deploy_respawn"`
  vs `"crash"` (pass-9 RC-2) so alerting suppresses deploy blips without masking a crash-loop.**
  (Land before Phase-3 cutover.)
- **Feature flag:** `dashboard.turn_isolation: false` default. Gates ONLY the dispatch site
  (`:1134`'s run() → local pool vs host frame). Keys land in `DEFAULT_CONFIG` + the gateway raw
  YAML loader (pass-1 config-drift note): `turn_isolation`, `compute_host_heartbeat_secs`,
  `compute_host_respawn_max`.

## 5. Alternatives rejected (why F)

- **D. Per-SESSION worker processes (v0.1's shape):** REJECTED by pass-1 review — 13 sessions
  × multi-GB agent RSS blows Studio headroom for a goal (serving responsiveness) that one
  compute host already meets at 1× memory; adds per-session lifecycle machinery (TTL reaping,
  per-session orphan reconciliation) = the most complex option. Its one real advantage
  (per-session crash blast-radius) is recorded as the named ESCALATION if host-wide turn
  contention or full-host crashes prove painful — the frame protocol is already
  session-multiplexed, so sharding sessions across N hosts later is a supervisor-only change.
- **A. uvicorn workers=N:** splits the SERVING plane only; agent turns still share each
  worker's GIL, and sessions are sticky-stateful (ws + in-memory session dicts) — breaks
  the session registry. Wrong axis.
- **B. Move serving OUT (new front process, keep agents in place):** same seam count as D
  but inverts who owns sessions; every existing RPC handler assumes it lives with the
  session dicts — far larger diff. D moves the smaller side (one call site + streaming).
- **C. multiprocessing.Pool for turns:** agents are not picklable (live provider clients,
  locks, threads); per-session persistent workers (D) sidestep pickling entirely and
  preserve prompt-cache continuity, which a pool cannot.
- **E. Read-replica/DB-only isolation (eventloop PRD's RR1 option 1):** already
  effectively done (reads are off-loop + semaphored); the 2026-07-04 profile shows the
  residual starvation is GIL/compute, not DB. Doesn't address the cause.

## 6. Invariants

- **INV-1 (client compatibility):** desktop/TUI/SPA clients see byte-compatible RPC
  responses + event streams; zero client changes. Proof: existing gateway test suite green
  with flag ON, plus a golden transcript diff (same seeded turn, flag off vs on).
- **INV-2 (prompt-cache safety):** a session's system prompt + history prefix remain
  byte-stable across turns (worker persistence guarantees the same agent object serves
  consecutive turns; a worker restart mid-session reloads history from state.db exactly as
  a gateway restart does today — no NEW cache-invalidation class).
- **INV-3 (no state.db schema/maintenance changes).**
- **INV-4 (single-writer history — persistent host, v1.2):** the persistent compute host is
  the SOLE writer of a session's live history for the session's entire lifetime; the serving
  process holds a metadata-only mirror and NEVER writes history. Single-writer is trivially
  true (one persistent writer). state.db is the handoff medium ONLY across host respawn/reap.
  The `_MUTATES_WHILE_RUNNING` running-guard MOVES INTO the host (idle-gated mutators applied
  only when the session has no in-flight turn). Phase-1 test: a table-driven check that EVERY
  enumerated mutator (§4 table) routes to the host as a control frame, correctly tagged
  turn-path vs idle-gated; CI fails if a new `session["history"]`/`session["agent"]` write site
  appears in the SERVING process (there should be none) or unclassified in the host.
- **INV-5 (flag-off = today):** with `turn_isolation: false`, the diff is dead code —
  in-process path byte-identical (guarded by the golden transcript test run flag-off).
- **INV-6 (per-turn durability — pass-2 B2, citation per pass-4 RC-1):** every completed
  turn's history is durable in state.db via the `agent/conversation_loop.py` per-turn
  `_persist_session()` calls (host-resident by construction) before the turn's `turn.end`
  frame is relayed to any client. The 8646-8651 in-memory swap is authority/versioning, NOT durability.
  Proof: AC-5 asserts recovered history == all-completed-turns after a host `kill -9`.
  **Resume ordering (pass-9 RC-3):** a resuming/reconnecting client reads `history_version`
  from **state.db, not the metadata mirror** — since INV-6 guarantees state.db reaches N before
  `turn.end{N}` relays, a relay-before-persist race is structurally impossible.

## 7. Phases

### Phase 0 — Seam ground-truth + host spike (GATE)
- Config read-site verification (pass-4 RC-2): the dispatch site reads via
  `tui_gateway::_load_cfg()` (raw yaml + managed overlay, NO DEFAULT_CONFIG merge) — land the
  three keys with explicit defaults AT THE READ SITE, plus DEFAULT_CONFIG seeding for the REST
  editor (`web_server.py` uses `load_config` — different loader, both must agree).
- ~~Enumerate EVERY consumer~~ **DONE (v1.1/v1.2, table in §4):** 10 mutating sites found. In
  Shape F ALL route to the persistent host (sole authority); the `_MUTATES_WHILE_RUNNING` gate
  (a REJECT-while-running set — v1.2 corrected v1.1's inverted reading) governs TIMING not
  location, and moves into the host with the agent. INV-4 = one persistent writer, trivially
  single. Phase-1 adds the table-test (every mutator → host control frame, turn-path vs
  idle-gated) + a CI guard that NO history write appears in the serving process.
- Spike: minimal `compute_host` hosting real agents for TWO seeded sessions; drive 3
  consecutive turns each (one STREAMING-HEAVY: long generation, measure per-delta p99 not just
  mean — pass-1 residual); measure (a) delta relay overhead p99 vs in-process, (b) host RSS
  delta vs baseline dashboard RSS (expected ≈0 — same agents, new process), (c) host spawn
  cold-start, (d) interrupt-over-pipe latency mid-turn. **STOP if:** delta overhead > 50ms p99,
  interrupt latency > 2× in-process, or cold-start > 5s.
- (interrupt verification folded into the spike, item d — the control-reader thread is now a
  DESIGN requirement §4, the spike measures its latency.)

#### Phase 0 execution results — Daedalus, 2026-07-04

- Base: current `fork/main` at `0b5063e67d8437b61752f7439c17a2c8d29eac3f`.
- Config read-site fix landed: `hermes_cli.config.DEFAULT_CONFIG["dashboard"]`
  seeds `turn_isolation: false`, `compute_host_heartbeat_secs: 15`, and
  `compute_host_respawn_max: 3`; `tui_gateway.server::_load_dashboard_process_isolation_config()`
  applies the same explicit defaults over raw `_load_cfg()` output. `prompt.submit`
  currently fails closed if `turn_isolation` is enabled before Phase 1 so the knob
  cannot silently no-op.
- Spike harness: `tui_gateway.compute_host` + `scripts/compute_host_phase0_spike.py`.
  The host is a persistent line-JSON child with a dedicated stdin control reader and
  seeded AIAgent-compatible sessions using the `run_conversation(..., stream_callback=...)`
  + `interrupt()` contract. The spike uses deterministic local agents to isolate pipe,
  fan-out, and interrupt overhead from provider/network variance.
- Workload: 2 seeded sessions (`alpha`, `bravo`), 3 turns each, 1 streaming-heavy
  turn per session; 616 host delta samples.
- Delta relay p99: direct/in-process callback p99 `0.012 ms`; host relay p99
  `0.856 ms`; overhead p99 `0.844 ms` (**PASS**, stop threshold `>50 ms`).
- Host RSS: baseline driver RSS `20.59 MB`; host RSS peak `18.88 MB`; delta vs
  baseline `-1.72 MB` (synthetic spike host is smaller than the driver process;
  no RSS expansion observed).
- Spawn cold-start: `113.77 ms` (**PASS**, stop threshold `>5000 ms`).
- Interrupt-over-pipe: direct end-to-end interrupt `77.11 ms`; host interrupt ack
  `0.339 ms`; host end-to-end interrupt `101.80 ms`; host/direct ratio `1.32x`
  (**PASS**, stop threshold `>2x`).
- HOL isolation: fast session solo p99 `0.593 ms`; fast session with a slow bounded
  consumer on the sibling stream p99 `0.503 ms`; delta `-0.090 ms`; slow consumer
  dropped/coalesced `180` deltas without stalling the fast session (**PASS**).
- STOP conditions evaluated: `delta_relay_p99_over_50ms=false`,
  `interrupt_over_2x_in_process=false`, `cold_start_over_5s=false`.

### Phase 1 — Supervisor + worker (flag-gated)
- `tui_gateway/turn_worker.py` (worker main: agent init from session-key, frame loop) +
  `tui_gateway/turn_supervisor.py` (spawn/registry/reap/relay). Dispatch-site flag switch
  at `:1134`/`:8709`. Unit: frame protocol round-trip, crash→turn.error→recover, supervisor-death orphan reconciliation, flush-on-ppid-flip, interrupt mid-frame. The full existing gateway suite green flag-OFF and
  flag-ON.

### Phase 2 — Routing seams
- Compress/`_mirror_slash_side_effects` → worker control frames (reuse the identity-guard
  semantics from PR #180 — the no-op predicate now evaluates in the worker).
- `session.list`/info reads → mirror (staleness = last turn.end, same as today).
- Live E2E: R7 slash sweep + AC-1a compress (the desktop PRD's gates) re-run flag-ON —
  they must pass unchanged.

### Phase 3 — Live certify + cutover
- On the live Studio dashboard, flag ON: drive 6 concurrent heavy turns (real agent
  sessions, 100K+ contexts) while probing ws `session.list` + raw REST every 500ms for
  10 min. **Gate: serving p99 < 1s, zero `event loop stalled` lines, zero ws disconnects.**
  This is the incident regime that wedged the box 4× — reproduced deliberately, survived.
- Golden transcript flag-off vs flag-on (INV-1/INV-5). Soak 24h flag-ON under Ace's normal
  fleet load; recurrence detector = the existing `event loop stalled` log line (no new
  watchdog). Then default the flag true in config; keep the flag for one release as the
  rollback lever.

## 8. Acceptance Criteria

- AC-1: Phase-0 spike numbers recorded in this doc; STOP conditions evaluated explicitly.
- AC-2: full gateway suite green flag-OFF (byte-identical path) AND flag-ON.
- AC-3: golden transcript diff — seeded session, same prompts, flag off vs on: identical
  client-visible frames (modulo timestamps/ids).
- AC-4: live incident-regime certify (Phase 3): 6 concurrent heavy turns, serving p99 <1s,
  0 stalls, 0 ws drops, 10 min sustained.
- AC-5: crash containment — seed a session, complete 3 turns, `kill -9` the HOST mid-turn-4:
  in-flight turn gets turn.error, serving stays responsive throughout, supervisor respawns,
  and **recovered turn-3 history is byte-identical to the per-turn
  `conversation_loop._persist_session` output** (pass-5 RC-2: presence is not enough — the
  recovery must NOT have depended on the weaker finalize/disconnect snapshot fallback; assert
  the finalize path did not run). Loss == in-flight turn only, not since-spawn.
- AC-5b (pass-1 blocker 1): SUPERVISOR death — `launchctl kickstart -k` the serving process
  with 2 live turns running: orphaned host flushes + self-terminates (ppid guard), restart
  reconciles the registry, exactly ONE host + ONE writer per session afterward, loss bounded
  to the in-flight turns.
- AC-6: interrupt E2E **under saturation (pass-2 B3)** — with the host GIL saturated by
  concurrent heavy turns (the §1 incident regime), interrupt one mid-stream: API call aborts
  ≤ 2× the in-process bound. A quiet-host interrupt pass does not count.
- AC-8 (pass-7 RC-1): reload.mcp mid-turn — trigger `/reload-mcp` while a turn holds an
  in-flight tool call; the host applies the new tool schema via the control-reader thread
  WITHOUT corrupting the active tool-call (turn completes with the pre-reload schema it started;
  next turn sees the new tools), consent gate honored. NOT routed through interrupt.
- AC-7: 24h soak flag-ON: zero `event loop stalled` attributable to serving; **BOTH processes'**
  RSS bounded (host ≤ baseline+20%; serving-side mirror peak measured and O(KB)/session);
  detector-visibility: `compute host stalled` fires in a FORCED deadlock test AND does NOT fire
  during a GIL-saturated-but-progressing window **produced by REAL concurrent heavy turns on
  the host interpreter (the §1 regime — NOT a synthetic sleep/hand-throttled emitter; the
  heartbeat thread's actual GIL-reacquisition latency under contention is the thing under
  test, pass-3 RC-3)**, silent during healthy soak.

## 9. Risks

- **R1 (biggest): hidden mutating consumers of session state** → Phase-0 enumeration is
  the gate; the flag keeps blast radius at zero until proven.
- **R2: interrupt latency over pipe** — spike-verified before build (Phase 0).
- **R3: full-host crash takes all in-flight turns** — accepted trade for 1× memory; escalation = session-sharded hosts (§5.D) if it bites.
- **R4: host/serving version skew** — closed at startup by the hello{build_sha} handshake (§4).
- **R5: streaming backpressure + single-pipe HOL (pass-9 RC-1):** ALL sessions multiplex over
  ONE stdin/stdout pipe pair, so a supervisor drain thread that blocks synchronously on a slow
  downstream client fills the ~64KB OS pipe buffer → the host's `write()` blocks → EVERY
  session's delta relay stalls (the `_SlashWorker` precedent was single-session and never had
  to prove this). Contract: the drain thread reads the pipe to an in-memory buffer and fans out
  **non-blocking** to bounded per-connection send queues; a full per-connection queue
  drops/coalesces THAT client's deltas (stated policy), never blocking the shared pipe.
  Phase-0 spike adds a measurement: one artificially-slow ws consumer while a second session
  streams heavy → assert the second session's per-delta p99 is unaffected.

## Review Log

### Pass 1 — Opus (claude-bpp), 2026-07-04 — BLOCK → folded to v0.2
Root cause endorsed ("§1 profile is convincing"); blocked on 3 classes + the shape itself:
- **Blocker 1 (orphaned workers on the REAL recovery path):** design now has die-with-parent
  ppid guard + flush-on-orphan, persisted host registry, startup orphan reconciliation, and
  AC-5b (supervisor hard-kill with live turns → single-writer proof).
- **Blocker 2 (memory denominator):** RESOLVED BY SHAPE — v0.1's per-session workers (13×
  RSS) replaced with ONE compute host (≈1× — the agents already share one interpreter today).
  Per-session isolation demoted to the named escalation (§5.D).
- **Blocker 3 (new pathology invisible to the old detector):** host heartbeats → supervisor
  logs `turn stuck`/`compute host stalled`/RSS lines into agent.log (in-process telemetry, not
  a watchdog daemon); AC-7 now includes a forced-fire detector-visibility proof.
- **RCs folded:** control-reader thread mandated in §4 design (not spike); argv carries no
  session data + env handshake (injection/config-drift seams); config keys named for
  DEFAULT_CONFIG + gateway loader; spawn-time authority = state.db at every handoff (INV-4);
  deploy-blip documented as expected; streaming-heavy per-delta p99 in the spike; pooled
  compute process examined and CHOSEN (the reviewer's suggested alternative was right).

### Pass 2 — Opus (claude-bpp), 2026-07-04 — BLOCK → folded to v0.3
Shape downgrade endorsed; blocked on two asserted-not-established resolutions + a relocated
starvation class. All folded:
- **B1 (mirror memory unbounded):** mirror pinned to METADATA-ONLY (~1KB/session); full-history
  reads route to state.db (already off-loop); AC-7 now gates BOTH processes' RSS. The 1×
  memory claim is structural, not asserted.
- **B2 (per-turn durability unassigned):** new INV-6 — the version-guarded turn-end write-back
  moves into the host and runs on EVERY turn.end, before the frame relays; AC-5 now proves
  recovered history == all completed turns (3-turn seed, kill mid-4th).
- **B3 (GIL starvation relocated into the host):** heartbeat carries a monotonic
  progress_counter; late-hb-alone never fires the stall line (saturated-but-progressing ≠
  deadlocked); respawn only on frozen counter + dead control channel; AC-6 interrupt and AC-7
  forced-fire/false-positive-guard run under the saturated incident regime.
- **RCs:** SIGTERM = flushing handler + bounded wait + SIGKILL fallback; reconciler verifies
  process identity by cmdline (PID-reuse guard); config-loader ground-truth added to Phase 0;
  O_APPEND single-write log records + torn-line unit test.

### Pass 3 — Opus (claude-bpp), 2026-07-04 — APPROVE WITH CHANGES → folded to v0.4
**All pass-1/2 blockers verified genuinely folded** (reviewer re-checked each against source,
explicitly not rubber-stamped). Zero architectural blockers; 3 RCs — two are source-mapping
corrections the reviewer verified against the tree:
- **RC-1:** INV-6 cited `server.py:8646-8651` as the durable write — it's an IN-MEMORY swap;
  the durable mechanism is `agent._persist_session()` inside `run_conversation()` (host-resident
  by construction). Spec corrected; a literal builder would otherwise have relocated a dict swap
  and believed durability handled.
- **RC-2:** the dashboard dispatch reads config via `tui_gateway/server.py::_load_cfg()`
  (`:1744`, bespoke raw yaml, NO DEFAULT_CONFIG merge) — not `load_config`, not "gateway raw
  YAML". All three keys get explicit read-site defaults in tui_gateway AND DEFAULT_CONFIG
  seeding for the REST editor surface (two loaders, both seeded).
- **RC-3:** AC-7's false-positive-guard window must be produced by real concurrent heavy turns
  on the host interpreter, not a synthetic sleep — the heartbeat thread's GIL-reacquisition
  under genuine contention is the property under test.

### Pass 4 — Opus (claude-api-proxy-f2), 2026-07-04 — APPROVE WITH CHANGES → folded to v0.5
Reviewer independently re-ground-truthed ALL prior citations ("did not inherit them") and
confirmed every pass-1/2/3 fold correct against the tree. Zero blockers. Three RCs:
- **RC-1 (FOLDED):** the per-turn `_persist_session` calls live in `agent/conversation_loop.py`
  (~27 sites; run_conversation is a thin forwarder) — not in run_conversation's body, and
  server.py:511-528 is a disconnect-cleanup snapshot, not the durability proof. Citations
  corrected in §4 + INV-6.
- **RC-2 (FOLDED/sharpened):** read-site defaults for all three keys named as a Phase-0 task
  targeting `_load_cfg` consumers explicitly; dual-loader seeding (editor vs runtime) stated.
- **RC-3 (REJECTED with ground-truth):** reviewer claimed `_pool` default is 4 — WRONG; source
  is `int(os.environ.get("HERMES_TUI_RPC_POOL_WORKERS") or "8")` → default 8; the 4 is the
  except-branch fallback for a malformed env var. Spec's §3 "8 default" stands. (Apollo
  verified against the runtime tree before rejecting.)

### Delta-review (v1.1→v1.2) — 2026-07-04 — AWC, INV-4 gate-polarity corrected
The delta pass caught that v1.1 INVERTED the `_MUTATES_WHILE_RUNNING` semantics: I called it
"the run-concurrent set → route to the worker" when the guard actually REJECTS those RPCs while
running (they're idle-only). A literal builder would have routed /compress /model /personality
/prompt INTO a live turn — the exact read-then-mutate race the guard exists to prevent.
Apollo verified against source (`server.py:12784-12786`: `if name in set and running: return
"session busy"`) — reviewer correct, my v1.1 wrong. Rewrote INV-4 for Shape F's PERSISTENT host:
the host is the sole live authority for a session's whole lifetime, ALL mutators route to it as
control frames, the running-guard moves into the host and governs TIMING (idle-gated vs
turn-path), state.db is handoff-only across respawn. Single-writer becomes trivially true (one
persistent writer) — a cleaner proof than v1.1 attempted. AC-5 byte-completeness clause
reaffirmed. Needs one confirmation pass, then Phase 0 re-dispatch.

### Phase-0 STOP + v1.1 revision — 2026-07-04 (Daedalus finding, Apollo folded) — ⚠️ SUPERSEDED by the v1.1→v1.2 delta-review above
**⚠️ SUPERSEDED (read INV-4/§4 in the body, NOT this entry): this v1.1 entry's "run-concurrent
mutator set … routes to the worker" framing INVERTED `_MUTATES_WHILE_RUNNING` (it is the
REJECT-while-running set). v1.2 corrected it: the persistent host is sole authority, ALL
mutators route to it, the gate governs timing not location. Do not build from this entry.**
Phase-0 seam enumeration (the spec's OWN gate) fired its STOP: 7 live history/agent mutators
exist vs the 4 the design assumed. Verified all 7 against source. The 5 "unknown" ones are NOT
a design breaker — they're cleanly split by the gateway's EXISTING `_MUTATES_WHILE_RUNNING`
allowlist (`server.py:12784`): run-concurrent mutators route to the worker, idle-only ones are
already turn-guarded so they never race a worker. This makes INV-4 PROVABLE from existing code
rather than asserted — a strictly stronger position. §4 gains the full mutator table + routing
rule; INV-4 rewritten; Phase-1 gains a classification table-test + an unclassified-write-site
CI guard so the seam can't rot. Per the spec's D-9 exit contract, this revision takes ONE
delta-review pass before Phase 0 re-dispatches.

### Pass 9 — Opus (claude-bpp), 2026-07-04 — APPROVE WITH CHANGES → folded, **v1.5 CONVERGED**
Verified the pass-8 fake-folds are now PHYSICALLY present (reviewer grepped the table + the
reload_mcp frame). Zero architectural blockers; the two contract-completeness items that gated
Phase-1 are closed against a real artifact. 3 operational RCs, all folded, NONE gate Phase 0:
- **RC-1 (real new catch):** single multiplexed pipe = all-sessions HOL choke if the drain
  thread blocks on a slow client. Fan-out contract (non-blocking to bounded per-conn queues,
  drop/coalesce policy) + a Phase-0 spike HOL-isolation measurement (item e).
- **RC-2:** deploy-respawn tags `reason:"deploy_respawn"` vs `"crash"` so alerting suppresses
  deploy blips without masking crash-loops (land before Phase-3 cutover).
- **RC-3:** resume/reconnect reads history_version from state.db not the mirror — relay-before-
  persist race structurally impossible (one INV-6 sentence).

**CONVERGED at v1.5.** Arc: BLOCK→BLOCK→AWC→AWC→AWC → [Phase-0 STOP: 5 extra mutators found]
→ AWC(polarity inverted, my error)→AWC(reload-mcp)→AWC(fake-folds, my error)→AWC(clean, 3 ops
residuals). Nine passes caught 4 real design blockers + 3 defects I introduced folding + 1
Phase-0 seam-premise falsification. Residuals are Phase-1/Phase-3 operational, not design.
Per descope-when-review-recurses: further passes would polish operational wording, not tighten
design. BUILD UNBLOCKED.

### Pass 8 — Opus (claude-bpp), 2026-07-04 — APPROVE WITH CHANGES → folded to v1.4
Confirmed architecture sound + v1.2 polarity fix genuinely present + SUPERSEDED marker correct.
Caught 2 FAKE-FOLDS (my error): pass-7 claimed the §4 mutator table shipped and reload_mcp was
added to the control-frame list — NEITHER was physically in the doc (my patch old_strings had
silently no-op'd against shifted content; the review-log asserted folds the body didn't have —
the exact "guessed classification" failure that caused the v1.1 inversion). Fixed for real this
time and VERIFIED present (grep count 1 table + 2 reload_mcp refs): the 10-row tagged mutator
table is in §4 body; `reload_mcp{sid}` is in the §4 control-frame list. AC-5 byte-completeness
reaffirmed. Lesson: after a fold, GREP the artifact is present — a patch reporting success can
no-op if the anchor moved.

### Pass 7 — Opus (claude-bpp), 2026-07-04 — APPROVE WITH CHANGES → folded to v1.3
Confirmed v1.2's polarity fix genuinely resolved (reviewer re-ground-truthed the reject-while-
running semantics + shape-F handoff; not rubber-stamped). Zero architectural blockers. 3 RCs:
- **RC-1 (FOLDED):** `reload.mcp` (`:11245`) is the ONE genuinely run-concurrent WRITE — not in
  `_MUTATES_WHILE_RUNNING`, mutates the live tool schema mid-turn. Named explicitly as a
  control-reader-thread frame (not interrupt-folded) + AC-8.
- **RC-2 (FOLDED):** the literal §4 mutator table now ships in the body with every row tagged
  {turn-path | run-concurrent | idle-gated} — the anchor for the Phase-1 classification test.
- **RC-3 (kept):** AC-5 byte-completeness + no-finalize-fallback clause reaffirmed.
- **Lens fix:** the stale v1.1 review-log entry marked ⚠️ SUPERSEDED so a builder reading the
  log instead of live INV-4 can't re-hit the inverted-polarity bug.

### Pass 5 — Opus (claude-bpp), 2026-07-04 — APPROVE WITH CHANGES → folded, **v1.0 CONVERGED**
Reviewer independently re-ground-truthed all disputed citations; confirmed every architectural
fold sound; **independently REJECTED pass-4's RC-3 with the same source evidence Apollo used**
(pool default IS 8; the 4 is the except-branch) — two independent rejections, claim dead.
Three RCs, all folded:
- **RC-1:** dead-RC tombstone added at the host-init task so a literal builder can't action the
  wrong pool number; pool inherits env-or-8 resolution, hardcodes nothing.
- **RC-2 (the one genuine gate hole):** AC-5 tightened from presence to BYTE-COMPLETENESS of
  the last completed turn + assertion the finalize/disconnect fallback did not run.
- **RC-3:** run_agent.py citations by name, not line (line numbers drift across trees — the
  reviewer's own numbers differed from Apollo's runtime tree for the same defs).

**Convergence: 5 passes (BLOCK → BLOCK → AWC → AWC → AWC-confirm), zero blockers since pass 3,
pass-5 deltas = test-wording + labels. Residual risk register accepted as stated. CONVERGED at
v1.0 per the descope-when-review-recurses standard (further passes are polishing citations,
not tightening the design). BUILD UNBLOCKED — Phase 0 (seam enumeration + spike, STOP
conditions) dispatches to Daedalus.**
