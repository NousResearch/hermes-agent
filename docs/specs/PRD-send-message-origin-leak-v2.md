# PRD — `send_message` origin-leak v2: subagent context-loss + instrumentation

**Status:** v0.3 (Pass-1 + Pass-2 review applied; APPROVED-WITH-CHANGES, fixes folded — ready to build)
**Author:** Apollo
**Date:** 2026-06-22
**Area:** `tools/delegate_tool.py`, `gateway/session_context.py` (new routing-only contextvars), `tools/send_message_tool.py` (resolver + instrumentation), tests
**Repo / track:** fork-internal PR against `Kyzcreig/hermes-agent`; carried as local patch on `main`.
**Predecessor:** PR #71 (v1) fixed the main-turn bare-target leak. This is the **follow-up** for a leak that recurred *after* #71 shipped + the gateway restarted.

**Changelog v0.3 (Pass 2 — APPROVE-WITH-CHANGES, fixes folded in):** Pass 2 confirmed the make-or-break mechanism works (the worker-thread `copy_context()` at tool-submit time, `tool_executor.py:590-596`, captures the send-origin vars set in `_run_with_thread_capture`, so they reach the child's actual `send_message` execution thread). Two changes folded in: **(C1, was a blocker)** the **grandchild/nested-orchestrator** case — spawn-capture reads `_SESSION_*` not the new `_SEND_ORIGIN_*`, so a grandchild under an `orchestrator`-role child (depth ≥2) would NOT inherit the real channel → its bare send would leak to home. **Fix:** the spawn-capture (`delegate_tool.py:1322-1325`) falls back to `get_send_origin()` so a child holding a bound send-origin chains the real grandparent channel down (viable — that capture runs on the child's tool thread which has the send-origin set, verified). **(C2)** pin `_has_messaging_origin` resolution order: `HERMES_CRON_SESSION` guard FIRST, then live `_SESSION_PLATFORM`, then send-origin; wrap `get_send_origin()` to fail toward home (no in-turn error), matching the existing `except: return False` posture.

**Changelog v0.2 (Pass 1 — BLOCK→fixed, verified vs code):**
- **B1 (security):** the v0.1 design (bind parent `HERMES_SESSION_PLATFORM` for the child run) would have **flipped approval gating, skill-disable lists, TTS format, terminal-notify routing, and the prompt-cache key** — because `_SESSION_PLATFORM` is the *sole* signal for `approval._is_gateway_approval_context()` (`approval.py:151`) and is read by `skills_tool.py:417/568`, `tts_tool.py:2066`, `terminal_tool.py:2278`, `prompt_builder.py:1164`, `skill_utils.py:344`. Cron *deliberately clears* this var for exactly these reasons (`cron/scheduler.py:1604-1609`); my design would have set it. **Fixed:** introduce **dedicated routing-only contextvars** (`_SEND_ORIGIN_PLATFORM/_CHAT_ID/_THREAD_ID`) read ONLY by the send-origin resolver — the child's approval/skill/TTS identity stays `subagent` (correct).
- **B2 (architecture):** dropped the "Phase-0 picks between propagate_context_to_thread vs narrow-bind" deferral — inspection shows `propagate_context_to_thread` is **decidably wrong** here (it `copy_context()`s the *entire* parent context incl. `_SESSION_ID`/`_SESSION_KEY`/task_id, corrupting the child's deliberately-distinct identity, `thread_context.py:78`). Resolved to the narrow routing-only bind.
- **Required:** corrected the async-path claim (it DOES reach `_run_single_child` via `_async_runner` `:2305`, and DOES carry `_blackbox_parent_*` — so one bind point covers sync/batch/async; session-key parse is only a fallback). Added CLI/cron-spawned subagent AC (where the approval-flip would've been most visible). Added side-effect non-regression ACs (approval/skills/TTS unchanged). Instrumentation now logs contextvar-vs-os.environ provenance + thread name.

---

## 1. Summary & Goal

**The recurrence (observed 2026-06-22 00:02 PT, post-#71, post-restart).** A progress update from the **#tdd** QA-campaign turn (session `…6a7a77e3`) landed in **#hermes** (Discord home) instead of #tdd. The v1 fix demonstrably works for the main turn (the *same* session sent correctly to #tdd at 23:36 PT, 26 min earlier), so a **second path** still leaks.

**Root cause #1 — CONFIRMED by code inspection (the load-bearing finding): subagent sends have no session origin.**
- `delegate_task` runs each child via a bare `ThreadPoolExecutor`: `_run_single_child` → `_timeout_executor.submit(_run_with_thread_capture)` (`delegate_tool.py:1648-1668`), and the batch path `executor.submit(_run_single_child, …)` (`:2358-2367`). **Neither wraps the child target in `copy_context()` / `propagate_context_to_thread()`.**
- A `ThreadPoolExecutor` worker starts with an **empty `contextvars.Context`** (documented in `tools/thread_context.py:4-14`). So inside a child run, `gateway.session_context`'s `_SESSION_PLATFORM` / `_SESSION_CHAT_ID` are **unset**.
- Therefore a subagent calling `send_message`/`react` with a **bare target** → `_has_messaging_origin()` returns `False` → `_resolve_send_target` returns `NOT_IN_TURN` → falls back to the **global home channel**. The exact v1 leak class, on the subagent path v1 didn't cover.
- The child agent already carries the parent's origin as `_blackbox_parent_platform` / `_blackbox_parent_chat_id` / `_blackbox_parent_chat_name`, captured on the parent thread at spawn (`delegate_tool.py:1319-1325`) — for blackbox attribution. **Nothing binds these into the child's session contextvars.** The data we need is already captured; it's just not wired to routing.

**Root cause #2 — UNCONFIRMED (the honest gap): the specific 00:02 leak was tagged with the PARENT session, not `[subagent-N]`.**
- A subagent logs under `log_prefix=[subagent-N]` + its own `session_id` (`delegate_tool.py:1230`, distinct child session). The 00:02 `send_message completed` line was tagged `[20260619_175552_6a7a77e3]` — the **parent** session, no `[subagent-N]`.
- That points at a **main-turn** send losing origin, which the inspected paths (`_set_session_env` wraps the whole turn `:8952→10121`; the parallel-tool propagator does `copy_context()` `thread_context.py:78`) say *shouldn't* happen.
- The actual 00:02 tool-call args + result were **pruned by a mid-turn context compaction** (that turn is at API call #85+), so the mechanism is **not provable from the surviving logs**. I will not fabricate a root cause for it.

**Goal.**
1. **Fix RC#1 (the proven seam):** a subagent's bare-target `send_message`/`react` must route to the **parent turn's origin channel**, never the global home — by binding the captured parent origin into **dedicated routing-only contextvars** (NOT `_SESSION_PLATFORM`) for the child's run. The child's approval/skills/TTS identity stays `subagent` (unchanged). Covers sync, batch, and async-background children (one bind point).
2. **Instrument RC#2 (the unproven seam):** add a `logger.warning` in `_resolve_send_target` that fires whenever it returns `NOT_IN_TURN` **but a session_key is nevertheless resolvable** (i.e. "we fell back to home while something says we're in a session") — capturing platform-bound state, session_key, thread id, and whether we're on a non-main thread. The next occurrence becomes a one-line root-cause instead of forensic archaeology. This is diagnostic, not a behavior change.
3. **Regression-test both** against the real delegation path (no-mock child run that asserts a bare send routes to the parent origin).

---

## 2. Non-Goals

- **No change to the v1 main-turn fix** (PR #71) — it stands and is verified.
- **No change to cron/CLI/TUI/desktop/ACP home-default** — preserved exactly (v1 invariant I2).
- **No new config keys / `HERMES_*` env vars.** Reuse the existing `_blackbox_parent_*` capture + `gateway.session_context.set_session_vars`.
- **No "fix" for RC#2 yet** — we instrument first and fix once the instrumentation captures the actual path. Speculatively rewriting main-turn context handling without a proven defect would violate the house "no speculative infra / verify the premise" rule. (If instrumentation later proves a main-turn seam, that's its own spec.)
- **Not** making subagents first-class messaging participants — they get the parent's origin for *bare* sends only; explicit targets and the home-default-when-no-parent-origin still work as before.

---

## 3. Constitution / Invariants

- **Invariant I1 — A subagent bare send never silently hits the global home when a parent messaging origin exists.**
  - *Why:* contract/privacy — the v2 leak class. A subagent's progress update must land in the turn's channel.
  - *Closeout proof:* a no-mock test that runs a real child agent (via `_run_single_child` / its thread path) with the parent origin = `discord:CHAN_A`, home = `CHAN_B` (A≠B), invokes a bare `send_message` inside the child, and asserts the resolved chat = `CHAN_A`, never `CHAN_B`.
- **Invariant I2 — No parent origin ⇒ unchanged behavior.** A child with no captured parent origin (e.g. CLI-spawned, cron) resolves a bare send to home exactly as today (no new error, no crash).
  - *Closeout proof:* test with `_blackbox_parent_platform=""` → `NOT_IN_TURN` → home.
- **Invariant I3 — Instrumentation is observation-only.** The new `logger.warning` changes no routing decision and never raises (wrapped, best-effort), and does not log secrets/PII (platform + chat-id + thread-id + booleans only, no message body).
  - *Closeout proof:* the log call is in a `try/except`; grep shows no message content in the log args; routing tests are byte-identical with/without the log.
- **Invariant I4 — Concurrency-safe + cache-safe.** Binding the child's send-origin uses task-local `set_send_origin`/`clear_send_origin` (contextvars) with a `finally`, scoped to the child run only; never mutates `os.environ`; no prompt-cache/alternation impact.
  - *Closeout proof:* the bind is inside the child-run wrapper with a `finally` clear; concurrent children with different parents don't cross (test).
- **Invariant I5 — Child's approval/skills/TTS/identity unchanged by the routing fix.** The routing bind uses dedicated `_SEND_ORIGIN_*` vars; `_SESSION_PLATFORM` stays unset/`subagent` in the child, so `_is_gateway_approval_context()` (approval mode + `delegation.subagent_auto_approve`), skill-disable lists, TTS opus/mp3 choice, terminal-notify routing, prompt-cache key, and blackbox attribution are all byte-identical to today.
  - *Why it matters:* security — Pass-1 caught that reusing `_SESSION_PLATFORM` would flip subagent dangerous-command gating (esp. CLI/cron-spawned children with no `HERMES_EXEC_ASK`).
  - *Closeout proof:* tests asserting, for a child run with a parent send-origin bound: `_is_gateway_approval_context()` is still False, the subagent approval callback path is unchanged, `_is_skill_disabled` resolves the child's (not parent's) platform, and `_blackbox_parent_*` attrs are intact — across gateway-spawned AND CLI/cron-spawned children.

---

## 4. Resolved Decisions

- **D-1 — Subagent bare send routes to PARENT origin (not error).** A subagent legitimately does the parent's work in the parent's channel, and the parent origin is already captured. Route to it. (The v1 cross-platform main-turn case still errors; this is a different, legitimate case.)
- **D-2 — Dedicated routing-only contextvars, bound in the child-run wrapper (Pass-1 B1).** Add `_SEND_ORIGIN_*` contextvars in `session_context.py` read ONLY by `send_message`/`react` origin resolution. Bind them from the child's already-captured `_blackbox_parent_platform/_chat_id` (`delegate_tool.py:1319-1325`) in `_run_with_thread_capture` with a `finally` clear. Do NOT reuse `_SESSION_PLATFORM` (it would flip approval/skills/TTS/cache — the child must keep its `subagent` identity for those).
- **D-3 — Instrument-first for RC#2.** Ship the `_resolve_send_target` tripwire now; do not speculatively rewrite main-turn context handling (house rule: verify the premise). Re-evaluate when it fires with real data.
- **D-4 — RESOLVED to narrow routing-only bind (Pass-1 B2); `propagate_context_to_thread` rejected.** Inspection (not a Phase-0 coin-flip) shows a full context-copy corrupts the child's distinct `_SESSION_ID`/`_SESSION_KEY`/task_id identity. Phase 0 keeps ONLY the RED-test reproduction step, not a mechanism bake-off.
- **D-5 — Ship as fork PR + local patch** (same workflow as v1).
- **D-6 — One bind point covers sync/batch/async (Pass-1 correction).** All three child paths funnel through `_run_single_child → _run_with_thread_capture` (`_async_runner` `:2305`), and all carry `_blackbox_parent_*`. No separate async bind; the session-key parse is a defensive fallback only.

---

## 5. Architecture / Design

### 5.1 The seam (confirmed)
```
delegate_task(sync/batch)
  └─ _run_single_child(...)                      # runs INSIDE a worker thread
       └─ _timeout_executor.submit(_run_with_thread_capture)   # bare ThreadPoolExecutor
            └─ child.run_conversation(goal)      # contextvars EMPTY here
                 └─ ...child calls send_message(target="discord")  # bare
                      └─ _has_messaging_origin() -> False (no _SESSION_PLATFORM)
                           └─ NOT_IN_TURN -> get_home_channel() -> LEAK
```

### 5.2 The fix (RC#1) — dedicated routing-only send-origin contextvars

**Do NOT reuse `_SESSION_PLATFORM`** (Pass-1 B1: it drives approval gating, skill-disable, TTS, terminal-notify, prompt-cache key — setting it for the child would change the child's security/skill/TTS behavior). Instead add **three new task-local contextvars** in `gateway/session_context.py`, read ONLY by the send-origin resolver:

```python
# gateway/session_context.py  (new — routing-only, NOT read by approval/skills/tts)
_SEND_ORIGIN_PLATFORM = ContextVar("HERMES_SEND_ORIGIN_PLATFORM", default=_UNSET)
_SEND_ORIGIN_CHAT_ID  = ContextVar("HERMES_SEND_ORIGIN_CHAT_ID",  default=_UNSET)
_SEND_ORIGIN_THREAD_ID= ContextVar("HERMES_SEND_ORIGIN_THREAD_ID",default=_UNSET)

def set_send_origin(platform, chat_id, thread_id="") -> list[Token]: ...
def clear_send_origin(tokens) -> None: ...
def get_send_origin() -> tuple[str,str,str]:   # ("","","") when unset
```
These are contextvars only (no `os.environ` write — preserves I4 + the concurrency property the module exists for).

At the child-run entry, bind the **captured parent origin** (already on the child as `_blackbox_parent_platform/_chat_id`, `delegate_tool.py:1319-1325`) into the send-origin vars, clear in `finally`:

```python
def _run_with_thread_capture():
    _worker_thread_holder["t"] = threading.current_thread()
    _origin_tokens = _bind_child_send_origin(child)   # NEW
    try:
        return child.run_conversation(user_message=goal, task_id=child_task_id)
    finally:
        _clear_child_send_origin(_origin_tokens)       # NEW (no-op if None)
```

`_bind_child_send_origin(child)`:
- platform = `getattr(child, "_blackbox_parent_platform", "")`, chat_id = `getattr(child, "_blackbox_parent_chat_id", "")`.
- If both present → `set_send_origin(platform, chat_id, thread_id=…)` → return tokens.
- Else (no parent origin: CLI/cron-spawned) → return `None` → child keeps unset send-origin → bare send falls to home per I2 (unchanged).

This is the **one bind point** covering sync, batch, AND async-background children — all three reach `_run_single_child → _run_with_thread_capture` (`_async_runner` `:2305`). (Pass-1 correction: async carries `_blackbox_parent_*` too; the session-key parse is only a defensive fallback, not the primary source.)

`_interactive_origin` in `send_message_tool.py` gains the send-origin vars as a resolution source, consulted **after** the live `_SESSION_PLATFORM` (so a true gateway turn still wins) and **before** giving up:

```python
def _interactive_origin(platform_name):
    # 1. live gateway turn (existing v1 path): _SESSION_PLATFORM == platform_name -> origin
    # 2. NEW: send-origin contextvars (subagent's captured parent origin)
    sp, sc, st = get_send_origin()
    if sp and sp.lower() == want and sc:
        return (sc, st or None)
    # 3. existing session-key parse fallback
    ...
```
And `_has_messaging_origin()` becomes True when EITHER `_SESSION_PLATFORM` is bound (gateway turn) OR a send-origin is bound (subagent with parent origin) — so the resolver returns ORIGIN, not NOT_IN_TURN, for a subagent bare send.

**Crucially, this touches nothing the child's approval/skills/TTS read** — `_SESSION_PLATFORM` stays unset/`subagent` in the child, so `_is_gateway_approval_context()`, skill-disable, and TTS format are byte-identical to today (I5). The send resolver is the *only* consumer of the new vars.

### 5.3 The instrumentation (RC#2) — unchanged intent, provenance-aware
In `_resolve_send_target`, when about to return `NOT_IN_TURN` **and** a session_key is nonetheless resolvable, emit a wrapped `logger.warning` capturing: `platform_bound` (contextvar vs os.environ provenance — the gateway leaves a stale `os.environ["HERMES_SESSION_KEY"]`, `run.py:15265`, so distinguish), `session_key[:64]`, `thread.name`, `cron` flag, and whether a send-origin was set. No message body, no secrets (I3). This is the tripwire for any *remaining* leak path (incl. the unproven RC#2).

### 5.5 Grandchild chaining (Pass-2 C1) — orchestrator subagents that spawn their own children
The spawn-time origin capture (`delegate_tool.py:1322-1325`) currently reads `_SESSION_*`. A `role="orchestrator"` child (depth ≥2 via `max_spawn_depth`) runs with `platform="subagent"` and no `_SESSION_*` bound, so its grandchild would capture `_blackbox_parent_platform="subagent"`, `_chat_id=""` → no bind → grandchild bare send → home (the v2 leak, one level down). **Fix:** the capture falls back to the send-origin var so a child that *holds* a bound send-origin chains the real grandparent channel to its grandchild:

```python
# delegate_tool.py spawn capture (runs on the child's tool thread, which has send-origin set)
from gateway.session_context import get_send_origin
_so_plat, _so_chat, _so_thread = get_send_origin()
child._blackbox_parent_platform = _gse("HERMES_SESSION_PLATFORM","") or _so_plat or getattr(parent_agent,"platform","") or ""
child._blackbox_parent_chat_id  = _gse("HERMES_SESSION_CHAT_ID","")  or _so_chat or ""
```
This makes origin transitive through any nesting depth: grandchild gets the real channel, not `"subagent"`. (Default `MAX_DEPTH=1` rejects grandchildren, so this is the opt-in-orchestrator case — but I1 is stated unconditionally, so we fix it rather than carve it out.)

### 5.6 Resolution order pin (Pass-2 C2)
`_has_messaging_origin()` checks in this fixed order, fail-closed-to-home throughout:
1. `HERMES_CRON_SESSION` set → **False** (cron always home; never treat a cron-spawned subagent as a messaging turn).
2. live `_get_session_platform()` truthy → **True** (real gateway main turn — wins over any send-origin).
3. `get_send_origin()[0]` truthy → **True** (subagent with a bound parent origin).
4. else → **False**.
`get_send_origin()` is wrapped so an import/lookup failure returns "" → home, never an in-turn error (matches `send_message_tool.py:167-176`'s existing `except: return False`).

---

## 6. Implementation Phases

### Phase 0 — Reproduce the seam (RED test first)
- *Unit/script check:* write a failing test FIRST: run a real child via `_run_single_child` with parent origin captured on the child (`_blackbox_parent_platform/_chat_id` = `discord:CHAN_A`) + home `CHAN_B`≠A, child invokes a bare `send_message` (with `_send_to_platform` stubbed to capture dest) → assert it currently routes to **HOME** (RED, proves the seam). No mechanism bake-off (D-4 resolved to the dedicated-var bind).
- *E2E:* the RED test above on the real delegation path.
- *Verify with:* `pytest tests/tools/test_subagent_send_origin.py::test_seam_red` fails pre-fix (routes home).

### Phase 1 — Implement the dedicated send-origin bind (RC#1) for sync + batch + async
- *Unit/script check:* `session_context.set_send_origin/get_send_origin/clear_send_origin` round-trip; `_bind_child_send_origin(child)` returns tokens when `_blackbox_parent_*` present, `None` when absent.
- *E2E/integration:* real child run (sync path) → bare `send_message` → routes to parent origin `CHAN_A`, not home `CHAN_B`. Repeat for batch (`delegate_task(tasks=[…])`) and async-background child (all funnel through `_run_with_thread_capture`).
- *Negative/adversarial:* (a) child with NO parent origin → bare send → home (I2, unchanged). (b) two concurrent children with **different** parent origins → each routes to its own parent (I4, no cross-bleed). (c) explicit-target subagent send → reaches the named target (unaffected). (d) a true gateway main-turn (`_SESSION_PLATFORM` bound) still wins over any stale send-origin (resolution order). (e) **grandchild** under an orchestrator child → bare send routes to the real grandparent channel, not `subagent`/home (§5.5 chaining).
- *Verify with:* `pytest tests/tools/test_subagent_send_origin.py -q` → all pass; the Phase-0 RED test now GREEN.

### Phase 2 — Instrumentation (RC#2)
- *Unit/script check:* calling `_resolve_send_target` with a session_key resolvable but no messaging-origin emits exactly one warning; with nothing bound (true cron/CLI, no session_key) emits none.
- *E2E:* `Not applicable (log-only).`
- *Negative/adversarial:* the warning is wrapped — a failure in the logging block never changes the returned state (test by monkeypatching the logger to raise). Confirm it does NOT spam on legitimate cron (cron clears platform + sets `HERMES_CRON_SESSION`; gate the warning to skip the cron case).
- *Verify with:* `pytest …::test_resolver_warns_on_unexpected_home_fallback` + `::test_warning_never_alters_state` + `::test_no_warning_on_cron`.

### Phase 3 — Security / identity non-regression (Pass-1 B1 guard)
- *Unit/script check:* with a parent send-origin bound for a child run, assert the child still sees: `_is_gateway_approval_context()` == False (unchanged), `skills_tool._get_session_platform()` resolves the child's platform not the parent's, TTS format unchanged, and `child._blackbox_parent_*` intact. Repeat the assertions for a **CLI/cron-spawned** child (no `HERMES_EXEC_ASK`, no gateway) — the case where reusing `_SESSION_PLATFORM` would have flipped dangerous-command gating.
- *E2E:* a child run still records under its own session_id (not the parent's) in the session store; a dangerous-command in a CLI-spawned child still honors `delegation.subagent_auto_approve` (not the gateway approval queue).
- *Verify with:* existing `tests/tools/test_delegate_blackbox_attr.py` + new `::test_origin_bind_preserves_child_approval_and_skills` (gateway + CLI/cron variants).

### Phase 4 — Ship
- Branch off `fork/main`, commit Phases 1-3 (+ tests), push to fork, open fork PR, real gates green (`check-attribution` cosmetic, ignore), squash-merge, cherry-pick onto live `main`, restart to activate (gated — ask Ace), verify live. Update patch tracker.
- *Verify with:* fork PR real gates green; `git log --oneline origin/main..main` shows the linear patch; live re-verify after restart (a real subagent bare send lands in the turn's channel).

---

## 7. Security, Privacy, Ops, Observability
- **Privacy:** closes the v2 cross-channel leak for subagent sends. Instrumentation logs no message body/secrets (I3).
- **No new secrets/config/public surface.**
- **Observability:** the new `NOT_IN_TURN`-despite-session warning is the standing tripwire for any *remaining* leak path (incl. the unproven RC#2). Greppable in `agent.log`.
- **Rollback:** revert the single delegate_tool + send_message_tool commit; drop the local cherry-pick. Pure code path, zero state/migration.

## 8. Risks & Mitigations
- **R1 — Routing bind corrupts child identity/security.** *Mitigation:* dedicated `_SEND_ORIGIN_*` vars read ONLY by the send resolver; `_SESSION_PLATFORM` untouched → approval/skills/TTS unchanged. Phase 3 proves it (gateway + CLI/cron).
- **R2 — Concurrent children cross-bleed origins.** *Mitigation:* task-local `set_send_origin` (contextvars, per-thread/per-task), `finally` clear, explicit Phase-1 concurrent-different-parents test.
- **R3 — RC#2 is a real separate main-turn seam we're not fixing yet.** *Mitigation:* instrumentation makes the next occurrence self-diagnosing; honest Non-Goal, deferred until proven. The subagent fix is the proven win regardless.
- **R4 — Async-background child uses a different runner** (`dispatch_async_delegation`). *Mitigation:* its `_async_runner` calls `_run_single_child → _run_with_thread_capture` (`:2305`), so the single bind point covers it; the child carries `_blackbox_parent_*` (not only session_key). Phase 1 covers the async path explicitly.
- **R5 — Stale send-origin leaking across runs.** *Mitigation:* `finally: clear_send_origin` on every child-run entry; resolution order puts a live `_SESSION_PLATFORM` (true gateway turn) ahead of the send-origin var, so a main turn never picks up a leftover. Phase-1 negative test (d) covers it.

## 9. Open Questions
- **OQ1 — RESOLVED (Pass 1):** dedicated routing-only contextvar bind; `propagate_context_to_thread` rejected by inspection (corrupts child identity). No Phase-0 bake-off.
- **OQ2 — RESOLVED (2026-06-22): the 00:02 leak WAS a subagent send, same RC#1 — there is no separate main-turn seam. Load-bearing proof is `agent.log`; store-state is corroborating.** (a) `agent.log` shows the #tdd turn was running a **16-way `delegate_task` fan-out** continuously through the 00:02:12 PT `send_message completed`; the `[6a7a77e3]` parent-session tag on that log line is inherited attribution, not main-thread proof — a subagent send logs under the parent session id. (b) Store corroboration: the #tdd session's transcript (gateway store `state.db` — that older session predates and is **not** in Apollo's `lcm.db`) shows **zero `send_message` tool-calls** across its retained 732 rows (a heavy `execute_code`/`patch`/`terminal` coding turn); had the leak been a *main-turn* send it would appear here. **Conclusion: the recurrence was the subagent seam (RC#1) the whole time; RC#2 (a hypothetical separate main-turn seam) does not exist.** *Caveats (honest):* `state.db` hygiene rewrites have since trimmed the actual 07:02-UTC leak minute (retained #tdd transcript now starts 07:48 UTC), so the leak send itself is no longer directly inspectable — store-absence is corroborating, the agent.log fan-out is the proof. The instrumentation tripwire stays as defense-in-depth, but the fix shipped in #83 closes the actual cause.
  - *Method note (for future forensics):* to recover a **compacted tool call's arguments**, query the right store — **compaction only trims the live context window, the underlying stores retain `tool_calls` (with verbatim `function.arguments`) by design.** Two stores, don't confuse them: Apollo's LCM raw store `~/.hermes/lcm.db` (`messages.tool_calls`; immutable raw rows, never auto-purged) holds *current/recent* LCM-era sessions; the gateway transcript store `~/.hermes/state.db` (`messages.tool_calls`, keyed by `session_id`) holds older/pre-LCM sessions but is subject to **hygiene rewrites** that can trim old turns. The earlier "args were compacted away → unprovable" framing was wrong — query the store first; just pick the store the session actually lives in.

## 10. Acceptance Criteria
- [ ] **AC1 (I1):** A real subagent bare `send_message` with parent origin `discord:CHAN_A` (home `CHAN_B`) routes to `CHAN_A`. Evidence: `pytest tests/tools/test_subagent_send_origin.py::test_bare_subagent_send_routes_to_parent_origin` (stubbed `_send_to_platform` captured `CHAN_A`).
- [ ] **AC1b:** Same for `react` and for the batch + async-background child paths. Evidence: the path-parametrized tests pass.
- [ ] **AC1c (resolution order):** A live gateway main-turn (`_SESSION_PLATFORM` bound) still routes to its own origin even if a stale send-origin var is present. Evidence: `::test_session_platform_wins_over_send_origin`.
- [ ] **AC1d (grandchild chaining, Pass-2 C1):** A grandchild under an `orchestrator`-role child routes a bare send to the real grandparent channel (not `subagent`/home). Evidence: `::test_grandchild_under_orchestrator_routes_to_real_origin`.
- [ ] **AC2 (I2):** Subagent with no parent origin → bare send → home, no error. Evidence: `::test_no_parent_origin_falls_to_home`.
- [ ] **AC3 (I4):** Two concurrent children with different parent origins each route to their own parent. Evidence: `::test_concurrent_children_no_origin_crossbleed`.
- [ ] **AC4 (I3):** Instrumentation emits on unexpected home-fallback, logs no body/secrets, never alters routing, silent on cron. Evidence: `::test_resolver_warns_on_unexpected_home_fallback` + `::test_warning_never_alters_state` + `::test_no_warning_on_cron`.
- [ ] **AC5 (I5):** Child identity/attribution unchanged by the origin bind (`_blackbox_parent_*` intact, own session_id). Evidence: `tests/tools/test_delegate_blackbox_attr.py` + `::test_origin_bind_preserves_child_identity`.
- [ ] **AC5b (I5 security — the Pass-1 blocker guard):** With a parent send-origin bound, the child's `_is_gateway_approval_context()` stays False, `subagent_auto_approve` is honored (not the gateway approval queue), skill-disable resolves the child's platform, and TTS format is unchanged — for BOTH gateway-spawned AND CLI/cron-spawned children. Evidence: `::test_origin_bind_preserves_child_approval_and_skills[gateway]` + `[cli_cron]`.
- [ ] **AC6 (explicit unchanged):** Explicit-target subagent sends reach the named target. Evidence: `::test_explicit_subagent_target_unchanged`.
- [ ] **AC7 (ship + live):** Fork PR real gates green; live re-verify after restart — a real subagent bare send lands in the turn's channel, and the instrumentation warning is silent on the happy path. Evidence: `gh pr` rollup + live delivery observation.

## 11. Review Handoff
1. `prd-review-pipeline` — **2 Opus passes** (review+fix each), Opus-only, vary role. Lenses: concurrency (contextvar bind across nested executors), security/privacy (leak closure + the approval-gate non-regression + no-PII logging), architecture (dedicated-var vs context-copy; child-identity preservation), testing (real-delegation no-mock e2e + concurrent cross-bleed + CLI/cron approval).
2. Implement approved phases (Phase 0 RED test first).
3. `prd-closeout` with live re-verify.
