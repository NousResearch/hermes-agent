# PRD: Thread the failover reason through the retry-exhaustion fallback site

**Status:** Draft → review → build
**Author:** Apollo
**Date:** 2026-07-12
**Repo:** hermes-agent (fork)
**Scope:** one bug, one component (`agent/conversation_loop.py` retry-exhaustion failover) + tests

---

## 1. Summary & Goal

**What changes:** The generic "max retries exhausted → fall back" site in the conversation
retry loop (`agent/conversation_loop.py:4144`, `if agent._try_activate_fallback():`) currently
calls the failover **bare** — it drops the already-computed `classified.reason`. As a result, a
provider failover that exhausts its retries announces a **reason-less** route-change line:

```
🔄 Model fallback: claude-apr/claude-opus-4-8 → claude-apx-1/claude-opus-4-8
```

instead of the intended reason-tagged form PR #280 established:

```
🔄 Model fallback (provider overloaded): claude-apr/claude-opus-4-8 → claude-apx-1/claude-opus-4-8
```

**Goal:** every failover announce carries a `(reason)` rider whenever the reason is knowable —
close the last un-threaded failover site so the PR #280 guarantee ("the announce always says WHY")
actually holds on the retry-exhaustion path, not just the eager-failover paths.

**Why now:** Ace observed a live reason-less fallback (2026-07-11 21:14–21:17, session
`20260710_182847_84cb906d`), during a `503 {"error":"no eligible sub"}` storm on the `claude-apr`
relay pool. 25 such 503s in the window; the pool was saturated by heavy fleet work. The failover
itself was correct (opus→opus, apr→apx-1, apx-1 answered fine). Only the *announce* was defective.

---

## 2. Non-Goals

- **NOT** changing failover *behavior* — the retry/backoff/chain-advance logic is correct and
  untouched. This is purely about threading the reason label into the announce.
- **NOT** touching the `claude-apr` relay pool capacity / `no eligible sub` root cause. That's an
  infra-load event, not a code defect; out of scope here.
- **NOT** re-architecting the reason-plumbing. PR #280's `_resolve_failover_reason` /
  `_fallback_reason_label` / `_FALLBACK_REASON_LABELS` machinery is correct and reused as-is.
- **NOT** adding new reason labels. `overloaded → "provider overloaded"` already exists.
- **NOT** stamping `_pending_stream_error_reason` on the pre-delivery streaming-fail path (an
  alternative fix considered and rejected — see §8 Risks).

---

## 3. Constitution / Invariants

- **Invariant — every knowable-reason failover announces its reason.**
  - *Why it matters:* Contract established by PR #280. A bare `A → B` line leaves the user guessing
    whether it was a policy refusal, a rate limit, or a transient blip — the exact ambiguity Ace hit.
  - *Closeout proof:* a test that drives a retry-exhausted classified failover (reason=`overloaded`)
    through the loop and asserts the emitted announce string contains `(provider overloaded)`.

- **Invariant — explicit reason wins; backfill only fills a None.**
  - *Why it matters:* `_resolve_failover_reason` already enforces "explicit-wins / backfill /
    consume-once." Passing `reason=classified.reason` at 4123 must NOT clobber a stamped
    `_pending_stream_error_reason` when the classified reason is None, nor be clobbered by it when
    classified is non-None.
  - *Closeout proof:* existing `_resolve_failover_reason` unit tests stay green; add a case asserting
    an explicit `overloaded` passed to `try_activate_fallback` survives even if a stale pending stamp
    exists.

- **Invariant — no behavior change for the un-reasoned floor.**
  - *Why it matters:* When neither a classified reason NOR a pending stamp exists (genuinely
    unknowable), the announce must still emit the honest `"connection issue"` floor, not regress to
    bare.
  - *Closeout proof:* a test with `reason=None` and no stamp asserts the announce renders
    `(connection issue)`, never a bare line.

- **Invariant — minimal diff.** Single-site change plus tests; no refactor of the surrounding loop.

---

## 4. Resolved Decisions

| Question | Decision | Why |
|---|---|---|
| Fix at 4144 (retry-exhaustion site) or stamp `_pending_stream_error_reason` on pre-delivery fail? | Fix at 4144: pass `reason=classified.reason`. | `classified` is already in scope at 4144 (computed at 2750). It's the direct, minimal, correct source. Stamping the stream path is a second, indirect route that only helps mid-stream fails and adds a new mutable-state write. |
| Is `classified` always in scope at 4144? | **Yes — PROVEN, not asserted.** There is exactly ONE `except ... as api_error:` at line 2478. `classified = classify_api_error(...)` (2750) and `if retry_count >= max_retries:` (4123) both sit at indent-16 directly inside it; the 4144 call is indent-20 nested in that `if`. 2750 executes unconditionally at the handler top before any branch that can reach 4144. | Ground-truthed: single handler, unconditional assignment precedes the call. Still add `getattr(classified, 'reason', None)` defensive access as cheap error-path insurance (RC2). |
| Any other un-threaded `_try_activate_fallback()` sites to fix in the same PR? | **10 sites total** (not 12 — see §5 audit). Thread every site with a knowable reason: **4144** (primary), **3935** (client-error block), **1241** (Nous rate-limit guard), **1991** (content-filter stream kill). Leave the 3 genuinely reason-less sites (1587/1660 empty/invalid-response, 5246 empty-response) on the `connection issue` floor. 1820/3386/3422 already threaded. | The floor is honest where no reason is computable; don't fabricate one. But 4 sites DO have a knowable reason and were bare — thread them. |
| Ground-truth of the live incident | `overloaded` (503) classified correctly; the label map has `overloaded → 'provider overloaded'`; the announce path (`_fallback_reason_label`) is airtight. The ONLY defect is the dropped `reason=` at 4144. | Traced end-to-end against the deployed runtime tree. |
| Does threading 3935 show a *misleading* reason (RC-Impl)? | **No — thread it.** Every reason reaching 3935 (`content_policy_blocked`, `ssl_cert_verification`, else non-retryable 4xx) is ALREADY named to the user in the block's own status prose above the call. Threading it into the announce rider just makes the route-change line consistent with the status line. | The block already tells the user the reason; the rider echoing it is not misleading, it's consistent. |

---

## 5. Architecture / Design

### The failover-reason data flow (as-built, PR #280)

```
API error raised
   │
   ▼
classify_api_error(...)  →  classified.reason  (e.g. FailoverReason.overloaded)   [conversation_loop.py:2750]
   │
   ├── eager transport/rate-limit failover  →  _try_activate_fallback(reason=classified.reason)   [3386]  ✅ threaded
   ├── auth failover                         →  _try_activate_fallback(reason=classified.reason)   [3422]  ✅ threaded
   ├── client-error (should_fallback) block  →  _try_activate_fallback()                            [3935]  ❌ bare (fix)
   └── retry-exhaustion floor                →  _try_activate_fallback()                            [4144]  ❌ bare (THE BUG)
                                                        │
                                                        ▼
                                          try_activate_fallback(reason=None)
                                                        │
                                          reason = _resolve_failover_reason(agent, None)
                                                        │   (backfill from _pending_stream_error_reason,
                                                        │    which is ALSO None for a pre-delivery fail)
                                                        ▼
                                          _emit_fallback_announce(..., reason=None)
                                                        │
                                          _reason_label = _fallback_reason_label(None)  →  None
                                                        ▼
                                          BARE announce:  "🔄 Model fallback: A → B"   ← the defect
```

### The fix

At line 4144, thread the reason that's already in scope (with defensive access per RC2):

```python
# BEFORE
if agent._try_activate_fallback():

# AFTER
if agent._try_activate_fallback(reason=getattr(classified, "reason", None)):
```

`classified` is **proven** in scope: exactly one `except ... as api_error:` (line 2478) wraps both
the unconditional `classified = classify_api_error(...)` (2750, indent-16) and the `if retry_count
>= max_retries:` block (4123, indent-16); the call (4144, indent-20) nests inside that `if`. 2750
precedes every branch that can reach 4144. The `getattr(..., None)` is cheap error-path insurance,
not a substitute for the proof. `_resolve_failover_reason` enforces explicit-wins, so a real
`overloaded` is safe even if a stale pending stamp exists; and a `None` (unclassifiable) still
falls to the honest `connection issue` floor.

### Site audit — all 10 `_try_activate_fallback` call sites (RC3, ground-truthed)

| Line | Context | Reason source | Action |
|---|---|---|---|
| 1241 | Nous portal rate-limit guard (pre-call skip) | knowable: `rate_limit` | **THREAD** `reason=FailoverReason.rate_limit` |
| 1587 | Empty/malformed response — eager fallback | no classification at site | leave (→ `connection issue` floor, by design) |
| 1660 | Invalid-response retry exhaustion | no classification at site | leave (→ floor, by design) |
| 1820 | Safety refusal | `content_policy_blocked` | ✅ already threaded |
| 1991 | Content-filter terminated stream | knowable: `content_policy_blocked` | **THREAD** `reason=FailoverReason.content_policy_blocked` |
| 3386 | Eager transport/rate-limit failover | `classified.reason` | ✅ already threaded |
| 3422 | Auth failover | `classified.reason` | ✅ already threaded |
| 3935 | Client-error `should_fallback` block | `classified.reason` in scope | **THREAD** `reason=classified.reason` (RC-Impl resolved: reason already shown in block prose) |
| 4144 | **Retry-exhaustion floor (THE BUG)** | `classified.reason` in scope | **THREAD** `reason=getattr(classified,"reason",None)` (primary fix) |
| 5246 | Empty-response provider switch | no classification at site | leave (→ floor, by design) |

**Reconciliation:** 10 sites total. 3 already threaded (1820/3386/3422). 4 threaded by this PR
(1241/1991/3935/4144). 3 left on the honest floor by design (1587/1660/5246) — each annotated with
a comment explaining why it has no classified reason. The prior "12" was a miscount; this table is
the authoritative enumeration and goes verbatim into the PR description.

---

## 6. Implementation Phases

- **Phase 1 — Thread `reason=classified.reason` at the retry-exhaustion site (4144).**  The one-line core fix.
  - *Unit/script check:* new unit test drives the retry loop to exhaustion with a mocked `503
    no eligible sub` (classifies `overloaded`), asserts `try_activate_fallback` is called with
    `reason=FailoverReason.overloaded` (not None/absent).
  - *E2E/integration check:* a loop-level test (mock transport raising 503 on primary, succeeding on
    fallback) captures the emitted status callback and asserts the announce string contains
    `(provider overloaded)` and the route `A → B`. Changes a user-facing announce path → e2e required.
  - *Negative/adversarial:* (a) `reason=None` + no pending stamp → announce renders `(connection issue)`,
    never bare. (b) explicit `overloaded` + stale non-None pending stamp → explicit wins, announce says
    `(provider overloaded)` not the stamp's label.
  - *Evals:* Not applicable — deterministic control-flow fix, no ML/heuristic.
  - *Verify with:* `cd ~/.hermes/runtime/hermes-agent && python -m pytest tests/ -k "fallback_reason and (retry_exhaust or overloaded)" -x -q` → all green.

- **Phase 2 — Thread the remaining 3 knowable-reason sites (1241, 1991, 3935).**  Apply the same
  threading to every site the §5 audit marks THREAD: 1241 (`rate_limit`), 1991
  (`content_policy_blocked`), 3935 (`classified.reason`). Annotate the 3 by-design floor sites
  (1587/1660/5246) with a comment stating why they have no classified reason.
  - *Unit/script check:* an **effect-gated** invariant test (NOT a paren-emptiness grep — RC1):
    the test imports the loop module, parses each threaded call site's AST (or asserts via a
    call-capturing mock at each site), and asserts the actual `reason=` argument is a non-None
    `FailoverReason` for all 4 THREAD sites. A site "fixed" as `reason=None` FAILS this gate.
  - *E2E/integration check:* per-reason loop tests — a `429 rate_limit` at the 1241 path asserts
    the announce contains `(rate limit)`; a content-filter kill at 1991 asserts `(safety refusal)`.
    Same proven announce path as Phase 1, exercised per site.
  - *Negative/adversarial:* the effect-gate test ALSO asserts the 3 by-design floor sites remain
    reason-less (so we don't over-fix and fabricate a reason where none is classified).
  - *Verify with:* `python -m pytest tests/ -k "fallback_reason_threading" -x -q` → the effect-gate
    test passes (all 4 THREAD sites carry a real reason; all 3 floor sites carry none).

---

## 7. Security, Privacy, Ops, Observability

- **No new surface.** No credentials, no network, no public posting. The change is a reason label on
  an already-emitted status line.
- **Observability improves:** the failover announce becomes self-describing again on the exhaustion
  path — operators (and Ace) can distinguish a `provider overloaded` blip from a `safety refusal` or
  `rate limit` at a glance, which is the whole point of PR #280.
- **Rollback:** trivial — revert the one/two-line diff. No migration, no persisted state.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `classified` not in scope at 4144 in some code path (NameError). | **PROVEN closed (RC2):** single `except ... as api_error:` at 2478; `classified = classify_api_error(...)` at 2750 (indent-16) assigns unconditionally at handler top, before the `if retry_count >= max_retries:` block (4123, indent-16) that contains the 4144 call (indent-20). Belt-and-suspenders: the fix uses `getattr(classified, "reason", None)` so even a hypothetical unbound-attr path degrades to the `connection issue` floor instead of crashing an error path. |
| Passing `reason=` clobbers a legitimately-stamped `_pending_stream_error_reason`. | `_resolve_failover_reason` enforces explicit-wins / backfill / consume-once. Covered by an explicit adversarial test (§6 Phase 1 negative case b). |
| Over-fix: threading reason at a site where the reason is genuinely wrong/misleading. | Only thread `classified.reason` where classification actually ran for THIS error. Empty/invalid-response eager sites (no classification) keep the honest `"connection issue"` floor — do not fabricate a reason. |
| False premise — maybe PR #280 never covered this site intentionally. | Ground-truthed: PR #280's own comment says "EVERY classifiable FailoverReason maps to a label so the announce always says WHY" and "a bare Model fallback: A → B left the user guessing." The retry-exhaustion site is an oversight, not an intentional carve-out — the reason-less floor is meant for *unclassifiable* faults, and `overloaded` is classifiable. |

---

## 9. Open Questions

None. Phase 2's site list is resolved (§5 audit: 4 THREAD, 3 floor). The 3935 "is the reason
misleading?" judgment call is resolved in §4 (no — the block already names the reason in prose).

---

## 10. Acceptance Criteria

1. A retry-exhausted `503 overloaded` failover announces `🔄 Model fallback (provider overloaded): A → B` — proven by an e2e loop test capturing the status callback. *(maps to Phase 1)*
2. `reason=None` + no pending stamp still renders `(connection issue)`, never a bare line. *(Phase 1 negative)*
3. Explicit reason wins over a stale pending stamp. *(Phase 1 negative)*
4. **Effect-gated** (not paren-grep — RC1): an invariant test asserts each of the 4 THREAD sites
   (1241/1991/3935/4144) passes a **non-None** `FailoverReason` as its `reason=` argument, and the
   3 by-design floor sites (1587/1660/5246) pass none. A `reason=None` "fix" fails this gate. *(Phase 2)*
5. Per-reason e2e: `(rate limit)` announces at the 1241 path, `(safety refusal)` at 1991. *(Phase 2)*
6. Full suite green; no regression in existing `_resolve_failover_reason` / announce tests. *(both phases)*
7. PR description includes the authoritative **10-site** audit table from §5 (RC3). *(Phase 2)*
