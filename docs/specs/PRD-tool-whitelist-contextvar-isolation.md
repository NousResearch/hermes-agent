# PRD — Isolate the background-review tool whitelist from the shared gateway thread pool (ContextVar migration)

**Status:** v0.2 — pass-1 review BLOCK folded (see §Review Log); implementation staged as UNCOMMITTED
working-tree edits + committed-shaped regression tests; pass-2 pending. → prd-plan → commit → PR.

> **§5's `_thread_tool_whitelist = threading.local()` "Current (buggy)" quote is the PRE-FIX baseline.**
> The fix is applied as an UNCOMMITTED working-tree edit (`git status`: `M hermes_cli/plugins.py`,
> `M tests/hermes_cli/test_plugins.py`; `git log -S _tool_whitelist_var` → nothing committed). The
> RED baseline is reproduced by `git stash push hermes_cli/plugins.py` (proven: recycle + nesting tests
> RED on stash, GREEN on pop). This resolves pass-1 CB3.
**Phase 0 (DONE):** fork checkout was 3578 commits behind `origin/main`; `origin/main` (`4da49fe02`)
already contains the runtime HEAD `d2eb81a3a` (no un-merged runtime commits). Local `main`
fast-forwarded to `4da49fe02`; `background_review.py` now matches the runtime (1037 lines, mem0
clause). Exact current line refs on the synced tree: `hermes_cli/plugins.py:2077`
(`_thread_tool_whitelist = threading.local()`), `:2087` `set_`, `:2095` `clear_`, `:2135` read site;
`agent/background_review.py:850` set-call, `:894` `finally: clear_`. R1/OQ1 resolved.
**Author:** Apollo · 2026-07-12
**Target repo:** `hermes-agent` (Kyzcreig fork; upstream NousResearch). **Upstream-bound:** yes.

---

## 1. Summary & Goal

The background-review fork restricts itself to memory/skill tools by arming a **process-shared
`threading.local()`** tool whitelist (`hermes_cli/plugins.py::_thread_tool_whitelist`), set at
`agent/background_review.py` and cleared in a `finally`. The gateway runs **every** session's agent
turn on ONE shared `ThreadPoolExecutor` (`gateway/run.py::_get_executor`, `max_workers=10`,
`thread_name_prefix="hermes-gateway"`). Because the whitelist is keyed to the **physical OS thread**,
a whitelist armed on a pooled worker that isn't cleared on that exact worker **leaks to the next
logical task the pool recycles onto that worker** — a later foreground or cron turn then has its
tools denied with `Background review denied non-whitelisted tool: <name>. Only memory/skill tools are
allowed.`

**Goal:** make tool-whitelist scope follow the **logical task/turn**, not the physical thread, so a
recycled pool worker can never inherit a stale whitelist. Fix = migrate `threading.local()` → a
**`ContextVar`** (set-with-token, `reset(token)` in `finally`).

## 2. Non-Goals

- NOT changing WHAT the review fork is allowed to call (still memory/skill[/mem0 when flagged]).
- NOT touching the bridge / provider layer. This bug is provider-agnostic (proven); the `claude-bpx`
  correlation is only a timing-window widener, not a cause.
- NOT removing or disabling background review.
- NOT altering `propagate_context_to_thread` or `_run_in_executor_with_context` semantics (they
  already `copy_context()`; the fix rides inside the context they already copy).

## 3. Constitution / Invariants

- **Invariant I1 — the review fork's tool restriction MUST still be enforced.** The fork cannot call
  non-whitelisted tools.
  - *Why:* removing the restriction lets the self-improvement fork run arbitrary tools (the original
    security/correctness reason it exists).
  - *Closeout proof:* a test arming the whitelist within a task context and asserting a
    non-whitelisted tool is blocked IN that context.
- **Invariant I2 — NO cross-task leak.** A whitelist armed for one logical task MUST NOT affect any
  other task, even when both run on the same recycled physical pool worker.
  - *Why:* this is the bug.
  - *Closeout proof:* the T3 regression test (arm on a pooled worker inside one `ctx.run`, submit a
    2nd task to the same worker, assert NOT blocked). RED on `threading.local`, GREEN post-fix.
- **Invariant I3 — clean-context default.** A task with no whitelist armed sees ALL tools (no
  accidental default-deny).
  - *Closeout proof:* baseline test — fresh context, dispatch check returns not-blocked.
- **Invariant I4 — nesting safe.** If two whitelist scopes ever nest in one context, exit restores
  the outer scope exactly (token-reset, not set-to-None).
  - *Closeout proof:* nested set/reset test asserts the outer value is restored after the inner
    `reset(token)`.
- **Contract invariant I5 — public function signatures preserved.** `set_thread_tool_whitelist(allowed,
  deny_msg_fmt=...)` and `clear_thread_tool_whitelist()` keep their names+signatures (they are imported
  by name in `background_review.py` and referenced in tests). Internal storage changes; the API does
  not. (If a token is needed for `reset`, store it context-internally so `clear_*()` stays arg-free —
  see Design.)

## 4. Resolved Decisions

- **D1 — ContextVar, not a same-thread-clear guard.** A "clear on the same thread" band-aid would still
  be a physical-thread mechanism and would break the moment a task hops threads mid-turn. ContextVar
  makes the leak structurally impossible and matches two prior fleet fixes of this exact class
  (`HERMES_CRON_SESSION` cron latch; session-id env leak). Chosen.
- **D2 — keep `clear_thread_tool_whitelist()` arg-free.** Rather than require callers to pass a token,
  store the reset token inside the ContextVar's own value (or a paired private ContextVar) so the
  existing `finally: clear_thread_tool_whitelist()` call site is unchanged. Preserves I5.
- **D3 — keep the `deny_msg_fmt` behavior** (per-arming custom deny message) — store fmt in the same
  ContextVar payload.
- **D4 — names.** Public API keeps `set_thread_tool_whitelist` / `clear_thread_tool_whitelist` for
  compatibility even though "thread" is now a misnomer; add a docstring note that scope is the
  contextvars Context (task), not the OS thread. (Renaming is an optional follow-up, not this PR — it
  would churn imports/tests for no behavior gain.)

## 5. Architecture / Design

**Pre-fix baseline (RC2 — reproduce via `git stash push hermes_cli/plugins.py`; NOT the live tree, which already holds the ContextVar fix):**
```python
_thread_tool_whitelist = threading.local()          # keyed to OS thread
def set_thread_tool_whitelist(allowed, deny_msg_fmt=...):
    _thread_tool_whitelist.allowed = allowed
    _thread_tool_whitelist.fmt = deny_msg_fmt
def clear_thread_tool_whitelist():
    _thread_tool_whitelist.allowed = None
# read: allowed = getattr(_thread_tool_whitelist, "allowed", None)
```

**Proposed (isolated):**
```python
import contextvars
# One ContextVar holding the whole payload (allowed set + deny fmt), default None.
_tool_whitelist_var: contextvars.ContextVar[Optional[tuple[frozenset[str], str]]] = \
    contextvars.ContextVar("hermes_tool_whitelist", default=None)
# A per-context stack of reset tokens so clear() stays arg-free and nest-safe.
_tool_whitelist_tokens: contextvars.ContextVar[tuple] = \
    contextvars.ContextVar("hermes_tool_whitelist_tokens", default=())

def set_thread_tool_whitelist(allowed, deny_msg_fmt=DEFAULT):
    payload = (frozenset(allowed) if allowed is not None else None, deny_msg_fmt)
    token = _tool_whitelist_var.set(payload)
    _tool_whitelist_tokens.set(_tool_whitelist_tokens.get() + (token,))

def clear_thread_tool_whitelist():
    toks = _tool_whitelist_tokens.get()
    if toks:
        _tool_whitelist_var.reset(toks[-1])          # restore outer scope (I4)
        _tool_whitelist_tokens.set(toks[:-1])
    else:
        _tool_whitelist_var.set(None)                # defensive: no token → clear

# read site (plugins.py ~1881):
payload = _tool_whitelist_var.get()
if payload is not None:
    allowed, fmt = payload
    if allowed is not None and tool_name not in allowed:
        return _PreToolCallDirective(action="block", message=fmt.format(tool_name=tool_name))
```

**Why this kills the leak:** the gateway's `_run_in_executor_with_context` and
`propagate_context_to_thread` both run the turn body inside `ctx.run(...)` on a `copy_context()`.
Each logical turn therefore executes in its OWN copied Context. A ContextVar set inside one turn's
`ctx.run` is visible only within that copy; when the pool recycles the worker to the next turn, that
turn runs under a DIFFERENT copied context whose `_tool_whitelist_var` is the default `None`. Physical
thread reuse becomes irrelevant.

**Edge — the `bg-review` daemon thread.** The async review is spawned via
`threading.Thread(target=propagate_context_to_thread(target), name="bg-review")`. `propagate_context_to_thread`
wraps the target in `ctx.run` on a context copied FROM the parent at spawn time. The fork's
`set_thread_tool_whitelist` therefore mutates the fork's OWN copied context — never the parent's, never
a sibling's. Confirmed safe by probe T2 (threading.local already didn't leak cross-thread; ContextVar
preserves that and additionally fixes the same-thread recycle T3).

## 6. Implementation Phases

- **Phase 1 — migrate storage in `hermes_cli/plugins.py`.** Replace `_thread_tool_whitelist =
  threading.local()` with the two ContextVars; rewrite `set_/clear_thread_tool_whitelist` and the read
  site in `_get_pre_tool_call_directive_details`. Preserve the public signatures (I5) and `deny_msg_fmt`
  (D3).
  - *Unit/script check:* existing `tests/hermes_cli/test_plugins.py` whitelist tests still green (set →
    denies non-whitelisted, clear → allows).
  - *E2E/integration check (RC1):* the load-bearing gate is the committed pytest
    `tests/hermes_cli/test_plugins.py::test_no_leak_across_recycled_pool_worker` — RED under
    `threading.local`, GREEN under the ContextVar. Run via
    `scripts/run_tests.sh tests/hermes_cli/test_plugins.py -k whitelist`. (The old
    `scripts/whitelist_leak_probe.py` in `bridge-session-key-collision-tool-loss-local` is an OPTIONAL
    human diagnostic only — NOT in the repo, NOT a CI gate; do not hunt for it.)
  - *Negative/adversarial:* the recycle test IS the adversarial case (recycled-worker inheritance, with
    turn-A deliberately skipping `clear` to model a crash/thread-hop). Plus: arm with a custom
    `deny_msg_fmt`, assert the blocked message uses it (D3 preserved).
  - *Verify with:* `scripts/run_tests.sh tests/hermes_cli/test_plugins.py -k whitelist` → all pass;
    RED baseline via `git stash push hermes_cli/plugins.py` → recycle + nesting tests fail → `git stash pop` → green.

- **Phase 2 — T3 regression test as a first-class unit test.** Port the probe's T3 into
  `tests/hermes_cli/test_plugins.py` (or a new `test_tool_whitelist_task_isolation.py`): arm the
  whitelist inside one `copy_context().run(...)` on a single-worker pool, submit a second task to the
  same worker, assert NOT blocked. Add I3 (clean default), I4 (nesting restores outer), and the custom-
  fmt assertion.
  - *Unit/script check:* the new test file RED on current `main` (stash the Phase-1 edit → prove it
    reds), GREEN after.
  - *E2E/integration check:* `Not applicable: pure in-process concurrency test IS the real path.`
  - *Negative/adversarial:* nesting test (I4) + clean-default test (I3).
  - *Verify with:* `git stash` the plugins.py edit → `pytest <newtest> -q` REDs on the recycle case →
    `git stash pop` → GREEN.

## 7. Security, Privacy, Ops, Observability

- **Security:** strictly IMPROVES isolation (I1 preserved, I2 newly guaranteed). No new surface, no new
  env var, no secret handling. No config knob (behavioral default, not user-tunable).
- **Ops/rollout:** harness-source change → fork PR → CI green → **deploy = gateway restart** (Apollo's
  own restart is the gated step; stage it, get Ace's approval, then `deploy.sh` + restart). E2E-verify
  first on the **Aegis rig** (un-gated) by running the committed whitelist tests against the deployed
  venv (RC1 — not a vendored probe).
- **Observability:** none added. (Optional, out of scope: add `threadName` + context tag to the deny
  log — deferred; the ContextVar fix removes the need to debug which thread was armed.)
- **Rollback:** single-commit revert restores `threading.local`; no data/schema migration, no state to
  unwind.

## 8. Risks & Mitigations

- **R1 — fork checkout is STALE vs the runtime tree** (fork `background_review.py` ~621 lines / set at
  L490; runtime ~1037 lines / set at L850 with the mem0 clause). *Mitigation:* Phase 0 (build prereq) —
  sync fork `main` to the runtime tree's state (or confirm the runtime's extra commits are already on
  fork `main` and this local checkout just needs `git pull`) BEFORE editing, so the PR lands on current
  code and the deploy doesn't regress the runtime's newer background_review logic. Ground-truth the diff
  first.
- **R2 — a SECOND armer exists that I haven't found.** Verified this session: `set_thread_tool_whitelist`
  has exactly ONE non-test caller (`background_review.py`). *Mitigation:* re-grep at build time on the
  synced tree; if a new caller appeared, it inherits the fix for free (same API) but re-confirm I2 holds
  for it.
- **R3 — a caller relies on the OS-thread semantics** (e.g. sets on thread A, expects to read on thread
  A via a different code path). *Mitigation:* none found; the only read is the dispatch check which runs
  in-context. Grep-confirm no direct `_thread_tool_whitelist.allowed` access outside the three sites.
- **R4 — perf regression.** *Mitigation:* none expected — `ContextVar.get()` is O(1) hash-cached; no NEW
  `copy_context()` calls (the fix rides existing copies); copy is shallow (one more reference in an
  already-copied mapping). Confirmed by design; note in PR.

## 9. Open Questions

- **OQ1 — does the runtime tree carry fork-un-merged commits** (i.e. was the runtime deployed from a
  branch ahead of fork `main`)? Resolve in Phase 0 by diffing. Decides whether this is a clean PR on
  `main` or needs a rebase onto the runtime's actual HEAD.
- **OQ2 — rename `*_thread_tool_whitelist` → `*_task_tool_whitelist`?** Deferred (D4) — churns imports
  for zero behavior gain; optional upstream follow-up.

## 10. Acceptance Criteria

- **AC1 (I2):** the T3 recycle-leak probe/test returns NOT blocked post-fix, and REDs on pre-fix `main`.
- **AC2 (I1):** an armed whitelist still blocks a non-whitelisted tool within its own context.
- **AC3 (I3):** a clean context (no arming) blocks nothing.
- **AC4 (I4):** nested arm→arm→clear restores the outer whitelist exactly.
- **AC5 (I5/D3):** public signatures unchanged; custom `deny_msg_fmt` still used in the block message.
- **AC6:** full `tests/hermes_cli/test_plugins.py` green; `background_review` toolset-restriction tests
  green; CI green.
- **AC7 (upstream):** PR opens cleanly against NousResearch after fork verification (whole-class fix +
  E2E-shaped regression test, matching their contribution bar).

## 11. Review Log

### Pass 1 — verdict BLOCK (Opus via claude-apx-1, 2026-07-12)
The review read the tree AFTER the implementation had been pre-staged during the review run, so it
correctly saw "future work" already resident with no visible RED baseline. Every blocker was real and
is resolved by committed-shaped work:

- **CB1 (no committed T3 recycle test; fake-green gate)** → FOLDED. Added
  `tests/hermes_cli/test_plugins.py::test_no_leak_across_recycled_pool_worker`, which models the REAL
  gateway dispatch (`copy_context().run()` per turn on a single-worker `hermes-gateway` pool) and
  asserts turn-B on the recycled worker is NOT blocked. **Proven RED on `git stash` of the fix, GREEN
  on pop.** (Note: an earlier ad-hoc probe used a raw `submit` without per-task `ctx.run` — that does
  NOT model the gateway and gave a false T3/T4; the committed test uses `copy_context().run` and is
  correct.)
- **CB2 (probe not in repo → unfalsifiable in CI)** → FOLDED. The acceptance gate is now the committed
  pytest above (runs in CI + on any clean checkout + by upstream), not the skill-vendored probe. The
  probe remains only as a human-run diagnostic.
- **CB3 (stale-premise / committed-or-working-tree?)** → RESOLVED. `git status` = `M plugins.py`,
  `M tests/...`; `git log -S _tool_whitelist_var` = nothing committed. The fix is an UNCOMMITTED
  working-tree edit; §5's `threading.local()` is the accurate pre-fix baseline, reproduced by
  `git stash`. Status header updated to say so.
- **RC1 (I3/I4 untested)** → FOLDED. Added `test_clean_context_blocks_nothing` (I3) and
  `test_nested_whitelist_restores_outer_scope` (I4, RED on old blanket-null clear, GREEN on token-reset).
- **RC2 (ValueError fallback untested)** → FOLDED. Added
  `test_clear_with_foreign_token_fails_safe_to_unset` — clear() from a bare worker thread must not raise
  and must fail-safe to unset.
- **RC3 (rename misleading `test_whitelist_is_thread_local`)** → FOLDED. Renamed to
  `test_whitelist_does_not_leak_into_a_bare_worker_thread` with a docstring noting ContextVar storage.

Result: full whitelist class 8/8 GREEN; `test_plugins.py` + `test_background_review_toolset_restriction.py`
+ `test_code_execution_modes.py` = 165 passed. Pending: pass-2 verification of these folds.

### Pass 2 — verdict APPROVE WITH CHANGES (Opus via claude-api-proxy, 2026-07-12)
Zero blockers. Every pass-1 fold verified against source (not the PRD's self-report): CB1 RESOLVED
(the recycle test faithfully models `_run_in_executor_with_context`'s per-turn `ctx.run` and is a real
I2 gate), CB2 RESOLVED (committed pytest is the gate; 0 code refs to the probe), CB3 DOWNGRADED to
RR1, RC1/RC2/RC3 all RESOLVED. Two doc-only Required Changes + one on-the-record proof:

- **RC1 (purge dead probe reference)** → DONE. §6 Phase-1 E2E check + §7 Ops now name the committed
  pytest (`scripts/run_tests.sh ... -k whitelist`) as the gate and mark the probe an optional human
  diagnostic (not in repo, not CI).
- **RC2 (relabel §5 "Current (buggy)")** → DONE. §5 header now reads "Pre-fix baseline (reproduce via
  `git stash push hermes_cli/plugins.py`; NOT the live tree)".
- **RR1 (paste the git-state + RED→GREEN transcript on the record)** → DONE:

```
$ git log -p -S '_tool_whitelist_var' hermes_cli/plugins.py
(empty — not committed; the fix is an uncommitted working-tree edit)

$ git stash push hermes_cli/plugins.py        # revert to threading.local baseline
$ pytest tests/hermes_cli/test_plugins.py -k 'recycled or nested' -q
  FAILED ...::test_no_leak_across_recycled_pool_worker
  FAILED ...::test_nested_whitelist_restores_outer_scope
  2 failed, 118 deselected            # RED baseline reproduced

$ git stash pop                                # restore the ContextVar fix
$ pytest tests/hermes_cli/test_plugins.py -k 'whitelist or recycled or clean_context or nested or foreign_token or bare_worker' -q
  8 passed, 112 deselected            # GREEN
```
- **RR2 (single armer)** → re-confirmed: `set_thread_tool_whitelist` has exactly one non-test caller
  (`background_review.py:850`, matched `finally: clear` at `:894`).

**Result: APPROVE.** Pipeline complete (spec → pass-1 BLOCK folded → pass-2 APPROVE-WITH-CHANGES →
changes applied). Ready to commit + open the fork PR. Deploy (Apollo gateway restart) stays gated on
Ace's explicit go; Aegis-rig e2e precedes it.
