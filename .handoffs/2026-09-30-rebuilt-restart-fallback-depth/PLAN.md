# Rebuilt-restart cap bounded by fallback-chain depth — Implementation Plan (local-model handoff)

## Architecture (decided — the builder changes none of this)
`agent/turn_iteration_prep.py::apply_retry_restarts` bounds two *different* restart
reasons — `restart_with_redirected_messages` (user interrupt/redirect) and
`restart_with_rebuilt_messages` (fallback-chain hop after a provider failure) — with the
**same** accumulator (`restart_count`) checked against the **same** ceiling
(`max_retries`, which is `agent._api_max_retries`, documented/tuned purely as "retries per
provider before fallback engages"). A 4-provider fallback chain needs up to 3 hops to
reach the last provider, but `api_max_retries` is commonly set to `1` (fast-failover
preference) or `3` by default — reusing it as the fallback-hop ceiling kills the chain
early on any deployment with more than `max_retries + 1` fallback providers, with zero
signal that the later providers were never tried. This was hit for real on 2026-09-30:
an Anthropic account-wide cap failure correctly hopped fable→sonnet (1 hop, within the
cap of 1), but the **second** hop (sonnet→gpt-5.6-sol, needed because Anthropic's cap is
account-wide so both Anthropic legs fail identically) tripped
`"Rebuilt-message restart limit (1) exceeded"` and ended the turn before any OpenAI leg
was ever called.

The fix: give `restart_with_rebuilt_messages` its **own** counter and its **own** ceiling,
sized off the actual configured fallback-chain length (`len(agent._fallback_chain)`) with
a small headroom constant, completely independent of `_api_max_retries`. The
redirect/interrupt restart path keeps using `max_retries` (`_api_max_retries`) exactly as
today — that semantic is correct for interrupts and is untouched by this change.

## Frozen contracts

### File map
- `agent/turn_iteration_prep.py` (edit) — the only implementation file. Add a new
  `rebuilt_restart_count` loop field threaded alongside the existing `restart_count`, and
  a new ceiling computed from the fallback chain length, used ONLY in the
  `restart_with_rebuilt_messages` branch (lines ~506-527 today). The
  `restart_with_redirected_messages` branch (lines ~448-471) is NOT touched — it keeps
  using `restart_count` / `max_retries` exactly as today.
- `agent/conversation_loop.py` (edit) — `_LoopState` dataclass needs one new field,
  `rebuilt_restart_count: int = 0`, next to the existing `restart_count: int = 0` (around
  line 1424). This is the ONLY change to this file — do not touch anything else in it.
- `tests/agent/test_turn_iteration_prep.py` (edit) — extend the existing parametrized
  test `test_restart_refunds_are_bounded_per_turn` so the two restart flags are checked
  against their own independent ceilings, plus one new test proving a 4-entry fallback
  chain survives 3 consecutive rebuilt-message restarts (previously impossible with
  `max_retries=1`).

Do NOT touch `agent/turn_retry_state.py`, `agent/conversation_loop.py`'s `_arm_fallback_restart`,
`agent/turn_api_error.py`, or any other file. Nothing else needs to change — the fallback
activation call sites (`turn_api_error.py:341-346`, `:373-378`, `turn_truncation.py:734-736`)
already arm `restart_with_rebuilt_messages` and already run through
`apply_retry_restarts`; they need no changes because the new ceiling is computed entirely
inside `apply_retry_restarts` from state already available on `agent`.

### Exact signatures (verbatim — current state, for reference; DO NOT change the public
signature of `apply_retry_restarts` beyond what phase 1 specifies)

Current `RetryRestartVerdict` dataclass (`agent/turn_iteration_prep.py` lines 401-416):
```python
@dataclass
class RetryRestartVerdict:
    """``action``: ``"fallthrough"`` (a response is ready — process it), ``"continue"``
    (a restart flag re-issues the iteration: redirect / compressed / rebuilt-for-fallback /
    length continuation) or ``"break"`` (turn ends: interrupted, non-actionable compaction
    handoff, or every retry exhausted without a response)."""

    action: str
    current_turn_user_idx: Any
    final_response: Any
    retry_count: Any
    restart_count: Any
    api_call_count: Any
    _preflight_compression_blocked: Any
    _turn_exit_reason: Any
```

Current `apply_retry_restarts` signature (`agent/turn_iteration_prep.py` lines 418-424):
```python
def apply_retry_restarts(
    agent: Any, *, _retry: Any, response: Any, interrupted: Any, messages: Any,
    conversation_history: Any, user_message: Any, api_kwargs: Any, current_turn_user_idx: Any,
    final_response: Any, retry_count: Any, max_retries: Any, api_call_count: Any,
    restart_count: Any, length_continue_retries: Any,
    _preflight_compression_blocked: Any, _turn_exit_reason: Any,
) -> RetryRestartVerdict:
```

Current `restart_with_rebuilt_messages` branch, exact text to replace
(`agent/turn_iteration_prep.py` lines 506-527):
```python
    if _retry.restart_with_rebuilt_messages:
        restart_count += 1
        if restart_count > max_retries:
            # A stall/failure keeps re-escalating to the fallback chain: stop refunding the
            # iteration budget and re-issuing, or a runaway turn holds the turn lease
            # indefinitely (rebuilt restarts previously had no bound).
            _turn_exit_reason = "rebuilt_restart_limit_exceeded"
            logger.warning(
                "Rebuilt-message restart limit (%s) exceeded; ending turn instead of "
                "refunding the iteration budget indefinitely.",
                max_retries,
            )
            return _verdict("break")
        # A stall/failure escalated to the fallback chain: re-issue against the
        # active fallback provider, refunding budget/count for the stalled attempt.
        api_call_count -= 1
        agent.iteration_budget.refund()
        _retry.restart_with_rebuilt_messages = False
        # Failover shrank the compressor window: clear the preflight block so
        # preflight re-runs before the first fallback call (single consumer).
        _preflight_compression_blocked = False
        return _verdict("continue")
```

Relevant existing attribute already set on `agent` by `agent_init.py::_init_fallback_chain`
(line 1097, unchanged by this plan): `agent._fallback_chain` — a `list` of fallback entry
dicts, in configured order. `len(agent._fallback_chain)` is the number of fallback hops
configured (NOT counting the primary provider). Read it with
`getattr(agent, "_fallback_chain", None) or []` (defensive — some test doubles are plain
`SimpleNamespace` objects without the attribute, see `tests/agent/test_turn_iteration_prep.py`'s
`_agent()` helper, which the new test must extend, not replace).

`_LoopState` dataclass, current state around the `restart_count` field
(`agent/conversation_loop.py` lines 1420-1426):
```python
    # Per-turn backstop for the refunding restarts (redirect / rebuilt-for-fallback).
    # Unlike ``retry_count`` (rebound to 0 each iteration) this accumulates for the whole
    # turn so a runaway interrupt/redirect that keeps re-arming a restart flag cannot
    # refund the iteration budget forever and hold the turn lease indefinitely.
    restart_count: int = 0
    _outer_error_count: int = 0  # outer-loop exceptions this turn (#92450), see _MAX_OUTER_LOOP_ERRORS
    truncated_tool_call_retries: int = 0
```

### Conventions
- Python, this repo's existing style (type hints as `Any` for loop-threaded locals, as
  already used throughout `turn_iteration_prep.py`).
- Tests: `pytest`, run via `.venv/Scripts/python.exe -m pytest <path> -q` (Windows venv
  already created at the repo root — `python -m venv .venv` then
  `.venv/Scripts/python.exe -m pip install -e . pytest` if a fresh worktree needs it).
- No new dependencies. No refactor beyond the two named files + the one test file.

## Rules & Tips
- `_run_phase` in `conversation_loop.py` (lines 1480-1495) copies every non-`action`/
  `result` dataclass field on a verdict back onto `_LoopState` by name via `setattr`. This
  means `RetryRestartVerdict` MUST gain a `rebuilt_restart_count` field (mirroring
  `restart_count`) for the new counter to survive between loop iterations — a plain
  function-local variable that isn't part of the verdict dataclass resets to its default
  every call and the bound would never actually accumulate. This is the single most
  important wiring detail in this phase; miss it and the gate's "it actually persists
  across iterations" test will show the bug clearly (counter always reads as 1).
- The headroom constant: use `len(agent._fallback_chain) + 2` as the new ceiling (not a
  bare `len(...)`) — same spirit as the existing `max_retries` off-by-one tolerance ("+1"
  because `restart_count > max_retries` trips only strictly after the Nth attempt), plus
  one extra hop of slack for a provider that gets skipped via
  `_should_skip_fallback_candidate` (unavailable key, cooldown) without consuming a "real"
  attempt. Floor it at `max_retries` itself so a chain with zero configured fallbacks
  (`_fallback_chain` empty or absent) does not regress below today's behavior — i.e.
  `ceiling = max(max_retries, len(agent._fallback_chain) + 2)`.
- Keep the log line's wording pattern consistent with the existing one two lines above it
  (`"Redirected-message restart limit (%s) exceeded..."`) — just swap in the new ceiling
  value so an operator reading logs sees which cap actually fired.
- Do not change `_turn_exit_reason` string `"rebuilt_restart_limit_exceeded"` — other code
  (`tests/agent/test_failed_turn_site_codes.py` line 133) asserts on that exact string.

## Phase map (ordered — each ends green before the next begins)
- [ ] 1. 01-independent-rebuilt-restart-ceiling — give `restart_with_rebuilt_messages` its
      own counter and a fallback-chain-sized ceiling — gate:
      `.venv/Scripts/python.exe -m pytest tests/agent/test_turn_iteration_prep.py tests/agent/test_truncated_tool_call_boost.py tests/agent/test_failed_turn_site_codes.py -q`

## Circuit breaker
If phase 1's gate fails twice in a row, halt and escalate back with the error + diff. Do
not attempt a third fix and do not touch any file outside the phase's scope.

## Real-environment verification (REQUIRED — not covered by any gate)
- The fallback-chain traversal itself (actually calling a real second/third/fourth
  provider in sequence on a live account-wide outage) — **ported from the 2026-09-30
  incident analysis, never re-exercised against a real multi-provider outage after this
  fix.** The unit tests below use a `SimpleNamespace` fake agent and never call a real
  model API. Flagged explicitly: this is a real, non-hypothetical risk the gate cannot
  cover — only a live multi-provider test (e.g. deliberately exhausting a real Anthropic
  account's cap again, or a provider-outage drill) proves the full chain actually reaches
  the 3rd/4th provider end-to-end.
- `agent._fallback_chain`'s shape/population — verified against real code
  (`agent_init.py::_init_fallback_chain`, read directly during planning), not guessed.

## Chain (REQUIRED — from `task-workflows`)
Task type: C Bug fix
Chain: bug-fix-minimal. Steps in order:

| # | Skill / step | Owner (builder / controller) | Evidence required |
|---|---|---|---|
| 1 | Root-cause investigation (code reading, this plan's analysis) | controller | this PLAN.md's Architecture section |
| 2 | qwen-handoff phase authoring | controller | this PLAN.md + phases/01 + gates/01 |
| 3 | Implementation | builder (qwen-worker / sonnet-worker) | diff + gate PASS output |
| 4 | verified-done | controller | completion statement in verified-done format |
| 5 | feature-demo-recording | controller | `SKIP(no deployed or browser-visible behavior — internal retry-loop bugfix, no UI/CLI-observable surface change)` |

The completion report for this handoff is the same table with the Evidence column filled
in and a PASS / SKIP(<reason>) column added.
