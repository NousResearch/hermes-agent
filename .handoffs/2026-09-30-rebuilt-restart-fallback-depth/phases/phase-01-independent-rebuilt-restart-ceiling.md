# Phase 01 — Independent ceiling for the rebuilt-message (fallback-hop) restart
Prereq: none — this is the only phase.

## Objective
`agent/turn_iteration_prep.py::apply_retry_restarts` currently bounds the
`restart_with_rebuilt_messages` restart (fired once per fallback-chain hop) with the same
`restart_count` accumulator AND the same `max_retries` ceiling used by the unrelated
`restart_with_redirected_messages` restart (user interrupt/redirect). `max_retries` is
`agent._api_max_retries`, tuned purely for same-provider retry-before-fallback — on a
deployment with `api_max_retries: 1` (a real, common setting — "fail over fast") and a
4-provider fallback chain, the chain can only take 1 hop before this guard kills the turn,
even though 3 hops would be needed to reach the last provider. Give the rebuilt-message
restart its OWN counter and its OWN ceiling, sized off the actual configured
fallback-chain length, completely independent of `max_retries`/`_api_max_retries`. Leave
the redirect/interrupt restart path (same function, different `if` branch) completely
unchanged — it keeps using `restart_count`/`max_retries` exactly as today.

## Files in scope — touch ONLY these
- agent/turn_iteration_prep.py   (edit)
- agent/conversation_loop.py     (edit — one field addition only)
- tests/agent/test_turn_iteration_prep.py   (edit — extend existing tests + add one new test)

Do NOT modify any other file. Do NOT refactor. Do NOT add dependencies. Do NOT touch
`agent/turn_retry_state.py`, `agent/turn_api_error.py`, or `agent/turn_truncation.py` —
none of them need changes for this fix.

## Contracts to honor (verbatim — do not change)

### 1. `RetryRestartVerdict` dataclass — add ONE new field
Current (`agent/turn_iteration_prep.py` lines 401-416):
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
Change to (add `rebuilt_restart_count: Any` right after `restart_count: Any`):
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
    rebuilt_restart_count: Any
    api_call_count: Any
    _preflight_compression_blocked: Any
    _turn_exit_reason: Any
```

### 2. `apply_retry_restarts` signature — add ONE new parameter (give it a default of
`0` so existing call sites that don't pass it, like `tests/agent/test_truncated_tool_call_boost.py`,
keep working unchanged)
Current (`agent/turn_iteration_prep.py` lines 418-424):
```python
def apply_retry_restarts(
    agent: Any, *, _retry: Any, response: Any, interrupted: Any, messages: Any,
    conversation_history: Any, user_message: Any, api_kwargs: Any, current_turn_user_idx: Any,
    final_response: Any, retry_count: Any, max_retries: Any, api_call_count: Any,
    restart_count: Any, length_continue_retries: Any,
    _preflight_compression_blocked: Any, _turn_exit_reason: Any,
) -> RetryRestartVerdict:
```
Change to (add `rebuilt_restart_count: Any = 0,` as the LAST parameter, after
`_turn_exit_reason: Any,` — it must have a default and come after all the other
keyword-only params so positional/keyword call sites that don't know about it yet don't
break):
```python
def apply_retry_restarts(
    agent: Any, *, _retry: Any, response: Any, interrupted: Any, messages: Any,
    conversation_history: Any, user_message: Any, api_kwargs: Any, current_turn_user_idx: Any,
    final_response: Any, retry_count: Any, max_retries: Any, api_call_count: Any,
    restart_count: Any, length_continue_retries: Any,
    _preflight_compression_blocked: Any, _turn_exit_reason: Any,
    rebuilt_restart_count: Any = 0,
) -> RetryRestartVerdict:
```

### 3. The internal `_verdict` helper — pass the new field through
Current (`agent/turn_iteration_prep.py` lines 439-446):
```python
    def _verdict(action: str) -> RetryRestartVerdict:
        return RetryRestartVerdict(
            action=action, current_turn_user_idx=current_turn_user_idx,
            final_response=final_response, retry_count=retry_count, restart_count=restart_count,
            api_call_count=api_call_count,
            _preflight_compression_blocked=_preflight_compression_blocked,
            _turn_exit_reason=_turn_exit_reason,
        )
```
Change to:
```python
    def _verdict(action: str) -> RetryRestartVerdict:
        return RetryRestartVerdict(
            action=action, current_turn_user_idx=current_turn_user_idx,
            final_response=final_response, retry_count=retry_count, restart_count=restart_count,
            rebuilt_restart_count=rebuilt_restart_count,
            api_call_count=api_call_count,
            _preflight_compression_blocked=_preflight_compression_blocked,
            _turn_exit_reason=_turn_exit_reason,
        )
```

### 4. The `restart_with_rebuilt_messages` branch — use the new counter + ceiling
Current (`agent/turn_iteration_prep.py` lines 506-527) — replace this exact block:
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
With this (new counter `rebuilt_restart_count`, new ceiling derived from
`agent._fallback_chain`, `max_retries` untouched elsewhere in the function):
```python
    if _retry.restart_with_rebuilt_messages:
        rebuilt_restart_count += 1
        # This restart fires once per fallback-chain hop, NOT once per same-provider retry
        # — reusing `max_retries` (tuned for "retries before fallback engages") as its
        # ceiling kills a deep fallback chain after its first hop when api_max_retries is
        # set low for fast failover (#<incident 2026-09-30>: a 4-provider chain died after
        # 1 hop, never reaching the 3rd/4th provider). Size the ceiling off the actual
        # configured chain length instead, with headroom for a skipped (unavailable/cooldown)
        # candidate that doesn't consume a "real" attempt; never regress below max_retries.
        _fallback_chain = getattr(agent, "_fallback_chain", None) or []
        _rebuilt_ceiling = max(max_retries, len(_fallback_chain) + 2)
        if rebuilt_restart_count > _rebuilt_ceiling:
            # A stall/failure keeps re-escalating to the fallback chain: stop refunding the
            # iteration budget and re-issuing, or a runaway turn holds the turn lease
            # indefinitely (rebuilt restarts previously had no bound).
            _turn_exit_reason = "rebuilt_restart_limit_exceeded"
            logger.warning(
                "Rebuilt-message restart limit (%s) exceeded; ending turn instead of "
                "refunding the iteration budget indefinitely.",
                _rebuilt_ceiling,
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

### 5. `_LoopState` dataclass (`agent/conversation_loop.py`) — add ONE new field
Current (lines 1420-1426):
```python
    # Per-turn backstop for the refunding restarts (redirect / rebuilt-for-fallback).
    # Unlike ``retry_count`` (rebound to 0 each iteration) this accumulates for the whole
    # turn so a runaway interrupt/redirect that keeps re-arming a restart flag cannot
    # refund the iteration budget forever and hold the turn lease indefinitely.
    restart_count: int = 0
    _outer_error_count: int = 0  # outer-loop exceptions this turn (#92450), see _MAX_OUTER_LOOP_ERRORS
    truncated_tool_call_retries: int = 0
```
Change to (add `rebuilt_restart_count: int = 0` right after `restart_count: int = 0`):
```python
    # Per-turn backstop for the refunding restarts (redirect / rebuilt-for-fallback).
    # Unlike ``retry_count`` (rebound to 0 each iteration) this accumulates for the whole
    # turn so a runaway interrupt/redirect that keeps re-arming a restart flag cannot
    # refund the iteration budget forever and hold the turn lease indefinitely.
    restart_count: int = 0
    # Separate accumulator for the rebuilt-for-fallback restart only: its ceiling is sized
    # off the fallback-chain depth, independent of `restart_count`'s `max_retries` ceiling
    # (see apply_retry_restarts). Redirect/interrupt restarts keep using `restart_count`.
    rebuilt_restart_count: int = 0
    _outer_error_count: int = 0  # outer-loop exceptions this turn (#92450), see _MAX_OUTER_LOOP_ERRORS
    truncated_tool_call_retries: int = 0
```

Nothing else in `conversation_loop.py` needs to change — `_run_phase` (lines 1480-1495)
already copies every verdict field back onto `_LoopState` by name generically; adding the
field to both the dataclass and the verdict is sufficient for it to flow through.

## Steps
1. Apply contract #1 (`RetryRestartVerdict` field) in `agent/turn_iteration_prep.py`.
2. Apply contract #2 (`apply_retry_restarts` signature) in the same file.
3. Apply contract #3 (`_verdict` helper) in the same file.
4. Apply contract #4 (the `restart_with_rebuilt_messages` branch body) in the same file.
5. Apply contract #5 (`_LoopState` field) in `agent/conversation_loop.py`.
6. Update `tests/agent/test_turn_iteration_prep.py`:
   a. In `_apply(agent, flag, restart_count)`, add a `rebuilt_restart_count` parameter
      (default `0`) and pass it through to `apply_retry_restarts(..., rebuilt_restart_count=rebuilt_restart_count, ...)`.
      Keep `restart_count` passed exactly as today for BOTH flags (the redirect branch
      still reads it; the rebuilt branch now reads `rebuilt_restart_count` instead, but the
      test helper must supply both so either branch has what it needs).
   b. In `_agent()`, add `_fallback_chain = []` to the returned `SimpleNamespace` (so
      `getattr(agent, "_fallback_chain", None) or []` resolves to an empty list — matching
      today's implicit ceiling of `max_retries` for a chain-less agent, keeping
      `test_restart_refunds_are_bounded_per_turn` passing unchanged for BOTH flags since
      `max(MAX_RETRIES, 0 + 2)` only differs from `MAX_RETRIES` when `MAX_RETRIES < 2` —
      this test uses `MAX_RETRIES = 3`, so behavior for the existing test is unchanged:
      `max(3, 0+2) == 3`).
   c. In the loop inside `test_restart_refunds_are_bounded_per_turn`, when
      `flag == "restart_with_rebuilt_messages"`, track `restart_count` from
      `verdict.rebuilt_restart_count` instead of `verdict.restart_count` (the redirect
      branch keeps reading `verdict.restart_count` as today). Concretely, change:
      ```python
      restart_count = verdicts[-1].restart_count
      ```
      to:
      ```python
      restart_count = (
          verdicts[-1].rebuilt_restart_count if flag == "restart_with_rebuilt_messages"
          else verdicts[-1].restart_count
      )
      ```
   d. Add a new test function (place it directly after
      `test_restart_refunds_are_bounded_per_turn`):
      ```python
      def test_rebuilt_restart_ceiling_scales_with_fallback_chain_depth():
          """A 4-provider fallback chain must survive 3 consecutive rebuilt-message
          restarts even when api_max_retries (max_retries here) is 1 — the ceiling for
          this restart is independent of max_retries and sized off the chain length
          (#2026-09-30 incident: chain died after 1 hop, never reaching providers 3-4)."""
          agent = _agent()
          agent._fallback_chain = [{"provider": "p1"}, {"provider": "p2"}, {"provider": "p3"}, {"provider": "p4"}]
          _retry = TurnRetryState()
          _retry.restart_with_rebuilt_messages = True
          rebuilt_restart_count = 0
          verdicts = []
          for _ in range(3):
              verdict = apply_retry_restarts(
                  agent, _retry=_retry, response=None, interrupted=False, messages=[],
                  conversation_history=[], user_message="hi", api_kwargs={}, current_turn_user_idx=0,
                  final_response=None, retry_count=0, max_retries=1, api_call_count=1,
                  restart_count=0, rebuilt_restart_count=rebuilt_restart_count, length_continue_retries=0,
                  _preflight_compression_blocked=True, _turn_exit_reason="unknown",
              )
              verdicts.append(verdict)
              rebuilt_restart_count = verdict.rebuilt_restart_count
              _retry.restart_with_rebuilt_messages = True  # simulate the next hop re-arming it
          assert [v.action for v in verdicts] == ["continue", "continue", "continue"]
          assert agent.iteration_budget.refunds == 3
      ```

## Gate — definition of done (objective)
Run: `.venv/Scripts/python.exe -m pytest tests/agent/test_turn_iteration_prep.py tests/agent/test_truncated_tool_call_boost.py tests/agent/test_failed_turn_site_codes.py::test_previously_bare_exit_reasons_now_carry_a_code -q`
(create the venv first if missing: `python -m venv .venv && .venv/Scripts/python.exe -m pip install -e . pytest` from the repo root)
(Note: `test_failed_turn_site_codes.py` has one unrelated pre-existing failure —
`test_persistence_failure_default_copy_is_actionable_and_profile_aware` — broken by an
environment/HERMES_HOME path assumption, nothing to do with this change. The gate
selects only the specific test in that file that actually covers this fix's exit-reason
string, not the whole file.)

Pass = command exits 0, all tests green, and the new test
`test_rebuilt_restart_ceiling_scales_with_fallback_chain_depth` is present and passing (not
skipped). If the gate fails: read the error, fix WITHIN the files in scope, re-run. Do not
proceed on a red gate. Do not edit out-of-scope files to force a pass.

## On success
Stop. Report: files changed, the gate command, and its output.
