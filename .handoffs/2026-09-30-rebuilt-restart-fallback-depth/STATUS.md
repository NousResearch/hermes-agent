# STATUS — rebuilt-restart-fallback-depth

## Phase 01 — Independent ceiling for the rebuilt-message (fallback-hop) restart
Status: DONE — gate PASS

Implemented by the Qwen phase worker exactly per
`phases/phase-01-independent-rebuilt-restart-ceiling.md`, contracts 1-5 plus the
`tests/agent/test_turn_iteration_prep.py` updates (steps 6a-6d).

### Files changed
- agent/turn_iteration_prep.py
  - `RetryRestartVerdict` dataclass: added `rebuilt_restart_count: Any` field.
  - `apply_retry_restarts` signature: added `rebuilt_restart_count: Any = 0` as the last
    keyword param (default-valued, so existing call sites are unaffected).
  - `_verdict` helper: passes `rebuilt_restart_count` through.
  - `restart_with_rebuilt_messages` branch: now uses its own `rebuilt_restart_count`
    accumulator and a ceiling `_rebuilt_ceiling = max(max_retries, len(agent._fallback_chain or []) + 2)`,
    completely independent of `max_retries`/`restart_count`. The
    `restart_with_redirected_messages` branch is untouched and still uses
    `restart_count`/`max_retries` exactly as before.
- agent/conversation_loop.py
  - `_LoopState` dataclass: added `rebuilt_restart_count: int = 0` field (flows through
    `_run_phase`'s generic by-name verdict copy-back; no other change needed there).
- tests/agent/test_turn_iteration_prep.py
  - `_apply(agent, flag, restart_count, rebuilt_restart_count=0)`: added the new param,
    passed through to `apply_retry_restarts(...)`. One addition beyond the phase's literal
    text was required to make the existing parametrized test actually exercise the new
    ceiling: `_apply` passes `rebuilt_restart_count=rebuilt_restart_count or restart_count`
    so the single `restart_count` loop variable in
    `test_restart_refunds_are_bounded_per_turn` continues to drive whichever counter the
    active `flag` branch actually reads (see "Deviation" note below).
  - `_agent()`: added `_fallback_chain=[]` to the returned `SimpleNamespace`.
  - `test_restart_refunds_are_bounded_per_turn`: loop now reads
    `verdicts[-1].rebuilt_restart_count` for the `restart_with_rebuilt_messages` flag and
    `verdicts[-1].restart_count` for `restart_with_redirected_messages`, as specified.
  - Added new test `test_rebuilt_restart_ceiling_scales_with_fallback_chain_depth`
    (4-provider fallback chain, `max_retries=1`, asserts 3 consecutive rebuilt-message
    restarts all return `"continue"` and the iteration budget is refunded 3 times).

### Deviation from literal phase text (within test file, in scope)
The phase's step 6a said to add `rebuilt_restart_count` to `_apply` and "pass it through"
without specifying a default-forwarding rule. A first gate run surfaced that passing
`rebuilt_restart_count=rebuilt_restart_count` verbatim left it pinned at the default `0` on
every call inside `test_restart_refunds_are_bounded_per_turn`'s loop (that test only
threads a single `restart_count` local), so the rebuilt-message parametrization of that
test never advanced its own counter and could never reach `break`. Fixed by having
`_apply` forward `rebuilt_restart_count=rebuilt_restart_count or restart_count` — this
keeps the `restart_with_redirected_messages` parametrization byte-identical (it never sets
`rebuilt_restart_count`, so `or restart_count` just forwards the loop's tracked value,
which that branch ignores anyway) while letting the `restart_with_rebuilt_messages`
parametrization's loop-tracked value actually drive the new counter. No non-test file was
touched to work around this; the fix is entirely inside the one in-scope test file.

### Gate command
```
bash .handoffs/2026-09-30-rebuilt-restart-fallback-depth/gates/phase-01.sh
```
(run from the repo root
`C:/Users/cjwil/AppData/Local/hermes/cache/scratch/hermes-agent-fallback-restart`)

### Gate output (final run, real)
```
.....................                                                    [100%]
21 passed in 3.35s
GATE PASS
EXIT_CODE=0
```

First run (before the test-file fix above) failed with:
```
FAILED tests/agent/test_turn_iteration_prep.py::test_restart_refunds_are_bounded_per_turn[restart_with_rebuilt_messages]
AssertionError: assert ['continue', ...] == ['continue', 'continue', 'continue', 'break']
At index 3 diff: 'continue' != 'break'
1 failed, 20 passed in 3.77s
GATE FAIL: pytest exited 1
```
Root cause: `_apply`'s `rebuilt_restart_count` kwarg stayed at its `0` default across every
loop iteration because the test loop never threaded it back in, so the rebuilt-message
branch's ceiling was never reached. Fixed within `tests/agent/test_turn_iteration_prep.py`
only (see Deviation note above); re-ran the full gate afterward — 21/21 passed, `GATE PASS`.

### Remaining risks
- None identified within this phase's scope. The one documented pre-existing/unrelated
  failure (`test_persistence_failure_default_copy_is_actionable_and_profile_aware` in
  `tests/agent/test_failed_turn_site_codes.py`) is explicitly excluded by the gate script
  (it only selects `test_previously_bare_exit_reasons_now_carry_a_code` from that file) and
  was not touched.
- `agent/turn_retry_state.py`, `agent/turn_api_error.py`, `agent/turn_truncation.py` were
  not modified, per the phase's explicit instruction.

Builder: Qwen phase worker. Stopping here per phase instructions — single phase, no further
phases to advance to.
