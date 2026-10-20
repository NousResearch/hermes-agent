# Review dispositions: fix/watch-notification-admission-conflict (#141562)

Controller: implementer session (Opus 5.5). Each rung was a fresh `claude -p` pass over
`git diff upstream/main..HEAD`, given only the fixed reviewer prompt. Raw outputs are in `review-rung-<n>.json`.
Severity is the reviewer's own. REAL/FP is the controller's ruling.

Branch base: `upstream/main` @ `fe78f95bbb`.

---

## Rung 1: `--model claude-haiku-5-5 --effort low`

The run succeeded: `is_error=false`, 3 turns, $0.44, and `modelUsage` lists `claude-haiku-5-5`. Stderr also
logged `[claude-code:unrecognized_model] {"model":"claude-haiku-5-5"}` (open question Q1).
Verdict: APPROVE WITH NITS. 9 findings.

| # | file:line | Sev | Ruling | Disposition |
|---|---|---|---|---|
| R1-1 | tests/gateway/test_background_process_notifications.py:345 (cherry-picked #140911 test) | MINOR | **REAL**. It froze the process-wide `time.monotonic`, which the running event loop's timers also read. It passes today only because `_inject_watch_notification` is stubbed, so no loop timer runs. | FIXED `03ff1e3a2d` |
| R1-2 | tests/gateway/test_completion_delivery.py:993 | MINOR | **REAL**. It advanced the global clock 60 s per read, which also moves every loop timer in `asyncio.run`. | FIXED `03ff1e3a2d` |
| R1-3 | tests/gateway/test_watch_notification_reset_pin.py:141 | MINOR | **REAL**. A global forward skew under a real SessionAuthority makes loop timers (to_thread waits) fire early. | FIXED `03ff1e3a2d` |
| R1-4 | gateway/run_notifications_ownership.py:146 | MINOR | **REAL**. "a /resume reopens the pinned session, which delivers again" reads as though the dropped notice is redelivered. It is not. | FIXED `03ff1e3a2d` |
| R1-5 | gateway/run_notifications.py:70, ~719 | MINOR | **FP**. Neither AGENTS.md nor the module bans issue references in comments. The same file cites `#52694`, `#87051`, `#9290`, `#139026` and `#114271` throughout. | REJECTED: matches the surrounding convention |
| R1-6 | gateway/run_notifications.py ~755 | NIT | **FP**. `evt.get("session_id", "unknown")` is the module's existing idiom (the "Dropping watch notification with no routing metadata" log, `_drain` neighbours). The code is cherry-picked #140911 code, kept verbatim to preserve authorship. | REJECTED |
| R1-7 | gateway/run_notifications.py:742-749 | NIT | **FP**. Sharing `_MAX_COMPLETION_INJECTION_ATTEMPTS` / `_COMPLETION_INJECTION_RETRY_MAX_DELAY` with the process watcher is deliberate: the handoff asks that the two drains share one policy, and the constant comment says so. | REJECTED: by design |
| R1-8 | gateway/run_notifications_ownership.py:153 | NIT | **REAL but accepted**. A `/resume` landing between the refusal and the classification drops one notice the resumed session could have admitted. The window is one SQLite read wide, and it is the same post-hoc window the completion pre-flight already accepts. A test would need to inject a race between two awaits on one task. | DEFERRED: noted for the controller (Q3) |
| R1-9 | gateway/run_notifications.py `_drain_watch_notifications` | NIT (unverified) | **FP**. The reviewer could not run the ratchet. The controller ran `python scripts/check --base upstream/main --only ruff,health,…`: `code health: 6 files vs fe78f95bbb29: 0 blocking, 0 advisory`. | REJECTED: verified clean |

### Receipts, rung 1 (`03ff1e3a2d`)

R1-1, tests/gateway/test_background_process_notifications.py (both #140911 tests).
Before:
```python
    clock = {"now": 1000.0}
    monkeypatch.setattr(run_notifications.time, "monotonic", lambda: clock["now"])
    ...
        attempt_times.append(clock["now"])
    ...
        clock["now"] += 2.0
```
After:
```python
    clock = install_module_clock(monkeypatch, run_notifications)
    ...
        attempt_times.append(clock.now)
    ...
        clock.now += 2.0
```

R1-2, tests/gateway/test_completion_delivery.py:~992.
Before:
```python
    real, minutes = time.monotonic, itertools.count()
    monkeypatch.setattr(run_notifications.time, "monotonic", lambda: real() + 60 * next(minutes))
```
After:
```python
    install_module_clock(monkeypatch, run_notifications, step=60)
```

R1-3, tests/gateway/test_watch_notification_reset_pin.py:~138.
Before:
```python
        # Only ever forward: the event loop keeps its own timers on the same clock.
        skew = {'s': 0.0}
        real = time.monotonic
        monkeypatch.setattr(run_notifications.time, 'monotonic', lambda: real() + skew['s'])
    ...
                skew['s'] += 60
```
After:
```python
        clock = install_module_clock(monkeypatch, run_notifications)
    ...
                clock.now += 60
```
The new helper is tests/module_clock.py. It swaps only the module's binding:
```python
def install_module_clock(monkeypatch, module, *, now: float = 1000.0, step: float = 0.0) -> ModuleClock:
    clock = ModuleClock(now, step)
    monkeypatch.setattr(module, "time", clock)
    return clock
```

R1-4, gateway/run_notifications_ownership.py:146.
Before:
```python
        would spin on it (#141562). This is the completion pre-flight's verdict, applied after the
        refusal: a /resume reopens the pinned session, which delivers again.
```
After:
```python
        would spin on it (#141562). This is the completion pre-flight's verdict, applied after the
        refusal. A dropped notice is gone for good; a /resume reopens the pinned session, so the
        process's later notices are admitted again.
```

Tests after the fix pass (`scripts/run_tests.sh tests/gateway/test_watch_notification_reset_pin.py
tests/gateway/test_completion_delivery.py tests/gateway/test_background_process_notifications.py`):
`=== Summary: 3 files, 82 tests passed, 0 failed (100% complete) in 43.7s (4 workers) ===`.

Fix passes used at rung 1: 1 of 2.

---

## Rung 2: `--model claude-haiku-5-5 --effort medium`

The run succeeded: `is_error=false`, 8 turns, $0.68, and `modelUsage` lists `claude-haiku-5-5`. Verdict: NEEDS CHANGES (on finding 1). 5 findings.

| # | file:line | Sev | Ruling | Disposition |
|---|---|---|---|---|
| R2-1 | gateway/run_notifications.py:757-764 | MAJOR | **FP as a defect; real policy question.** After 8 refusals the drain drops a watch/heartbeat event with no plain-notice fallback, while the process watcher falls back for completions. This is the deliberate cap from the cherry-picked #140899/#140911: the issue asks for "backoff/cap/TTL", where main requeued forever. The watcher's fallback exists because a completion is the process's final result and has a human formatter (`_format_process_final_message`). A watch match or heartbeat is an interim, agent-facing signal; its output stays in `process(action='log')`, and the completion (with its own fallback) still fires on exit. Sending a plain notice through the same refusing adapter would hit the same outage. | REJECTED as a defect, NOT silently: raised as open question Q2 for the controller/maintainer, since the fallback policy is #140911's call |
| R2-2 | gateway/run_notifications.py:937 / run_notifications_ownership.py:141-163 | MINOR | **REAL but accepted.** Duplicate of R1-8 (the /resume race between refusal and classification). | DEFERRED (Q3), same as R1-8 |
| R2-3 | gateway/run_notifications.py:937 | MINOR | **REAL.** No test covered the `not raise_not_accepted` guard. Mutation check: removing the guard left every test green before this fix. | FIXED `43f0d07ce4` |
| R2-4 | gateway/run_notifications.py:757-762 | MINOR | **FP.** The cap warning already names the type: `"Watch notification for %s chat=%s (type=%s session_id=%s) was refused %d times; dropping the queued event"`. The `"unknown"` fallback repeats R1-6. | REJECTED |
| R2-5 | tests/gateway/test_watch_notification_reset_pin.py (`_foreign_pin`; compression assertion) | NIT | **REAL.** The parameter was never varied, and a truthiness-only assertion would pass for any row. | FIXED `43f0d07ce4` |

### Receipts, rung 2 (`43f0d07ce4`)

R2-3, a new test in tests/gateway/test_watch_notification_reset_pin.py:
```python
@pytest.mark.asyncio
async def test_durable_caller_still_gets_the_refusal_for_a_gone_pin(monkeypatch, caplog):
    ...
        with caplog.at_level(logging.WARNING, logger='gateway.run'):
            with pytest.raises(WakeNotAccepted, match='admission_conflict'):
                await runner._inject_watch_notification(
                    'notice', _notice(entry, retired), raise_not_accepted=True)

        assert not [r for r in caplog.records if 'permanently gone' in r.getMessage()]
```
Mutation proof: `gateway/run_notifications.py` was edited to
`if await self._watch_pin_gone(evt, str(exc)):` (guard removed). Result:
`=== Summary: 1 files, 5 tests passed, 1 failed ===` with
`FAILED ...::test_durable_caller_still_gets_the_refusal_for_a_gone_pin`. Then the file was restored with `git checkout`.

R2-5. Before:
```python
        assert _queued_admissions(authority, parent) + _queued_admissions(authority, child)
...
async def _foreign_pin(runner, adapter_source_chat='43'):
...
    other = adapter.build_source(chat_id=adapter_source_chat, chat_type='dm', user_id='42')
```
After:
```python
        admitted = _queued_admissions(authority, parent) + _queued_admissions(authority, child)
        assert [row['status'] for row in admitted] == ['queued']
...
async def _foreign_pin(runner):
...
    other = adapter.build_source(chat_id='43', chat_type='dm', user_id='42')
```
Tests: `scripts/run_tests.sh tests/gateway/test_watch_notification_reset_pin.py`, giving
`=== Summary: 1 files, 6 tests passed, 0 failed ===`.

Fix passes used at rung 2: 1 of 2.

---

## Rung 3: `--model claude-haiku-5-5 --effort high`

The run succeeded: `is_error=false`, 14 turns, $1.31, and `modelUsage` lists `claude-haiku-5-5`. Verdict: NEEDS CHANGES (on finding 1). 7 findings.

| # | file:line | Sev | Ruling | Disposition |
|---|---|---|---|---|
| R3-1 | gateway/run_notifications.py:744-756 | MAJOR | **REAL in part.** This repeats R2-1. The reviewer offered two remedies: "add that fallback or make the cap an explicit, documented loss". The documented-loss half is real and cheap: the cap warning named no place to recover the output. The fallback half stays FP for the reasons given at R2-1. | PARTIALLY FIXED `975fca9559` (the loss is now explicit in the log). Fallback policy remains open question Q2 |
| R3-2 | tests/gateway/test_background_process_notifications.py:366 | MINOR | **REAL.** `all(gap >= base)` passes for a flat 4 s cadence, so it never proved the backoff grows. | FIXED `975fca9559` |
| R3-3 | tests/gateway/test_completion_delivery.py:992 | MINOR | **REAL.** A per-read 60 s step also advanced `_async_delegation_watcher`'s orphan-sweep throttle (`last_orphan_sweep`), an undocumented coupling. | FIXED `975fca9559` (the clock now moves per watcher tick, and the unused `step` knob was removed from `tests/module_clock.py`) |
| R3-4 | gateway/run_notifications_ownership.py:150-151 | MINOR | **REAL.** The `if not pinned: return False` branch had no test. | FIXED `975fca9559` |
| R3-5 | gateway/run_notifications.py:754 vs run_notifications_ownership.py:161 | NIT | **FP.** This repeats R1-6/R2-4. `"<unknown>"` in `_watch_pin_gone` matches the completion pre-flight's log in this module (`evt.get("session_id") or "<unknown>"`), and the other line is #140911's verbatim code. | REJECTED |
| R3-6 | gateway/run_notifications_ownership.py:144-151 | NIT | **FP.** Docstrings in this mixin explain the cross-path reasoning and cite issues: `_resolve_async_delegation_session` cites #92611 and #55578. The docstring carries WHY (why a post-refusal verdict, the /resume caveat), not WHAT. | REJECTED |
| R3-7 | tests/gateway/test_completion_delivery.py:18-21 | NIT | **REAL.** `tests.` sorted after `tools.`. | FIXED `975fca9559` |

### Receipts, rung 3 (`975fca9559`)

R3-1, gateway/run_notifications.py (cap warning).
Before:
```python
                        "Watch notification for %s chat=%s (type=%s session_id=%s) was refused "
                        "%d times; dropping the queued event",
```
After:
```python
                        "Watch notification for %s chat=%s (type=%s session_id=%s) was refused "
                        "%d times; dropping the queued event (output remains available via "
                        "process(action='log'))",
```

R3-2, tests/gateway/test_background_process_notifications.py.
Before:
```python
    # Backoff, not one attempt per drain tick: successive retries are >= the base delay apart.
    gaps = [b - a for a, b in itertools.pairwise(attempt_times)]
    assert gaps
    assert all(gap >= _WATCH_INJECTION_RETRY_BASE_DELAY for gap in gaps)
```
After:
```python
    # Backoff, not one attempt per drain tick: each retry waits twice the last, up to the cap.
    gaps = [b - a for a, b in itertools.pairwise(attempt_times)]
    assert gaps == [min(_WATCH_INJECTION_RETRY_BASE_DELAY * 2 ** n, _COMPLETION_INJECTION_RETRY_MAX_DELAY)
                    for n in range(_MAX_COMPLETION_INJECTION_ATTEMPTS - 1)]
```

R3-3, tests/gateway/test_completion_delivery.py.
Before:
```python
    # The refused event backs off (#140899); each read is a minute later, so the retry is due.
    install_module_clock(monkeypatch, run_notifications, step=60)
```
After:
```python
    # The refused event backs off (#140899): each watcher tick is a minute later, so the retry is due.
    clock = install_module_clock(monkeypatch, run_notifications)
    bounded_sleep = asyncio.sleep

    async def minute_later(delay):
        clock.now += 60
        await bounded_sleep(delay)
    monkeypatch.setattr(asyncio, "sleep", minute_later)
```
In tests/module_clock.py, `ModuleClock.__init__(self, now: float, step: float)`, which had a per-read `self.now += self.step`, became `ModuleClock.__init__(self, now: float)` with `monotonic()` returning `self.now`.

R3-4, a new test in tests/gateway/test_watch_notification_reset_pin.py:
```python
async def test_unpinned_refusal_requeues(monkeypatch, caplog):
    """A notice that names no pinned session proves nothing gone: its refusal is retried."""
    ...
        notice = _notice(entry, '')
    ...
        assert pending.get_nowait() is notice
        assert notice['_watch_injection_attempts'] == 1
        assert not [r for r in caplog.records if 'permanently gone' in r.getMessage()]
```

R3-7. Before:
```python
from tools.process_registry import ProcessRegistry, ProcessSession
from tests.module_clock import install_module_clock
```
After:
```python
from tests.module_clock import install_module_clock
from tools.process_registry import ProcessRegistry, ProcessSession
```

Tests: `scripts/run_tests.sh tests/gateway/test_watch_notification_reset_pin.py tests/gateway/test_completion_delivery.py
tests/gateway/test_background_process_notifications.py` gave `=== Summary: 3 files, 84 tests passed, 0 failed ===`.
`scripts/check --only ruff,health,os-marker-fakes,footguns` gave `0 blocking, 0 advisory` and `4 checks, ok`.

Fix passes used at rung 3: 1 of 2.

---

## Rung 4: `--model claude-haiku-5-5 --effort xhigh`

The run succeeded: `is_error=false`, 33 turns, $3.15, and `modelUsage` lists `claude-haiku-5-5`. Verdict: NEEDS CHANGES. 9 findings.

| # | file:line | Sev | Ruling | Disposition |
|---|---|---|---|---|
| R4-1 | gateway/run_notifications.py:744-758, :886 | MAJOR | **REAL policy risk, not a correctness defect.** This is the third rung on R2-1/R3-1. New detail: "no adapter" (`return False`) also spends the 8-attempt budget, so an adapter outage longer than about 4 min loses watch notices that main would have delivered on reconnect (at the cost of a 2 s spin). This is the cap's documented trade-off in `281005ee93` (#140911), not this fix's code. Changing it would fork the policy away from the open upstream PR, which the handoff told us to align with. The loss became explicit in the log at R3-1. | **DEFERRED, explicitly and not silently**, as open question Q2 for the controller and #140911's maintainer. Not frozen: there was no failed fix pass, the call is a policy decision |
| R4-2 | gateway/session_automation.py:130 (+ process_registry_notifications.py:389) | MAJOR | **REAL, severity downgraded to MINOR.** A heartbeat tick held by the backoff can come due after a newer tick of the same process was admitted. `prepare_heartbeat` then folds `join_heartbeat_outputs(earlier=<newer stored>, event=<stale>)`, so the output order is reversed and the stale `seq`/`elapsed` win. Preconditions: the stale tick's refusals must outlast the ≥60 s heartbeat interval (4+8+16+32 s of holds) and the outage must clear between its last attempt and the newer tick's arrival. Impact: one folded turn shows two chunks in the wrong order. The same reorder already existed at 2 s scale on main (a requeued tick lands behind ticks enqueued during the pass). A correct fix needs the tick's `seq` in the stored `heartbeat_tick` metadata (`_tick_metadata`), which changes the heartbeat ledger format and is outside this fix's scope. | DEFERRED as open question Q4. Not frozen (no failed fix pass) |
| R4-3 | gateway/run_notifications.py:717-721 | MINOR | **FP.** Each watch_match is its own turn, with no ordering contract between notices. Held-event overtaking is inherent to any backoff and is #140911's design. | REJECTED |
| R4-4 | gateway/run_notifications_ownership.py:158-161 | MINOR | **REAL.** `_lookup_completion_owner`'s `terminal` also covers a missing row (`:114`) and an owner-less subagent row (`:133`), so the warning misnamed the cause. | FIXED `ad089ba5f4` |
| R4-5 | tests/gateway/test_watch_notification_reset_pin.py:98-117 | MINOR | **REAL.** The refused-compression-ancestor branch had no test. | FIXED `ad089ba5f4`. Mutation proof below |
| R4-6 | tests/gateway/test_background_process_notifications.py:365 | MINOR | **REAL.** The R3-2 fix re-derived the formula, which is change-detector-shaped. Invariants also prove growth (answering R3-2) without restating the implementation. | FIXED `ad089ba5f4` |
| R4-7 | gateway/run_notifications.py:70-72 | NIT | **FP.** "Keyed to the 2 s drain cadence" states the rationale (base = 2× the idle drain's interval, as the watcher's base is 2× its check interval). This is #140911's verbatim comment. | REJECTED |
| R4-8 | gateway/run_notifications.py:752-758 | NIT | **FP.** Every `False` the drain counts is a failed delivery attempt, which is what "refused" means here. The `"unknown"` fallback repeats R1-6. | REJECTED |
| R4-9 | tests/gateway/test_background_process_notifications.py:20 | NIT | **REAL.** The helper import sat in its own block. | FIXED `ad089ba5f4` |

### Receipts, rung 4 (`ad089ba5f4`)

R4-4, gateway/run_notifications_ownership.py:158.
Before:
```python
            "Dropping %s notification of process %s: pinned session %s is permanently gone (user "
            "boundary such as /new); the route now runs %s and refuses it (%s). Output remains "
            "available via process(action='log').",
```
After:
```python
            "Dropping %s notification of process %s: pinned session %s is permanently gone (closed "
            "at a user boundary such as /new, or no longer an owner); the route now runs %s and "
            "refuses it (%s). Output remains available via process(action='log').",
```

R4-5, a new test in tests/gateway/test_watch_notification_reset_pin.py:
```python
async def test_refused_notice_pinned_to_a_compression_ancestor_is_retried(monkeypatch, caplog):
    """A compression ancestor is not gone: whatever refused its notice, the notice stays queued."""
    ...
        async def refuse(*_args):
            raise RuntimeStoreError('admission_conflict')
        monkeypatch.setattr(session_automation, 'admit_process_notice', refuse)
    ...
        assert pending.get_nowait() is notice
        assert not [r for r in caplog.records if 'permanently gone' in r.getMessage()]
```
Mutation proof: the `if verdict != "terminal": return False` guard was removed from `_watch_pin_gone`. Result:
`5 tests passed, 3 failed`, including `FAILED ...::test_refused_notice_pinned_to_a_compression_ancestor_is_retried`.
The file was then restored. `git diff` afterwards showed only the R4-4 wording.

R4-6, tests/gateway/test_background_process_notifications.py.
Before:
```python
    # Backoff, not one attempt per drain tick: each retry waits twice the last, up to the cap.
    gaps = [b - a for a, b in itertools.pairwise(attempt_times)]
    assert gaps == [min(_WATCH_INJECTION_RETRY_BASE_DELAY * 2 ** n, _COMPLETION_INJECTION_RETRY_MAX_DELAY)
                    for n in range(_MAX_COMPLETION_INJECTION_ATTEMPTS - 1)]
```
After:
```python
    # Backoff, not one attempt per drain tick: the wait starts at the base, grows, and stops at the cap.
    gaps = [b - a for a, b in itertools.pairwise(attempt_times)]
    assert gaps[0] >= _WATCH_INJECTION_RETRY_BASE_DELAY
    assert all(b >= a for a, b in itertools.pairwise(gaps)) and gaps[-1] > gaps[0]
    assert max(gaps) <= _COMPLETION_INJECTION_RETRY_MAX_DELAY
```

R4-9. Before:
```python
import pytest

from tests.module_clock import install_module_clock

from gateway.config import GatewayConfig, Platform
...
from gateway.run_notifications import INTERNAL_NOTIFICATION_FOOTER
```
After:
```python
import pytest

from gateway.config import GatewayConfig, Platform
...
from gateway.run_notifications import INTERNAL_NOTIFICATION_FOOTER
from tests.module_clock import install_module_clock
```

Tests: `scripts/run_tests.sh tests/gateway/test_watch_notification_reset_pin.py tests/gateway/test_background_process_notifications.py`
gave `=== Summary: 2 files, 43 tests passed, 0 failed ===`. `scripts/check --only ruff,health,os-marker-fakes,footguns`
gave `4 checks, ok`.

Fix passes used at rung 4: 1 of 2.

---

## Rung 5: `--model sonnet --effort high`

The run succeeded: `is_error=false`, 25 turns, $0.72, and `modelUsage` lists `claude-sonnet-5`. Verdict: APPROVE. **0 findings.**
That count is recorded as zero findings, not as evidence of quality. No fix pass was needed.

---

## Summary

| Rung | Model / effort | Found | REAL | FP | Fixed | Partial | Deferred | Fix passes |
|---|---|---|---|---|---|---|---|---|
| 1 | claude-haiku-5-5 / low | 9 | 5 | 4 | 4 | 0 | 1 (R1-8) | 1 |
| 2 | claude-haiku-5-5 / medium | 5 | 3 | 2 (incl. R2-1 MAJOR) | 2 | 0 | 1 (R2-2 = R1-8) | 1 |
| 3 | claude-haiku-5-5 / high | 7 | 5 | 2 | 4 | 1 (R3-1 MAJOR) | 0 | 1 |
| 4 | claude-haiku-5-5 / xhigh | 9 | 6 | 3 | 4 | 0 | 2 (R4-1 MAJOR, R4-2 MAJOR→MINOR) | 1 |
| 5 | sonnet / high | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

R2-1 is counted as FP-as-defect. Its policy half carries forward through R3-1 (PARTIALLY FIXED) and R4-1 (DEFERRED).
- Frozen rungs: none. No REAL BLOCKER/MAJOR survived a failed fix pass. The two MAJOR-flagged items still open,
  R4-1 (the cap/fallback policy, raised by rungs 2, 3 and 4) and R4-2 (heartbeat fold order), are deliberate,
  reasoned deferrals to the controller.
- Orphaned findings: none. All 30 findings have a disposition.

## Open questions for the controller

- **Q1.** `claude-haiku-5-5` triggers `[claude-code:unrecognized_model]` on stderr, yet every Haiku rung ran with
  `modelUsage` keyed `claude-haiku-5-5`. Confirm the id resolves to the intended model and not a fallback.
- **Q2 (MAJOR, raised by rungs 2–4).** #140911's cap drops a watch/heartbeat notice after 8 failed attempts (about 4 min),
  including "no adapter" outages that main would deliver on reconnect. The completion watcher falls back to a
  plain notice; the drain does not. Options: accept as documented (the warning now names `process(action='log')`),
  lengthen the budget for watch events, or add a plain fallback for `watch_match`. This is a policy call for the
  controller or #140911's maintainer. Changing it here would fork the policy away from the open upstream PR.
- **Q3 (MINOR).** A `/resume` landing between a refusal and `_watch_pin_gone`'s classification drops one notice
  the resumed session could have admitted. The window is one SQLite read, the same post-hoc window the completion pre-flight accepts.
- **Q4 (MINOR, reviewer said MAJOR).** A heartbeat tick held by backoff can fold behind a newer admitted tick in
  reverse output order. A real fix needs `seq` in the stored `heartbeat_tick` metadata (a ledger format change).
