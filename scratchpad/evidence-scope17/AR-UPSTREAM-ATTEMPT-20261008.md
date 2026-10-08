# Independent attempt lifecycle AR — 2026-10-08

FINDINGS

Reviewed exact HEAD `0e85d306fc2b38db79408373308f039942c2fc4a`, parent `8a33891bdd58c3e0795ebfb277bcdde1003e91ea`. No source/test edits, network, credentials, model calls, push, merge or deployment.

## F1 — Explicit cancellation loses when the owner observes deadline expiry before the watchdog

`agent/auxiliary_client.py:1313-1316`, reached by newly added post-acquire (`1575`) and post-consume (`1586`) checks. If cancellation is already set and deadline expired, but the timer has not yet published `timed_out`, `_close_client_on_timeout()` correctly chooses cancellation and avoids destructive client cleanup. The caller nevertheless unconditionally raises `TimeoutError`. The adapter then sees `timed_out` and raises another timeout. The frozen explicit cancellation exception is lost; normal timeout retry/fallback is permitted. The new early `timed_out` branch handles only watchdog-first ordering.

Deterministic offline reproduction (timer deliberately suppressed to represent the owner-first ordering):

```python
import threading
from types import SimpleNamespace
from unittest.mock import patch
import agent.auxiliary_client as a
cancel = threading.Event()
response = SimpleNamespace(output=[], usage=None)
def acquire(**kw):
    cancel.set()
    return response
leaf = SimpleNamespace(base_url="https://chatgpt.com/backend-api/codex",
    responses=SimpleNamespace(create=acquire), close=lambda: None)
original = a._CodexStreamGuard.check_cancelled
checks = []
def check(g):
    checks.append(1)
    if len(checks) == 2:
        g._progress_deadline = 0
    return original(g)
with patch.object(a._CodexStreamGuard, "_arm_timer", lambda *x: None), \
     patch.object(a._CodexStreamGuard, "check_cancelled", check), \
     a.aux_interrupt_protection(cancel_event=cancel):
    a._CodexCompletionsAdapter(leaf, "m").create(messages=[], timeout=30)
```

Observed: `TimeoutError`; required: `AuxiliaryExplicitCancellation`. Add owner-first expired-deadline cancellation checks for both acquisition and consumption, including the `_AuxiliaryCancellationDecision` frozen source. Reevaluate the chosen cancellation outcome immediately after timeout cleanup before raising a timeout.

## Evidence

- Exact new test file: 8 passed at HEAD. Loaded parent source in an isolated in-memory module and ran the same file: 8 failed (all eight expected regressions). Worktree source was never replaced.
- Existing no-progress plus explicit-cancellation suites: 21 passed, including the 13 existing explicit-cancellation cases.
- Expanded FD-ownership/no-progress/explicit-cancellation run: 1 failed, 23 passed. FD suite alone repeats 1 failed, 2 passed: `test_stalled_stream_timeout_shuts_down_from_timer_and_closes_from_owner` observes only owner `client.close`, no socket shutdown. This requires resolution or demonstrated cold-import/test-timing explanation; it is not counted as a proved production bug. Added pre-consume deadline check can reject before mocked consumption starts.
- Codex sync/async wrappers bypass insertion into the shared cache; other provider caching is unchanged. Finalizer targets the real synchronous leaf, not the wrapper, avoiding a wrapper retention cycle. Async worker retains the adapter/leaf while running, but does not retain the finalizer-owning async wrapper; see F2 below.
- Sync/async timeout retry reacquires a Codex wrapper. Other exceptions retain existing retry/fallback behavior. No stale outer guard added; all new checks use the one adapter-local guard.
- Source line count: 8400, diff 24 added / 24 removed. Test 204 added lines.
- Source SHA-256: `e7d5b91491c07d02c4615d71295e1c63275f291022734904461c0726ece52601`.
- Test SHA-256: `b1dedb0f69849452bef4be41d9184ddf18d6ccede2eaf47592e0d0a6655bd800`.
- Full check and health results pending at initial report creation; appended below when complete.

## F2 — Async wrapper finalizer closes active worker transport from the wrong thread

`agent/auxiliary_client.py:6071-6073` registers cleanup on the outer async wrapper, while `_AsyncCompletionsAdapter.create()` hands only its sync adapter to `asyncio.to_thread()`. Cancelling the await does not stop that worker. Once the caller releases the outer wrapper, the new finalizer closes `_real_client` on the event-loop thread while the worker remains blocked in its SDK request. This violates the explicitly documented FD ownership rule and can release an FD still referenced by the worker TLS BIO. Cleanup must remain owned by an object retained throughout the worker request, or defer close until that worker exits.

Deterministic offline reproduction:

```python
import asyncio, threading, gc
from types import SimpleNamespace
from unittest.mock import patch
import agent.auxiliary_client as a
started = threading.Event()
release = threading.Event()
events = []
class Leaf:
    api_key = "test"
    base_url = "https://chatgpt.com/backend-api/codex"
    def __init__(self):
        self.responses = SimpleNamespace(create=self.create)
    def create(self, **kw):
        started.set()
        release.wait(3)
        return SimpleNamespace(output=[], usage=None)
    def close(self):
        events.append(("close", threading.get_ident(), release.is_set()))
async def main():
    with patch.object(a, "resolve_provider_client", lambda *x, **kw:
            (a.AsyncCodexAuxiliaryClient(a.CodexAuxiliaryClient(Leaf(), "m")), "m")):
        client, _ = a._get_cached_client("openai-codex", "m", async_mode=True)
        task = asyncio.create_task(client.chat.completions.create(messages=[], timeout=30))
        while not started.is_set():
            await asyncio.sleep(.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        del client
        gc.collect()
        print("before worker released:", events)
        release.set()
        await asyncio.sleep(.1)
asyncio.run(main())
```

Observed: `before worker released: [('close', <event-loop thread id>, False)]`. No network or SDK transport required. Add regression asserting cleanup does not run before the worker finishes and that timeout cleanup still belongs to its request thread.

## Final checks

Full `python scripts/check --commit HEAD`: 11 checks, ok. Health: 2 files vs parent, 0 blocking, 0 advisory (22.0s). These checks do not resolve F1/F2.
