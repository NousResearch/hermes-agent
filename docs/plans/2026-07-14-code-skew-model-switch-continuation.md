# Code-skew `/model` continuation — completed 2026-07-14

## Problem

The code-skew guard is deliberately process-global: after the checkout changes, a long-lived gateway may lazy-import modules from a mismatched checkout. A `/model` request in one Telegram topic could previously trigger a process restart while an unrelated topic still had an active agent, losing that selection and interrupting unrelated work.

## Implemented behavior

1. A stale `/model` never runs on the stale process. It is converted into a `PendingModelSwitch` and persisted atomically under the active Hermes home.
2. Persisted payload is intentionally narrow: allowlisted string routing fields (including Telegram `thread_id`), requested model/provider, scope, and an intent ID. It never stores a raw command, callback, event/runtime object, credentials, or nested adapter metadata.
3. If agents are active, the gateway defers restart. When the actual final active slot releases, it latches `_draining` synchronously before requesting the service-aware restart.
4. The inbound path checks the drain state both at initial admission and after its final pre-claim topic lookup await. A turn that was already suspended cannot claim a new slot after the safe restart decision.
5. Fresh startup and platform reconnect replay durable requests serially. Replay invokes the standard `/model` handler directly and acknowledges only after the normalized origin session contains the requested runtime model override.
6. Replay does not treat adapter task completion, delivery, queueing, or swallowed exceptions as success. Adapter absence, handler errors, a no-commit result, or a second code drift retain the original intent for retry.
7. A replay marker is runtime-only and excluded from persistence. If checkout drift recurs during replay, the stale guard retains the existing intent rather than enqueueing a duplicate.

## Delivery semantics

The continuation is deliberately **at-least-once**. A crash after switch commit but before durable acknowledgement can repeat the same model selection; repeating an identical selected model is safe. The implementation prefers this over silently dropping the user's request.

## Verification

Final local gates:

```text
29 passed  tests/test_code_skew.py
 3 passed  tests/test_stale_utils_module_import.py
 3 passed  tests/gateway/test_model_picker_persist.py
 7 passed  tests/gateway/test_telegram_model_picker.py
40 passed  tests/gateway/test_command_bypass_active_session.py
50 passed  tests/gateway/test_telegram_thread_fallback.py
ruff: passed
py_compile: passed
git diff --check: passed
```

The five focused gateway suites were run in separate Python processes. Their combined invocation has an unrelated pre-existing test-order contamination: `test_telegram_thread_fallback.py` passes independently both here and against clean `upstream/main`, but three `chat_type` assertions fail after preceding picker suites in the same process.

## Review record

Independent reviews identified and the final code addresses:

- acknowledge-after-dispatch data loss;
- zero-agent restart/admission TOCTOU race;
- acknowledgement of queued or swallowed-failure adapter work;
- duplicate durable intents after a second checkout drift during replay.

The final focused review passed with the global stale guard, routing isolation, scalar persistence, and no-duplicate replay behavior confirmed.
