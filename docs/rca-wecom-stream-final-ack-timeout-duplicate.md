# RCA — WeCom stream final-frame ack-timeout duplicate send

Status: FIXED 2026-08-30 (card t_9851e2d0)
Product: gateway/run.py + gateway/stream_consumer.py + plugins/platforms/wecom/adapter.py

## Symptom

32 duplicated user messages in 7 days (measured 2026-08-30), all same-session,
across both desktop and Slack WeCom. Includes a doubled "credits added. go." which
risks double-dispatching an approval. The gateway logs the signature:

    Normal final-send NOT suppressed despite active stream consumer for session ...
    possible duplicate send (see wecom ack-timeout RCA)

7 such occurrences on 08-30 alone (gateway/run.py ~30986).

## Root cause — a timeout-inversion race

On WeCom native streaming, the consumer's turn-final delivery happens through a
`finish=true` stream frame whose server ack is awaited. Two independent timers
vie against each other:

1. **Consumer finalize ack window.** `plugins/platforms/wecom/adapter.py`
   `_REPLY_ACK_TIMEOUT=15.0` bounds how long a final frame waits for its ack, and
   the final frame first waits up to another 15s draining a pending intermediate
   ack. So a finalize can legitimately be in flight up to ~30s.
   On ack-timeout the adapter synthesises delivery (returns a success-shaped
   response, `errmsg="ack_timeout_assumed_delivered"`) so the caller treats the
   message as delivered — matching the official wecom-openclaw-plugin.

2. **Gateway flush bound.** The gateway `_run_agent` finally block joined
   `stream_task` with a hardcoded `timeout=5.0` (`gateway/run.py` flush block)
   and then `stream_task.cancel()` on timeout.

When the ack is slower than the gateway's 5s join window, the gateway cancels the
consumer **before** it finishes finalizing. The finalize frame's bytes, however,
were already written to the wire (the adapter write went out) and WeCom has
rendered them. But because the consumer coroutine was cancelled mid-await, `run.py`
reads `final_content_delivered=False`, the normal suppression predicate
(`final_response_sent` OR `final_content_delivered`) does NOT fire, and the
gateway emits a second, duplicate bubble.

## Fix

Two complementary changes in `gateway/run.py`:

1. **Longer flush bound for native streaming (root cause).** When the stream
   consumer resolved native streaming (`_use_native_streaming`), the flush now
   waits up to 32s instead of 5s — covering the adapter's ~30s worst-case
   finalize ack window. The consumer thus completes its finalize and sets
   `final_content_delivered` before the join fires, so the existing suppression
   predicate works as designed. Edit/draft consumers keep the fast 5s bound
   (their finalize is a local API round-trip; the happy path returns the moment
   the task completes regardless of this cap).

2. **Delivery-boundary dedup (safety net).** In the case where a stream consumer
   existed for the turn but suppression still did NOT fire (the ack-pending race
   where the flag was never set), the gateway now dedupes on CONTENT before
   emitting the normal final-send: if the consumer has already delivered the
   exact final text (`has_delivered_text` — visible prefix / recorded segments
   match), the redundant send is suppressed. This only suppresses on an exact
   content match, so genuinely distinct content is never dropped and delivery
   semantics for distinct messages are unchanged.

## Verification

- `tests/gateway/test_wecom_double_send.py` — real `GatewayStreamConsumer`
  lifecycle against the real `WeComAdapter` (only WS byte-writer + ack timing
  controlled) exercising the ack-timeout path. `TestDeliveryBoundaryDedup`
  asserts deterministic ack-pending state (content on wire, flag unset) closes
  with exactly one delivery; and that genuinely distinct content is NOT deduped.
  Full suite: 5 passed, 1 xfailed.
- Follow-up (human): after gateway restart + a day of live run, confirm zero new
  `possible duplicate send` lines in gateway.error.log:
  `grep 'possible duplicate send' <log>`.

## Deployment

- **Gateway restart required** for the fix to take effect (code in `gateway/run.py`).