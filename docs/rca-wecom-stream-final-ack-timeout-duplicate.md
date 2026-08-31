# RCA — WeCom stream final-frame ack-timeout duplicate send

Status: FIXED 2026-08-30 (card t_9851e2d0) — defense-in-depth on top of B2
Product: gateway/run.py + gateway/stream_consumer.py + plugins/platforms/wecom/adapter.py

## Symptom

32 duplicated user messages in 7 days (measured 2026-08-30), all same-session,
across both desktop and Slack WeCom. Includes a doubled "credits added. go." which
risks double-dispatching an approval. The gateway logs the signature:

    Normal final-send NOT suppressed despite active stream consumer for session ...
    possible duplicate send (see wecom ack-timeout RCA)

7 such occurrences on 08-30 alone (gateway/run.py ~30986).

## Root cause — a timeout-inversion race (the ack-await window)

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

2. **Gateway flush bound.** The gateway `_run_agent_inner` finally block joined
   `stream_task` with a hardcoded `timeout=5.0` (`gateway/run.py` flush block)
   and then `stream_task.cancel()` on timeout.

When the ack is slower than the gateway's 5s join window, the gateway cancels the
consumer **before** it finishes finalizing. The finalize frame's bytes, however,
were already written to the wire (the adapter write went out) and WeCom has
rendered them. Depending on where the cancel lands, `run.py` may read
`final_content_delivered=False`, the normal suppression predicate
(`final_response_sent` OR `final_content_delivered`) does NOT fire, and the
gateway emits a second, duplicate bubble.

## Important prior fix — B2 (ALREADY on main, landed 08-29)

This window was partially fixed on main *before* this card, by the **B2
optimistic-mark patch** (`gateway/stream_consumer.py`, `_final_response_sent` /
`_final_content_delivered` set to True **before** `send_stream_frame` blocks on
the final-frame ack). B2 closes the pure ack-await cancel: once the optimistic
mark has run, a gateway cancel that lands during the ack wait leaves the flags
already True, so suppression fires and no duplicate occurs.

Consequence: **the "7 occurrences on 08-30" evidence predates B2 being loaded**
— the gateway was not restarted after B2 landed (08-29), so the 08-30 run was on
a binary carrying the old hardcoded-5s behaviour. Post-B2 restarts, that primary
window should already be quiet. Anyone investigating new duplicates must first
confirm the gateway is on a B2-inclusive build.

## Residual gaps B2 does NOT cover (the reason this card still matters)

B2's optimistic mark is set only in the native finalize send path, right before
the ack await. It does **not** cover cancellation or dispatch failure that lands
*earlier*, before the optimistic mark, while WeCom has already rendered content:

1. **`_abandon_native_stream` on cancellation (native draft path).** When the
   consumer is cancelled before reaching its finalize send, the native-draft
   branch (`_message_id is None`) seals the stream via `abandon_open_draft` and
   deliberately sets **no delivery flags** (comment: "an abandoned turn's text
   was partial, and the gateway's normal paths still own whatever happens
   next"). If a gateway flush-cancel lands in this pre-finalize window and the
   rendered draft equals the final text, the normal final-send duplicates it.

2. **Best-effort "DO NOT mark" finalize.** On a definitive native dispatch
   failure (stream never opened, 846608 expired, errcode 6000, or the send
   raised), the optimistic mark is **rolled back** and the consumer sends a
   best-effort finalize frame to close the bubble, explicitly *not* marking
   `_final_content_delivered` ("WeCom may not actually render the content").
   If that frame did render, the fallback then re-sends — a second duplicate
   path that the optimistic mark cannot cover by construction.

3. **Delivery recorded under a different record.** The final answer may have
   been delivered earlier in the turn as a segment break / commentary (exact
   content match) without the turn-final flag; the flag-independent
   `has_delivered_text` sees it, the flag predicate does not.

Axiom: for WeCom, a stream frame that reached the wire is rendered by the client
—the same premise the ack-timeout-as-success path and B2 rely on.

## Fix (defense-in-depth on top of B2)

Two complementary changes in `gateway/run.py`:

1. **Longer flush bound for native streaming.** When the stream consumer
   resolved native streaming (`_use_native_streaming`), the flush now waits
   `2 * adapter._REPLY_ACK_TIMEOUT + 4s` (~34s by default) instead of 5s. This
   stops the gateway from cancelling a native consumer *before* it reaches B2's
   optimistic mark at all — closing the residual paths above (pre-finalize
   abandon, DON'T-mark fallback) by letting the consumer finalize normally.
   Edit/draft consumers keep the fast 5s bound (their finalize is a local API
   round-trip; the happy path returns the moment the task completes regardless
   of this cap). Deliberate trade-off: a slow WeCom ack holds the session
   "busy" (release of `_running_agent_state`) up to the cap instead of 5s —
   correctness over best-case latency, and only in the slow-ack case.

2. **Delivery-boundary dedup (extractable decision).** In the case where a
   stream consumer existed for the turn but suppression still did NOT fire
   (any residual path where content was reached the wire but the flag was not
   observed), the gateway now dedupes on CONTENT before emitting the normal
   final-send: the decision lives in
   `gateway/stream_consumer.py::should_suppress_duplicate_final_send` (imported
   by run.py and exercised directly by the test), which returns True only when
   the consumer has already delivered the exact final text
   (`has_delivered_text` — `_last_sent_text`/`_visible_prefix` or recorded
   segment/commentary match). Exact-content match only — genuinely distinct
   content is never dropped and delivery semantics for distinct messages are
   unchanged. The legacy logged-warning-only branch becomes: suppress if the
   content matches, else log the "possible duplicate send" warning.

## Verification

- `tests/gateway/test_wecom_double_send.py` — real `GatewayStreamConsumer`
  lifecycle against the real `WeComAdapter` (only WS byte-writer + ack timing
  controlled) exercising the ack-timeout path. `TestDeliveryBoundaryDedup`
  asserts deterministic ack-pending state (content on wire, flag unset) closes
  with exactly one delivery via the **imported** `should_suppress_...` helper
  (the same code run.py calls, not a mirror); and that genuinely distinct
  content is NOT deduped. Full suite: 5 passed, 1 xfailed.
- Follow-up (human): after gateway restart + a day of live run, confirm zero new
  `possible duplicate send` lines in gateway.error.log:
  `grep 'possible duplicate send' <log>`.

## Deployment

- **Gateway restart required** for the fix to take effect (code in `gateway/run.py`).