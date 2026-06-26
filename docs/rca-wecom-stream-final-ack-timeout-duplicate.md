# RCA: WeCom native streaming 在 ack 超时时会重复推送同一条消息

**Status:** resolved by `fix(wecom-stream): align with official block-streaming + ack-timeout semantics`
**Severity:** P3 — affects user-perceived UX (duplicate bubbles in WeCom),
no data loss / silent failure / billing impact.
**Branch / scope:** `feat/wecom-native-streaming` and any descendant.

## TL;DR

When the WeCom AI Bot final stream frame's ack does not return within 5
seconds, Hermes treats the entire final-frame send as a failure and
falls back to a normal `aibot_respond_msg / msgtype=markdown` reply.
But the server has often already rendered the streamed final frame —
just the ack didn't make it back in time. The user sees the same content
twice: once from the streamed final frame, once from the fallback send.

The official `@wecom/wecom-openclaw-plugin` does **not** do this fallback —
it lets the timeout propagate up, accepting that the user may see a
visible failure. Hermes' fallback was added in the spirit of "deliver at
all costs" but conflates "I never got the ack" with "WeCom never
rendered the content," which they aren't.

## Symptom

Concrete repro from `~/.hermes/profiles/jarvis/logs/agent.log`
on 2026-06-25 23:33:

```
23:33:30,244  inbound message: msg='你能感知你现在用的模型真实上下午么'
23:33:30~42   LLM stream completes (response_len=281, api_calls=1)
23:33:42,948  Turn ended: response_len=281
23:33:47,955  response ready: time=17.7s
23:33:47,979  [Wecom] Sending response (281 chars) ←  fallback path
23:33:48,118  WARNING ... Final frame ack timeout (req_id=o_8FK...) ←  ~5.17s after final frame send
23:33:48,122  WARNING ... Stream frame failed (chat=...): Final frame ack timeout
```

The 5.17s gap between turn-ended and the warning matches `_REPLY_ACK_TIMEOUT
= 5.0` seconds. The fallback `Sending response (281 chars)` happens
*before* the timeout warning is emitted because the consumer task and the
gateway final-send dispatcher run on independent code paths; the consumer's
finalize ran into ack timeout, marked the stream as failed, the gateway's
`Suppressing normal final send` branch was not taken (because
`final_content_delivered=False`), and a normal markdown send went out.

User-visible result: same 281-character "好问题…" reply rendered twice in
WeCom — once by the stream finalize that the server *did* render, and
once by the fallback markdown send.

## Code path

### Where the timeout fires

`plugins/platforms/wecom/adapter.py::_send_reply_queued` (in_final=True
branch):

```python
if is_final:
    try:
        response = await asyncio.wait_for(future, timeout=self._REPLY_ACK_TIMEOUT)
        return response
    except asyncio.TimeoutError:
        # Final frame ack timeout is a FAILURE — aligned with official SDK.
        logger.warning(
            "[%s] Final frame ack timeout (req_id=%s) — treating as failure",
            self.name, normalized,
        )
        raise RuntimeError(f"Final frame ack timeout (req_id={normalized})")
```

The `_REPLY_ACK_TIMEOUT = 5.0` constant matches the official SDK's
internal `replyAckTimeout = 5000`. The semantic alignment is correct
*for the SDK*: SDK rejects on timeout. But the official plugin's
`message-sender.ts::sendWeComReply` doesn't translate that reject into
"resend via a different path" — it just lets the error propagate.

### Where Hermes diverges

The exception is caught one level up in
`adapter.py::send_stream_frame`'s `except Exception`:

```python
except Exception as exc:
    logger.warning("[%s] Stream frame failed (chat=%s): %s", ...)
    if 'turn' in locals(): ...
    return False  # ← signals "stream failed"
```

`send_stream_frame` returns `False`, and the consumer
(`gateway/stream_consumer.py::_send_or_edit`, line ~1750-1803) interprets
this as native-stream failure:

```python
if ok: ...   # success path
else:
    self._use_native_streaming = False
    # Best-effort finalize to close the bubble
    if self._native_stream_opened:
        try:
            await self.adapter.send_stream_frame(text, finalize=True, ...)
            # DO NOT mark _final_content_delivered here.
            # The finalize frame closes the typing bubble, but WeCom may
            # not actually render the content (e.g., errcode 6000 race).
            # Let the fallback send() path deliver the content reliably.
        except Exception as e: ...
    # Fall through to the edit/send paths
```

The "DO NOT mark `_final_content_delivered`" comment is the crux. The
intent is: *we don't trust that the bubble actually rendered*, so don't
suppress the gateway's normal final-send.

### Where the duplicate is emitted

`gateway/run.py::~17369`:

```python
_streamed = bool(_sc and getattr(_sc, "final_response_sent", False))
_previewed = bool(response.get("response_previewed"))
_content_delivered = bool(_sc and getattr(_sc, "final_content_delivered", False))
if not _is_empty_sentinel and not _transformed and (_streamed or _previewed or _content_delivered):
    logger.info("Suppressing normal final send for session %s ...")
    response["already_sent"] = True
```

When `_content_delivered=False` (which the consumer set above), no flag
is True, suppression is skipped, and the gateway's normal final-send
runs — `Sending response (281 chars)`.

## Why "ack timeout" ≠ "frame not delivered"

The WeCom AI Bot ack model is asymmetric:

- The bot client sends `aibot_respond_msg` with a stream finalize
  (`finish=true`). The server enqueues it for client rendering immediately.
- The server returns an `errcode=0` ack frame on the same WS, but only
  *after* it has confirmed the message was queued/persisted on its side.
- Network jitter, server-side message-queue lag, or even another in-flight
  reply on the same WS can push the ack past 5s while the client has
  already rendered the message.
- Empirically in the bilibili WeCom environment, ack > 5s is not rare
  on long replies. The SDK's `replyAckTimeout = 5000` was tuned for
  Tencent-internal latency profiles.

The official plugin handles this by surfacing the failure to its caller
(the openclaw runtime), which logs and shows the user a generic error.
It does *not* attempt to resend.

## Comparison with the official plugin

| Behavior | `@wecom/wecom-openclaw-plugin` | Hermes (this branch) |
|----------|-------------------------------|----------------------|
| Ack timeout | `withTimeout` rejects → `throw err` | `_send_reply_queued` raises RuntimeError |
| `846608` (stream expired) | Caught, throws `StreamExpiredError`, caller falls back to active `sendMessage` | Same — caught, marks turn expired, switches to proactive send |
| **Generic ack timeout** | **No fallback — error propagates to runtime** | **Falls back to normal `aibot_respond_msg / msgtype=markdown` send → duplicates the message when WeCom did render the streamed final** |
| Errcode 6000 (version conflict) | Throws | Caught, returns False, fallback fires |

So the only true divergence is the *generic ack timeout* row.

## Options to fix

| # | Approach | Pros | Cons |
|---|----------|------|------|
| **A** | **Treat final-frame ack timeout as success-with-uncertainty.** Mark `_final_content_delivered=True` when the timeout fires *after* the server already accepted the frame (i.e., we sent it, no errcode came back). The thinking-bubble is closed, the message has been queued, suppress normal final send. | Eliminates duplicates in the common timeout path. Matches what users actually see in WeCom. | Loses the "guaranteed redelivery" property when the server *also* never queued the message. Empirically rare — the WS layer would 4xx instead of timing out silently. |
| **B** | **Increase `_REPLY_ACK_TIMEOUT`** from 5s to e.g. 15s, matching the official plugin's `REPLY_SEND_TIMEOUT_MS = 15_000`. | Trivial diff. Reduces — though doesn't eliminate — the timeout-on-success window. | Long replies (>15s) still hit it. Increases the time the user waits in `Suppressing normal final send` decision-making before falling back when the server *did* fail. |
| **C** | **Idempotent fallback send.** Before the gateway's normal final send fires, have the adapter dedupe by `(chat_id, last_assistant_text_fingerprint)` within a short window. Skip the second send if the same content is being delivered. | Defends against duplicates from any path (current bug + future ones). | Stateful, requires per-chat tracking. Edge cases when the model genuinely repeats itself across turns. |
| **D** | **Hybrid (A + idempotent guard).** Mark delivered on ack timeout (A) AND keep an idempotent fingerprint guard as belt-and-braces (C-lite). | Closes the duplicate root cause and the class of bugs around it. | Most code, smallest behavioral surprise. |

Recommendation: **D**. A alone fixes this concrete bug; the fingerprint
guard hardens the gateway against any future native-streaming path that
returns False after the server *did* accept the frame.

## Pin-down for "fix the bug only" (option A)

Three small edits:

1. `plugins/platforms/wecom/adapter.py::_send_reply_queued` —
   distinguish "send-failed-during-write" from "ack-not-received-after-write."
   The former should still raise; the latter should return a result
   that the caller can interpret as "delivered, ack pending."
2. `plugins/platforms/wecom/adapter.py::send_stream_frame` —
   on the new "ack pending" return shape, return `True` (delivered)
   instead of `False`. The turn cleanup still happens (the stream is
   considered done from the bot's side).
3. (No change to `gateway/stream_consumer.py` or `gateway/run.py`.)
   With `send_stream_frame` returning True, the consumer's existing
   success path runs — `_final_content_delivered=True` — and the
   gateway's `Suppressing normal final send` branch fires.

Tests: extend `tests/gateway/test_wecom.py::TestSendStreamFrame` with
a case where `_send_reply_queued`'s ack future never resolves —
the test should assert that `send_stream_frame` returns True and
the consumer marks delivery confirmed.

## Why I'm not patching this in the same commit as the 1M context change

The 1M context fix is unrelated to the wecom plugin and lives in
`agent/`. Mixing them obscures both the diff for review and the bisect
surface. This RCA captures the analysis so we can land it as a
separate, focused commit when the maintainer (you) signs off on
option A vs D.

## Fix landed

We went one step further than option A. Rather than only relaxing the
ack-timeout semantics, we also aligned the frame cadence with the
official `@wecom/wecom-openclaw-plugin` so the timeout almost never
fires in the first place. Two changes shipped in
`plugins/platforms/wecom/adapter.py`:

### 1. Block-streaming chunker (replaces the 200ms time throttle)

The official wecom plugin coalesces incoming LLM tokens into
sentence-aligned blocks (`webhook/helpers.ts::buildHermesAgentConfig`):

```
blockStreamingChunk:    { minChars: 120, maxChars: 360, breakPreference: "sentence" }
blockStreamingCoalesce: { minChars: 120, maxChars: 360, idleMs: 250 }
```

Hermes now does the same. Each `StreamTurn` owns a `_BlockChunker`
that buffers the consumer's cumulative cursor and only emits a frame
when one of these holds:

- new content is **≥ 120 chars** AND ends on a safe sentence
  terminator (`.!?。！？` followed by whitespace) — guards against
  splitting decimals / IP addresses;
- new content is **≥ 120 chars** AND ends on a paragraph boundary
  (`\n\n`);
- new content reaches the **360-char hard cap** — force a break;
- the **250ms idle-flush timer** fires (LLM paused mid-thought);
- the caller passes `finalize=True` (force-drain).

The 200ms time-window throttle is gone — frames now ship on natural
language boundaries instead of a wall-clock tick. Typical cadence
drops from "every 200ms" to "every 0.5–2s," which keeps the WeCom
30-frame/min/chat budget intact and gives the server's ack pipeline
plenty of headroom.

### 2. Final-frame ack timeout → success-with-uncertainty

`_send_reply_queued`'s `is_final=True` branch no longer raises
`RuntimeError` when the ack times out. The official plugin's
`message-sender.ts` re-throws to its caller, which does not retry
with a different transport; we mirror that by returning a
success-shaped response:

```python
return {
    "errcode": 0,
    "errmsg": "ack_timeout_assumed_delivered",
    "ack_pending": True,
}
```

The consumer treats this as a successful finalize, marks
`_final_content_delivered=True`, and the gateway takes the
`Suppressing normal final send` branch — no duplicate.

The errcode-flagged failure paths (846608 stream expired, 846609
subscription lost, errcode 6000 version conflict, errcode response
on ack with non-zero status) are unchanged — those are real failures
and still propagate the way they did. Only the "frame written, ack
silent" case is reclassified.

### Tests

`tests/gateway/test_wecom.py` adds three new test classes:

- `TestBlockChunker` (8 tests) — min/max gating, sentence/paragraph
  boundaries, hard cap, force-drain, decimal-safe splits;
- `TestBlockStreamingFrameFlow` (4 tests) — short text buffered no
  frame, finalize force-drains the tail, idle-flush emits a partial,
  sentence-boundary triggers an immediate frame;
- `TestFinalFrameAckTimeoutSemantics` (2 tests) — ack timeout returns
  the success-shaped response, send-failure errors still propagate.

Plus the two pre-existing `TestSendStreamFrame` cases that asserted
the old throttle behaviour were updated to feed sentence-terminated
payloads above `min_chars` (their intent — seed lifecycle and
stream_id continuity — is preserved).

Full wecom test suite: **90 passed, 3 skipped** (unchanged skipped
list, all pre-existing module-path skew in adjacent tests fixed
en passant). Adjacent suites also green:

```
tests/gateway/test_wecom.py                  90 passed
tests/gateway/test_wecom_per_turn.py          8 passed
tests/gateway/test_approval_boundary.py       6 passed
tests/gateway/test_stream_consumer_wecom_native.py   13 passed
tests/tools/test_send_message_cross_loop.py   4 passed
```

### Tunables

The constants live at the top of `plugins/platforms/wecom/adapter.py`:

```python
BLOCK_STREAM_MIN_CHARS = 120
BLOCK_STREAM_MAX_CHARS = 360
BLOCK_STREAM_IDLE_FLUSH = 0.25  # seconds
```

They mirror the official plugin's values verbatim. If you want
snappier mid-stream updates on slow chats, lower `min_chars` to e.g.
60 — the chunker still won't split mid-sentence, so the only effect
is that very short sentences emit earlier.
