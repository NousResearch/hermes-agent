# WeCom Native Streaming

How the WeCom AI Bot adapter turns `GatewayStreamConsumer` deltas into
progressive reply-channel stream frames — and why it does things a bit
differently from every other platform adapter.

> **Credit.** The initial implementation is adapted from [PR #7041][pr-7041]
> by @fantasticKe, with a simpler mapping onto the existing consumer API
> plus tool-boundary and test coverage.
> The `<think></think>` thinking-placeholder pattern was investigated after
> reading the [OpenClaw WeCom plugin][openclaw] but is deliberately NOT
> enabled in this adapter (see *Why no thinking animation* below).

[pr-7041]: https://github.com/NousResearch/hermes-agent/pull/7041
[openclaw]: https://github.com/openclaw

## Why

WeCom's AI Bot client displays a reply-channel *stream* message
(`msgtype: stream`) as a single, progressively-updated bubble in the chat
— this is the platform's native analogue of Telegram's "edit a message in
place" trick that most streaming adapters use. Without it, a user sees
nothing for the full duration of the reply, then the full text appears
as one block.

This adapter wires the consumer's generic `send` → `edit_message` →
`finalize_stream` cycle onto that native stream protocol, so the WeCom
user sees the same progressive-typing UX that Telegram and Discord users
get.

## Protocol

The reply-channel stream is correlated by two ids:

- **`reply_req_id`** — identifies the inbound request being replied to.
  WeCom assigns exactly ONE stream lifetime per `reply_req_id`; once a
  frame is sent with `finish=true`, that id is exhausted. Attempting to
  open a second stream (a new stream_id) against the same `reply_req_id`
  fails with `errcode 6000` ("data version conflict").
- **`stream_id`** — identifies which stream inside a reply. Chosen by the
  sender. All frames of one stream (`finish=false` continuations and the
  closing `finish=true`) must share the same `stream_id`.

Frame shape (delivered over the existing WebSocket `aibot_respond_msg`):

```json
{
  "msgtype": "stream",
  "stream": {
    "id":      "<stream_id>",
    "finish":  false,
    "content": "<full accumulated text>"
  }
}
```

Notes:

- `content` is the FULL response so far, not a delta. WeCom's client
  does the diffing.
- Only the final frame (`finish=true`) is sent through
  `_send_reply_request` (which awaits an ACK). Intermediate frames are
  fire-and-forget via `_send_json` — WeCom does not ACK per-frame and
  awaiting one would stall the stream behind network RTT.

## Mapping the consumer API

```
GatewayStreamConsumer              WeComAdapter
──────────────────────────         ──────────────────────────
first send(text, streaming=True)   → stream_id = new_id()
                                     _send_reply_stream(finish=False)
                                     _active_streams[reply_req_id] = stream_id
                                     return SendResult(message_id=reply_req_id)

edit_message(id, text)             → look up stream_id
                                     _send_reply_stream(finish=False)

finalize_stream(id, text)          → look up stream_id
                                     _send_reply_stream(finish=True)
                                     _active_streams.pop(reply_req_id)
```

`message_id` is deliberately set to `reply_req_id` in streaming mode so
that subsequent consumer calls can find the right stream_id without
needing a separate lookup table.

## Unified-stream and tool boundaries

Hermes' `GatewayStreamConsumer` inserts a *segment break* at every tool
call, which on most adapters finalizes the current message so tool
progress messages can appear "between" assistant text messages. On
WeCom this would be catastrophic: finalizing closes the stream's only
lifetime, and the next segment's `send()` would try to open a second
stream against the same `reply_req_id` — triggering `errcode 6000`.

To avoid that, the WeCom adapter sets:

```python
class WeComAdapter(...):
    native_streaming_unified = True
```

The consumer checks this flag in its segment-break branch and skips
both `finalize_stream` and the internal `_reset_segment_state`. The
entire response — across any number of tool calls — is delivered
through a single stream frame sequence.

Consequence: tool-progress messages (sent via the proactive
`aibot_send_msg` channel) appear in the chat timeline as independent
messages, BELOW the streaming reply bubble. The default
`_PLATFORM_DEFAULTS["wecom"]["tool_progress"] = "off"` already
suppresses those for typical users; operators who opt into tool
progress on WeCom accept that they see the progress after the streaming
bubble has already been pinned to its timeline position.

## Why no thinking animation

WeCom renders a `<think>...</think>` prefix as a thinking animation,
and it is tempting to open the stream the moment an inbound message
arrives (in `send_typing`) so the user sees an immediate "typing"
signal while the agent reasons and calls tools.

This is how OpenClaw's WeCom plugin works, and an earlier draft of this
patch did the same. It was reverted after live testing because the
stream bubble is pinned to its *initial* position in the chat timeline,
which means any tool-progress messages sent after the thinking frame
appear BELOW it, and the eventual real reply appears ABOVE those
progress messages. The result is confusing: the final reply looks like
it was said *before* the tools ran.

Given the choice between a visible "typing" animation with wrong
ordering, and no animation with correct ordering plus visible tool
progress, we picked the latter. `send_typing` is intentionally a no-op
on WeCom; the first streaming delta itself serves as the "bot is
responding" signal.

If you need a thinking animation AND correct ordering, you also need to
suppress `display.platforms.wecom.tool_progress` and route all
intermediate signalling through the stream frames themselves. That is
out of scope for this PR.

## Failure modes

- **`finalize_stream` returns `success=False`** — the consumer's
  `_finalize_native_stream` helper treats that as a signal to fall
  through to the generic `_send_or_edit` path, which will attempt to
  deliver the final text as a continuation `edit_message` or, on
  failure, a fallback standalone message.
- **Adapter-level exceptions** (network, timeout) — swallowed and
  logged inside the `finalize_stream` / `edit_message` methods; caller
  sees a failed `SendResult` and handles it as above.
- **Proactive (non-reply) streaming send** — the `streaming` flag is
  honored only when a `reply_req_id` is available. Proactive sends to
  a chat with no reply context deliver as a single block message, and
  `_active_streams` is deliberately not populated (so a later
  `edit_message` correctly reports "no active stream").
