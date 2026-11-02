# Draft send-state verification — did it already go? (verified 2026-10-23)

The failure mode this prevents: you tell NDR "six drafts are ready, press send", he presses send
on all of them himself, then later says **"send"** and you either (a) re-create drafts that already
went out, or (b) report as pending things that are long gone. Both happened in one session.

## Rule: never report a draft as unsent without a fresh read

NDR works in parallel with you. Between your turn and his next message he routinely opens Gmail
and presses send himself. Treat every "draft ready" statement as expiring the moment you hand it
over.

**Before any claim about draft state, run both:**

```
gws_gmail_draft_list(service_name="google-draas", max_results=25)
gws_gmail_search(query="in:sent newer_than:1d", service_name="google-draas", max_results=20)
```

- A **`count: 0`** from `draft_list` is not "nothing was ever drafted" — it usually means
  **he already sent them**. Cross-check `in:sent`.
- `in:sent` returns `from`, `to`, `subject` and an exact `date` per message. That is the proof
  of what left, and it lets you tell him "all six went out between 10:58 and 14:22".
- Do **not** trust a draft ID you were given earlier in the session. `gws_gmail_get` on a
  draft-id-shaped id (`r-…`) fails `Invalid id value` — draft ids are not message ids; list the
  drafts folder instead of probing ids.

## WhatsApp: a draft leaving the pending list is not proof of delivery

- `wa_draft_status()` with **no argument** lists what is still awaiting approval, with each
  draft's expiry. After `wa_send_confirmed` the draft disappears from that list — that only proves
  it left the queue.
- **Confirm delivery by reading the thread**: `wa_read_messages(chatId=..., limit=2)` and check
  your text appears as `me:` with a timestamp. That is the only honest basis for "sent and
  verified in the chat".
- Bridge drafts carry an expiry (default 30 min, max 240 via `ttlMinutes`). Before confirming a
  staged draft, compare its expiry against `date -u` — an expired draft must be re-created, not
  re-confirmed.

## Interpreting a bare "send"

When he says only **"send"**, it refers to *everything currently staged*, across channels — not
just the last thing discussed. Enumerate what is staged first, act on the WhatsApp drafts (those
you can actually send), and state plainly that email drafts remain his to press. Then report:
what was sent, what was already sent by him, and what is still staged and needs a decision.

## Reporting shape he responds to

- Lead with **"Sent — both delivered and verified (time)"**, then one bullet per recipient and the
  substance of what they were asked.
- Then a separate **"already out"** block for the things he sent himself, with times — it stops him
  re-checking his own Sent folder.
- Close with what is *still* pending, one line each, and the explicit send-or-drop ask.
