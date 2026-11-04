# Replying to NDR's mail — Anbu's "yes, I will take it up" delegation reply

**Trigger:** NDR mails Anbu a job; Anbu forwards the text to Hermes and dictates
"reply back to this mail stating that…" or "this is the mail I have to respond to
Nishant sir — prepare the draft mail and keep it ready so that I can send."

Verified end-to-end 2026-10-17 (Servant room – tile repair work at 2401, Prestige
Hermitage; 6-point reply).

## Steps

1. **Find NDR's original in Anbu's OWN mailbox** — `gws_gmail_search`,
   `service_name='google-draas'`. Search on a **content phrase**, never a guessed
   subject: the flat number, the contractor's name, a distinctive phrase
   (`"Prestige Hermitage"`, `"servants room"`). NDR's subject often bears no
   resemblance to what Anbu dictated.
2. **`gws_gmail_get` the hit and READ THE HEADERS BEFORE DRAFTING.** Two traps:
   - The original's `to:` can be a **different address from the mailbox session's**
     (original went to `anbarasan@draas.com`; the session token is
     `pm2.blr@draas.com`). Same person — both are Anbu. Not an error.
   - **NDR may ADDRESS a third person inside the body who is NOT a recipient.**
     His mail said "Sarthak, you will please see this job through…" while `to:`
     was Anbu alone. Never assume the addressed person is on the thread — check
     the headers, and tell Anbu to CC them manually if he wants them looped.
3. **`gws_gmail_reply_draft(message_id=<original id>, service_name='google-draas')`.**
   The tool takes **no `to` parameter** — it sets To = the original's From
   (`ndr@draas.com`). That is exactly what Anbu wants: **To NDR only, never
   reply-all**, even when the thread carries other people. Do not pass `cc`
   unless he asks.
4. **Read the draft back** with `gws_gmail_get` on the `message_id` returned by
   the reply-draft call (which also returns `draft_id` and `thread_id`). Confirm
   To, the `Re: …` subject, and every numbered point. Never tell him a draft is
   ready without this read-back.
5. **Own-token path only.** This is the session user's own mail — native `gws_*`
   tools / `service_name='google-draas'`. Never DWD.
6. **Never send.** The deliverable is a draft in his Drafts folder; he presses send.

## Body shape

Numbered points, ALL-CAPS point headers, one concern per point — same discipline as
his review mails. Keep his construction terms verbatim.

Signature block he expects (exact):

```
Thanks & Regards,
Anbarasan M
Vice President
DRA Realty Private Limited
M: +91 81500 29900
```

## Flagging discipline — every time, in the Telegram reply, NEVER in the mail

- **Additions** — any point you added that he did not dictate. Name it and offer
  deletion: *"point 6 was my addition; delete it if you'd rather not commit."*
  (Verified: the official-bill line, which only reaffirmed NDR's own ask.)
- **Omissions** — anything NDR asked for that the dictated reply drops, especially a
  **substitution**. When Anbu replaces NDR's suggested source with his own vendor
  (Guru instead of Jairam sir), say so plainly rather than silently dropping NDR's
  instruction: *"Jairam sir is not mentioned — NDR specifically asked to check with
  him. Want a line added, or leave it out deliberately?"*
- **Corrections** — never `drafts().update()` (fails HttpError 400 "Missing draft
  message"). `gws_gmail_draft_delete` the old draft, recreate the reply draft, read
  it back.
- **Drafts go in the Telegram reply text**, never via `send_message` — cross-user
  targeting is blocked in this session and you are already replying to him.

## Voice garbles seen in this exchange (2026-10-17)

| Heard | Means |
|---|---|
| "Akhil the driver" | **Atik** — the *contractor*; the driver is who found him, not the contractor |
| "Satak" | **Sarthak** (Sharma, admin3.blr@draas.com) |
| "Nishan sir" | **Nishant** sir |

Rule holds: confirm a garbled proper noun against the artifact (the original mail,
the contacts sheet) before it reaches the draft body.
