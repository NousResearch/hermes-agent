# Verify draft state before reporting "pending" — or re-drafting (verified 2026-10-23)

## The trap
NDR replies **"send"** to a message that listed several staged items. Two failure modes:

1. **He already pressed send himself.** On 23-Oct-2026 the reply said "send"; the Gmail
   draft was gone and `in:sent` showed the ITR mail had gone out at 14:22:09 — one minute
   *before* his message arrived at 14:23. Reporting "draft still pending, press send" would
   have been wrong, and re-creating the draft would have produced a duplicate.
2. **The whole Drafts folder is empty because every draft was sent.** `drafts().list()`
   returning `count: 0` is not proof the drafts were lost — check `in:sent` for the same
   day before saying anything.

## Order of operations when he says "send"
1. `date -u` + `TZ=Asia/Kolkata date` — confirm IST now. WhatsApp bridge drafts carry a
   UTC expiry (e.g. `09:19Z` = `14:49` IST); a draft can be alive with minutes to spare.
2. WhatsApp drafts: `wa_send_confirmed` per draft, then `wa_draft_status` (empty pending
   list = both left the queue), then `wa_read_messages(chatId, limit=2)` to see the
   message sitting in the thread — that is the only real proof of delivery.
3. Email: search `in:sent newer_than:1d` and match subject/recipient/time **before**
   telling him a draft is still waiting. Only then report what remains.
4. Report the sent-timestamp of anything he sent himself, so the record is unambiguous
   ("the ITR email is already out — sent 14:22:09, you pressed it yourself").

## Reporting shape that landed
Sent/verified items first with the chat or timestamp proof, then one line on anything
still genuinely pending, then the caveat about what he still has to do himself. Do not
re-list all six drafts he already sent as if they were outstanding.

## Side note
`drafts().list()` via `tools.gws_auth.build_service("gmail","v1",service_name="google-draas")`
and the native `gws_gmail_draft_list` both returned 0 here and both were correct. When the
count is 0, believe it and go looking in `in:sent` — don't assume the tool failed.
