# WhatsApp — a contact exists in the bridge only once there is a chat

Two durable traps from landowner correspondence, both hit in the same session.

## 1. The person is not missing, the conversation is

`wa_list_chats` and `wa_search_messages` only return conversations the bridge has actually
seen. A counterparty who has never exchanged a message with NDR does not appear at all — not
by name, not by number. Searching his phone number returns nothing.

Symptom: after scanning every chat you conclude "no WhatsApp conversation with \<person\>
exists". What you *will* find is an unsaved number that looks plausible — a near-miss on the
number, no profile name. **Do not send to it on that basis.** A wrong number in a landowner
dispute is unrecoverable.

Correct handling: report the blocker plainly and offer the two real options —

- give me the mobile number, confirmed to be on WhatsApp, and I'll reach it with
  `wa_send_new_number` (the only tool that works with no existing chat); or
- skip WhatsApp and let the email carry the briefing alone.

Never quietly substitute a guessed number, and never quietly drop the WhatsApp half and let
the user assume it went.

**It clears itself the moment NDR messages them.** He comes back with "I just sent him a
WhatsApp so you will find his contact now" — re-run `wa_list_chats` and the chat is there,
with full history. Re-read it before drafting rather than drafting from what you already
believed. In this case the thread contained the position that actually mattered ("call won't
be sufficient, we can meet in your office … fix a meeting sometime next week"), which
contradicted the message NDR had just asked for (proposing a call tomorrow). Surface that
contradiction instead of writing the message as briefed.

## 2. Never release a message that claims an email already went

The standard landowner pattern is email-with-attachment plus a short WhatsApp pointing at it,
and the WhatsApp text says *"I have sent you the workings by email"*. That sentence is only
true once the draft is actually sent.

Check before releasing the text, every time:

- **Gmail drafts list empty, message sitting in SENT with today's timestamp** → the email went,
  the WhatsApp is accurate, release it.
- **Draft still in Drafts** → either have him send it first, or reword to *"I am sending you the
  email and the Excel now"*. Do not release on the assumption he will send it in the next
  minute; he may not, and the landowner then hunts for an email that does not exist.

Note the same check backwards: a draft vanishing from the Drafts list is **not** proof it was
sent. Confirm the SENT label and timestamp before you tell the recipient it is in his inbox.

## 3. Auto-send list decides whether he still has an action

`wa_list_auto_send` → if the chat is on it, the message goes out on `wa_send_message` with no
approval. If it is not, it stages and needs his one-word release. Say which in the report, so
he knows whether he has anything left to do.

Draft ids expire (default 30 min, max 240). Re-list before releasing rather than trusting an
id from earlier in the conversation — and **pass the full `draft_id`**; a truncated one errors.

## 4. Long briefing plus a file is two drafts

A WhatsApp caption caps near 1,024 characters, so a real briefing cannot ride as a caption.
Stage the text with `wa_draft_reply` and the file with `wa_send_media` — separate `draft_id`s,
separate approvals — and release the **text first**, then the file, so the sentence saying
"attached" is never ahead of the attachment. See
`ledger-allocation-and-workbook-verification.md` §11b for the byte-identity check on the file
itself (attach the live Drive bytes, not your local working copy).
