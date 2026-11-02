# Cross-user sends, and how to hand a message over (verified 2026-10-16)

## The block

You **cannot** deliver a message into a *third party's* chat from the requester's session. Verified twice in one session, on a Telegram request to "share this with Nishant sir, as well as Prakash":

| Attempt | Result |
|---|---|
| `send_message(target="telegram:<Other Person> (dm)")` | **error**: `Cross-user send blocked. Deliver results to the requesting user only. Use target='telegram' (no explicit ID) or target='origin'` |
| `send_message(recipient_name="<Other Person>", platform="telegram")` | returns `success` + a `chat_id` — but that `chat_id` is the **requester's own** chat. Silent fallback, no error. |
| `send_message(target="telegram")` (bare home) afterwards | also refused with the same `Cross-user send blocked` error |

**Rule: compare the returned `chat_id` against the session's home-channel ID before telling anyone a message went out.** A `success: true` here is not evidence of delivery.

## What to do instead — pick the cheapest one that works

1. **Hand the text over to the requester to copy or forward.** This is almost always enough: the text is already sitting in their own chat with the bot, so a long-press → Forward / Copy is a two-tap action. Say that explicitly — users do not realise the message is already in their hands.
2. **`whatsapp_link` tap-to-send link.** Opens WhatsApp with the text pre-filled; the user picks the contact and sends. Needs no phone number if you omit `phone` (contact picker). Watch the 4096-char Telegram cap: two long `display_link`s in one reply will get split and a link can be cut in half — deliver each link as its own message, or skip the links and offer them.
3. **Unsent Gmail draft** via `gws_gmail_draft_create` when the content suits email.

## Answering "why can't you DM like he did?"

The other person did it **themselves** — they opened their own Telegram/WhatsApp app and typed or forwarded. That is a human action. The assistant is a bot wired to the requester's Hermes, and the platform bars it from posting into a third party's chat. There is no permission missing on the requester's account and nothing for them to fix.

Say this plainly, in two or three lines, then give the copy/forward steps. Do **not** re-attempt the send, and do **not** speculate about the other person's setup.

## Handover format the user expects

When the user asks for "the entire message with links and all", they want a **finished, forwardable message**: real Drive URLs inline (folder link first, then the individual files), no placeholders, no "insert link here". Resolve the links yourself before presenting the draft — search Drive for the collateral set rather than asking the user for the URLs.

Two checks worth making before the draft goes out:
- **Permission reality** — files owned by someone else (e.g. marketing's PDFs) may not be openable by the recipient. You usually cannot read the ACL (403 `insufficientFilePermissions` when you are only a viewer), so say so and name the owner who must share, rather than claiming the links work.
- **Dates/names** — voice-dictated event names, dates and counts must be confirmed with the user before the message goes out.

## Related trap

`contact_resolver` fails with `User '<telegram id>' has no gws_service configured in their profile` in sessions where the caller's Google account is not the session default — pass `account_email="sales1.blr@draas.com"` (or the correct account) explicitly.
