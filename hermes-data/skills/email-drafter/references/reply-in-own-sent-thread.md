# Replying inside a thread where YOU are the sender

Trigger: "reply all in that thread" pointing at a mail NDR himself sent (it lives in
`SENT`, `From: Nishant Ranka <ndr@draas.com>`).

## The trap

`gws_gmail_reply_draft(message_id=<id of a message in your own SENT>)` addresses the
draft **to NDR himself**:

```
To: Nishant Ranka <ndr@draas.com>        # WRONG
Subject: Re: <original subject>          # correct
In-Reply-To / References: <original>     # correct
```

Gmail's reply target is the message's `From`. When the original is your own outbound
mail, `From` == you. The threading headers are right; only the recipient is wrong.
Always verify `To:` before reporting a draft as ready — a self-addressed draft looks
completely fine in the Drafts list.

## The second trap — do NOT fix it with `drafts().update()`

Rewriting the raw and calling `drafts().update()` **re-threads the draft onto a brand
new thread**. Verified 2026-10-17: draft reported `threadId 1a130a838b74e29f` before the
update and `1a149620599b465f` after, so it vanished from the original thread even though
`In-Reply-To`/`References` still pointed at the original `Message-ID`.

`threads().get(<original>)` showing 1 message is the tell.

## Working procedure

Run in `execute_code` with `sys.path.insert(0, "/opt/hermes")` then
`build_service("gmail", "v1", service_name="google-draas")` (never `terminal()`).

1. Create the reply normally so Gmail builds the MIME + threading headers:
   `gws_gmail_reply_draft(message_id=<original>, body=...)` → keep `draft_id`.
2. `drafts().get(userId="me", id=<draft_id>, format="raw")`, base64url-decode.
3. Replace the `To:` header line with the real recipient(s). Leave
   `In-Reply-To`/`References` untouched — they already point at the original.
   (`assert raw.count(old_to) == 1` first; if the count isn't 1, inspect before editing.)
4. `drafts().delete(userId="me", id=<draft_id>)`.
5. `drafts().create(userId="me", body={"message": {"threadId": <ORIGINAL_THREAD_ID>,
   "raw": base64url(raw2)}})`. **Passing `threadId` here is what keeps it in the thread.**
6. Verify, in this order:
   - `threads().get(<original>)` → message count went up by 1 and the new id carries
     label `DRAFT`
   - `messages().get(<new id>, format="full")` → print `From / To / Cc / Subject /
     In-Reply-To / References` and the decoded body
   Only then report the draft as ready.

## Reply-all specifics

Read `To` + `Cc` from the **original message headers**, drop your own address, keep the
rest. In the 2026-10-17 Kanta visa case the original was `To: Sarthak Sharma
<admin3.blr@draas.com>` with no Cc, so reply-all == reply to Sarthak alone. Say that
explicitly in the reply to NDR so he can see the recipient list was considered.

## Also worth checking before drafting

- **Sibling drafts.** NDR often has an earlier unsent draft on the same topic (different
  `threadId`, label `DRAFT`). Search the mailbox for the topic first; if you find one,
  tell him it exists and offer to delete or merge instead of leaving two overlapping
  mails to trip over.
- **Access claims.** If the mail says "you have viewer access", actually call
  `gws_drive_permissions_list` on the folder and confirm the `reader` role for that exact
  address before writing the sentence.
