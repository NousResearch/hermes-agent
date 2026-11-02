# Gmail Threaded Reply Drafts — Bridge Pitfalls

Applies to `tools.gws_skill_bridge.call("draft_reply_create", ...)`.

## Correct signature
```python
call("draft_reply_create", service_name="google-draas",
     message_id="<id of the message being replied to>",
     to="Vardhan Ventures <info@vardhanventures.in>",
     cc="Dharmesh Ranka <dharmesh.ranka1@gmail.com>, Manish Ranka <mdr@draas.com>",  # optional
     body="...")
```

**`cc` IS supported (verified from bridge source 2026-09-23).** The bridge
reads `args.cc` and sets the `Cc` header when non-empty. For a "reply all"
draft, pass the primary recipient in `to` (the person the reply is directed
at) and the other thread participants comma-separated in `cc`. Confirm after
creation with `drafts().get(format='full')` — the To/Cc headers render
exactly as passed.

## Pitfall 1 — recipient goes in `to`, NOT `reply_to` (CRITICAL)
The bridge reads `args.to` and falls back to the original message's From
header when it's empty. If you pass `reply_to=...`, the arg is ignored →
on a thread NDR sent himself, the draft gets addressed to
ndr@draas.com (i.e. a reply to himself). Always pass `to` explicitly with
the vendor/contact's name + email.

## Pitfall 2 — use `message_id`, not `thread_id`
`draft_reply_create` calls `messages().get(id=args.message_id)`. Passing
`thread_id` raises `TypeError: Missing required parameter "id"`. Get the
message id from `gmail_thread_get` (correct op name, NOT `gmail_get_thread`)
or from the earlier `gmail_search` result.

## Pitfall 3 — verify headers after creating
`drafts().get(format='full')` — NOT `format='metadata'` with
`metadataHeaders` (drafts().get rejects metadataHeaders with TypeError;
that param is only for messages().get). Check: To == intended recipient,
In-Reply-To / References non-empty, threadId == original thread.
If wrong, `draft_delete(draft_id=...)` then recreate.

## Pitfall 4 — draft_get shows the reply "to" as the mailbox owner
`draft_get` reports the To header of the draft message; if the draft is a
reply created via the threaded path it may display NDR's own address in
some views. Always confirm the raw header via drafts().get(format='full')
before telling the user the recipient is correct.

## Pitfall 5 — reply-ALL is supported: `cc=` is a real kwarg (verified 2026-09-23, against bridge source)
`draft_reply_create` (in `tools/gws_skill_bridge.py`) reads
`getattr(args, "cc", "")` and sets the `Cc:` header whenever it is non-empty.
Supported kwargs are exactly: `message_id`, `to`, `cc`, `body`, `html`,
`from_header`. So a genuine reply-all is a single call — no raw-MIME fallback
needed:

```python
call("draft_reply_create", service_name="google-draas",
     message_id="<id of the OTHER PARTY's latest message>",
     to="Amit Jain <amitkjain@deloitte.com>",
     cc="Dharmesh Ranka <dharmesh.ranka1@gmail.com>, Manish Ranka <mdr@draas.com>",
     body="...")
```

Two rules that follow:
- Reply to the **other party's** last message id, not your own sent reply — the
  bridge derives `In-Reply-To`/`References` from that message's `Message-ID`.
- "Reply all" = put the primary counterparty in `to` and *everyone else on the
  thread* in `cc`. When NDR says "reply all Amit, Manish please respond, and a
  separate note for Manish", the extra addressee belongs in the SAME draft's
  `cc` plus a paragraph addressed to them — not a second email.
- After creating, confirm To, Cc, threadId and In-Reply-To via
  `drafts().get(userId='me', id=..., format='full')`. `drafts().get` accepts
  `format` but NOT `metadataHeaders`.