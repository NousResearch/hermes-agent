# Reply-on-the-SAME-thread discipline (user-corrected 2026-09-29)

NDR corrected a standalone email: *"The email prepared just now for Dharmesh and Manish is not part
of the chain. It should be part of the same thread… reply all on the same thread."*

**Rule:** when NDR references a prior exchange — "we sent a series of emails about X", "put it back
on email", "reply on the same thread where I raised Y", "it should be part of that chain" — the
deliverable is a **reply-all threaded draft on the ORIGINAL thread**. A standalone draft is a
miss, even with identical recipients.

## Find the original thread first (search, don't ask)

DRAs on the draas vault mailboxes:
1. `in:sent <name>` and plain `<name>` in `google-draas` (subject usually lacks the employee's name — search sender keywords instead).
2. Recipient sweep: `(to:mdr@draas.com OR to:dharmesh.ranka1@gmail.com OR to:ddr@draas.com) newer_than:90d` etc.
3. **NDR often MAILS these from his OTHER identity `ndr@drahomes.in`, which is NOT in the gws vault.**
   The thread copy is still visible in the draas mailbox because a participant replied and the
   reply landed in draas — search by the participant's reply (`from:<participant>`), not by the
   sender. Worked example: the Hemant thread "Snags and repairs at 6th floor Prestige hermitage"
   was sent from ndr@drahomes.in on 24-Mar; found via Dharmesh's 25-Mar reply in draas.
4. If still missing after both sweeps, tell NDR exactly what you searched and ask for the thread
   subject/date — do NOT guess a thread.

## Reply-all recipient set (verbatim rule)

Take the LAST message of the thread, read its To + Cc, and reproduce that set minus NDR's own
addresses. Preserve the CC participants as Cc. Do not silently add/remove people.

**Sensitive-content flag:** a reply-all includes every thread participant. When the body is
sensitive (e.g. firing an employee) and a participant wasn't named by the user, say so in the
delivery message and let him opt out — worked 2026-09-29 (Khushroo was on the repair-thread Cc).

## Build + verify (raw MIME, drafts().create)

```
anchor := last message of thread (format='full' → metadataHeaders)
msg['In-Reply-To'] = anchor Message-ID
msg['References']  = anchor References (kept angle-IDs only) + anchor Message-ID
drafts().create(body={'message': {'raw': raw_out, 'threadId': THREAD}})
```

Verify via `drafts().get(format='full')`: labelIds contains DRAFT, `threadId` == original thread,
In-Reply-To set, To/Cc/Bcc correct, attachment present in payload parts. Then `drafts().delete`
any orphan standalone draft created earlier in the session (NDR authorizes this: "either delete
it and make a new draft or make this draft part of the same thread").

## Pitfall — the orphan

A standalone draft created before the thread was located stays in Drafts as a misleading
"new thread". Delete it the moment the threaded rebuild is verified. Confirm deletion by
attempting `drafts().get` (404/error = gone) and count drafts afterward.

**`drafts().delete` can return a transient 404 while the draft still exists** (verified
2026-10-16, Hermitage khata thread): the first delete raised
`404 Requested entity was not found` and the draft was *still listed* by `drafts().list()`
immediately after. Retry in a short loop (3 s apart) — the second attempt succeeded and
`drafts().get` then 404'd for real. Never report "deleted" off the back of the failed call,
and never conclude from the first 404 that the draft is gone.

## Merging two drafts in one thread into ONE email (verified 2026-10-16)

When NDR says *"just update the draft with this additional information … making a meaningful
[email] understanding both the content and creating one new content"*, the deliverable is
**one** draft in the thread, not two:

1. Read the OLDER draft in full (`drafts().get(format='full')` — walk `parts` for
   `text/plain` *or* `text/html`; an earlier draft may be plain-text while the new one is
   HTML). Keep every fact from it — NDR wants both contents understood and fused, not the
   newer one winning.
2. Rebuild as one HTML email: the older draft's detail as numbered headed sections, with the
   newer ask (options + vote request) folded in as its own section, plus the attachment.
3. **`drafts().update(id=<the older draft>, body={'message': {'raw': raw, 'threadId': THREAD}})`**
   — updates in place, keeps the id, no third draft.
4. `drafts().delete` the now-redundant newer draft, then list drafts to prove exactly ONE
   remains in the thread.