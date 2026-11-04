# Reply-All Threaded Draft via Raw Gmail API (verified 2026-09-29)

Use when the user dictates a **reply-all** to a working email thread (internal project
tracking, contractor/architect chains) that must: (a) land in the SAME conversation,
(b) carry a correct In-Reply-To/References chain, (c) go to the full counterparty set,
and (d) stay a DRAFT (never send — hard rule).

Worked example: NDR replied-all on the "Ranka Udaya (Serenity Estate) – GFC Drawings…"
thread — To: Bhuvanesh + Sinchana, Cc: Prakash + Vinay + Bhuvanesh's second address,
after Bhuvanesh's message that closed the thread.

## Recipe

1. Resolve the service: `tools.gws_auth.build_service('gmail', 'v1', service_name=<resolved>)`
   in `/opt/hermes/.venv/bin/python3` with `sys.path.insert(0,'/opt/hermes')` and
   `HOME=/opt/hermes`. Assert `getProfile().emailAddress` is the expected mailbox.

2. Get the anchor = **the LAST message in the thread** (the one being replied to):
   `threads().get(userId='me', id=THREAD_ID, format='metadata')` → `msgs[-1]['id']`.

3. Fetch the anchor raw and parse Message-ID + References:
   `messages().get(userId='me', id=anchor_id, format='raw')` → b64url-decode →
   `email.message_from_bytes(..., policy=email.policy.default)`.

4. Build the recipient set from the ANCHOR's headers (not from memory):
   - `To` = the anchor's `From` **plus** everyone in the anchor's `To` except yourself
   - `Cc` = the anchor's `Cc` except yourself
   - Keep EVERY address the counterparty used — e.g. Bhuvanesh appears twice
     (`bk@findingform.design` and `mail.bskrishnan@gmail.com`); drop neither or the
     thread fragments for one of his inboxes.

5. Threading headers:
   - `Subject`: anchor's subject, ensure a single leading `Re:`.
   - `In-Reply-To` = anchor Message-ID (`<...>` form).
   - `References` = angle-bracket IDs extracted from the anchor's References, plus the
     anchor Message-ID appended if not already present. Sanitize: keep only
     `re.findall(r'<[^>]+>', ...)` tokens.

6. Create the draft — the `threadId` is what pins it into the conversation:
   `drafts().create(userId='me', body={'message': {'raw': b64url, 'threadId': THREAD_ID}})`
   (raw = base64.urlsafe_b64encode of the `email.message.EmailMessage` bytes).

7. VERIFY before handing over (drafts are cheap to check, embarrassing to skip):
   `drafts().get(id=draft_id, format='full')` → labelIds contains `DRAFT`,
   `threadId` equals the target, `To`/`Cc`/`Subject`/`In-Reply-To`/`References` headers
   present, body decodes to the dictated text; `drafts().list()` contains the id.

## Body conventions on these chain-closing replies

- Address each counterparty by first name in the body; the user's dictated tone is
  firm + urgent with a concrete ask ("by what time?", "chase X", "target: end of day").
- Keep the three-bullet rule for file lists; merge near-duplicate dictated categories
  (e.g. "GFC Revit AutoCAD drawings" + "GFC Revit drawings" → one bullet) and FLAG the
  merge in the delivery message so the user can veto before sending.
- Quote pipeline/approval context the user gives verbally (who needs to sign off, what
  approval is pending today) — it is the reason the message is urgent.
- Deliver to the user as draft-only: recipients, subject, body summary, verification
  evidence, and the "review and send when ready" line.