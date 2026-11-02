# Thread Response-Status Audit ("has X replied?")

Answers the recurring question: **"I emailed / WhatsApped <person> about <matter> — have they responded at all?"** Produces a dated timeline + a verdict, then offers (doesn't auto-write) the next nudge.

## Workflow

1. **Identify the person** — `contact_resolver(query=<name>, context=<project>)`. Capture ALL emails + phones; people routinely have 2 emails (e.g. Raghu Iyer: `oz.iyer@gmail.com` + `rmiyer@bitanz.com`). A reply can arrive on either.

2. **Locate the thread** — two routes:
   - `session_search(query="<name> <topic>")` first: past sessions usually carry the thread subject, an `#inbox/<id>` link, or the draft id.
   - Gmail fallback: `messages().list(q='(to:<email> OR from:<email>) ...')`.

3. **CRITICAL PITFALL — inbox/draft URLs carry a MESSAGE id, not a THREAD id.**
   Past sessions surface links like `https://mail.google.com/mail/u/0/#inbox/1a04b9f165bbdef8`. That long hex is a **message id** (confusingly, it looks like a thread id). Calling `threads().get(userId='me', id=<that>)` 404s with `Requested entity was not found.` — observed live Sep 2026.
   Fix:
   ```python
   m = svc.users().messages().get(userId='me', id=MSG_ID, format='full').execute()
   tid = m['threadId']
   th = svc.users().threads().get(userId='me', id=tid, format='full').execute()
   ```

4. **Read the thread chronologically** — print From / Date / Subject / body preview per message. Build the chain: our send → their ack → our reminder → silence.

5. **Sent vs draft verification** — `threads().get()` returns ONLY real messages; a draft created with a `threadId` does NOT appear in the thread. If the "reminder" shows as the last thread message, it was actually **sent**. When in doubt, `drafts().list()`.

6. **Cross-account reply sweep** — replies can sit on any of NDR's accounts. Sweep all three rather than assuming:
   ```python
   for svc_name in ('google-draas', 'google-gmail', 'google-ahfl'):
       svc = build_service('gmail', 'v1', service_name=svc_name)
       res = svc.users().messages().list(userId='me',
           q='from:(<email1> OR <email2>) after:<YYYY/MM/DD>', maxResults=20).execute()
   ```
   Zero hits on all = no reply since that date. Report as *checked everywhere*, not assumed.

7. **WhatsApp leg** — `whatsapp_link` deep links never send, so there is no delivery/reply record server-side. Recover the GENERATION date from session memory:
   `session_search(query="<name> whatsapp_link", role_filter="assistant,tool", sort="newest")`
   — the tool-call record carries the timestamp; the session date is the answer. Report honestly: "link generated <date>; no record of delivery or reply."

## Deliverable shape (Nishant's preference)
- Lead with the verdict: **"He has NOT responded since <date>"** / "He replied on <date>: <quote>".
- Then the compact timeline (date → who → one-line content), then the WhatsApp-date answer.
- End with an offer to draft the next nudge — do NOT auto-draft a follow-up when the user only asked to check.

## Example (Sep 2026 — Raghu Iyer / Ranka Amber SSA registration)
- 28 Aug: NDR email — asks Raghu to fly in for the one-day SSA registration (bank won't release project finance)
- 29 Aug: Raghu — "Thank you for your mail. I will review over the weekend and revert." (his last word)
- 1 Sep: NDR reminder SENT (thread-verified; CC Manohar Singh + Prakash Singh; CC application + 4-slab/₹4-4.5Cr funding case)
- WhatsApp deep link generated 28 Aug into the Ranka Amber landowner group; no reply record
- Verdict: no response for 8 days past his own promise — nudge warranted