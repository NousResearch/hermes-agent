# Update an existing threaded draft: DELETE + RECREATE via bridge + bold-unanswered style

Verified 2026-09-10 (Amber MEP delay-email amplification). Applies when NDR asks to change/amplify a draft
already created with `draft_reply_create` (e.g. "add the work-order-no-timelines question to the Amber email").

## Why not in-place update
`drafts().update` with rebuilt raw MIME is a re-thread trap: rebuilding via MIMEText silently DROPS the
original Message-ID, Gmail assigns a new threadId and orphans the draft; it can also hit
`HttpError 400 "Message not a draft"`. Avoid the whole class by delete + recreate through the bridge,
which re-derives In-Reply-To/References/threadId automatically.

## Sequence (worked)
1. `gmail.users().drafts().get(userId='me', id=DRAFT_ID, format='metadata').execute()` → record To/Cc, note the threadId.
2. `gmail.users().drafts().delete(userId='me', id=DRAFT_ID).execute()` (old draft id from the earlier `draft_created` response).
3. Recreate with the SAME source message:
   ```python
   from tools.gws_skill_bridge import call
   r = call("draft_reply_create", service_name="google-draas",
            message_id="<same counterparty message_id as the original draft>",
            to="<same to>", cc="<same cc csv>", body="<NEW html body>", html=True)
   # r may be a dict {"status","draft_id","threadId"} OR a JSON string — parse defensively
   ```
4. Verify: `drafts().get(id=new_draft_id, format='full')` → `message.threadId == original thread`;
   walk payload for the new bolded text (`walk()` on `payload.body.data` then `parts[]`);
   `drafts().list(q='subject:"<kw>"')` and assert exactly ONE draft remains for that subject.

Note: the bridge's return type has been observed as both dict and string; do not assume `r.get(...)`
without checking `isinstance(r, dict)`.

## NDR's "bold the unanswered queries" follow-up style
When replying to a reportee's status email (Anbu pattern) where NDR's specific questions went unanswered:
- Acknowledge, then: "The following points from my earlier mail are <b>still unanswered</b> and I need them closed:"
- One `<b>bolded</b>` bullet per open query, each ending with an explicit marker
  (`NOT PROVIDED`, `NOT ANSWERED`, `NOT DELIVERED`, `— give a specific date`).
- Close "Please respond on this email." — keeps thread as single record.
- Delay/penalty emails: cite exact document refs (WO ref/date, fee, milestone), state the consequence
  ("we will apply the penalty..."), invite counter-reason in writing. Use `html=True` so the bold renders.

## Tracker reconciliation
After (re)creating drafts, log the NEW draft id into the tracker row Notes (append-only) with a dated
entry, e.g. `[2026-09-10] ... recreated as r-2387428385854081675 on same thread 1a055808539b1185 (verified, no duplicate)`.