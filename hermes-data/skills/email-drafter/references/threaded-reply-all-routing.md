# Reply-all routing on an existing thread (don't start a fresh mail)

> **Read this file, not SKILL.md.** The parent SKILL.md is ~100 KB and `skill_view` on it returns a
> persisted temp file instead of inline content, costing a turn. For any threaded reply, these two
> references are enough: this one (routing / headers / threadId) and
> `references/co-director-deadlock-thread-reply.md` (reply shape for the family-director wind-down
> thread). Read them straight off disk — `find / -name "<file>.md" -path "*email-drafter*"` — or pull
> the `content` field out of the persisted JSON with `json.load()`.

## Trigger
NDR says any of:
- "draft that as a reply-all to the email X already sent"
- "he sent it as part of some chain of feedback — the conversation should not be lost"
- "continue that thread"

If the subject matter is already in a Gmail thread, **never** create a standalone draft. A
standalone mail loses the chain, the quoted history and the participants.

## Step 1 — find the thread, not the message
Colleagues route through a thread whose subject has nothing to do with the topic, so search by
**author + topic words**, not by subject:

```
gws_gmail_search(query="from:anbarasan@draas.com (drawing OR plan OR changes OR feedback)")
gws_gmail_search(query="<project name>")     # second pass, catches the external side
```

Anbu SENDS as `anbarasan@draas.com` even though the contacts sheet says `pm2.blr@draas.com`.
Addresses always come from Gmail headers.

## Step 2 — read every message's participant + Message-ID
`gws_gmail_thread_get` gives the bodies but not the `Message-ID` header. For headers, one cheap
call via execute_code:

```python
t = g.threads().get(userId="me", id=THREAD_ID, format="metadata").execute()
for m in t["messages"]:
    h = {x["name"].lower(): x["value"] for x in m["payload"]["headers"]}
    print(m["id"], h.get("date"), h.get("message-id"), h.get("from"), "->", h.get("to"))
```

## Step 3 — anchor on the LAST message that reached the external party
This is the pitfall that bites. The internal colleague's review mail usually went **only to NDR**
(Anbu → ndr@draas.com), so a literal reply-all to it reaches nobody outside the company. The
anchor must be the tail message that was actually addressed to the architect / lawyer / vendor —
typically the colleague's *forward* of that review.

Recipe:
- **Anchor** = the thread tail addressed to the external party.
- **To** = external recipients from that message.
- **Cc** = the internal author(s) of the review (so they see it land) + any internal coordinator.
- `In-Reply-To` = anchor's Message-ID.
- `References` = every prior Message-ID in the thread, oldest first, space-separated.
- `drafts().create(body={"message": {"raw": raw, "threadId": THREAD_ID}})` — pass **both** raw and
  threadId; raw alone will fork a new thread.

## Step 4 — verify before reporting
```python
m = g.users().drafts().get(userId="me", id=DRAFT_ID, format="full").execute()["message"]
# assert m["threadId"] == THREAD_ID, and check To / Cc / Subject / In-Reply-To headers
```
Say in the reply which thread it now sits in and why that anchor was chosen.

## Special case — instructing ONE colleague on a thread whose To-list was wide

Verified 2026-10-20, "Deliverables checklist - RANKA AMBER and RANKA OASIS"
(thread `1a11006090f5ce2a`): Sinchana's original mail went To Bhuvanesh (both
addresses), Anbu, Kantesh **and** NDR — a flat To-list, no Cc. NDR then said
*"let's send Sinchana a message… respond to the email where she shared the
sheet link itself."*

The instruction is aimed at **one internal colleague**, but the external
architect must see the priority ask inside it. Recipe used:

- **To** = the internal colleague being instructed (`sgowda@draas.com`).
- **Cc** = the external party from the original To-list (both of Bhuvanesh's
  addresses) + the other internal staff who were on it (Anbu, Kantesh).
- Anchor = the thread's tail message; `In-Reply-To`/`References` = its
  Message-ID; `drafts().create(..., threadId=...)` with **both** raw and
  threadId.

This reads as "instructions from the boss to the coordinator, with the
consultant informed" rather than "boss replying to the consultant", which is
what a reply-all would have implied. Tell NDR in the hand-off which way you
routed it and why, and offer to drop the external party to a separate mail.

Do **not** strip the consultant off such a thread just because the ask is
addressed to the colleague — he is the person who has to act on the timeline,
and NDR's brief was explicitly that the colleague *sits with him*.

## Pitfalls
- **The bridge sets `References` to the parent's Message-ID ONLY — not the ancestor chain.** On a
  thread deeper than two messages (`A → B → C`, replying to `C`) the draft comes back with
  `References: <C>` where Gmail itself would write `<A> <B> <C>`. `In-Reply-To` and `threadId` still
  file it correctly in Gmail, but the chain is incomplete for every other client and for anyone
  reading the raw. Fix it in place, then prove the body survived:

  ```python
  d   = g.users().drafts().get(userId="me", id=DRAFT_ID, format="raw").execute()
  msg = email.message_from_bytes(base64.urlsafe_b64decode(d["message"]["raw"]))
  body_before = <decode the text/plain part>
  del msg["References"]; msg["References"] = " ".join(all_message_ids_oldest_first)
  g.users().drafts().update(userId="me", id=DRAFT_ID,
      body={"message": {"raw": base64.urlsafe_b64encode(msg.as_bytes()).decode(),
                        "threadId": THREAD_ID}}).execute()
  # re-read format="raw": assert body_after == body_before, References == full chain,
  # labelIds == ['DRAFT'], threadId unchanged, and only ONE draft in the thread.
  ```

  The message id changes on update; quote the new one. Verified 2026-10-27, thread
  `1a085f3c1c1c409e`.
- **A dictated recipient name that finds nothing in Gmail is usually a mis-hearing, not a missing
  mail.** NDR said "Asauddin"; the mailbox is `h assudani <assudanirtr@gmail.com>` (Hirendra
  Assudani). `asauddin` / `asaduddin` / `asaudin` / `sauddin` all returned nothing or irrelevant
  hits. Search **project or subject words** (`newer_than:60d serenity`) and read the From line off
  the hits — then take the address from those headers, never from the dictated spelling.
- **Address parity beats the address map.** On a reply-all inside a thread NDR himself sent,
  preserve that thread's exact To and Cc even where the address map says the person "sends as"
  a different address (Anbu is `pm2.blr@draas.com` on his own BOQ review but
  `anbarasan@draas.com` elsewhere). Swapping adds a second copy to an aliased mailbox and
  breaks the parity NDR reads down the thread. Use the "better" address only for a NEW mail,
  or when he asks. Verified 2026-10-22, "1500 East T3 BOQ" (`1a156633ba1531cc`).
- **His recollection of who he sent it to is not the record.** Before writing *"I am
  resending to those who were missed"* or *"as I sent to X, Y, Z"*, read To/Cc off his own
  last message via the metadata call in Step 2. He is sometimes right and sometimes has
  misremembered. Correct the RECORD without implying carelessness — *"a correction to your
  recollection: the original review did go to all four; they simply have not responded"* —
  and say so in the hand-off, because the correction changes the draft.
- **Don't nest triple quotes when building the raw MIME.** A builder script composed inline
  via `execute_code` that itself contains triple-quoted HTML dies with
  `SyntaxError: invalid syntax` on the first HTML line — the inner `'''` closes the outer
  string. Write the builder to `/tmp/mk_<thing>.py` with `write_file`, run it with
  `terminal`, then run a separate `/tmp/ver_<thing>.py` that re-reads the draft with
  `format="full"` and prints From/To/Cc/Subject/In-Reply-To plus asserts `threadId`.
  `drafts().list()` has been seen to return fewer drafts than exist moments after
  creation — confirm by fetching the draft id directly, not by list count.
- **`gws_gmail_get` returns an empty body for a message that is still a DRAFT.** To read an
  existing draft's content use `drafts().get(id=..., format="full")` and walk
  `payload.parts`, base64url-decoding `body.data`. Do not conclude the draft is blank.
- Reply to the tail, not the middle: Gmail will still file it correctly when threadId is passed,
  but the quoted history reads backwards.
- Delete the superseded standalone draft once the threaded version exists, so there is only one
  live draft and NDR cannot send the wrong one.
- If the thread's external parties are on Yahoo / a firm domain, keep their display names
  (`Arvind Jain Architect <...>`) — Gmail's reply-all parity matters less than the human reading it.

## Replying when NDR's OWN mail is the thread tail — the native reply tool drops the Cc

Verified 2026-10-24, "Call for a meeting" (`1a1178935be29c75`). NDR had already sent the agenda
mail that morning; the ask was a short follow-up in the same thread to Dharmesh + Manish.

- **`gws_gmail_reply_draft` takes no `to` / `cc` at all.** It reply-alls whatever the parent
  message's headers say. Anchor it on NDR's own last sent mail and the draft comes out
  **`To: ndr@draas.com`** with no Cc — a mail to himself. Verify with
  `drafts().get(format="raw")` and read the To:/Cc: lines; `gws_gmail_get` shows `to` as the
  sender and hides Cc entirely, so it is not a verification.
- **Use the bridge instead whenever recipients must be explicit:**
  ```python
  from tools.gws_skill_bridge import call
  print(call("draft_reply_create", service_name="google-draas",
             message_id="<the counterparty's last message id>",   # anchor, not NDR's own
             to="dharmesh.ranka1@gmail.com", cc="mdr@draas.com",
             body=body, html=False))
  ```
  This sets `To`, `Cc`, `In-Reply-To`, `References` from the anchor and keeps the threadId, and
  it still never sends. Anchoring on the counterparty's message (not NDR's reply to it) is what
  makes reply-all land on the other party rather than on NDR.
- **Delete the bad draft before creating the replacement**, and confirm afterwards that exactly
  one live draft remains in the thread.

## The thread contains a question addressed to someone who is NOT on it

Common shape: the colleague writes *"FOR SARTHAK — please confirm the credit into Kotak A/c
0051732561"*, but Sarthak was never copied and has no address in the contacts or employees sheet.
NDR's instruction will still be *"do a reply-all — not only to X but to whoever is on it."*

- **Do not invent or guess an address to satisfy the ask.** An unresolved address silently drops
  the one person whose action matters, or bounces the whole reply.
- **Preserve the thread's exact To/Cc** and route the missing person's action **through a mailbox
  that is already on the thread** (here `admin3.blr@draas.com`), naming them in the body
  ("Sarthak, through admin, should pull the … statement"). The work gets assigned without
  changing the participant list.
- **Flag it in the hand-off to NDR**, with the offer to amend the draft: *"Sarthak isn't on the
  thread and has no email in the contacts/employees sheet — I routed that action through
  admin3.blr and named him. Give me his address and I'll copy him in."* He resolves it or accepts
  the routing; either way he is not surprised at send time.

## When the counterparty asked permission for something that needs none

If NDR's dictation starts *"I am not sure I understand your question, but I request you to…"*,
the counterparty has asked for a decision that was already his to take. The reply shape is:

- **Line one answers the over-ask and dismisses it** — *"you do not need to wait for anything
  further from me"* — so the colleague stops queuing on an approval that will never be needed.
- **Then the substance, numbered**, each section carrying the instruction and the reason.
- **Close by releasing him**: *"Nothing above needs a further approval from me."* An internal
  colleague who over-asks will over-ask again unless the reply closes that door explicitly.

Verified 2026-10-27, Ranka Udaya Plot No. 31 (`1a17cabc7b3b35a4`): Bharat asked permission merely
to execute an Agreement for Sale on stamp paper. NDR's answer granted it, then went further —
collect the Rs 44,670 balance, execute the Sale Deed reciting the **full** consideration, hand it
over, registration optional.

## Pitfalls (delivery mechanics)

- **MIMEText hard-wraps plain body lines at ~72 chars.** A long sentence comes back visibly broken
  mid-phrase (`for whichever\r\nday we fix`) in the delivered mail. Keep every body line of a
  bridge-built draft **≤ ~60 characters** — break on clause boundaries you would accept as line
  breaks in the mail — and re-read the raw draft to confirm no sentence was split.
- **Don't run the same builder subprocess twice.** `subprocess.run(...)` once for stdout and again
  for stderr *executes the script twice* — the first run created a draft, the second created a
  duplicate (and one copy of the delete+create pair timed out at 300 s). One `run()` call, then
  print `r.stdout` and `r.stderr` from that same result object; add an explicit `timeout=` so a
  hung vault call surfaces as a timeout instead of a 300 s script kill.
- **`drafts().get(format="raw")` is the only check that catches all of this** — To, Cc,
  In-Reply-To, References and the body text in one read.

## See also
- **Same-day internal team update** ("send an update email to everyone on the same thing") — one
  name-sectioned mail carrying two dictated person-specific asks, participant list taken from the
  original mail's headers rather than the dictation, the "nobody has replied yet, so ask" rule,
  uncorroborated third-party facts, and the bridge's top-level `draft_id`:
  `references/internal-team-same-day-update.md`.
- **The reply must carry a file** (invoice, WhatsApp split, sheet) — none of the draft tools take an
  attachment, so build the multipart MIME and create a DRAFT: routing + code + the byte-size proof,
  and the `Re: Fwd:` subject trap: `references/threaded-reply-with-attachment.md`, with a runnable
  builder at `scripts/draft_reply_with_attachment.py`.
- **Landowner / plot-buyer status reply** — an external counterparty replying to a commitment NDR
  put "on record", asking for an update: apology + blocking fact + what is already done + the new
  expectation, the salutation/sign-off copied from the thread, and the deadline-arithmetic rule
  (never restate a slipped deadline in a figure the thread's own trail mail contradicts):
  `references/external-counterparty-status-apology-reply.md`.
- **Co-director / sibling deadlock thread**
  already reviewed and enhanced it: five-section HTML shape, today-deadline wording, naming
  the colleague who owns the freeze, the "a rate is not a rate without the material and the
  method" framing, and the dictation garbles seen in this class:
  `references/deliverable-freeze-followup-email.md`.
- **Co-director / sibling deadlock thread** — accusatory reply-all between family directors
  (DRA Projects wind-down), NDR's five-move reply shape, the deliberate "none of us" concession,
  and the office-sit-down ask: `references/co-director-deadlock-thread-reply.md`.
  Three variants — **A** accusation-with-substance (rebut, five moves), **B** vague one-liner /
  "agenda" deflection (define the agenda, demand specifics, give three disagreement rules),
  **C** the returning co-director who offers dates and asks to *set the agenda first* (answer the
  date in line one, concede the points are valid, name the third party's pending plan, defer the
  office/vakalat question into the sit-down). Pick by what the other side actually sent.

## Worked example (2026-10)
"Ranka Northstar Sanction" thread `1a114e58152cdc40`: Shrinidhi's 7-Oct drawing → Sinchana's
chase → resent drawing → Anbu's 16-point review of 9-Oct (to NDR only) → Sinchana's 14-Oct
forward of that review to Arvind + Shrinidhi. The reply-all was anchored on Sinchana's forward,
To: Arvind + Shrinidhi, Cc: Anbu + Sinchana, subject `Re: Ranka Northstar Sanction`.
