# Same-day internal team update ("send an update email to everyone")

Verified 2026-10-31, DRA Realty Saturday Review thread `1a18d15f4fc5cefd`.

## Trigger shape

NDR dictates what reads like TWO separate messages — one addressed to person A, one to person B —
and closes with an instruction to the assistant:

> "Accordingly, send an update email to everyone on the same thing: X is not in the office today,
> so I am doing this one-on-one with A … separately I am doing one with B when he comes in at 1 …
> Then I will send an email out to everyone so everyone is aware of what is going on after the
> meetings."

This is ONE mail to the whole team, not two mails and not a WhatsApp to each. The two dictated
"messages" become **sectioned, name-headed blocks inside a single update**, because the mail's job
is to put everyone (including the people being talked about) on the same page before the day's
meetings start.

## Reader model — write for the team, not for the addressee

- Every section is headed by the person's name in caps (`BHARAT - NOT IN OFFICE TODAY`,
  `GAURI - ONE-ON-ONE, 11:30 AM TO 12:00 NOON`, `ANBU - 1 O'CLOCK, ENGINEERING DELIVERABLES`).
  The absent person and the person being chased both read it; a headed block reads as "this is
  where things stand", not as a memo addressed over their head.
- Keep every dictated number and third-party name **verbatim** — `Rs 2 crore a month`, the `1500`
  target, `Bhuvanesh`, `Kantesh`, `Bhuvanesh's drawings`. Those are the accountability hooks the
  thread will be measured against later. Do not generalise them into "monthly target".
- Close with the promise NDR actually made: *"I will send a mail out to all of you once these
  meetings are done, so everyone is aware of what is going on and where each item stands."* That
  line is what makes the mail a schedule rather than a status report — it commits a follow-up.
- The chase (value-of-work data not flowing, overhead exceeding work delivered) stays in, but
  framed as the reason the data is needed ("that has to be remedied now"), never as a rebuke of the
  individual in front of the team.

## Sequence that worked

1. **Find the source thread** by author + topic (see `threaded-reply-all-routing.md`), never by
   guessing the subject. Here: `subject:(all-hands OR all hands)` returned 0 — the real subject was
   "Saturday Review — what we discussed last week, and the agenda for tomorrow". `from:me
   newer_than:2d` found it in one call.
2. **Read who is actually on it, off the headers.** The dictated list is not the participant list:
   NDR named six people; Sinchana was also on the original mail (she joins online). A reply-all in
   this thread keeps her — and the hand-off says so explicitly (*"Sinchana kept because she is on
   the thread and joins online"*) so NDR can drop her if he disagrees. Never silently narrow a
   reply-all to the names in the dictation.
3. **Check for replies before writing a single line.** `newer_than:1d -from:me` and
   `from:(a@draas.com OR b@draas.com) newer_than:3d`. If nobody has confirmed a time, the mail must
   **ask** for confirmation ("please confirm that 11:30 - 12:00 is okay, and tell me what time you
   are coming in") — it must not open with "as discussed" or assume a slot was agreed.
4. **Check WhatsApp for the scheduling fact the mail depends on** — the person's arrival time is
   often only in WhatsApp (Anbu's daily DPR messages), and the answer is frequently absent. Note in
   the hand-off that the arrival time was *not* supplied, so the mail is genuinely the first ask.
5. **Draft as an in-thread reply** anchored on the counterparty's tail message (the bridge, with
   explicit `to=`), not a fresh standalone mail — the chain and quoted history are the point.
6. **Companion WhatsApp only where a live chat exists.** Anbu has a direct thread → draft the same
   timing ask there. Gauri has **no WhatsApp chat in the bridge at all** → email is the only channel;
   say that in the hand-off rather than inventing a number or asking NDR for one he already knows.

## Uncorroborated facts about third parties

NDR dictated "Bharat is not in the office today because he has an event at Hosur". Nothing in mail
or WhatsApp corroborated it. Write it as his statement of fact, exactly as dictated — do not attach
a verification claim ("confirmed with him"), and do not soften it. Flag the gap in the hand-off:
*"No trace anywhere of Bharat's Hosur event — it is your word only as of now."* He either accepts it
or supplies the source.

## Mechanics (bridge, not native reply tool)

```python
sys.path.insert(0, "/opt/hermes")
from tools.gws_skill_bridge import call
r = call("draft_reply_create", service_name="google-draas",
         message_id=ANCHOR_MSG_ID,          # the counterparty's tail message
         to="A <a@draas.com>, B <b@draas.com>, ...",   # explicit, full list
         body=BODY, html=False)
draft_id = r["draft_id"]       # TOP-LEVEL keys, not r["draft"]["id"]
```

Returned shape: `{"status": "draft_created", "draft_id": ..., "message_id": ..., "threadId": ...}`.
Reading `r.get("draft", {}).get("id")` prints `None` and costs a turn — take `r["draft_id"]`.

- The bridge writes `References` as the **parent Message-ID only**. On a 3-message thread, rewrite it
  to the full chain oldest-first via `drafts().update(...)` and re-read `format="raw"` to prove the
  body survived (`threaded-reply-all-routing.md` has the exact loop).
- The ≤60-char line rule in that file applies to **hand-built `EmailMessage` MIME**, not to the
  bridge. Bridge-drafted bodies survived intact with lines up to 74 chars, breaking on clause
  boundaries that read naturally in the mail. Check `max(len(l) for l in body.splitlines())` and read
  the stored body back; don't pre-wrap to 60 for no reason.
- Verify with `format="raw"`: To, Cc, Subject, In-Reply-To, References, `threadId`, and
  `labelIds == ['DRAFT']`. Then say "not sent" in the hand-off.

## Hand-off content

Say who was found on the thread, who has replied and who has not, which dictated facts are
uncorroborated, which person has no WhatsApp channel, and the one or two judgement calls the draft
made on his behalf (extra recipient kept, channel used). NDR reads the calls, not the prose.
