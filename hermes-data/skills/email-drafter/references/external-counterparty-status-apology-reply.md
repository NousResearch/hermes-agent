# Apology + status reply to a landowner / plot buyer on a commitment thread

## Trigger
An external counterparty — landowner, plot buyer, farmer-collective member — replies to a mail in
which NDR put a delivery commitment **on record**, typically asking only *"may I request for an
update on the same"*. NDR dictates the answer: an apology for the silence, the true blocking
reason, what is already done, and what happens next.

Worked case (27 Oct 2026): Air Cmde Hirendra Assudani, plot buyer in Serenity Hill View, replied to
NDR's 9 Sep "Development Works – On Record" mail that had committed site start from 1 October.
Thread `1a085f3c1c1c409e`.

## Step 1 — the dictated name may not be the mailbox name
NDR dictated "Asauddin". Nothing matched in Gmail under `asauddin`, `asaduddin`, `asaudin`,
`sauddin` — the man is **Hirendra Assudani**, mailbox `assudanirtr@gmail.com`, display name
`h assudani`. A `from:`/name search is therefore the wrong first move for a garbled name.

Search by **project or subject words** instead, then read the sender off the hits:

```
gws_gmail_search(query="newer_than:60d serenity")
# → h assudani <assudanirtr@gmail.com>  "Re: Serenity Hill View: Development Works - On Record"
```

Cross-check with the contacts sheet / `entity_resolver` only after you have the thread — the sheet
name is the formal name, the mailbox is what the reply must go to, and the two often differ.

## Step 2 — read the whole thread before writing a word
`threads().get(format='full')` on the matched thread gives, for each message: Date, From, To, Cc,
Subject, Message-ID, In-Reply-To, References and the plain-text body. Three things come out of it
that you cannot get anywhere else:

- **How NDR addressed him before.** Here `Dear Air Commodore Assudani,` and the sign-off
  `Warm regards, / Nishant Ranka / DRA Group`. Copy that salutation and that sign-off; do not
  demote a service rank to "Dear Mr Assudani" and do not invent a new signature block.
- **Exactly what was promised and when.** The 9 Sep mail committed *work from 1 October* and
  completion in *6 months plus 2 months grace*. That is the yardstick the counterparty is using.
- **Who else is on it.** Reply-all parity, per `references/threaded-reply-all-routing.md`.

## Step 3 — the deadline arithmetic rule (the one that matters)
**Never restate a slipped deadline in a figure that the thread itself contradicts.**

NDR's dictation said *"we are already about a week past our own deadline."* The thread's own trail
mail says 1 October; the reply is being written on 27 October. To the man quoting that trail mail
back at you, "a week" reads as either carelessness or an attempt to soften it, and it undermines
the rest of the status.

Two acceptable forms:
- **Accurate and explicit:** "we are now close to four weeks past the 1st October date".
- **Relative, no figure:** "we are some weeks past the 1st October date we had committed to you".

Use the relative form by default — it is honest, it cannot be caught out, and it avoids opening
with a number that invites a complaint. Then say plainly *"I do appreciate your patience"* and move
straight to the new fact.

Where the dictation date is right, keep it verbatim. The check is one subtraction: dictation
slippage vs (today − the date in the thread), both read in the same turn. If they disagree, the
thread wins — and **tell NDR in the hand-off** that you changed it and why, so he is not surprised
when he opens the draft.

## Step 4 — shape of the body
Five short paragraphs, plain text, lines ≤ ~60 characters (the bridge hard-wraps at ~72 and will
split a clause mid-phrase):

1. **Thanks + apology, one line each.** *"Thank you for your mail, and apologies for not sending
   you an update earlier."* He asked for an update; apologise for the silence, not for the question.
2. **The blocking fact, concretely.** Where the file physically is, who is holding it, that it is
   taking longer than expected, that discussions are ongoing, expectation "any time now", and the
   honest *"we are at the last leg, and this is the only thing we are waiting on."*
3. **What happens the moment it clears + what is already done.** Work starts at site; teams ready;
   contractor advances already issued; *"everything now is pending on that one clearance."*
   (NDR's own phrasing: "advances have been issued and everything is pending".)
4. **The date, handled per Step 3**, plus appreciation of his patience.
5. **"I will keep you posted"** and the standard sign-off.

Do not add sections, headings, bullet lists, tables, or a list of the seven development scope items
from the original mail — this is a letter to a person, not a status report. Do not offer a revised
completion date unless NDR gave one; every date in this class of mail becomes a new commitment.

## Step 5 — routing
Reply-all on the counterparty's latest message:

- **To** = the counterparty exactly as his mail shows it (`assudanirtr@gmail.com`).
- **Cc** = whoever was Cc on that mail (`nishantprakash1@gmail.com` — Nishant Prakash Zara). No
  additions, no Bcc, nothing to the internal team unless NDR asks.
- **Anchor** = his latest message, never NDR's own earlier mail (anchoring on your own mail gives
  `To: ndr@draas.com`).
- **Subject** = `Re: <his subject>` unchanged.

```python
from tools.gws_skill_bridge import call
call("draft_reply_create", service_name="google-draas",
     message_id=ANCHOR_MSG_ID, to="assudanirtr@gmail.com",
     cc="nishantprakash1@gmail.com", body=body, html=False)
```

## Step 6 — verify, then report
`drafts().get(id=DRAFT_ID, format='raw')` and read back **To / Cc / Subject / In-Reply-To /
References** plus the decoded plain body; assert `threadId` and `labelIds == ['DRAFT']`. Confirm the
thread now holds exactly one DRAFT message. The `References` chain needs an extra fix on threads
deeper than two messages — see the first pitfall in
`references/threaded-reply-all-routing.md`.

In the hand-off give him: thread subject, To line, the one wording change you made (and the reason),
and confirmation that nothing was sent.

## Pitfalls
- **Do not answer this from the internal team's status update alone.** The internal note said
  "about a week past our own deadline" because it was written around 8 October; repeating it in a
  mail dated 27 October is the exact error this reference exists to prevent.
- **"Advances have been issued" is not "advances have been paid".** Keep NDR's verb. If he says
  issued, write issued; a payment claim to a buyer is checkable and he may not have the UTRs.
- **No gratitude inflation and no new promises** — no "thank you for your understanding", no "we
  assure you", no date. One apology, one status, one expectation, one promise to update.
- **A same-day reply still needs the apology**: the lapse is the month of silence since 1 October,
  not the hours since his mail.
