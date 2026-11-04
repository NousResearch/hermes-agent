# "I sent it to all of them and nobody replied" — the follow-up mail

Verified 2026-10-22, **"1500 East T3 BOQ"** thread `1a156633ba1531cc` (Bhuvanesh / Anbu / Kantesh /
Sinchana, Ranka Oasis villa BOQ review).

NDR dictated: *"There was an email I have sent to Bhuvanesh, Anbu and Kantesh and Sinchana maybe
regarding this BOQ … I just want to follow up and ask all of them whether they have had a chance to
take a look at it."* His recollection that some of them had **not** been copied was wrong. A
`format="metadata"` read of the thread showed his own earlier mail went:

- **To:** Bhuvanesh S Krishnan `<mail.bskrishnan@gmail.com>`, `BK@fnw.homes`
- **Cc:** Sinchana `<sgowda@draas.com>`, Kantesh `<kanteshbg@draas.com>`, Anbarasan `<pm2.blr@draas.com>`

Everybody was on it. They had simply not responded — which is a different mail.

## The rule

**Pull the headers of NDR's own last mail in the thread before writing a single sentence about who
received it.** Never assert "nobody was left out", "you were all copied", or "I sent this only to X"
from his memory. He commissions several mails a day across three mailboxes and his reconstruction of
any one of them is approximate.

One cheap call, before drafting:

```
gws_gmail_thread_get(thread_id=...)          # bodies
# then metadata for Message-ID / To / Cc:
t = g.users().threads().get(userId="me", id=THREAD, format="metadata").execute()
for m in t["messages"]:
    h = {x["name"].lower(): x["value"] for x in m["payload"]["headers"]}
    print(m["id"], h.get("date"), "->TO:", h.get("to"), "|CC:", h.get("cc"))
```

## What to do with the finding

- **Tell NDR in the hand-off, not just in the draft.** "First, a correction to your recollection: the
  original review *did* go to all four… so nobody was left off — they've simply not responded." He
  needs this because it changes what the follow-up is allowed to say.
- The draft then states the fact neutrally and moves straight to ownership:
  *"Has each of you had a chance to go through it? I have had no response so far, so I am setting out
  what has to happen today and who owns which part of it."*
- **Never** "I did not copy you earlier" / "you were not on the list" when the headers say otherwise.

## Recipient preservation on the reply-all

**Preserve the thread's own Cc list verbatim.** Anbu sat on `pm2.blr@draas.com` on this thread; the
standing "Anbu SENDS as `anbarasan@draas.com`" rule applies to a *fresh* mail to him, **not** to
reply-all parity. Adding `anbarasan@` alongside `pm2.blr@` risks double-delivery, and dropping
`pm2.blr@` silently removes him from a chain he was already on.

## Make the follow-up carry ownership, not just a nudge

A follow-up that only says "please review" gets another silence. The shape that works, verified here:

1. **Where it stands** — the numbers, so the recipient cannot claim they did not know the scale
   (84 line items, 57 at PARTIAL/NO/CHECK, 40 ASK ARCHITECT, M1–M8 reconciliation, N1–N11 areas).
   Link straight to the tab, not the workbook root.
2. **Per-person section** — name each person and what is theirs. "Bhuvanesh — today is the target to
   fix the BOQ. Not to review it once more. To *close* it." "Anbu — you drive the freeze with
   Bhuvanesh and the team."
3. **The commercial why**, in NDR's own framing — here: a definitive baseline, a practical doable
   cost, total cost structure; the Oasis go-to-market is value for money at a great ticket price;
   achieved by looking at cost in every way without compromising the intended look.
4. **What I need back by end of day** — reviewed yes/no · what is closed, item by item · what is
   open, who owns it, the date it closes.

Format: HTML, numbered sections with the blue `border-left:4px solid #2a6496; background:#f5f5f5`
header blocks (see SKILL.md "Template — HTML email structure"). A four-recipient coordination mail is
exactly the class that must not go out as plain running text.

## Pitfall

`drafts().list()` immediately after creating a draft can under-report — it returned only the newest
draft while two were live in different threads. Verify a specific draft with
`drafts().get(id=..., format="metadata")` before concluding a draft was lost or sent.
