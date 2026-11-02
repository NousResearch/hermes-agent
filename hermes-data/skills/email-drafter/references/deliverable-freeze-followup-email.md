# Deliverable-freeze follow-up: chase a consultant after NDR has already reviewed

Verified 2026-10-22, thread **"1500 East T3 BOQ"** (`1a156633ba1531cc`).
Class: NDR reviewed and *enhanced* a consultant's deliverable (BOQ / spec sheet / costing),
nobody responded, and he now wants a follow-up that (a) chases everyone, (b) puts a
**today** deadline on the consultant, and (c) names a colleague to own the freeze.

## Step 0 — reconcile his stated recipient list against the headers. Always.

He will say *"I sent it to Bhuvanesh, Anbu, Kantesh and Sinchana"* — sometimes he is
right, sometimes he has misremembered (same session, the Sarthak meeting mail really did
omit one person). **Pull the thread metadata first** and check To/Cc on his own last mail:

```python
t = g.users().threads().get(userId="me", id=THREAD, format="metadata").execute()
for m in t["messages"]:
    h = {x["name"].lower(): x["value"] for x in m["payload"]["headers"]}
    print(m["id"], h.get("message-id"), h.get("from"), "TO:", h.get("to"), "CC:", h.get("cc"))
```

Then **report the reconciliation in the hand-off**: *"a correction to your recollection —
the original review did go to all four; they simply have not responded."* He accepts this
without friction and it changes the draft (you cannot write "I am resending to those who
were missed" if nobody was missed).

Be careful how the correction lands: correct the RECORD, never imply he was careless.

## Step 1 — routing

Anchored on **his own last review message** on the thread, reply-all with the thread's
**exact** To and Cc. Do not upgrade an address just because the person "sends as" a
different one elsewhere (see the address-parity pitfall in `SKILL.md`).

## Step 2 — shape that landed: five numbered HTML sections

1. **Where it stands** — the numbers of the review, stated flatly, with a link to the
   review tab. Counts, not adjectives: *84 line items · 57 of 84 in PARTIAL/NO/CHECK ·
   40 ASK ARCHITECT · M1–M8 reconciliation · N1–N11 area schedule*.
2. **<Consultant> — today is the target to fix it.** Then *"Not to review it once more.
   To close it:"* and the three specific closures. Offer the low-friction channel:
   *"Voice notes are fine — I will process them into the sheet myself."*
3. **<Colleague> — you drive the freeze with <consultant> and the team.** Name Kantesh
   and Sinchana as working the drawings and quantities. This is the ownership sentence.
4. **Why this matters now** — the commercial reason, in NDR's own framing (below).
5. **What I need back by end of day** — three one-line answerable asks: reviewed yes/no ·
   what closed, item by item · what is open, who owns it, the date it closes.

Open with *"Has each of you had a chance to go through it? I have had no response so far,
so I am setting out what has to happen today and who owns which part of it."* — this is
the chase without a complaint.

## Step 3 — the two framings NDR dictated, and why they work

- **"A rate is not a rate without the material and the method."** NDR's own words:
  *all elements I raised as missing, and corrections on material and methodology detail,
  which are essential to arriving at a rate, have to be frozen.* Render as: *"A number
  without the material and the method of execution behind it is not something we can
  commit to."*
- **Cost structure → go-to-market.** *Definitive baseline, practical doable cost for our
  villas, total cost structure*, then the pricing logic: value for money at a great
  ticket price, a sharply priced product, achieved by looking at cost in every way
  **without compromising the overall look we intend to deliver**. Land on the two
  versions off one frozen BOQ: **base version** (what we build to sell) and **marketing
  version** (what we build to show).

## Garbles seen in this class

- **"Balu"** → **Anbu**. Dictation mangles "Anbu" into "Balu"/"Anbu"/"Anbo"; take Anbu
  when the sentence is *"<name>, you need to work with <consultant> and the team"*.
- **"Paya, Rav, a Grit Ticket Price"** → *"value for money at a great ticket price"*.
  Render the sense, then **flag the garble in the hand-off with your reading** and offer
  to swap in his exact phrase. Do not silently invent brand language.
- **"Aamir, Kandil"** → resolved nowhere. Leave the name out of the body and say so.

## Rules

- HTML, inline styles, numbered blue-header sections (`border-left:4px solid #2a6496;
  background:#f5f5f5`) — this is a multi-point internal coordination mail.
- Numbers are load-bearing. Never write "several" where you have 84 / 57 / 40 / 12.
- Every section must end in something one person can answer in one line.
- Report back to NDR: routing, the recipient reconciliation, and every garble you resolved
  or dropped.
