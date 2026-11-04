# Outbound denial / counterparty-position mail — attribution discipline

Verified 2026-10-24, Ranka Iris / Shankar Electricals (WO 616) thread, NDR → Bhavik Ranka,
Cc Piyush + Sanjeev Ranka + Manish + Dharmesh.

## The correction (NDR, verbatim)

> "I don't want the email to read saying that Anbu has come back and Anbu is denying it. No, I just
> want it. We don't want to even say that we formulated it out of Anbu's response. We just want to
> say, Bhavik, we have spoken to Nanjesh slash Shankar Electrical and he is denying all of this and
> these are his [replies] below. Okay, let's put it that way. Let's not bring Anbu into this picture."

## Why this class recurs

NDR frequently routes a **colleague's** written analysis (Anbu's reconciliation, Bhavik's snag list,
Sarthak's numbers) into an **outbound** mail to a counterparty or to another colleague. The natural
drafting instinct — "X says Y" — is wrong twice over:

1. It exposes the internal colleague to the counterparty as the author of the position, which NDR
   does not want. NDR owns the position; the colleague only supplied the working.
2. It reads as second-hand. NDR's mail lands harder as **"I have spoken to him and he is denying
   it"** — first person, first hand, on the record.

## Rewrite table

| Wrong | Right |
|---|---|
| "Anbu has now come back on the reconciliation…" | "I have spoken to Nanjesh and to Shankar Electrical and put every one of the points to him." |
| "Anbu does not accept this debit." | "He is denying all of it." |
| "Anbu's position: earthing had to be provided…" | "His position: earthing had to be provided…" (he = the counterparty) |
| "Anbu's reconciliation is attached." | "The reconciliation on record is attached." |
| "He is getting your facts in writing." (when it was the colleague) | "He has given no factual basis at all and is disputing it outright." then ask the RECIPIENT for the facts |

Grep the finished body for the colleague's first name, surname and any `X's` possessive before you
create the draft. Introspect the stored draft too — a quick
`body.count("Anbu")` on the draft's `format='full'` payload catches prose you thought you'd fixed.

## Do not invent a decision

First pass of this mail said, of the core-cutting debit:

> "I find that difficult to argue with… I am inclined to drop this debit."

NDR had said only *"he is denying it… he is unwilling to take that debit"* — he had **not** decided.
Rewritten as:

> "That is his case as it stands. If you have something on record that changes the sequence, this is
> the point to put it, because on the record as it is today the cost of making that opening sits
> with us and not with him."

Rule: a denial is a FACT to relay; a resolution is NDR's DECISION and never yours to pre-empt.

## Two specifications from the same source document — surface, don't arbitrate

Bhavik recorded the earthing strip deviation as **2.5 m WO vs 1.5 m supplied**; the colleague
recorded the same WO as **2 m strip vs 1 m plus brick bat**. Same work order, two readings. Do not
choose. Write the conflict explicitly and ask the recipient to settle it off the BOQ in writing —
that is the one debit both sides agree is clean, so it must not be argued on a wrong figure.

## Names that legitimately stay

NDR dictated the payment sequence himself: *"getting the PDC delivered in Anbu's hand as I promised
him"*. Those two mentions stay, because they are an operative handover step, not attribution. State
in the reply to NDR which mentions you kept and why, and offer to genericise them ("into your team's
hand") rather than stripping silently.

## Other things this mail had to carry

- **The ask, made concrete**: "talk to \<counterparty\> directly and close out X, and whichever other
  debits you believe you can actually convince him on; come back with the agreed net figure."
- **The four facts needed in writing** before any defect is asserted against a vendor: when first
  noticed and by whom; what was connected and what load was on it; dated photographs beyond the set
  already circulated; whether any other agency worked on that equipment afterwards.
- **The commercial sequence as an ordered list** — clear accounts → PDC ready → PDC handed over →
  activation/charging → cheque last, "not before that". Ordered list, not prose; it is what gets
  executed.
- Do **not** fold in the colleague's own fair caveat unless NDR says so ("deductions must be raised
  at execution stage, not at settlement"). Surface it to NDR separately instead.

## Mechanics used

- Reply-all on the counterparty's own latest message in the thread (In-Reply-To = that Message-ID,
  References = prior chain + it), `threadId` passed on `drafts().create`.
- Attachment relayed from a colleague's mail on the SAME thread — download
  `messages().attachments().get(messageId=<colleague's msg>, id=<attId>)`, re-attach, then
  byte-verify on the stored draft (~`len(data)*3//4` vs source filesize).
- Delete-and-recreate the draft when the body changes materially; do not stack two drafts on one
  thread. Pattern: `drafts().delete(id=<old>)` at the top of the rebuild script.
