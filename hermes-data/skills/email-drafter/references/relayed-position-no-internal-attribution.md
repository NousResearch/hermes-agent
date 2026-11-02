# Presenting a relayed internal position — strip the attribution

NDR correction, 2026-10-24. Ranka Iris / Shankar Electricals final-payment thread.

## What happened

NDR dictated a mail to **Bhavik** (Cc Piyush, Sanjeev Ranka, Manish, Dharmesh) carrying
Anbu's item-by-item response to Bhavik's debit list. The first draft opened with
*"Anbu has now come back on the Shankar Electricals final-payment reconciliation"* and then,
section by section, *"Anbu does not accept this debit"*, *"his position: …"*.

NDR rejected it outright:

> *"I don't want the email to read saying that Anbu has come back and Anbu is denying it.
> No, I just want it. We don't want to even say that we formulated it out of Anbu's response.
> We just want to say, Bhavik, we have spoken to Nanjesh / Shankar Electrical and he is
> denying all of this and these are his answers below. Okay, let's put it that way.
> Let's not bring Anbu into this picture."*

## The rule

An **internal colleague is a source, not a party** to the correspondence. The moment his name
appears in front of a counterparty — or in front of the family/board group — NDR is shown to
be relaying rather than holding a position, the internal split is exposed, and the other side
gets an argument ("your own VP agrees with us").

- ✗ *"Anbu has now come back … Anbu is denying it. His position: the earthing had to be
  provided at those locations…"*
- ✓ *"I have spoken to Nanjesh and to Shankar Electrical and put every one of the points to
  him. He is denying all of it. His responses are below, item by item."*

Same discipline in reverse for externals: *"we have spoken to X and he denies it"*, never
*"we were told that…"*.

## Which mentions survive

Do a pass over every occurrence and classify:

| Kind | Example | Keep? |
|---|---|---|
| Attribution | "Anbu says", "Anbu denies", "per Anbu's email", "Anbu has come back" | **Delete** |
| Framing | "based on our discussions with Anbu", "from his feedback" | **Delete** |
| Operational handover | "PDC delivered into Anbu's hand", "Anbu to collect" | **Keep** |
| Assignment | "Anbu will pick this up" | Keep, unless it reads as evidence |

In the accepted draft exactly two mentions survived, both in the payment-sequence section
(NDR's own dictated step: *"PDC into Anbu's hand"*). Report the surviving mentions in the
hand-off so NDR can strike them too if he wants them gone.

## Second, separate rule — never assert a decision NDR has not taken

The same draft initially carried *"I find that difficult to argue with … I am inclined to
drop this debit"* and *"He is right that the photographs are not evidence of causation."*
NDR had said **present** the denial, not decide it.

When presenting a denial:

- **Allowed:** state the record as it stands — *"on the record as it is today the cost of
  making that opening sits with us and not with him"* — and hand the reader the rebuttal
  window: *"if you have something on record that changes the sequence, this is the point to
  put it."*
- **Forbidden:** *"I am inclined to drop this debit"*, *"I agree with him"*, *"he is right
  that…"*, *"we should concede"*. NDR decides; the draft presents.

This is not a soft style point — it forces a full delete-and-rebuild of the draft, because the
MIME body cannot be patched after creation. Get it right the first time.

## Rebuild mechanics (delete-and-recreate, not patch)

An approved-but-wrong draft cannot be edited in place. The working pattern used here:

1. `drafts().delete(userId='me', id=<old>)` at the top of the build script, guarded in
   try/except so a re-run is idempotent.
2. Rebuild the MIME from scratch and `drafts().create(... 'threadId': THREAD)` again.
3. Print the new draft id and re-verify: threadId, To, Cc, In-Reply-To, MIME tree,
   attachment byte-count.

Never leave two drafts on the same thread — NDR will pick the wrong one.

## Verification checklist before handing back

- [ ] Zero *"X says / X denies / per X's response"* constructions naming an internal colleague.
- [ ] Any surviving colleague mention is operational, and is called out in the hand-off.
- [ ] No decision, concession or agreement sentence NDR did not dictate.
- [ ] Rebuttal windows offered where a denial is presented ("this is the point to put it").
- [ ] Discrepancies between the two sides' figures flagged as **questions**, not silently
      resolved in either direction (e.g. one recorded 2.5 m earthing rods vs 1.5 m supplied,
      the other 2 m vs 1 m — the mail asked the counterparty to settle it off the BOQ).
