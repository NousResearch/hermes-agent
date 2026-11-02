# Scheduling an external professional into a brothers' thread (Deloitte / advisory presentations)

Recurring class: NDR is coordinating a meeting between an EXTERNAL professional
(Deloitte M&A partner, tax advisor, consultant) and his two brothers, and the
date keeps shifting. Each shift produces BOTH (a) a threaded email to the
brothers and (b) a WhatsApp to the outsider. Validated 2026-09-16 → 2026-09-19
on the Amit Jain / Deloitte LLP-conversion thread (three successive rounds:
propose Friday → Amit asks for Thursday → confirm date).

## The two-channel split (mandatory, NDR's pattern)

| Channel | Recipients | Content |
|---------|-----------|---------|
| **Email draft** | brothers only (Dharmesh + Manish) — the outsider is NOT on it | Who has confirmed, who is pending, the shifted date, the priority ask |
| **WhatsApp link** | the outsider alone | Thanks for the revert, "I've asked both to confirm X", "I'll revert before end of day" |

Never CC the external professional on the internal scheduling email — it exposes the
family back-and-forth. This is the same audience-split rule as the general
"external-meeting follow-up" variant in SKILL.md, applied to *scheduling* rather
than *summarisation*.

## Canonical people / addresses (verified from thread headers, not the sheet)

- **Amit Jain** — Partner, M&A / M&A Tax, **Deloitte India**. Contacts sheet row
  **4238**: `amitkjain@deloitte.com`, **+91 97044 04000**. Note there are at least
  three Deloitte/India "Amit Jain" partners (Gurgaon M&A-tax; Hyderabad M&A-tax ex-PwC/
  KPMG/EY; also a Carlyle India MD who is NOT him) — `contact_resolver` on bare
  "Amit Jain" returns the right sheet row (score 100) so trust the row, not LinkedIn,
  when they disagree. His own outreach is by phone/WhatsApp, not necessarily email.
- **Dharmesh Ranka** — `dharmesh.ranka1@gmail.com` (threads) / `dharmesh.ranka@gmail.com`
  (older). Replies from iPhone; terse ("I am available to meet this Friday").
- **Manish Ranka** — `mdr@draas.com`. Frequently the one who has NOT replied; the
  draft should name him as pending rather than implying consensus.
- **Thread** — `1a05bdbce4e46b9c`, subject
  `Deloitte - Amit Jain: Pvt Ltd to LLP conversion for non-operating entities`.
  Started 1-Sep-2026 by NDR. Anchor replies used for threading:
  Dharmesh's Friday-confirmation `1a0a94b409c72fd3`.

## The scheduling constraint that must be carried forward

Amit's standing ask: **anything after 1 PM, post-lunch**; he travels in from out of
town, so he needs **3–4 days' notice** to plan travel and prepare. Every draft on
this thread should preserve that constraint — it is the reason NDR presses the
brothers for a fast confirmation.

## Draft shape that NDR accepted

Short, HTML, one idea per line:
1. Opening line: what the outsider's latest revert is ("has just come back with a
   request that **Thursday** would suit him better than **Friday**").
2. Per-person status: "Dharmesh – you had confirmed Friday. Manish – I am still
   awaiting your confirmation."
3. The ask, with a face-saving fallback: "could you please see if we can
   **accommodate him on Thursday**... if not, he is happy to do **Friday** as
   originally proposed."
4. **Priority** line: "Request you both to please respond on a priority so I can
   confirm the date with him today."
5. `Regards, Nishant` — no signature block.

Keep it this short. NDR's earlier correction on this thread ("it's too long, make it
shorter") applies here too: do not restate the Deloitte context, the entity structure,
or the family background in a scheduling mail.

## New pattern 2026-09-22: FINAL confirmation goes TO the outsider, brothers on CC (user-requested override)

Earlier rounds kept the outsider OFF the internal email. But when NDR says "send a
message to [outsider], mark Dharmesh and Manish as well / fresh email to [outsider]
marking Dharmesh Manish" — that is an explicit override of the two-channel split:
draft the threaded reply (same `Re:` subject, In-Reply-To/References = last NDR
message, threadId = `1a05bdbce4e46b9c`) with **To = outsider, Cc = both brothers**.
Body formula that worked (Amit Jain / Deloitte, Thu→Fri revert):

1. "Thank you for suggesting Thursday. Unfortunately, Dharmesh has a conflict on
   Thursday, so can we please stick with **Friday** if you don't mind? That also
   gives you adequate time to plan your travel."
2. "Can I pencil in **Friday at 2:00 PM**? Does that work for you?"
3. Location block: office name + full address + a **clickable Google Maps link**
   (`https://www.google.com/maps/search/?api=1&query=<url-encoded address>`).
   Ranka Chambers = 4, Ranka Chambers, 31 Cunningham Road, Bengaluru 560052.
4. "As soon as you confirm, I will send you a calendar invitation for 2:00–4:00 PM."
   — i.e. do NOT create the calendar event until the outsider confirms.

Brothers' numbers for the parallel WhatsApp nudges (verified contacts sheet):
Dharmesh Ranka +91 98441 09010; Manish Ranka +91 98866 29074. The revert also
changes the per-brother nudge text (Dharmesh already confirmed Friday → the revert
SUITS him; Manish still unconfirmed → ask him directly).

## Pitfalls

1. **Delete the superseded draft before creating its replacement.** Each date shift
   rewrites the same mail; leaving the previous round in place puts two drafts on one
   thread and the user sees a duplicate. Enumerate drafts unfiltered, match on
   `threadId`, delete, then create.
2. **This thread lives at `threadId 1a05bdbce4e46b9c` while `Winding up matters`
   (Manish's trigger, `1a089af60712afd3`) is a DIFFERENT thread** that quotes the
   Deloitte mail. Searching `amit jain deloitte` returns both, interleaved. Pick the
   thread by its actual participants and Message-ID chain, not by which search
   surfaced it first.
3. **Don't write a bare weekday without checking the date.** NDR dictates "this
   Friday"/"Thursday" and the draft should mirror his wording, but the true date
   shifts fast — state the weekday as he said it AND flag the resolved calendar date
   to him in the chat reply so he can decide.
4. **BCC the wives only on the family/business threads, not the Deloitte scheduling
   thread.** The BCC pattern in `dra-family-windup-matters.md` is for the wind-up
   correspondence; it was not requested here.

## Follow-up when the brothers have NOT replied (2026-09-22)

After the "respond on a priority so I can confirm the date with him today" ask gets
NO reply from either brother, NDR's pattern is **two SEPARATE WhatsApp messages, one
per brother** (NOT one email to both, NOT a group) — polite nudge, personalized per
brother, carrying: (a) the outsider's current preference (Amit prefers Thursday;
Dharmesh had confirmed Friday earlier; Manish is the one pending), (b) the
"lock either Thursday or Friday — whichever works for you — so I can confirm with
him today" ask, (c) the motivation line "let's get this over with and take a
decision; a lot of the DRA projects planning becomes easier once we settle the
structure question. The sooner we do it, the better." Verify each brother's
WhatsApp number from the contacts sheet row (Dharmesh row 870 = +91 98441 09010;
Manish row 1902 = +91 98866 29074) — do not reuse numbers from stale memory.
Generate via the `whatsapp_link` tool, never hand-built URLs.

## Round 2026-09-22: outsider stuck Thu+Fri, proposes Saturday — reply-into-the-thread ask

New shift direction: Amit pinged NDR by WhatsApp saying he is stuck
Thursday AND Friday and asked for a **Saturday** meeting. NDR's reply
(WhatsApp-only, no new email needed) formula that worked:

1. "Thank you for letting me know — I completely understand if Friday
   doesn't work for you."
2. "Saturday works for me, but I am not sure if it will work for everyone
   else." — never claim consensus for the brothers (Dharmesh/Manish) that
   doesn't exist.
3. The ask: "Could you please reply on the email thread itself with 2-3
   suggested alternative dates, so that everyone can confirm which one
   they are available on?" — this deliberately delegates the date
   proposal to the outsider while keeping the family back-and-forth off
   WhatsApp, and re-anchors the coordination back onto the email thread.
4. No caption (external professional). Resolve the weekday→date on the
   reply (Fri 25-Sep / Sat 26-Sep) so NDR can sanity-check.

Voice-garble note re-confirmed this round: **"Aamir Jain" = Amit Jain**
(Deloitte M&A partner) — not Aamir Khan (Millers Road) and not Arvind
Jain (architect). Same contacts row 4238, +91 97044 04000.
