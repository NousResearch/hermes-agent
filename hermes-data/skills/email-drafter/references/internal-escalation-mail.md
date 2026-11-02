# Internal escalation / course-correction mail — stern but not offensive

When NDR dictates a complaint about a colleague's *conduct* (a payment escalation aimed at a junior,
a threat that "work will stop", a process breach) and says any of:

- *"a little stern email but without offending"*
- *"it should not be offensive to anyone"*
- *"be very clever in your language"*
- *"I wanted to document it because this is what came back to me"*

…he is asking for a **documented, fact-anchored correction**, not a rebuke. The mail must be
quotable in a later dispute and must not read as an attack.

Verified 2026-10-16 — NDR → Anbu (Anbarasan), Vardhan Ventures / Ranka Amber invoice escalation.

## The structure that worked

1. **Open with a meta-line that buys room, then attribute the report — never assert it.**
   > *"I want to raise something with you directly, and I would rather do it on the facts than on
   > what was reported to me."*
   > *"I am told that you spoke to <person> and conveyed that …"*
   Then immediately hand the recipient the benefit of the doubt as a **question**:
   > *"Before I take that at face value, two questions: is that a position coming from <vendor
   > principal / outside>, or is that something being said at our end?"*
   Naming the possible external source (e.g. *"is that a position coming from Jayaram sir?"*) is what
   makes the mail non-offensive — it offers the recipient a clean exit.

2. **Exactly ONE plain sentence, and only conditionally** — *"If it is coming from our end…"*:
   > *"…then I have to say it plainly: it is not correct, and at your level it is not the way we work."*
   No second sentence. No adjectives about the person.

3. **Numbered FACTS block.** Head it *"Now the facts of the item, because they do not support
   <the claim>"* and give each point a date:
   1. record id / vendor / what it is for / amount / PO number / invoice date / when it was posted;
   2. **when the recipient themselves approved it** (exact date + time);
   3. what the actual discrepancy was, and how it was read/reconciled;
   4. any secondary point noted and accepted (brand, grade, spec);
   5. **when NDR approved, and for how much** + what step is the only one still pending.
   Close the block with the arithmetic of the accusation:
   > *"So: your approval was three days ago, and mine two days ago. Nothing has been sat on."*
   Facts come from the record — see `kelsa-mcp` →
   `references/dictated-payment-escalation-reverse-lookup.md`.

4. **The policy statement, in the plural "we"** — this is the durable part of the mail, the bit that
   is meant to be re-read in future:
   > *"We have said this repeatedly, and it holds for every payment: the approval cycle takes three to
   > four working days. We are not going to speed finance up. Push entries through and we invite
   > accounting errors and tax issues, and that costs us far more than a few days. The process will be
   > allowed to run its course."*

5. **Close with a call-me, then the single firm sentence LAST.**
   > *"Please go through the above and call me. I am raising this with you directly and leaving it
   > there — but at your level, the company's work cannot be presented as something that will come to
   > a stop, and certainly not to a colleague."*
   *"At your level"* is the standard-setter — it appeals to seniority rather than scolding it. That is
   what makes it stern without being offensive.

## Tone rules (all from NDR's dictation)

- Never accuse in the second person. Report → ask → state the standard.
- The firm sentence goes **last**, never first. Opening with it makes the mail a reprimand.
- No consequences, no escalation threat, no "I am disappointed", no Cc-for-shame.
- Do not re-litigate the vendor's commercial position; the mail is about **our** process, not the
  vendor's grievance.
- Keep the recipient's own good action in the record ("you approved it the same day, at 18:08") —
  giving credit inside the facts block is a large part of what keeps it non-offensive.

## Recipient discipline — two shapes NDR asks for

- **"Draft a reply all email … keep marking it only to <X>."** Thread it onto the original
  (`threadId` + `In-Reply-To`/`References`), but put **only `<X>` in To and nobody in Cc**. The
  original may have had no other recipients anyway, in which case reply-all and To-only are the same
  thing — say so in the report rather than silently choosing.
- **"Draft it for me to him."** A NEW message (no `threadId`), To = that person alone. These are
  framed as a direct conversation; do not add Cc on your own initiative.
- In both cases **report the addressing decision explicitly** ("Addressed to Prakash only, no Cc —
  as you instructed") so NDR can add a Cc in one word if he wants one.

## Build

Same build as any internal multi-point mail: ASCII-only Subject (use `Rs.` and `-`, never `₹`/`—`),
`EmailMessage` + `set_content(plain)` + `add_alternative(html, subtype="html")`, HTML entities for
dashes/quotes, omit `threadId` for a new thread. Verify with the five checks in
`references/new-email-html-attachment-nonascii.md` § 5 — `labelIds == ['DRAFT']`, To/Cc/Subject exact,
`text/html` part actually starts with `<div`, and the draft is present in `drafts().list()`.

## DRA staff — SENDING address map (verified from Gmail headers)

Contacts-sheet addresses are the *mailbox*; these are what people actually mail **from**. Always
confirm by searching `from:<candidate>` before addressing.

| Person | Sends as | Note |
|---|---|---|
| Anbu / Anbarasan M | `anbarasan@draas.com` | Sheet says `pm2.blr@draas.com` — do NOT use |
| Gowri Singh | `gsingh@draas.com` | Marketing + **content-approval gatekeeper**. Contacts sheet row 1156 carries only `gowrisingh72@yahoo.com` / `gowrisingh1341@gmail.com` (personal) — both wrong for company mail |
| Sarthak Sharma | `admin3.blr@draas.com` | Voice "Sartak"; contact_resolver row 4215 |
| Prakash Singh | `psingh@draas.com` | Also DRA Realty sales/marketing lead |
| Sinchana Gowda | `sgowda@draas.com` | DRA architect (Oasis) |

**Approval-ownership map that recurs in escalation mails:** Gowri approves marketing/content
collaterals and in turn takes Sinchana's sign-off on floor plans, SBA and carpet area (obtained from
the project architects as a GFC process); the checklist is RERA number on every piece, verified
phone/QR/location, and AI enhancement limited to lighting and greening — no further embellishment,
that call being Gowri's.
