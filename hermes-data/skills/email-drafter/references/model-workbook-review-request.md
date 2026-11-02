# Model / workbook review-request email

**Trigger.** "Send X the model and the briefing, give him editor access, ask him
to check every formula and write back." Recipient is usually a finance reviewer
(Sarthak Sharma, admin3.blr@draas.com). NDR dictates the brief by voice; the model
already exists; the mail has to make the brief legible to someone who was not in
the room.

## Sequence

1. **Grant access first, then verify by read-back.** `drive_permissions_add`
   role=`writer`, then `permissions().list(fields="permissions(emailAddress,role)")`.
   State the verified access level in the mail ("Your access: Editor") and tell
   NDR you verified it rather than that you granted it.
2. **Read the file back before writing a word.** Pull actual values from every tab
   (`UNFORMATTED_VALUE`) and quote only those. The dictation is the brief; the
   model is the truth. Never restate the model from your own previous summary.
3. **Structure: one numbered section per element, in the order NDR dictated it.**
   Headings in the deal's accent colour. Typical section list:
   what the model is for → the asset → the parties (table: who / puts in /
   takes out) → how the transaction is structured (numbered steps) → the
   control-panel build standard → statutory / FSI cost → purchase → JD terms →
   cost stack → timeline & phasing → sales / pricing / collections → S&M →
   waterfall (formula in monospace) → outputs table → WHAT I NEED YOU TO DO →
   open items.
4. **The ask must be explicit and blunt:** "write back with a clear verdict —
   **correct or not correct** — and a list of every error, gap or disagreement."
   Then numbered specifics, which is what makes the review real:
   - check every formula and every cell, trace each line to the control panel;
   - recompute the big totals **independently**, do not accept the model's own;
   - check quarter-by-quarter placement — nothing double counted, missing, or
     in the wrong period;
   - check the waterfall releases exactly the net cash with nothing stranded;
   - check the three party cash flows tie, and the landowner's two roles are
     separated then correctly added;
   - tell me what is missing that a reader of this deal would want.
5. **Turn the build standard into a testable rule** so the reviewer can fail it:
   "If you find a hardcoded number anywhere other than a yellow cell, that is a
   defect. Please flag it."
6. **List the open items you already know about** — placeholder guideline value,
   an assumed percentage, an ambiguous interpretation, unconfirmed phasing — so
   the reviewer checks them instead of assuming they are settled. This single
   section is what separates a useful review request from a rubber stamp.
7. Close with why it matters, in NDR's voice: "a model we can put in front of
   anyone and defend line by line — so please be blunt. A clean bill of health is
   only useful if you have actually opened every cell."
8. **Draft only. Never send.** Tell NDR the draft is in his Drafts folder and he
   sends it.
9. **Offer the disclosure choice.** If the mail states a conclusion that flatters
   or embarrasses a party (e.g. an investor IRR that is thinner than the deal
   implies, or the price per saleable sqft), offer to cut those lines so the
   reviewer reaches it himself.

## Build

`build_service("gmail", "v1", service_name="google-draas")` →
`EmailMessage` → `base64.urlsafe_b64encode(msg.as_bytes())` →
`drafts().create`. HTML part via `msg.add_alternative(html, subtype="html")`,
with a short plain-text fallback carrying the link. Verify with
`drafts().get(id=..., format="full")` and probe the decoded HTML for the key
figures and the verdict phrasing — a truncated or mis-encoded body has been the
failure mode before.
