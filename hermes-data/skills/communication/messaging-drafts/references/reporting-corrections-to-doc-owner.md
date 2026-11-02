# Reporting corrections to the colleague who owns a document

NDR correction, 2026-10-05. Context: a review of two customer letters (Ranka Iris possession
letter + registration-delay letter) turned up errors, and the message had to go to the colleague
who issues those letters.

NDR's instruction, near-verbatim: *"give me a WhatsApp message of all that we have fixed in the
two documents, then giving the links to the documents, rather than telling him what all is wrong
in the document and then asking him to just review it once more."*

## The rule

Frame the message as **what has been FIXED**, plus the document links, plus a close-out ask:

> Both letters have been corrected on Drive. Please issue these versions.
> **WHAT HAS BEEN FIXED** — 1. … 2. …
> **DOCUMENTS** — <link per document>

Not: a numbered list of defects with "please review once more". The recipient is the owner/issuer
of the document, not a peer reviewer — he wants to know the version is now safe to issue.

Corollaries:

- **Corrections you deliberately chose NOT to make are omitted entirely, not reported.** A
  review usually produces findings that get downgraded on instruction (stylistic inconsistencies,
  buyer-favouring drafting, inherited defects). Listing them re-opens a closed decision. See the
  correction-priority split in
  `productivity/legal-document-drafting` → `references/customer-project-letters-possession-and-delay.md` §2.
- **Anything you could not fix goes to NDR, not into the colleague message.** Unconfirmed
  document numbers, missing bay numbers, blank dates — those are open items for the user.
- **State each fix in one line with the old value and the new one** ("cited BBMP/Addl.Dir/JD
  North/0037/2013-14, which is the 2013 sanction — corrected to ACTP/BCCC/OC/001/2026-27"). The
  recipient needs to know what changed, not why you think the drafter erred.
- **Retract your own false alarms explicitly.** If an earlier list flagged something that the
  record later disproved, say so in the message ("ignore that point from my earlier list") — do
  not leave it standing, and do not silently drop it if it was already sent.

## Before sending

- **The recipient must be able to OPEN the links.** A Drive link to a file they have no permission
  on shows "You need access". Check `permissions` and grant at least reader before the message
  goes out. Say which level you granted, and offer writer if the recipient may edit.
- **Check the channel actually exists.** Don't assume Telegram — verify the person is in the
  channel registry; fall back to a WhatsApp deep link if not, and say which you used.
- **Confirm the phone-number label** you are dialling from (NDR's contacts carry labels such as
  `Work` / `Mobile`; the labelled-work number is the one to use).
- Build any WhatsApp link with the `whatsapp_link` tool only. Long messages come back as a `parts`
  array — deliver each part as its own separate message.
- Deep links and Gmail drafts are inert: generate them in the same turn as any confirmation ask.
