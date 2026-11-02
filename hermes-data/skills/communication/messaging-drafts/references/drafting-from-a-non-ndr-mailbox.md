# Drafting email from a NON-NDR session mailbox (Bharat / sales1.blr)

The `email-drafter` skill is written for ndr@draas.com. A large share of drafting traffic
comes from other session users — **Bharat Hawaldar (sales1.blr@draas.com)** drafts to Gowri
Singh and Roshini Ranka constantly (hoardings, creatives, event branding, design requests).
Same rules and the same verification discipline, different account resolution.

## 1. Resolve the account FIRST — never assume `google-draas` means NDR

`gws_resolve_account()` with NO argument lists every service and its live status. In a Bharat
session (Oct-2026) it returned:

- `google-draas` → `sales1.blr@draas.com`, `has_token: true`  ← the one to use
- `google-ahfl` → no email, `has_token: false`
- `google-gmail` → no email, `has_token: false`

So `service_name='google-draas'` is the vault KEY meaning "the DRA Workspace account of the
**session user**", not a hardcoded identity. Pass it explicitly to every call
(`gws_gmail_search(..., service_name='google-draas')`,
`build_service('gmail','v1', service_name='google-draas')`) and confirm on the draft
read-back that `From:` is the session user, **not** ndr@draas.com.

## 2. Contact/entity lookup needs `account_email` for a non-NDR user

`contact_resolver(query='Gowri')` fails in a Bharat session with:

```
Sheets connection failed: User '<telegram id>' has no gws_service configured in their
profile. An admin must set 'gws_service', or pass account_email explicitly
```

Fix: pass `account_email='sales1.blr@draas.com'` (or the resolved label) and re-run. This is
a per-session argument, **not** an admin action — do not escalate it to the user as a config
problem. Addresses verified this way:

- **Gowri Singh** (DRAAS marketing / creatives) = `gsingh@draas.com`. Her contacts-sheet rows
  carry only personal addresses (`gowrisingh72@yahoo.com`, `gowrisingh1341@gmail.com`) — the
  WORK address is the one on the live Gmail threads, so trust the thread header.
- **Roshini Ranka** = `rnr@draas.com` (contacts row 2989). Dictation renders her as
  "Roshini ma'am" / "Roshni ma'am".

## 3. Bharat's drafting voice and standing conventions

- Opens `Hi ma'am,` / `Hi Ma'am,` — never the name. Closes `Thanks & Regards,` or `Regards,`
  then `Bharat Hawaldar` (or just `Bharat`), usually with `9900029200`.
- Support-team convention: address **Sir / Ma'am, never by first name.**
- **Numbered ask, not prose**: "Kindly let us know: 1. … 2. … 3. … 4. …" — he dictates the
  ask as loose speech; the mail must land as a scannable list.
- **Default recipients for his marketing mails:** To Gowri Singh, Cc Roshini Ranka when he
  says "keep Roshini ma'am in CC". A follow-up voice note in the same drafting flow keeps that
  Cc — but SAY in the reply that you carried it over, so he can correct it in one word.
- **State scope honestly — this was an explicit correction (Oct-2026).** He said: *"we were
  not supposed to do the posters but any which we have done — you can just give the
  reference, the attachment is also correct."* The mail must say plainly that the work was
  **not our scope**, that it was done anyway so the spec is settled, and label the artefact
  **"reference only"**. Do not soften that into a neutral "please find attached".
- **Never invent the purpose.** If he hasn't said what a deliverable is for (digital campaign /
  site office / event screens), leave it out and flag it as an open question rather than
  guessing a use case into the body.
- Always leave the draft **unsent**, report To/Cc/Subject + attachment verification back, and
  name every assumption in 2-3 bullets at the end.

## 4. Cross-check a numbered ask against its own reference artefact

When the mail carries BOTH a rendered artefact (poster PDF) and a table of the same numbers,
extract the rendered numbers and compare before reporting the draft ready:

```bash
pdftotext -f <N> -l <N> poster.pdf -   # per page, one page at a time
```

Oct-2026: page 8 of the 13-page poster PDF read `6 mins away`; the email table had a blank
(`Roma Bakery —  mins away`). Both were reconciled by re-deriving the column (see the `maps`
skill, `references/mymaps-distance-from-anchor.md`) — the PDF was right and the table had lost
the digit. Report the AGREEMENT explicitly; a broken cell in a spec table is a real defect and
the user is relying on you to catch it.
