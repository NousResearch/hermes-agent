---
name: email-archive-research
description: "Gather business/docs facts from an email archive."
version: 0.1.0
author: Abe Perl (abeperl), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Email, IMAP, Gmail, Research, Extraction, Tax]
    related_skills: [himalaya, email-inbox-triage]
---

# Email Archive Research

Pull facts and documents out of an email archive (Gmail IMAP) and answer with a cited summary — no sending, no live-inbox triage. Use for "check my email and gather the info I need to give [company] 1099 info", "find the paperwork/incorporation/EIN docs for entity X", or any request to locate past mail and extract structured facts from it.

Don't use for live-inbox triage or drafting replies — that is `email-inbox-triage`.

## Quick Path

The reusable search script already encodes the working recipe:

```bash
python3 ~/.hermes/skills/hermes-email/email-archive-research/scripts/imap_search.py \
  --query '(BODY "Abe & Judith")' --query '(BODY "EIN")' \
  --save-attachments --out /tmp/attach
```

Then parse saved PDFs in Python: `pypdf.PdfReader(path)` + `page.extract_text()` per page.

## Procedure

1. **Connect.** Read `email =` from `~/.config/himalaya/config.toml`, password from `~/.config/himalaya/.imap-password` (strip whitespace), then `imaplib.IMAP4_SSL("imap.gmail.com").login(...)`.
2. **Search ALL mail, not the inbox.** `mail.select('"[Gmail]/All Mail"')` — archived/filtered mail lives off-INBOX. Keep the literal quotes around the folder name in the argument so the brackets survive.
3. **Run several narrow AND-queries and union the ids.** One broad `BODY "1099"` returns hundreds of false positives; `(BODY "Judith" BODY "1099")` narrows. IMAP OR form: `(OR (BODY "x") (BODY "y"))`. Dedupe with a Python set and sort uid-descending (newest first).
4. **Filter mailbox noise before reading bodies.** A shared family/business mailbox pollutes keyword searches: marketing addressed to various family members, daily digests, USPS/medical/job-bot mail. Drop by sender-domain and subject pattern — see `references/abe-mailbox.md` for the current list.
5. **Triage headers cheaply, then fetch full messages.** `(RFC822.HEADER)` fetch gives date/from/subject for cheap screening; `(RFC822)` for the promising uids only.
6. **The facts live in PDF attachments, not the email body.** For every part with `get_filename()`, save `part.get_payload(decode=True)` to /tmp and extract its text. Confirmation emails repeat entity name + application IDs; the EIN-class detail (IRS CP 575 A: EIN, name control, tax classification) is a PDF attachment.
7. **Assemble the answer with source citations.** Name the document each fact came from (e.g. "IRS CP 575 A notice, 11/12/2025").

## Pitfalls

- Show candidate list to the user before fetching full bodies — this mailbox returns 1,000+ candidates for family/corporate keyword sets, and the user cares about exactly which sensitive docs get opened, not about an exhaustive crawl.
- Never trust an IMAP BODY hit count — "1099"-style keywords match modern HTML marketing mail almost everywhere. Filter, then inspect; filter again.
- Long inline `python3 -c` scripts with nested quotes break on shell quoting (a `mail.select('"[Gmail]/All Mail"')` one-liner threw a SyntaxError mid-run). Write the script to a file with `write_file` once it exceeds a few lines — you will iterate on it anyway.
- Street names vary in case across official docs (`140 Middleton St`, `140 MIDDLETON ST`, `140 middleton st`); normalize case before matching.
- For W-9/1099-info requests the anchors are the IRS CP 575 A notice (EIN, name control, classification) and the NYS DOS filing receipt (DOS ID, service address, filing date) — the business-express confirmation emails alone are not enough.

## References

- `references/abe-mailbox.md` — noise-sender patterns + known business entities for this mailbox
- `scripts/imap_search.py` — the working search + extract script (usage above)
