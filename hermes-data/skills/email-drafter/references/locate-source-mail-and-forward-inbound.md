# Locate the source email, then forward INBOUND (bank / third-party) mail as a draft

Companion to `forward-existing-sent-mail.md` (a SENT mail) and
`forward-received-mail-photos-and-draft-diff.md` (a received mail with photos + draft diff).
This one covers the commonest inbound case: **a bank / consultant / authority email that NDR
received, that he now wants passed to a colleague, with one or two people marked on it.**

Verified 2026-10-23 — Kotak balance confirmation forwarded to Eshwari, cc Roshni + Sarthak.

> **Housekeeping:** `email-drafter/SKILL.md` is at the 100,000-character cap — `skill_manage
> action=patch` on it fails with a size error. New detail must go into `references/` (this file
> pattern), and the pointer line can only be added by first trimming SKILL.md. The same is true
> of `confirm-before-actions/SKILL.md`. Both need splitting by the curator.

Ask shape (voice): *"There are a bunch of emails yesterday on balance confirmations from Kotak
Bank. Nilesh, I think, has sent it. Can you check those emails and forward it to Eishwari,
marking Roshni and Sartak on it? All the emails individually?"*

## 1. Count first — the ask is often bigger than the mailbox

"**A bunch of emails**" was **one** Kotak email, inside a three-message thread, with **zero
attachments**. Build from what exists and lead the reply with the real count. Never pad the
forward to match the plural in the request, and never quietly drop the "all the emails
individually" instruction — explain the count and offer to add more.

Run in this order:

1. `gws_resolve_account()` with **no arguments** — enumerate every vault account
   (this session: `google-draas`, `google-ahfl`, `google-gmail`, `google-o3infotech`) and repeat
   the query on each. The default account is not always the mailbox he means.
2. Topic + sender on the primary account: `(Kotak) (balance confirmation) newer_than:3d`.
3. `in:anywhere` — catches archive/all-mail, not just INBOX.
4. `in:spam` — corporate mailboxes bury vendor mail here constantly.
5. **Alternate sender domains.** Banks send from several: `kotak.com`, `kotak.bank.in`,
   `kotakmahindra`. One domain alone silently returns "nothing found".
6. Attachment census for the date window:
   `after:2026/10/21 before:2026/10/23 has:attachment -in:sent -in:spam`.
7. Grep the distinctive reference number (request ID / RERA ticket / acknowledgement no.) —
   finds the sibling mails a keyword search misses.

If every rung comes back clean, that is a real answer: say "there is no such set in the mailbox"
*and* show what does exist, rather than asking him to re-describe it.

## 2. Bank / statutory confirmations are frequently NOT in the mailbox

Kotak's branch manager replied only:

> Classification: Public — Sent on registered mail ID with request ID 50536257

The signed confirmation had been dispatched to the account's **registered email ID**, not
attached to the thread. So the deliverable is the covering note + the **request ID**, and the
reply to NDR must state plainly that no attachment exists in the mailbox.

- Never write "please find the confirmation attached" when nothing is attached.
- Never reconstruct/invent a PDF. Carry the request ID, the date and the officer's name instead.
- If he needs the document itself, the honest next step is for the recipient (who holds the
  registered mailbox) to pull it — say so.

## 3. Light path: one native call, no raw MIME builder

When the source has **no attachments**, skip the `EmailMessage` + base64url recipe entirely:

```
gws_gmail_draft_create(
  service_name="google-draas",
  to="echamundeshwari@draas.com",
  cc="rnr@draas.com, admin3.blr@draas.com",   # comma-separated string
  subject="Fwd: <original subject verbatim>",  # keeps the Fwd: prefix
  body="<covering note>\n\n---------- Forwarded message ---------\nFrom: …\nDate: …\nSubject: …\nTo: …\n\n<quoted body>",
)
```

Returns `draft_id` + `message_id`. Draft only — never send. Use the heavy recipe only when
attachments must be carried.

## 4. Voice-name resolution: mail traffic beats the contacts sheet

`contact_resolver` was pure noise here — for "Eishwari" it ranked *Eshwari Jio* (no email),
*Eshwar Mayakumar*, *Amiteshwar Bulls*; for "Sartak" it ranked *S Advocate* and *Tarak Madhani*.

What actually resolved them: **who is on the thread and who has been mailing recently.**

- **Eishwari → Eshwari Chamundeshwari `<echamundeshwari@draas.com>`** — TEAM-ACCOUNTS &
  FINANCE. The other Eshwari in the org is *Eshwari Anand `<eshwarianand2024@gmail.com>`* — a
  different person, do not substitute.
- **Sartak → Sarthak Sharma `<admin3.blr@draas.com>`** — NDR's admin (proved by Roshni
  replying to `Sarthak Sharma <admin3.blr@draas.com>` the same week).

Rule: resolve from thread participants → recent mail traffic → contacts sheet, in that order.
Then **name the address you used in the reply and offer the alternate in one line.** A draft is
reversible; do not stall the task waiting on a confirmation you can source yourself.

## 5. "All the emails individually" = one draft per message

Never merge several source messages into one forward. When the real set is one email, build that
one and say so; keep the per-message shape for when he confirms a larger set.

## 6. Do not forward a person their own email without saying why

Eshwari was the **original recipient** and had already replied "Noted sir, thank you for the
confirmation." Forwarding it back to her is only worth doing as a **file copy**, so the covering
note must say that (keep it with the audit file / ITR working papers) — and the reply to NDR must
flag the redundancy in one line, with the alternative ("or is it Eshwari Anand?") offered.

## 7. Report shape

- Line 1: what actually exists (count, thread name, timestamps, senders) — including the words
  "one email, not a bunch" when that is the truth.
- Then the two caveats worth his time: no attachment anywhere + the registered-mail-ID detail.
- Then the draft block: To / Cc / Subject / what the body asserts / draft is in Drafts, not sent.
- Then one short line offering the alternative reading (different person, different mailbox).
- Bullets, no tables. He reads this on Telegram.
