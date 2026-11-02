# Forwarding an ALREADY-SENT email to a NEW person (attachments carried, original quoted)

**See also — `references/locate-source-mail-and-forward-inbound.md`** (verified 2026-10-23):
forwarding an **INBOUND** bank/consultant email to a colleague with others marked on it — the
exhaustive locate-the-source search ladder (all vault accounts, `in:anywhere`, `in:spam`,
alternate sender domains, attachment census, reference-number grep), why "a bunch of emails" is
often one email, why bank balance confirmations live on the "registered mail ID" with a request
ID and carry no attachment, the one-call `gws_gmail_draft_create` light path when there is
nothing to attach, and voice-name resolution from mail traffic ("Eishwari" → echamundeshwari@,
"Sartak" → Sarthak Sharma admin3.blr@).

**See also — `references/forward-received-mail-photos-and-draft-diff.md`** (verified 2026-10-16):
forwarding a **RECEIVED** mail (not a sent one) plus the sender's photos to a new person, with
the agent's analysis on top; `vision_analyze` every photo before captioning and state which
claims have NO photographic proof; and PART B, diffing your prepared draft against the copy NDR
actually edited-and-sent (the Drafts row is gone — read the SENT message with `format="raw"`).
PART C there also covers reading `Cc` from raw (the native `gws_gmail_get` never returns Cc) and
PART D the rule that you never speculate about which account a draft sits in.

Verified 2026-10-15 (the two Kanta Ranka passport representations sent to the MEA Public
Grievance Officer and RPO Bengaluru → forwarded to Manohar Singh with a covering note).

Ask shape: *"forward that email to <person>"*, *"forward the mail we have been sending to
the departments, and in the forward tell him we've already sent it to these people and are
awaiting a reply."* This is **not** `new-email-html-attachment-nonascii.md` (which covers a
brand-new email with an attachment) and **not** `html-reply-with-attachments-threaded.md`
(which replies onto a thread). Here the recipient is new, so there is no parent thread, and
the body must carry a *quoted copy of an existing sent message* plus its full attachment set.

## 0. First establish what actually exists — the false-premise check

NDR often asks this as *"we were supposed to forward that mail — if it's not already sent or
drafted, let me know."* Answer that binary question from Gmail, not from memory:

- `gws_gmail_search(query="<address or distinctive subject>")` → find the SENT messages that
  constitute "the mail". Note each one's `date`, `to`, `subject` and `labelIds`.
- `gws_gmail_search(query="from:mailer-daemon OR subject:\"Delivery Status Notification\"")` →
  separate **delivered** from **bounced** addresses. A representation "sent to the departments"
  is usually a mix: two live addresses plus several discontinued ones. The covering note should
  state the split (That split is the most useful thing in the forward — see §3.)
- `gws_gmail_draft_list(...)` → if `count == 0`, nothing was drafted. Say so plainly in the
  first line of the reply (*"the mail was not drafted — it is now"*), then build it.
Never assert a WhatsApp message was "sent" — Hermes only generates deep links. Say
"the note needs your tap".

## 0a. The dictated recipient address — history-check it, then build to the LITERAL address

Verified 2026-10-27 (two sent mails — the 26-Oct meeting note and the Gunjur term-sheet
approval mail — forwarded to a dictated "rnr@drast.com"). NDR dictates addresses by voice and
the domain is the fragile part: `rnr@drast.com` vs the well-known `rnr@draas.com`, one letter.

Before you build, run three cheap queries against the sending account:

```python
for q in ["drast.com", "rnr@drast.com", "subject:drast"]:
    svc.users().messages().list(userId="me", q=q, maxResults=10).execute()
```

Zero hits on all three is a **finding, not a detail** — it means this address has never been
used from this mailbox in either direction, which for an internal family/company forward is
improbable and points at a dictation slip. Do NOT silently "correct" it to the domain you
recognise, and do not refuse: build the draft to the address **exactly as dictated** — a draft
is reviewable before it goes anywhere, so the human sees the To field — and put the evidence in
the report as a question:

> `rnr@drast.com` has no mail history at all in ndr@draas.com (searched the address, the domain
> and the subject — 0 hits). The same local part on the company domain, `rnr@draas.com`,
> appears all through your mail. Both drafts are built to the address exactly as you gave it;
> say the word and I'll repoint them.

Name the *same local part* at the familiar domain and quote the most recent sighting. That
gives him a one-word decision instead of a re-typed address. Same rule as §5.3 — one address
per person, his choice, raised as a question rather than settled by you.

## 0b. Two or three mails in one ask — one draft per source, one builder

"forward that … forward that too" = N source mails → N separate drafts. Never merge them into
one forward, even when they are the same topic and the same recipient: each must carry its own
verbatim forward block and its own attachment set. Build them from one `JOBS` list (tag, source
message id, subject, forward-block headers, attachment filename list, cover note) in a single
`.py` file run once with `terminal`, and write the returned `draft_id`/`message_id` pairs to
`drafts.json` so the verify pass can walk them. Do not build them one at a time — the second
build is where a copy-pasted attachment list or the wrong source message id gets in.

## 1. Pick the message to carry

When several SENT versions exist (e.g. a 13-Oct MEA version and a 14-Oct RPO version of the
same representation), choose one, state which in the reply, and offer the alternative
explicitly: *"the forward currently carries the 14 Oct RPO version — say the word if you'd
rather it carried the MEA one, or both."* Do not silently merge them.

## 2. Download the source message whole

```python
m = svc.users().messages().get(userId="me", id=MID, format="full").execute()
plain_txt = html_txt = None
att_meta = []
for p in m["payload"]["parts"]:
    if p["mimeType"] == "multipart/alternative":
        for q in p["parts"]:
            data = base64.urlsafe_b64decode(q["body"]["data"]).decode("utf-8")
            if q["mimeType"] == "text/plain": plain_txt = data
            else:                             html_txt  = data
    else:                                   # a sibling part = an attachment
        a = svc.users().messages().attachments().get(
                userId="me", messageId=MID, id=p["body"]["attachmentId"]).execute()
        raw = base64.urlsafe_b64decode(a["data"])
        ...
```

Write both bodies to files (`_orig_plain.txt`, `_orig_html.txt`) and every attachment to disk
with its `sha256` recorded. **Carry the FULL attachment set in the ORIGINAL order** — a forward
of a government representation that silently drops attachments is worse than useless. Preserve
order explicitly with a hard-coded filename list rather than `os.listdir`.

## 3. Body shape — cover note, then a real forward block

The covering note is the actual deliverable. For a "we've already done this, please help"
forward it must state, in this order:

1. The issue in one line.
2. **A delivered / bounced status list** — one line per authority: `✔ Delivered — MEA Public
   Grievance Officer (passport.pg@mea.gov.in) 13 Oct` / `✘ Bounced — three other addresses
   returned as discontinued, so these two are the only live channels I have`. Use ASCII HTML
   entities (`&#10004;`, `&#10008;`) — never raw non-ASCII glyphs.
3. The silence + the deadline (*"no reply yet, and there is an appointment on 22 Oct, so we
   are working against a deadline"*).
4. The specific ask (a connect with a named class of officer), in a highlighted box.
5. Signature block with NDR's address and phone.

Then the forward block, in both parts:

```
---------- Forwarded message ----------
From: Nishant Ranka <ndr@draas.com>
Date: Wed, 14 Oct 2026 at 09:45
Subject: <original subject>
To: rpo.bangalore@cpo.gov.in
Cc: Roshini Ranka, Sarthak Sharma
```

Plain part: append `orig_plain` verbatim after this block. HTML part: reuse `orig_html`
**inside a left-bordered quote div** (`border-left:2px solid #dfe3e8;padding-left:16px`) — do
not re-render the original from its plain text, or you lose its formatting.

## 4. Build with `EmailMessage` + `add_attachment` (same as the new-email recipe)

```python
msg = EmailMessage()
msg["From"] = "Nishant Ranka <ndr@draas.com>"
msg["To"] = TO
msg["Subject"] = SUBJ          # ASCII only, prefixed "Fwd: " + the original subject
msg["Date"] = formatdate(localtime=True)
msg.set_content(plain_body)
msg.add_alternative(cover_html, subtype="html")
for p in ATTACH:
    ctype, _ = mimetypes.guess_type(p)
    main, sub = (ctype or "application/octet-stream").split("/", 1)
    msg.add_attachment(open(p, "rb").read(), maintype=main, subtype=sub,
                       filename=os.path.basename(p))
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
svc.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
```

Size reference: 10 attachments ≈ 7.6 MB on disk → **13.7 MB base64 raw**, comfortably inside
Gmail's 25 MB. Fine for a draft; don't bother with Drive links.

## 5. FIVE pitfalls that cost real time

1. **Never `%`-format an HTML template — CSS kills it.** `cover_html % orig_html` raises
   `ValueError: unsupported format character ';' (0x3b)` because of `width:100%` and every
   other `%` in the inline styles. Use a placeholder and `.replace()`:
   `cover_html = cover_html.replace("__ORIG_HTML__", orig_html)`.
2. **Subject must stay pure ASCII** and keeps the `Fwd: ` prefix plus the *original* subject
   verbatim (§3 of `new-email-html-attachment-nonascii.md`). Check `msg["Subject"].isascii()`.
   **Exception — the source subject is a bare thread subject.** When the mail being carried sits
   on a long conversational thread, its subject says nothing about the content: the 26-Oct
   meeting note went out as `Re: Call for a meeting`. `Fwd: Re: Call for a meeting` tells the
   new recipient nothing, so make the Fwd subject descriptive and keep the original verbatim
   inside the forward block (§3): `Fwd: Call for a meeting - note of the meeting of 26 October
   2026`. Drop `Re:`/`Fwd:` prefixes from the descriptive form, use a plain ASCII hyphen, and
   say in the report that you renamed it — never let a chain of `Re:`/`Fwd:` accumulate.
3. **ONE address per person — do not Cc the alternates as "delivery insurance."** NDR corrected
   this directly: given a person with several live mailboxes, he wants the ONE he names. Ask or
   take his last-used address; if you used a Cc, flag it as a question (*"keep, or send only to
   X?"*) rather than presenting it as settled. (Manohar Singh = `msingh@redsoul.co.in`, not
   `msingh@kelsa.io` — NDR: *"redsoul.co.in is better."*) The same rule bit again on 2026-10-26
   (Prestige Hermitage DOD → Dharmesh, who has `dharmesh.ranka1@gmail.com`, an older
   `dharmesh.ranka@gmail.com` where his replies come from, and `ddr@draas.com`): a personal-address
   To plus a work-ID Cc is still two addresses — pick his last-used one and raise the Cc as a
   question in the report, don't settle it yourself. Check which address the thread's replies
   actually came FROM before choosing.
4. **No draft-update API — a header fix means delete + recreate.** Deleting the previous draft
   first also keeps Drafts at exactly one row, which is the §6 check.
5. **Write the builder to a `.py` file and run it with `terminal`,** not a long `execute_code`
   block — the draft call returns a draft id you must actually see, and long code-exec blocks
   silently lose their tail.

## 6. Verify before reporting (mandatory, same as the sibling recipe)

- `drafts().list(userId="me")` → print the To/Subject of every draft; assert there is **exactly
  one** for this recipient/topic. Unrelated drafts to the *same person* are normal and must be
  left alone — on 2026-10-27 the two new forwards to `rnr@…` sat alongside a pre-existing draft
  to `rnr@draas.com` about a staff separation. Filter on the `Fwd:` subject/topic and say
  "exactly these two carry a `Fwd:` subject" rather than asserting the Drafts folder holds only
  your work; never delete someone else's draft to make the count tidy.
- Read back `messages().get(id=<returned message_id>, format="raw")` →
  `message_from_bytes` → assert `labelIds == ['DRAFT']` (**never `['SENT']`**), then print
  `To` / `Cc` / `Subject` through `make_header(decode_header(...))`. `Cc` is invisible in every
  JSON summary — the raw read-back is the only proof a Cc did or did not land.
- Walk `parsed.walk()` for filenames and **compare `sha256` per file** against the local copy.
  Print the per-file comparison (`f"{fn}: sha256 match: {sha == local}"`) — never accumulate it
  into a flag initialised to `False`.
- Keyword-check BOTH `text/plain` and `text/html` for: the salutation, each authority address,
  the deadline date, the ask wording, and `Forwarded message`.

## 7. Report shape

Lead with the binary answer to what he asked (was it drafted — yes/no, now done), then the
header block (To / Cc / Subject / attachment count + byte-verified), then *what the covering note
says* as bullets, then the hard reminder that **drafts never send themselves** — he has to open
Drafts and hit send. Close with a short confirm list of every inference you made (which sent
version is carried, any deadline date you pulled from another thread, whether a Cc should stay).
