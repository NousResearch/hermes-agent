# Forwarding a SENT mail to a third party — cover note + original body + its attachments

Verified 2026-10-16 (NDR → Manohar "Sahabji" Singh: the Kanta Ranka passport representation that
had already gone to the MEA Public Grievance Officer and RPO Bengaluru).

Use when NDR says *"forward that mail to \<person\>, and in the forward tell him we have already
sent it to these departments and are awaiting a reply"*.

This is **not** `draft_reply_create` — the new recipient is not on the thread. And it is **not**
`references/new-email-html-attachment-nonascii.md` — that builds a fresh email with nothing to
carry over. Both of those files still supply the `EmailMessage` mechanics; this file is the delta.

## 0. Answer the question NDR actually asked, with evidence

His phrasing is usually *"if that mail is not already sent or not drafted, please let me know and
we need to send one now"* — so never answer from memory or from the session summary of a previous
day. Two cheap calls settle it:

```python
svc.users().drafts().list(userId="me", maxResults=25)                 # already waiting?
svc.users().messages().list(userId="me", q="<case keywords>")         # what actually went out?
```

Then two searches separate **live** channels from **dead** ones:

- search the case keywords (`"Kanta Ranka"`) → every SENT copy and its `To:`;
- search `from:mailer-daemon` / `subject:"Delivery Status Notification"` → the bounces.

This matters because a sent-items list flatters you: three sends to RPO/MEA grievance addresses
look like three live departments and were in fact three `550 5.1.1` bounces. The cover note must
name only what was **delivered** — that is the whole difference between "I have written to four
departments" and the truth. Here the honest list was two delivered, three dead.

## 1. Choose WHICH sent copy to forward

When the same representation went out more than once (different subject, different recipient),
forward the **most recent** copy, then say which one it is and put it in the confirm block. Here:
13-Oct to `passport.pg@mea.gov.in`, 14-Oct to `rpo.bangalore@cpo.gov.in` — the 14-Oct copy was
forwarded. Their attachment sets were identical; **verify that before assuming it**, since a later
send may carry an extra annexure the earlier one did not.

## 2. Pull the source message's bodies AND its attachment bytes

```python
m = svc.users().messages().get(userId="me", id=SRC_ID, format="full").execute()
for p in m["payload"]["parts"]:
    if p["mimeType"] == "multipart/alternative":
        for q in p["parts"]:                       # cache BOTH parts
            data = base64.urlsafe_b64decode(q["body"]["data"]).decode("utf-8")
            # keep text/plain for the plain body, text/html for the quoted block
    else:                                          # an attachment
        a = svc.users().messages().attachments().get(
                userId="me", messageId=SRC_ID, id=p["body"]["attachmentId"]).execute()
        raw = base64.urlsafe_b64decode(a["data"])
```

Keep the **original HTML part**. It is what makes the forwarded block render; reusing the sender's
plain-text part instead shows the raw `*asterisks*` from his own emphasis style.

## 3. Assemble

- cover note — plain + HTML — the actual message to the new recipient;
- a `---------- Forwarded message ----------` header block (From / Date / Subject / To / Cc);
- the cached original HTML wrapped in a quote container, e.g.
  `<div style="border-left:2px solid #dfe3e8;padding-left:16px">…</div>`;
- every attachment carried over in original order via `add_attachment`.

Subject: `Fwd: ` + the original subject verbatim, ASCII only (see the ASCII-Subject rule in the
main skill).

Write the cover note to four beats, in this order — it is what makes a favour request land:
1. the issue in one line, framed as *not a change of name*;
2. the delivered/dead status list, with dates and the actual addresses;
3. the live deadline and "no reply yet";
4. the ask, then the signature block.

## 4. Pitfall — never `%`-format an HTML template in Python

Splicing the original HTML into the cover HTML with `%` blows up:

```
ValueError: unsupported format character ';' (0x3b) at index 1109
```

because CSS is full of `%` (`width:100%`). Use an explicit placeholder and `.replace()`:

```python
cover_html = "<div> ... __ORIG_HTML__ ... </div>"
cover_html = cover_html.replace("__ORIG_HTML__", orig_html)
```

Same family of trap when reading hrefs back out of an existing WhatsApp card:
`u.replace("&amp;", "&")` **before** `urllib.parse.parse_qs`.

## 5. Size sanity check

Raw MIME inflates ~1.33× under base64: 7.6 MB of attachments → **13.7 MB** raw. Gmail's draft
ceiling is 25 MB, so it fits — but print the raw size before creating rather than discovering it.
Download attachments one at a time (as above); never pull ten base64 blobs through a tool result.

## 6. Verify, then report

Run `scripts/verify_gmail_draft.py <message_id> --local-dir <dir> --keyword <kw> …`. It proves
`labelIds == ['DRAFT']`, reads `To`/`Cc`/`Subject` from the raw MIME (`Cc` is invisible in every
JSON summary), sha256-compares each attachment to the local copy, and keyword-checks both body
parts. Report only from its output.

## 7. Report shape

- say plainly whether it already existed, and that a **draft is not a send** — he sends it;
- To / Cc / Subject, with the Cc flagged for a keep-or-drop decision;
- attachment count, and that they are byte-identical to what the authority received;
- what the cover note asserts (delivered/bounced list, bearing it names, the deadline, the ask);
- close with an **N things to confirm** block naming every inference — which sent copy was
  forwarded, any deadline lifted from a different mail, the Cc choice.
