# Forwarding an already-sent mail WITH its attachments (covering note + true forward)

Verified 2026-10-15 — Kanta Ranka passport: forward the mail already sent to MEA/RPO, with a
covering note asking Manohar Singh for a network connect. To `msingh@redsoul.co.in`, **no Cc**,
10 attachments, 13.7 MB raw, draft `r1659211161496050422` / msg `1a13e5b66f09c902`.

Trigger: NDR says *"forward that email to X, and in the forward also tell him …"* / *"forward it
and in the forward also tell him that we have already sent this to these departments"*.

This is a **NEW thread**, not a reply — `draft_reply_create` is the wrong tool.

## 0. First establish whether it already exists — answer the question he actually asked

`gws_gmail_draft_list` on the sending account is the fastest check, plus a
`gws_gmail_search` for `to:<recipient>`. An empty Drafts folder and no SENT to that recipient
means nothing was ever drafted — say so plainly and build it, rather than assuming something
exists. NDR's phrasing is usually *"if that mail is not already sent or not drafted please let me
know and we need to send one now"* — so the deliverable is the draft, this turn.

## 1. Pull the original once, in one pass

```python
m = svc.users().messages().get(userId="me", id=ORIG_MSG_ID, format="full").execute()
for p in m["payload"]["parts"]:
    if p["mimeType"] == "multipart/alternative":
        for q in p["parts"]:
            data = base64.urlsafe_b64decode(q["body"]["data"]).decode("utf-8")
            ...                       # keep BOTH text/plain and text/html
    else:                             # an attachment part
        a = svc.users().messages().attachments().get(
            userId="me", messageId=ORIG_MSG_ID, id=p["body"]["attachmentId"]).execute()
        open(f"/tmp/<work>/{p['filename']}", "wb").write(
            base64.urlsafe_b64decode(a["data"]))
```

Keep the original's **HTML** part — splice it into the forward so the government mail renders
exactly as it was sent. Keep the plain part for the `text/plain` alternative.

Writing a one-shot script to disk (`write_file` → `terminal python3 …`) beats re-pasting a long
builder into `execute_code` when you expect a second pass (e.g. a Cc change).

## 2. Build the forward as EmailMessage — never a hand-made MIME dict

Cover note → `---------- Forwarded message ----------` block carrying the original's
From / Date / Subject / To / Cc → the original body inside a quoted container. Attach the
original's files in their **original order**.

`os.listdir` order is arbitrary — hold an explicit ordered filename list and iterate that.

## 3. PITFALL — never `%`-format an HTML template

```python
cover_html = """<div style="width:100%">…</div>""" % orig_html
# ValueError: unsupported format character ';' at index 1109
```

CSS is full of `%`. Use a sentinel:

```python
cover_html = cover_html.replace("__ORIG_HTML__", orig_html)
```

## 4. Changing just the Cc = delete + recreate

There is no draft-update API. `drafts().delete` the old id, recreate, re-verify. Confirmed:
NDR replied *"Redstone.co.in is better"* (i.e. `redsoul.co.in`, and drop the Cc entirely);
that was one script edit plus a delete call — cheap, so never leave a wrong Cc sitting there.

## 5. Verify before reporting — all five checks

1. `drafts().list` — exactly ONE draft for this recipient/topic.
2. `messages().get(format="raw")` → `labelIds == ["DRAFT"]` (never `["SENT"]`).
3. Print `To` / `Cc` / `Subject` through `make_header(decode_header(...))` — the JSON draft
   summary never shows `Cc`, so an unverified Cc is invisible everywhere else.
4. **Byte-verify every attachment**: sha256 of the part off the created draft vs the local file.
   Print per-file `sha256 match: True` — do NOT accumulate into a flag initialised to `False`.
5. Keyword-check **both** `text/plain` and `text/html` for the facts that matter: the salutation,
   each authority address, each date, the ask, the deadline.

## 6. Covering-note shape NDR expects for an escalation-favour forward

Order that worked, and that he accepted without edits:

1. One line of context — who he is writing to and why it is a favour ("I am writing to you on a
   personal matter, and I would be grateful for your help").
2. The issue in one or two sentences.
3. **A status list of who has already been written to**, separating delivered from bounced:
   ✔ Delivered — MEA Public Grievance Officer (`passport.pg@mea.gov.in`), 13 Oct
   ✔ Delivered — RPO Bengaluru (`rpo.bangalore@cpo.gov.in`), 14 Oct
   ✘ three other RPO/MEA grievance addresses bounced back as discontinued
   …so he can see the channels are exhausted and the network is the only route left.
4. Explicit "no reply yet" plus the deadline (the visa appointment date).
5. The ask, isolated in its own highlighted box.
6. Signature block, then the forwarded original below it.

Note: a bounced address is evidence, not noise — grep `from:mailer-daemon` to get the delivery
failures, because they are what lets you write "these two are the only live channels I have".

## 7. Recipient addresses — corrections from this session

- **Manohar Singh ("Sahabji") = `msingh@redsoul.co.in`, and only that.** A Cc to
  `msingh@kelsa.io` was rejected: *"redsoul.co.in is better."* His `msingh@kelsa.io` and
  `msingh@o3infotech.com` mailboxes are the Kelsa / O3 work addresses — never Cc them on a
  personal or favour request.
- **Do not hedge a personal/favour email by Cc'ing the same person's second mailbox.** One
  address, the one that matches the context. Two addresses for one human reads as automated.
- For a *personal favour* ask, use the person's own-firm address (M/S Redsoul), not the current
  employer's corporate address.
- The contacts sheet had the wrong Manohar entirely (an IRCA row) — Gmail headers, not the sheet,
  remain the authority on which address a person actually mails from.
