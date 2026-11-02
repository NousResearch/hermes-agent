# Revising a draft that already exists (Gmail) — and the section-numbering trap

When NDR comes back on a draft you already created ("please update the email for the X section",
"you dropped it", "add this to the draft"), the deliverable is **the same draft, revised** — not a
second draft sitting beside the first.

## 1. Update in place — `drafts().update`, never draft a twin

The native tools only *create* drafts (`gws_gmail_draft_create`). To revise one, use the API through
the sanctioned auth layer:

```python
import sys, base64
sys.path.insert(0, '/opt/hermes')
from email.mime.text import MIMEText
from tools.gws_auth import build_service

g = build_service('gmail', 'v1', service_name='google-draas')   # never a raw email as service_name

msg = MIMEText(html, 'html', 'utf-8')
msg['To'] = 'Sarthak Sharma <admin3.blr@draas.com>'
msg['From'] = 'Nishant Ranka <ndr@draas.com>'
msg['Subject'] = '<unchanged subject>'
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

res = g.users().drafts().update(userId='me', id='r-1836553301263656547',
                                body={'message': {'raw': raw}}).execute()
```

Why not delete-and-recreate: the **draft id stays the same**, so the id you quoted to NDR last turn
still points at the live draft, and his Drafts folder never shows two near-identical mails.
`drafts().update` replaces the body wholesale — rebuild the whole HTML, do not try to append.
(Delete-and-re-draft is the fallback only when the old draft cannot be addressed, e.g. already sent.)

### 1b. EXCEPTION — delete-and-recreate when the existing draft sits in the WRONG THREAD (2026-10-30)

An in-place update keeps the draft wherever it already is, **including a thread it should never have
been in**. Verified on the VGRO mail: the 29-Oct draft to `sudheer@vgrofinserv.com` ("Re: Reg :-
information required") sat in thread `1a18834e62f88eb4`, while Sudheer's actual inbound request was
thread `1a1861f22d87333c` — so an `update()` would have preserved a reply that starts its own
conversation in the lender's mailbox. Before choosing update-vs-rebuild:

```python
d = g.users().drafts().get(userId='me', id=DRAFT_ID, format='full').execute()
print(d['message']['threadId'])            # must equal the INBOUND mail's threadId
```
If it differs → `drafts().delete(userId='me', id=OLD)` then `drafts().create(...)` with the inbound
threadId. Either way the user must end up with **exactly one** draft for that recipient/subject, so
re-list the folder afterwards and report the new draft id (§6 of the threading-traps ref: name the
account, draft id, message id and thread in the same sentence).

Also: `drafts().list()` needs `userId='me'` — omitting it raises
`TypeError: Missing required parameter "userId"` from the discovery client. That is a call-shape
error, not an auth failure.

## 1c. He has ALREADY rewritten the draft himself — his text is the draft (2026-10-30)

Signal: *"I did not send out the draft email; I have updated it. Please review that … please add
this, update the meeting minutes, and update the draft for me with this."*

Rule: **his version is authoritative.** Review it, then add to it. Never restore a point he deleted
(on the Saturday Review mail he removed the "Sartak has nothing for tomorrow" agenda line and left
the list numbered 1,2,3,4,6 — the deletion stands; inserting a new item as **5** repairs the gap
without resurrecting his deletion). Never regenerate the mail from your own earlier version, and
never "tidy" his wording or his `* HEADER *` / `<b>` structure.

How you know he edited it: the **draft id is stable, the underlying message id changes on every
save**. `drafts().list()` showing `r-3344798209455834652` against a message id different from the
one you recorded at creation is proof. Read his text with `gws_gmail_get(message_id=…)` before
touching anything.

Then update the SAME draft id (§1). Any body edit must land in **both** the `text/plain` and the
`text/html` parts, or the two renders disagree by recipient.

### The MIME rebuild traps — each one silently produces a broken draft

Naive surgery on the raw message failed three times in a row on this mail. The failures and their
tell-tales:

1. **Re-joining `raw.split("--"+boundary)` with `"--"+boundary+"\r\n"+part`** when the part still
   carries its own leading `\r\n` inserts a blank line after the boundary; the part loses its
   headers and Gmail stores the whole MIME as the message *body*. Tell: `gws_gmail_get` returns
   **empty `from`/`to`/`subject`** and a body starting `Content-Type: text/html; charset="UTF-8"…`.
2. **Filtering header lines with `startswith("Received:")`** and keeping the rest. `Received:` and
   `To:` are folded — continuation lines start with a TAB, survive the filter, and terminate the
   header block early. Same tell as (1). Fix: **build the top-level headers explicitly** —
   `Date`, `Subject` via `email.header.Header(subject,"utf-8").encode()`, `From`, `To`,
   `MIME-Version`, `Content-Type: multipart/alternative; boundary="…"`.
3. **Pulling a QP part out with `.lstrip("\r\n")` before splitting on `"\r\n\r\n"`** — you then
   split on a paragraph break *inside* the body and truncate the mail. Split on `"quoted-printable"`
   first, then on the FIRST `"\r\n\r\n"`, no lstrip.
4. **Leaving the old boundary token in the decoded text** — the draft ends with a stray
   `--0000000000004434f4065f0eff98` line the recipient can see. Strip `^--[0-9_=A-Za-z]{8,}` from
   the tail before rebuilding.

Verification must go through the **rendered** view, not the raw you just built: decode-by-hand is
how the broken build passed its own check. `gws_gmail_get` on the new message id must show
populated From/To/Subject, a body that starts with his first line, and no trailing boundary token.

Runnable implementation (does all of the above and aborts if an anchor matches ≠ 1 time):
`email-drafter` → `scripts/revise_gmail_draft.py`; narrative version →
`email-drafter` → `references/revising-a-draft-ndr-has-edited.md`.

## 2. Read the old draft's body before rewriting it

`gws_gmail_get` / `gws_gmail_thread_get` return `"body": ""` for a draft — they don't decode the MIME
part. Pull it explicitly with `drafts().get(..., format='full')` and walk
`d['message']['payload']['parts']`, `base64.urlsafe_b64decode(part['body']['data'])`.

Use the existing HTML as the template: same font stack (`Calibri,Arial,sans-serif`, 11pt), same
`border:1px solid #bfbfbf` tables, `&#8377;` entities, same sign-off. A revision that changes house
style reads as a different author.

## 3. The section-numbering trap (Oct-2026)

NDR: *"Please update the email for the land bank section. **Section 4 is still to come; you dropped
it.**"* The draft's own numbering had land bank as section 3 — but the recipient (VGR) had asked for
five things in a fixed order: shareholding, unsold inventory, upcoming projects, land bank,
borrowings. **"Section 4" meant item 4 of the recipient's list, not section 3 of my draft.**

Rule: when the mail answers a numbered request from a third party, **number the mail to the
requester's order**, with not-yet-ready items present as one-line "to follow" placeholders so nothing
they asked for is silently missing. Here: 1 inventory (DRA Projects) · 2 inventory (DRA Developers) ·
3 upcoming projects — to follow · 4 land bank · 5 shareholding and borrowings — to follow. Say in the
chat reply that you renumbered to match his list, so the change isn't a surprise.

Corollary: "you dropped it" almost always means a **section he expected is thin or absent**, not that
he wants a new mail. Re-read the previous draft against the original request item by item first.

### 3b. A "to follow" placeholder is a DEBT — re-verify it before re-shipping it (2026-10-30)

Same VGRO mail, redrafted a day later ("draft this mail again … now in my drive there is one company
folder with the reconstitution deeds"). The earlier draft had closed item 3 with *"The reconstitution
deed is being retrieved from our records and will be shared with you separately."* It was not
being retrieved — the executed deeds were sitting in the sender's own Drive folder **`Company
Records`**, uploaded that morning under a different name pattern (`20250801 Reconstitution of
Partnership Deed Thindlu land Partners (3).pdf`, `20250227 Reconstitution Of Partnership Deed SPL
(Retiring Of SK) (3).pdf`), fully downloadable.

Rule: **every placeholder in the draft being revised is a claim that may already be false.** Before
re-issuing it, run the retrieval again — a bare `name contains '<keyword>'` sweep with no folder or
owner filter. Then fill the section from the primary document and say in the chat reply which source
each block came from, so the user can see the gap actually closed rather than being re-promised.
Structure data for the DRA entities lives in `dra-corporate-records` →
`references/shareholding-pattern-and-partnership-deeds.md`.

Also check the **principal's** instruction in the underlying mail before the revised draft goes out.
NDR's own 29-Oct mail ended *"please keep this mail on hold with Sudheer until the full set is ready,
so he gets one complete response rather than it going out in pieces."* When the requester wants
speed and the principal wants one consolidated reply, deliver the draft **and surface the conflict**
— never quietly send a partial pack, and never suppress the draft either.

## 4. Verify by reading the draft back

Never report a revision from the `update()` response alone. Re-fetch and assert:

- `labelIds` contains `DRAFT` (it did not get sent);
- `To` / `Subject` unchanged unless he asked otherwise;
- every dictated figure appears in the decoded text — grep the plain-text render for each number
  (`1,500 acres`, `50,095`, `15,757.93`, `76,000`, `1,580`) and print True/False per token.

## 5. Close the loop honestly

The reply lists what changed, then the **check-before-sending** items: unit assumptions you made
(76,000 treated as sq ft, not sq m), figures resting on his dictation rather than a document, any
parcel whose status a lender could misread, and any conflict with our own records (Ranka Iris 11th
floor per his instruction vs 9th floor in the liquidation master). He reads that block before he hits
send — it is the whole value of drafting it for him.
