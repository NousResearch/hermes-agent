# Reply-all with a changed To/Cc set, multiple attachments, and the originals embedded in the body

Verified 2026-10-03 (Balaji Metal ↔ Ranka Oasis balcony railing ↔ Sinchana Gowda).
The governing NDR instruction, near-verbatim:

> *"I want us to reply all to that email ... attaching all the original attachments I had
> sent to Balaji, including Balaji's own picture that he has attached ... the content of my
> original mail should also be forwarded, it should be visible in the forward. And the
> attachments from the original mail. And then explaining to Sinchana to coordinate with
> Bhuvanesh, Anbu, Kantesh and respond to all the points that the vendor would need to be
> able to quote us on."*

Four separate requirements are buried in that one sentence. Miss any one and the draft is
wrong: **reply-all**, **all original attachments + the vendor's attachment**, **the original
message visible in the body**, and **an instruction to an internal colleague**.

## When to use this

- A vendor/counterparty replied, and the next action is internal (a colleague must supply
  information or resolve a decision) while the counterparty stays on the thread.
- The recipient set must CHANGE from a plain "reply to sender" — the colleague is added to
  `To`, the vendor stays on `To`, others go to `Cc`.
- The thread needs to be self-contained: the reader gets the original + the reply inline,
  without scrolling or hunting their mailbox.

## Why the native tools cannot do it

| Requirement | `gws_gmail_reply_draft` | `gws_gmail_draft_create` | Hand-built MIME |
|---|---|---|---|
| Correct threading (In-Reply-To/References) | ✅ | ❌ | ✅ |
| Attachments | ❌ | ❌ | ✅ |
| Explicit To ≠ auto-sender | To only | ✅ | ✅ |
| Embedded original body | ❌ | manual text | ✅ |

`draft_reply_create` (skill-bridge) auto-fills `To` from the original sender and takes **no
attachment parameter** — so it cannot carry files. `gws_gmail_draft_create` can set To/Cc but
offers no threading and no attachments. **Build the MIME by hand**; this is the same class of
exception described in `references/html-body-email-drafting.md`.

## Recipe

```python
import sys, base64, mimetypes
sys.path.insert(0, "/opt/hermes")
from tools.gws_auth import build_service
from email.message import EmailMessage

svc = build_service("gmail", "v1", service_name="google-draas")

THREAD   = "1a0f8762c49ce641"                    # from the thread you read
ORIG_MSGID = "<223525870167034880.0.v2@titan.email>"  # BALAJI'S Message-ID -> In-Reply-To

msg = EmailMessage()
msg["To"]      = "Sinchana Gowda <sgowda@draas.com>, BALAJI METAL <export@decorativecasting.com>"
msg["Cc"]      = "Bhuvanesh Krishnan <bk@findingform.design>, Anbarasan M <pm2.blr@draas.com>, Kantesh B G <kanteshbg@draas.com>"
msg["Subject"] = "Re: <exact original subject>"
msg["In-Reply-To"] = ORIG_MSGID
msg["References"]  = ORIG_MSGID

msg.set_content("Please view this email in an HTML-capable client.")
msg.add_alternative(HTML_BODY, subtype="html")

for path, fname in ATTACHMENTS:                   # originals + the counterparty's file
    ctype, _ = mimetypes.guess_type(fname)
    maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
    with open(path, "rb") as f:
        msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=fname)

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
res = svc.users().drafts().create(
    userId="me", body={"message": {"raw": raw, "threadId": THREAD}}).execute()
```

`EmailMessage` + `add_attachment` is cleaner than assembling `MIMEMultipart` by hand and
auto-sets Content-Type/Transfer-Encoding. Both work.

## Embedding the originals in the body

Put the original enquiry and the counterparty's reply **verbatim** in the body, after the new
text. Quoting them inside `<pre>`, with `white-space:pre-wrap` so the plain-mail paragraphing
survives, and keep the wrapper full-width per NDR's standing rule (no `max-width`):

```html
<h3>----- Original enquiry -----</h3>
<pre style="font-family:inherit;white-space:pre-wrap;">{ORIGINAL_TEXT}</pre>
<h3>----- {Counterparty} reply, {date} -----</h3>
<pre style="font-family:inherit;white-space:pre-wrap;">{REPLY_TEXT}</pre>
```

Extracting the body: **always decode the raw MIME, not the convenience fields.**
`messages().get(format="full")` returned an **empty `body: ""`** for the vendor's message
(it was HTML-only inside `multipart/related`). Two working routes:

- `format="full"` and walk the payload for the `text/html` part, base64url-decode it, then
  strip tags: `re.sub(r"<br\s*/?>", "\n", h)` → `</(p|div|tr|li|h[1-6])>` → `"\n"` →
  `<[^>]+>` → `""` → `html.unescape`. This gave the full readable reply.
- For header values (`Message-ID`, `In-Reply-To`, `References`) the same walk is required —
  read them from `payload.headers`, not from any top-level dict.

## Verification (mandatory before reporting ready)

**Never report "the draft is ready" on the strength of a successful `drafts().create()` call.**
A draft has been reported created and then found absent from the mailbox — the user had to ask
"what is my draft? that is still not done". Read the draft back out of the mailbox and confirm it
is really there before you tell the user it is. If you are redoing a draft, also confirm the
superseded one is gone — two unlabelled drafts on one thread is its own failure.

```python
d = svc.users().drafts().get(userId="me", id=DRAFT_ID, format="full").execute()
m = d["message"]; hh = {h["name"]: h["value"] for h in m["payload"]["headers"]}
# assert To / Cc / Subject / In-Reply-To / References
# assert m["threadId"] == THREAD          -> it landed on the existing conversation
# walk payload: assert each expected filename + its size
```

Three things must be true, and each has failed in a past session:

1. **`threadId` equals the original thread** — otherwise it opens a new conversation.
2. **Every attachment is present with the right size** — the original two *and* the
   counterparty's file. Filenames are NOT visible via `format="metadata"`; use `"full"`.
3. **`In-Reply-To` is the counterparty's most recent Message-ID**, not the original sender's.

## Pitfalls

- **A reply that only matches the category of the request is the common failure.** "Reply all
  to that email" also carried: attach *all three* (including his), embed the original, and add
  the colleague — with an explicit instruction to coordinate. Re-read the dictated ask and
  enumerate its clauses before drafting.
- **Quote from the actual asked-for content, not from memory.** The vendor's reply and NDR's
  original both go in verbatim; paraphrase loses the technical detail the colleague needs.
- **Do not "answer" for the counterparty.** If the reply conflicts with the brief, the draft
  raises the conflict and asks the colleague to resolve it — it does not quietly drop the
  requirement or silently accept the vendor's position.
- **Read the existing Cc off the counterparty's message headers and CONTINUE it.** A reply-all
  is not only "add the colleague" — if the vendor already put one of ours on Cc (e.g. Anbarasan
  on a vendor's quotation mail), the reply-all must carry that same Cc forward or you silently
  drop a colleague who was already on the thread. Pull `To`/`Cc` from the message you're replying
  to (`payload.headers`) rather than reconstructing the recipient set from memory. Same class of
  check as "enumerate every clause of the dictated ask before drafting".
- **Do not guess colleague addresses** when the contacts sheet has none. Search the mail
  corpus (`from:<name>` / `to:<name>`) to get the real ones; a guessed pattern may deliver to
  a stranger.
- **Draft only, never send.** State plainly that the draft is in Drafts and the user presses
  send.