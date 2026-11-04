# HTML draft shows up as "HTML code" — the DOUBLE-ESCAPING variant

Two different bugs produce the same user complaint ("the draft shows up as HTML
code / raw HTML in the email body"). They have the same symptom and completely
different fixes, so diagnose by decoding the MIME part — not by guessing.

## Which of the two is it?

| | Bug A — `html=False` | Bug B — double-escaped |
|---|---|---|
| Cause | `html=True` not passed to the bridge | already-escaped body passed *with* `html=True` |
| `text/html` part | absent — body is `text/plain` | present **and correct** |
| Decoded body | real markup: `<p>Hello</p>` | escaped: `&lt;p&gt;Hello&lt;/p&gt;` |
| `payload.mimeType` check | catches it | **PASSES — bug hides** |

Bug A is already documented in SKILL.md ("Pitfall — `html=True` is required for
HTML drafts"). **Bug B is the one that slips through**, because every
`mimeType`-style check says the draft is fine.

## Bug B — confirmed case

2026-10-17, reply to Bhavik Ranka on *Re: Ranka Iris — Transformer Charging
Today*. The draft was created with `html=True` correctly set, yet the raw MIME
showed:

```
Content-Type: text/html; charset="utf-8"
Content-Transfer-Encoding: base64

Jmx0O3AmZ3Q7QmhhdmlrLCZsdDsvcCZndDsK...      <- decodes to "&lt;p&gt;Bhavik,&lt;/p&gt;"
```

Gmail rendered the literal text `<p>Bhavik,</p>`. Root cause: the body handed to
the bridge was already `&lt;`-escaped. `MIMEText(body, "html")` escapes
**nothing** — it stores the string verbatim — so pre-escaped input is faithfully
preserved as visible tag soup.

**Rule:** with `html=True`, pass RAW markup (`<p>…</p>`). Never pre-escape it,
never `html.escape()` it, and never pass the output of a markdown→HTML converter
that escapes entities.

## Detection — the only reliable check

`mimeType` is not enough. Decode the part and assert on the content:

```python
import base64, email, sys
sys.path.insert(0, "/opt/hermes")          # REQUIRED before importing tools.gws_auth
from tools.gws_auth import build_service

svc = build_service("gmail", "v1", service_name="<resolved-service-name>")
d = svc.users().drafts().get(userId="me", id=DRAFT_ID, format="raw").execute()
msg = email.message_from_bytes(base64.urlsafe_b64decode(d["message"]["raw"]))

for part in msg.walk():
    if part.get_content_type() == "text/html":
        body = part.get_payload(decode=True).decode("utf-8", "replace")
        assert "<p>" in body and "&lt;p&gt;" not in body, "DRAFT IS DOUBLE-ESCAPED"
```

**Do NOT hand-split the raw text on `Content-Transfer-Encoding: base64`.** The
base64 payload wraps across many lines and a naive
`raw.split(...)[1].split("\n\n",1)[1]` picks up boundary/whitespace and produces
mojibake (verified: 13 chars of garbage instead of the 1,809-char body). Let the
`email` module parse it.

## Fix procedure for an already-broken draft

1. Snapshot — decode each part of the broken draft and `html.unescape()` the
   `text/html` payload to recover the intended markup.
2. Create a **fresh** reply draft: `draft_reply_create(message_id=<the INBOUND
   message you are replying to>, body=<unescaped HTML>, html=True)`. Target the
   inbound message id, not the broken draft — the bridge then rebuilds
   `In-Reply-To`/`References`, and Gmail fills `References` with the **full
   chain** (better than the single-id chain the bridge writes).
3. `drafts().delete()` the broken draft.
4. Re-verify the new draft's decoded part (assert above), then report the **new**
   draft id to the user — the id changes, so any id you quoted earlier is dead.
5. Keep the body content byte-identical; this is an encoding fix, not a rewrite.
   Only flag separate content/design changes as a distinct question.

## Finding the inbound message id to reply to

`threads().get(userId="me", id=<threadId>, format="metadata",
metadataHeaders=["From","Subject","Message-ID","Date"])` lists every message with
its `Message-ID` header and Gmail id. Match the `In-Reply-To` of the broken draft
against those `Message-ID` values to identify which inbound mail to thread
against. Choose the **latest inbound** from the counterparty (skip NDR's own
messages, which are also in the thread).
