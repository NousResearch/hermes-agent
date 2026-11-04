# Gmail Draft Formatting Pitfalls

## Draft Creation — Use the Bridge, Not Raw API

**Lesson (Jul 2026, Skylark Zenith reply):** Gmail drafts created via the raw API (`users().drafts().create()` with a `raw` MIME message + `threadId`) often do NOT appear inline in the Gmail thread view. The draft shows in the Drafts folder but when the user opens the email thread, it's invisible.

**Fix:** Always use `draft_reply_create` from `tools.gws_skill_bridge.call()` — it properly nests the draft in the Gmail thread with the `DRAFT` label.

```python
from tools.gws_skill_bridge import call
call('draft_reply_create', service_name='google-draas',
     thread_id='<threadId>', message_id='<messageId>',
     body='Your reply text', body_type='plain')
```

## HTML Rendering — Keep It Minimal

**Lesson (Jul 2026, Skylark Zenith reply):** Complex HTML with CSS colors, spans, nested divs, and `style` attributes does NOT render well in Gmail drafts created via the API. The HTML tags may show as raw text in the compose window.

**The user's preference (Nishant Ranka, Jul 2026):**
- Plain text format (body_type='plain' or minimal HTML)
- Simple bold tags `<b>` for emphasizing response content — no colors, no CSS
- The user explicitly rejected blue-font styling and complex HTML

**What works:**
```
# Use plain text with section markers:
call('draft_reply_create', ..., body='''Hi X,

--- Question 1 ---
Your bolded response here.

--- Question 2 ---
Your bolded response here.
''', body_type='plain')

# OR minimal HTML with ONLY <b> tags:
call('draft_reply_create', ..., body='''<b>Your bolded response here.</b>''', body_type='html')
```

**What to avoid:**
- `<span style="color:#...">` — rendered HTML tags as visible text
- Nested `<div>` with CSS — broke the compose view
- `<hr>`, styled elements — can cause rendering issues

## Reply-All and BCC — neither tool does both (verified 2026-10-17)

Two different tools, two different holes, and each one fails **silently**:

- **`gws_skill_bridge`'s `draft_reply_create`** accepts `to=` and `cc=` — and has **no `bcc=`** at
  all. It also nests the draft correctly in the thread.
- **The native `gws_gmail_reply_draft` / `gws_dwd_gmail_reply_draft`** accepts `bcc=` — and
  **silently ignores `to=`**. It hard-codes To to the original sender. It was called with
  `to="mdr@draas.com, dharmesh.ranka1@gmail.com"` and the created draft came back addressed to
  **only Manish**, with no Cc and no error. The second participant vanished.

So **any request for reply-all and/or BCC means hand-built MIME**:

```python
from email.mime.text import MIMEText
from tools.gws_auth import build_service
svc = build_service("gmail", "v1", service_name="google-draas")
orig = svc.users().messages().get(userId="me", id=ORIG_ID, format="metadata",
    metadataHeaders=["From","Subject","Message-ID"]).execute()
H = {h["name"].lower(): h["value"] for h in orig["payload"]["headers"]}
m = MIMEText(body, "html", "utf-8")          # minimal HTML only — see above
m["To"] = "Manish Ranka <mdr@draas.com>, Dharmesh Ranka <dharmesh.ranka1@gmail.com>"
m["Cc"] = "…"; m["Bcc"] = "rnr@draas.com"
m["Subject"] = H["subject"] if H["subject"].startswith("Re:") else "Re: " + H["subject"]
m["In-Reply-To"] = H["message-id"]; m["References"] = H["message-id"]
raw = base64.urlsafe_b64encode(m.as_bytes()).decode()
svc.users().drafts().create(userId="me",
    body={"message": {"raw": raw, "threadId": orig["threadId"]}}).execute()
```

- **Verify by reading the draft BACK** — `drafts().get(format='full')` and assert on `to` / `cc` /
  `bcc` before telling NDR it is ready. **Correction to the older note above: Gmail DOES return the
  `bcc` header on an *unsent draft*** — it is stripped from *sent* mail, not from drafts. Checked
  twice on the same draft.
- Reply-all rule: To = original sender + the other original To/Cc participants, minus yourself.
  For Manish → NDR + Dharmesh that is `To: mdr@draas.com, dharmesh.ranka1@gmail.com`.
- If a bad draft was already created, `drafts().delete(id=...)` it **before** creating the corrected
  one, so the thread does not carry two drafts.

## Minimal HTML for long multi-point briefings — this is the shape that works

`<p>` paragraphs with `<b>` on key phrases only — no `<div>`, no `style`, no `<hr>` — produced clean
drafts for a 6-paragraph internal directive and a 7-paragraph internal briefing, both correctly
threaded (2026-10-17). That satisfies NDR's "HTML for internal briefings / multi-point
coordination" preference without tripping the compose-view raw-tag bug above. Do not reach for CSS
just because the message is long.

## Recipient control — the NATIVE reply tool IGNORES `to=` (2026-10-17)

`gws_gmail_reply_draft` (and `gws_dwd_gmail_reply_draft`) hard-codes `To` to the original sender
and **silently discards the `to=` argument**. Verified: called with
`to="mdr@draas.com, dharmesh.ranka1@gmail.com"` + `bcc="rnr@draas.com"`; the created draft came back
`to: Manish Ranka <mdr@draas.com>` only — second recipient dropped, no Cc, **no error raised**.
The native tool DOES honour `bcc`; the bridge's `draft_reply_create` honours `to=`/`cc=` but has
**no** `bcc`. So:

- **Plain single-recipient reply** → the native tool is fine, or the bridge.
- **Reply-all and/or BCC** → hand-build the MIME. `tools.gws_auth.build_service("gmail","v1",
  service_name=...)`, fetch the original's `Message-ID` (metadata format — not the thread id), set
  `To`/`Cc`/`Bcc`/`Subject` (`Re: ` prefix if absent) / `In-Reply-To`/`References`, then
  `drafts().create(body={"message": {"raw": raw, "threadId": orig["threadId"]}})`.
- **Reply-all set** = original sender + the other original To/Cc participants, **minus yourself**.
  Read the To/Cc off the actual message headers — never from the contacts sheet.
- **Always verify** with `drafts().get(format="full")` and print To/Cc/Bcc/Subject/In-Reply-To
  before telling the user it's ready. That verification is the only way this silent drop gets
  caught.
- **Delete the bad draft first** (`drafts().delete(id=...)`) so two drafts don't sit on the thread.

## Bcc on a DRAFT is readable back — corrects an earlier claim (2026-10-17)

An earlier note said Gmail strips Bcc from API responses. True for *sent* messages; on an **unsent
draft** `drafts().get(format="full")` returns the `bcc` header verbatim, so you can assert on it
directly — `hh.get("bcc") == "rnr@draas.com"`, verified twice on the same draft.

## Bridge calls from `execute_code` DO work (2026-10-17)

`sys.path.insert(0, "/opt/hermes")` then `from tools.gws_skill_bridge import call` /
`from tools.gws_auth import build_service` worked from `execute_code` — three drafts created that
way in one session (`draft_reply_create`, raw-MIME `drafts().create`, `drafts().get` verification),
no ImportError. If `ImportError: cannot import name 'gws_fetch_token'` does appear, the fix is the
`sys.path` line — not abandoning the sandbox for `terminal`.

## ⚠️ Scope correction — the above is NOT a blanket rule (Sep 2026)

The Jul 2026 lesson above applies to **short threaded replies created via the bridge**
(`draft_reply_create`). Do **not** generalize it into "rich HTML doesn't work in Gmail" —
that reading is wrong and will block legitimate work.

**Counter-evidence (Sep 2026, DRA Liquidation Inventory dossier):** a full standalone
**HTML+CSS document** (gradient header, bordered asset cards, two-column tables, styled
callout boxes, 27 inline hyperlinks) built with `EmailMessage` +
`set_content(plain)` + `add_alternative(html, subtype="html")` and created via
`users().drafts().create(body={"message": {"raw": raw}})` rendered **correctly** —
verified by reading the draft back and by WeasyPrint render. NDR reviewed it and asked
for *more* HTML, not less.

**The discriminator is the delivery shape, not HTML itself:**

| Shape | Use |
|---|---|
| Short threaded reply → `draft_reply_create` bridge | plain text / minimal `<b>` |
| **Standalone new email → `EmailMessage` + `add_alternative`** | **full HTML+CSS is fine and often required** |

The `add_alternative` multipart shape is what makes rich HTML render reliably. A
single-part raw `text/html` MIME is the shape that historically broke. If the user asks
for a formatted HTML email, build the multipart alternative — do not refuse on the
grounds of this reference.

**Correctly reading back a single-part `text/html` draft:** the body sits at the
**top-level** `payload.body.data` with **no `parts[]`**. A recursive walker that only
descends `parts` returns empty and looks like a failure when the body is perfect. Check
`payload.body.data` BEFORE walking parts.
