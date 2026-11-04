# Creating a Gmail draft with a LARGE HTML body (and verifying it)

For any draft whose body is too big to type into a native tool call — a full document emailed inline
(entitlement statements, proposals, circulars). Worked example: a ~73 KB HTML landowner statement
(Oct-2026). Everything here is draft-only; `gmail_send` / `gmail_reply` are permanently blocked in
`tools/gws_skill_bridge.py` and a draft is the only compose path.

## 1. Build the body as a file, then create the draft through the bridge

```python
import sys; sys.path.insert(0, "/opt/hermes")
from tools.gws_skill_bridge import call

body = open("/tmp/<job>/email.html").read()
call("draft_create",
     service_name="google-draas",          # resolve with gws_resolve_account, never guess
     to="one@recipient.com",               # explicit — never rely on a default
     cc="a@x.com, b@y.com",
     subject="…",
     body=body,
     from_header="",
     html=True)                            # MANDATORY, or the tags ship as literal characters
```

Run it as `cd /opt/hermes && env -u PYTHONPATH /opt/hermes/.venv/bin/python3 script.py` — the
`env -u PYTHONPATH` matters, because `PYTHONPATH=$HOME/.local` shadows the venv packages and `tools.*`
stops importing. Returns
`{"status":"draft_created","draft_id":"r-…","message_id":"…"}` — keep both ids; the delivery message
should carry them so the user can find the draft.

## 2. Verify by reading the draft back — through the drafts endpoint

**The native `gws_gmail_get` returns an EMPTY `body` for an HTML-only draft.** It looks for a
`text/plain` part, and an HTML-only message has none. `body: ""` is therefore NOT a failure and NOT
evidence the draft is broken — do not "fix" it by recreating the draft. Confirm the draft exists with
`gws_gmail_draft_list` (it shows `labelIds: ["DRAFT"]`, and `gws_gmail_get` on the message id does return
correct To/Cc/Subject/From), then read the actual HTML through the Gmail API:

```python
import base64
from tools.gws_auth import build_service
svc = build_service("gmail", "v1", service_name="google-draas")
d = svc.users().drafts().get(userId="me", id=DRAFT_ID, format="full").execute()
msg = d["message"]
h = {x["name"].lower(): x["value"] for x in msg["payload"]["headers"]}   # assert to/cc/subject/from

def find_html(p):
    if p.get("mimeType") == "text/html": return p
    for c in p.get("parts", []) or []:
        r = find_html(c)
        if r: return r

html = base64.urlsafe_b64decode(find_html(msg["payload"])["body"]["data"]).decode("utf-8", "replace")
```

Also note: the bridge's own `gmail_get` operation routes to the underlying skill module's own auth path
(it can answer `Not authenticated. Run the setup script first`) — `draft_create` is implemented in the
bridge directly against the Gmail API, so creation succeeds where that read path does not. Use the native
tool or `build_service` for reads; the bridge for the write.

## 3. Assert on the decoded HTML, not on the JSON

Probe for content regressions rather than eyeballing a 70 KB string:

```python
assert h["subject"] == EXPECTED and "DRAFT" in msg["labelIds"]
print(len(html), html.count("<table"), html.count("drive.google.com/file/d/"))
for probe in HEADLINE_FIGURES + OPENING_PHRASES:   # every figure + the greeting/sign-off
    assert probe in html, probe
```

A healthy result for the worked example: 72,825 chars, 10 tables, 26 Drive links, all figure and phrase
probes True. Also confirm `gws_gmail_draft_list` shows exactly **one** draft — two drafts for the same
send confuse the recipient and the user.

## 3.5 The flag trap, and repairing a batch that already went in broken (2026-10-19)

The bridge's `html=True` is mandatory (§1). The **native tools are a separate trap**: 
`gws_gmail_draft_create` / `gws_gmail_reply_draft` / the `gws_dwd_*` equivalents also take an
`html` flag and it **defaults to false**. Omit it and the call still succeeds — but Gmail files the
markup as a `text/plain` part, and the user opens Drafts to see `<p>`, `<ol>`, `<b>` as literal text.
NDR's words: *"All of them show up in the draft with HTML code rather than rendered HTML."*
Eight drafts in one batch were all broken this way; the flag is per-call, so one forgotten argument
breaks one draft and leaves its siblings fine.

**Always verify the whole batch before reporting drafts as ready** — `drafts().get(format="full")`,
then per draft assert a `text/html` part exists and that no `text/plain` part contains `<p>`/`<ol>`/`<ul>`.
(§2 already warns that header names come back **lower-cased** on these messages — `h.get('Subject')`
returns `None` and falsely looks like a missing subject. Compare case-insensitively.)

**Repair procedure:**
1. `drafts().delete(userId='me', id=<draft_id>)` for each broken draft.
2. Recreate with the flag set; for replies pass the **same `message_id`** so In-Reply-To/References
   survive and the draft lands on the right thread.
3. Assert the returned `thread_id` equals the source thread for every reply.
4. **Rewrite the columns/notes that carried the old draft ids** (replace the id string, don't merely
   append a note) — dead ids open a blank compose window when the user clicks them.
5. Cancel any superseded WhatsApp draft on the same subject (`wa_draft_cancel`) so only one live
   draft exists per intent, then re-run the verification loop and report.

**Prevention:** when a task batch generates several drafts, build them through one helper that
hard-codes the html flag — the omission is always an accident, never a choice.

## 4. Email-specific HTML rules

- **Inline styles only.** Strip `<link rel=stylesheet>` and any `@page`/print-page contract; put the
  colours, padding and borders on the elements themselves (`style="background:#fafcfe;padding:8px 10px;
  border:1px solid #e5ebf1"`). External stylesheets and `@page` are for the printable artifact, not the
  mail.
- **Keep the document's visual identity** — a ~900px wrapper, the brand header fill on `<th>`, and the
  same row-background conventions the artifact uses, so the email and the artifact cannot drift.
- **Source the tables from the published artifact's own HTML**, not from a re-typed copy.
- **Do not link a private artifact to external recipients.** If the deliverable also lives in a private
  content system, a link there is a dead end for the recipient — either grant them access first or omit
  the link and say which you chose.

## 4.5 Status / coordination mails must be the BRANDED layout, not bullets (user correction, Oct 2026)

The user rejected a clean plain-text bulleted draft of a registration-coordination mail with
*"I don't see this is in a proper format … it has to look a little bit more professional"*.
For any mail that reports status, coordinates people, or lists items to be closed, build the
branded HTML layout instead of bullets:

- 680px white card on `#f4f5f7`, 1px `#e3e5e8` border
- navy header band `#1a3a5c` — bold white title + light subtitle (project · date)
- numbered sections, each an UPPERCASE `#1a3a5c` heading with a 1px bottom rule
- particulars as a two-column table with a shaded `#f7f8fa` label column
- **anything the recipient must act on** in an amber callout (`#fff8e1`, 4px `#e0a800` left border)
- links inline and clickable; sign-off with name + entity; small grey footer with the reference line

Ready-to-copy skeleton: **`templates/professional-coordination-email.html`** (placeholders only —
change the content, not the layout). Always send `multipart/alternative` with a real `text/plain`
fallback so a plain client still reads cleanly.

## 4.6 The draft can vanish between turns — keep the HTML, verify before reporting

`drafts().get()` returning **404 `Requested entity was not found`** means the user deleted (or
already sent) the draft in Gmail — not that the id is malformed. Check `drafts().list()` and the
thread's own messages before concluding anything: if the thread still shows only the earlier sent
mails, the draft was deleted and must be **recreated**, not "updated".

So always keep the built HTML on disk (`/tmp/<job>/mail_preview.html`); recreating from the saved
file reproduces the approved layout exactly, whereas re-typing it drifts. Recreate into the **same
thread** with the original `In-Reply-To` / `References`, and report the NEW draft id — the old id
is dead and any note or sheet cell carrying it opens a blank compose window.

Before reporting links as ready, confirm each one opens for **every** recipient: an index/folder
link pasted for an external Cc (a partner on another domain) needs an explicit reader grant, and an
internal admin needs writer. Verify the permission list per file — "shared with the team" does not
cover a newly added address.

Then verify the render: `browser_navigate` to `file:///tmp/<job>/mail_preview.html` followed by
`browser_vision` asking specifically about broken markup, misaligned table columns and missing
sections is a fast alternative to the playwright route in §5.

## 5. Optional visual check

Render the HTML with playwright (present in `/opt/hermes/.venv`) and read the top back — text-bearing
crops come back as OCR, which is exactly the evidence needed for "no raw tags leaking, no overlap,
links visible, tables aligned":

```python
import os; os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/tmp/pw'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width':1000,'height':1400})
    pg.goto('file:///tmp/<job>/email.html')
    pg.screenshot(path='/tmp/<job>/full.png', full_page=True); b.close()
```

Crop with PIL, then `vision_analyze`. Treat it as a legibility confirmation only — the §3 assertions are
the correctness check.

## 6. Reporting

Give the user: draft id + message id · exact To / Cc / Subject · the read-back evidence (length, table
count, link count, figures verified) · anything deliberately not done (a colleague left off the Cc, a
private link omitted). **Never say it was sent** — say it sits in Drafts and they send it.
