# Styled HTML email — packaging, plain-text alternative, and render verification

Companion to `references/html-email-templates.md` (design/CSS patterns). This file is about
**getting a hand-authored HTML email into a real Gmail draft correctly and proving it renders**
before you tell the user it is ready. Verified end-to-end on 2026-10-27 (Gunjur / Prestige
approval email, ndr@draas.com).

---

## 0. Job / API-server sessions: bootstrap the identity FIRST

In an `api_server` job session `HERMES_SESSION_USER_ID` exists in the environment but its value is
**empty**. Every native `gws_*` tool then answers:

```
No session user context (HERMES_SESSION_USER_ID not set) -- cannot check token status.
```

That is *not* a vault outage and *not* a missing token. Set the session context in-process, then
import the auth layer:

```python
import sys; sys.path.insert(0, '/opt/hermes')
from gateway.session_context import set_session_vars
set_session_vars(platform="api_server", user_id="ndr", session_key="<job key>")

from tools.gws_skill_bridge import call        # or: from tools.gws_auth import build_service
```

Order matters — `set_session_vars` must run **before** `build_service` / the bridge are used, because
`gws_auth` reads the identity ContextVar at call time.

Verify you landed on the right mailbox before doing anything else:

```python
gmail = build_service('gmail', 'v1', service_name='google-draas')
print(gmail.users().getProfile(userId='me').execute()['emailAddress'])   # ndr@draas.com
```

Use this only for the job owner's OWN account. Never use it to act on another person's mailbox.

---

## 1. `draft_create(..., html=True)` produces ONE part — no text/plain

The bridge call:

```python
call("draft_create", service_name="google-draas", to=TO, cc="", from_header="",
     subject=SUBJ, body=html_body, html=True)
```

returns a draft whose structure is a **single `text/html`** part. Fine in Gmail, but text-only
clients, some corporate gateways and digest/forward paths get nothing readable.

**Fix without creating a second draft:** create it with the bridge (that gets the account and From
right), then rebuild the SAME draft id as `multipart/alternative`:

```python
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header

msg = MIMEMultipart('alternative')
msg['To'] = TO
msg['Subject'] = Header(SUBJ, 'utf-8')
msg['From'] = 'Nishant Ranka <ndr@draas.com>'
msg.attach(MIMEText(text_body, 'plain', 'utf-8'))
msg.attach(MIMEText(html_body, 'html', 'utf-8'))
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

gmail.users().drafts().update(userId='me', id=DRAFT_ID,
                             body={'message': {'raw': raw}}).execute()
```

`drafts().update` keeps the same draft id, so the user still sees exactly one draft. Re-run this
whenever the HTML or text changes — it is idempotent.

Note the asymmetry: `drafts().create` + `threadId` is the pattern for *threaded replies*; here it is
a **new** email, so no `In-Reply-To` / `References` / `threadId` — it should open its own thread.

---

## 2. Plain-text alternative that actually reads

A naive `re.sub(r'<[^>]+>', '', html)` leaves table rows as `Structure |` fragments on separate
lines. Use `scripts/html_to_text_email.py` (in this skill):

```bash
python3 scripts/html_to_text_email.py email.html email.txt
```

How it works, and why:

- `</td>` → an em-dash **marker**, not a literal pipe, so a two-cell row becomes
  `Label — value` on one line.
- Rows whose two cells sit on **different source lines** (very common when the HTML is formatted
  by hand) are merged: a line ending in the marker pulls the next non-empty line up as the value.
- Trailing separators from the final empty `<td>` are stripped, and paragraphs keep their blank lines.
- Result is checked by eye before it goes into the draft — a mangled text part is worse than none.

---

## 3. Verify the render before delivering

There is no playwright in the system python on this host, and installing one for a preview is not
the job. Use the browser tool against the file on disk — it is faster and needs nothing installed:

```
write the HTML to /data/.../email.html
browser_navigate("file:///data/.../email.html")
browser_vision("Is the sheet centred, tables aligned, no overflow or cut-off text?")
browser_scroll("down")
browser_vision("Is the signature and footer clean, and the last table intact?")
```

Two vision calls (top + bottom) catch both classes of defect: table overflow/half-width columns at
the top, and a broken signature/footer at the bottom. Copy the returned `screenshot_path` into the
job folder and deliver it as previews.

**Gmail-safe HTML rules used (all confirmed rendering correctly):**

- Outer `role="presentation"` table, `width="700"`/`"760"`, `style="width:100%;max-width:NNNpx;"`
  so mobile shrinks it; the sheet is a `<td>` with `padding:34px`.
- Every style **inline**; no `<style>` block, no `var(--x)`, no flex/grid, no `box-shadow`.
- Zebra striping done with an inline `background` on the even `<td>`s, never `nth-child`.
- Section heads: `font-family:'Helvetica Neue',Arial,sans-serif; font-size:13px; letter-spacing:.13em;
  text-transform:uppercase; color:#8a1c1c; border-bottom:1px solid #d8d2c6`.
- Body: `Georgia,'Times New Roman',serif; font-size:15.5px; line-height:1.6`.
- Callout boxes are `<div>` with `background` + `border-left:4px solid` (gold `#c9a227` for a
  request, green `#0f5c46` for a decision/vote, grey `#d8d2c6` for context).
- Images would need CID parts — avoid them unless essential; a text-only brand block reads fine.

**Read the draft back and assert the substance**, not just the structure:

```python
d = gmail.users().drafts().get(userId='me', id=DRAFT_ID, format='full').execute()
# walk payload.parts, decode the text/html part, then assert the key figures are in it:
assert '52%' in body and '20 crore' in body
```

and print the structure (`multipart/alternative` → `text/plain` + `text/html`) plus the To/Subject
headers back to yourself. Never report "the draft is ready" from the create call alone.

---

## 4. The approval / buy-in email shape (a distinct content class)

When NDR wants the commercials put on record and everyone's affirmative before someone signs
(term sheet, JDA, authority to act), the email that works on him has these parts, in this order:

1. **One-line purpose + one explicit request** up front ("reply with a one-line affirmative;
   nothing else is needed at this stage"), repeated in a gold callout so it cannot be missed.
2. **A commercial-terms table** — one row per term, label in small caps on the left, value right.
   Every figure comes from his dictation or the minutes; nothing rounded or inferred.
3. **The one thing being attempted, separated from what is being approved.** Say plainly "this is an
   attempt, not a condition — if it fails we proceed on the terms in section 2". This keeps the
   approval clean when there is an open ask (e.g. moving from area share to a 50:48 revenue share).
4. **My vote, for the record** — a green callout that states the authorisation in one sentence.
5. **Per-person ask table** — one row per approver, each with what they must reply, not a generic
   "please confirm".
6. **The limit of the authority** — "it covers these terms and nothing beyond them; if any of them
   moves, it comes back to all of us". NDR values this; it is the sentence that makes the email
   safe to sign against.
7. **Sequence table** (now / next / before the JDA / then), then the sign-off and a footer citing
   the source (minutes item number + date).

**Recipient discipline:**

- Derive the approver set from the minutes' action line, not from the greeting. Here the task table
  read *"sign once all three approve"* → exactly three approvers, no Cc.
- If the user says "...and everyone", do **not** pad the To line with non-approvers. Keep the
  approvers, and state in your report which reading you applied plus a one-line offer to widen.
- Resolve addresses from earlier mail **about the same matter**, by printing To/Cc of the matching
  threads — not by recency. Mamta Rathod appears as `mamatadr@gmail.com`, `maadi32@gmail.com` and
  `mamta.r@drahomes.in`; the July landowner-cooperation mail on this very land used
  `mamatadr@gmail.com`, so that is the one to use.

---

## Pitfalls

- **`drafts().list()` empty + a previous job "updated a draft" = it was SENT.** Search the subject,
  then read `labelIds` on the hit. `['SENT']` means the earlier work went out; `['DRAFT']` means it
  is still waiting. Do not re-create it.
- **The content system can be missing an artifact you have already rendered.** `content_find`
  returning 0 results (and an artifact URL 404ing) does not mean the document is gone from the
  world — the earlier job's build files live in `/data/hermes/tmp/jobs/<job>/*.html|*.txt`, and the
  SENT copy in Gmail carries the same text. Use those, and say in your report which source you
  used and that the content-system copy was unavailable.
- **Don't invent mechanics the user did not state.** "A 50:48 revenue-share proposal for the
  landowners" was dictated; who it is offered to (Prestige, or the landowners under us) was not.
  Keep his wording, flag the ambiguity in the report, and offer to sharpen it.
