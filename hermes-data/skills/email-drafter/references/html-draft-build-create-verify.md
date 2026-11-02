# HTML email drafts — build, create, verify (job / api_server sessions)

Use when NDR asks for an email "in HTML/CSS so it is easy to read", "presented using that HTML
rendered itself, not this difficult to read text only", or any follow-up where the body must look
like a document rather than prose. Also the runbook for revising an existing HTML draft in place.

## 0 · Session identity (job / api_server sessions have no session user)

Native `gws_*` tools fail closed there. Use a script file:

```python
import os, sys
os.environ['HERMES_SESSION_USER_ID'] = 'ndr'      # BEFORE importing gws_auth; setdefault fails
sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import build_service
svc = build_service('gmail', 'v1', service_name='google-draas')
```

## 1 · Before writing a word

1. `svc.users().drafts().list(userId='me')` — is there already a draft for this matter? If yes,
   **update it** (`drafts().update`, same draft id, new raw). A second draft for the same subject
   is the failure NDR notices.
2. Search the matter's history so the email's facts match the record:
   `subject:(Hermitage) newer_than:45d`, `"car park" OR "car parking" newer_than:45d`,
   `subject:(khata) newer_than:120d`. Pull the prior note/minutes and re-use its figures.
3. Addresses from GMAIL HEADERS, not the contacts sheet. Dharmesh Ranka = `dharmesh.ranka1@gmail.com`
   (Ndr often Ccs `ddr@draas.com` and `dharmesh.ranka@gmail.com`); Manish Ranka = `mdr@draas.com`.
4. "Addressed to X with Y on CC" is literal — X in To, Y in Cc, nobody else.

## 2 · Email-safe HTML rules

- Inline style on every element. **No** `<style>` block, `var()`, flex, CSS grid or media queries —
  Gmail collapses them.
- Layout with `<table role="presentation" cellpadding="0" cellspacing="0" border="0">`,
  `width="700"` plus `style="width:700px;max-width:700px"` on the sheet.
- Data tables: `border-collapse:collapse`, explicit widths on the fixed columns, zebra rows via
  inline `background` on `<tr>`.
- DRA palette used in practice: deep green `#1f4d3a` (header band, table head, unit labels),
  body `#222a26`, muted `#6b7772`, panel `#f4f7f5`, pale green total row `#eef3f0`, amber callout
  `#fbf7e9` with `border-left:4px solid #c9a227`. Font Arial/Helvetica; 15px/24px body, 21px title,
  11–12px uppercase section labels with `letter-spacing`.
- Callout box = one-cell table with background + border (and a coloured `border-left` for emphasis).
- Always ship a plain-text alternative: `msg.set_content(txt)` then
  `msg.add_alternative(html, subtype='html')`.
- Entities for typographic characters (`&middot;`, `&nbsp;`, `&mdash;`); keep the text part ASCII.

## 3 · Create or update the draft

```python
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
svc.users().drafts().create(userId='me', body={'message': {'raw': raw}})            # new
svc.users().drafts().update(userId='me', id=draft_id, body={'message': {'raw': raw}})  # revise
```

`scripts/create_or_update_html_draft.py` in this skill does exactly this — pass `--html`, `--text`,
`--to`, `--cc`, `--subject`, and optionally `--update <draft_id>`. It writes `draft_id.txt` so a
later revision reuses the same draft.

## 4 · Verify before reporting (never describe, always read back)

1. Read the message back and walk the payload: expect `multipart/alternative` → `text/plain` +
   `text/html`, `labelIds == ['DRAFT']`, sane `sizeEstimate`.
2. Assert the facts are in the HTML: every party/unit name and every computed figure;
   assert `'<style' not in html` and `'var(' not in html`.
3. Render a real preview: `browser_navigate('file:///abs/path/email.html')` then
   `browser_vision(...)`. The returned `screenshot_path` is a PNG — copy it next to the source as
   `<name>_preview.png` and list it as a job artifact so NDR can see the layout without opening
   Gmail. This is the only local render path available (no playwright / weasyprint / chromium binary
   in the container); do not fight it, just use the browser tool.
4. Re-render after any edit that touches visible text (the first screenshot goes stale silently).

## 5 · Dictation handling — the part that earns trust

- NDR dictates the body. Keep **his wording and his figures verbatim**, even where they differ from
  an earlier note. Never silently "correct" a number you think is wrong.
- Finish every computation he sketches. Example from the Hermitage khata mail: he gave
  "typically 5.5 × 2.5, ours 6 × 3 for abundant caution" → the email carries 13.75 sq m per park,
  18 sq m per park, per-unit totals (4 × 18 = 72; 6 × 18 = 108), the aggregate (72 + 4×108 = 504
  sq m on 28 parks) **and** the delta against the convention (4.25 sq m more per park, 119 sq m
  more in aggregate).
- If his numbers contradict a document you found, put the discrepancy in the **job summary**, never
  inside the email — he decides. Same for an unclear dictation (e.g. "F number 4"): carry it
  verbatim, and say in the summary that you could not verify it against a source.
- Date line: do not trust the clock. The Gmail API `Date` header comes back in the server's zone
  (`-0600`) while the business date is IST; a draft built at 18:41 IST on the 27th first rendered
  as "28 October". Take the date from the conversation/session and check it in the screenshot.

## 6 · Pitfalls

- `messages().get` needs the **message** id (first column of a search hit), not `threadId` — using
  the thread id returns `404 Requested entity was not found`.
- A plain-text file that carries a `Subject:` line for reference must have that line stripped before
  `set_content()`, or the subject appears twice in the message body.
- After `drafts().update` the **message id changes, the draft id does not** — re-read the returned
  ids rather than reusing the old message id for verification.
- Nothing is ever sent. Close the summary with "it is a draft; you review and send".
