# Matching the HTML format of an earlier email in the thread

**Trigger phrases from NDR:** "redo the draft using the same HTML formatting as the
earlier email in that thread", "so that it doesn't look very different", "make it look
like the original one", "the content is fine, just the formatting", "branded like the
last one".

**Rule:** when a follow-up / revision lands in a thread whose earlier message was a
branded HTML card, a plain-text draft looks broken sitting underneath it. Do NOT
hand-design a new look and do NOT reach for a generic template. **Extract the earlier
message's HTML part verbatim, keep its shell byte-for-byte, and re-flow the new
content into the same blocks.** "Same formatting" means identical container, not a
similar one.

## Steps

1. **Resolve the account** with `gws_resolve_account` (ndr@draas.com = `google-draas`).
   Never guess the vault key.
2. **Fetch the earlier message in full and pull both MIME parts.** The `gws_*` thread
   tool returns text bodies only, so go to the API:

   ```python
   import sys, base64, json
   sys.path.insert(0, "/opt/hermes")
   from tools.gws_auth import build_service
   svc = build_service("gmail", "v1", service_name="google-draas")

   m = svc.users().messages().get(userId="me", id=MSG_HEX_ID, format="full").execute()

   parts = []
   def walk(p):
       b = p.get("body", {})
       if b.get("data"):
           parts.append((p.get("mimeType", ""),
                         base64.urlsafe_b64decode(b["data"]).decode("utf-8", "replace")))
       for sp in p.get("parts", []) or []:
           walk(sp)
   walk(m["payload"])
   ```

   Also read `payload.headers` and keep: `message-id`, `subject`, `from`, `to`,
   `date`. The **RFC** `message-id` (`<CAPD...@mail.gmail.com>`) is what goes into
   `In-Reply-To` / `References` — the hex id `1a17...` is Gmail's internal id and is
   NOT valid there.
3. **Record the shell invariants** (see `templates/dra-branded-email-card.html`):
   `#eef1f0` page background, 700px white card with `1px #d8dedb` border, 3px
   `#1f4d3a` rule under the header, uppercase 11px eyebrow with 2px letter-spacing,
   21px `#1f4d3a` bold title, 13px `#6b7772` sub-line, 15px/24px `#222a26` body,
   `#f4f7f5` header-strip section boxes, amber callout `#fbf7e9` / `#e8d9a6` with a
   4px `#c9a227` left bar, table header row `#1f4d3a` white 12px, alternating rows
   `#f7f9f8`, totals row `#eef3f0`, grey note box `#f6f8f7`, footer 12px `#7d8a83`
   above a `#e4e9e6` rule. Font stack is always `Arial,Helvetica,sans-serif`.
4. **Map the new content onto the SAME block types, 1:1 with the original.** Header →
   intro paragraphs → grey-header decision/summary box → amber callout → table →
   worked-out note → grey note box → closing paragraphs → sign-off → footer. Do not
   invent a block type the original didn't have. If the original had a computation
   table and the follow-up restates the same numbers, reproduce the table; that is
   what makes the two mails read as one series.
5. **Build `multipart/alternative` raw MIME** with a plain-text fallback that mirrors
   the HTML content (Gmail clients that strip HTML must not see an empty mail):

   ```python
   from email.mime.multipart import MIMEMultipart
   from email.mime.text import MIMEText
   from email.utils import formatdate

   msg = MIMEMultipart("alternative")
   msg["From"]    = "Nishant Ranka <ndr@draas.com>"
   msg["To"]      = "Dharmesh Ranka <dharmesh.ranka1@gmail.com>, Manish Ranka <mdr@draas.com>"
   msg["Subject"] = "Re: " + ORIGINAL_SUBJECT          # keep the original punctuation
   msg["In-Reply-To"] = ORIG_RFC_MESSAGE_ID
   msg["References"]  = ORIG_RFC_MESSAGE_ID
   msg["Date"] = formatdate(localtime=True)
   msg.attach(MIMEText(TEXT, "plain", "utf-8"))
   msg.attach(MIMEText(HTML, "html", "utf-8"))

   raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
   d = svc.users().drafts().create(
       userId="me",
       body={"message": {"raw": raw, "threadId": THREAD_HEX_ID}},
   ).execute()
   ```

   Passing `threadId` is what keeps the draft in the same conversation.
6. **Delete the draft it supersedes** (`drafts().delete(userId="me", id=OLD_DRAFT_ID)`)
   so the thread holds exactly one draft. Leaving both means NDR has to work out
   which one to send.
7. **Verify by reading it back** — do not trust the create response:
   headers (`from`, `to`, `subject`, `in-reply-to`, `references`), `labelIds` must be
   exactly `["DRAFT"]`, `threadId` unchanged, both MIME parts present, and the HTML
   part still carrying the shell markers (`#1f4d3a`, the table header text, `#fbf7e9`).
   Compare the HTML length against the original's — within a few dozen characters of
   it is the signal the shell survived intact.

## Pitfalls

- **`.json()` on a Google API call fails** with `'HttpRequest' object has no attribute
  'json'`. Resource methods return a request object — call `.execute()`.
- **Hex id vs RFC message-id.** `In-Reply-To` / `References` take the RFC header value
  from `payload.headers`, with angle brackets. Using the hex id silently breaks
  threading.
- **Keep the original subject's punctuation.** If the original used em dashes (—),
  the reply must too. Swapping them for hyphens is exactly the "looks different"
  NDR complained about.
- **`build_service` needs `sys.path.insert(0, "/opt/hermes")`** and only works where
  `hermes_tools` is importable — i.e. inside `execute_code`. A bare
  `python3 -c` subprocess fails on `No module named 'hermes_tools'`; don't spend a
  round diagnosing the vault when it is just an import-path problem.
- **Content stays as-is.** When NDR says "the content is fine", the only thing that
  may change is the container. Don't take the opportunity to rewrite sentences, and
  don't add claims or numbers the original draft didn't have.
- **Don't send.** Always draft-only; report the draft id and what was verified.
