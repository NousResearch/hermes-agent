# WhatsApp document sharing — links, access, and message shape

Trigger: *"I need to share both the documents through WhatsApp… create me a
link… drop a message with both the deeds."*

## Delivery preference (Bharat, Oct 2026) — plain text, not a deep-link blob

The `whatsapp_link` tool returns a `display_link` that renders the entire
message as one clickable block. Bharat's correction: *"can you just give me a
normal text message like you know including that link… if you click on that link
that document has to be visible for anyone."*

So for **document-sharing** messages, deliver:

1. A fenced ``` block of the message text — clean, copy-pasteable, links inline
   as plain URLs.
2. The links listed separately, so he can paste them one at a time.
3. The `display_link` as a small "Open WhatsApp → <name>" tap link, as a
   convenience only — never as the primary payload.

Deep-link blobs are fine for short replies; they are wrong for a multi-paragraph
brief with three URLs in it.

## Make the links actually open for a stranger

The recipient is often outside the org and may not be signed in. A restricted
Drive link is useless over WhatsApp.

```python
import os, sys
os.environ["HERMES_SESSION_USER_ID"] = "sales1_blr"      # the owning user's slug
os.environ["GWS_VAULT_SOCKET"] = "/run/gws-vault/vault.sock"
sys.path.insert(0, "/opt/hermes")
from tools.gws_auth import build_service
svc = build_service("drive", "v3", service_name="google-draas")

# 1) dedicated share folder inside the project's existing folder — easy to revoke
FID = svc.files().create(
    body={"name": "Plot 36 - TN Sale Deed: Our Draft vs Portal Draft",
          "mimeType": "application/vnd.google-apps.folder", "parents": [KIT_FOLDER_ID]},
    fields="id,webViewLink").execute()["id"]

# 2) folder-level anyone-with-link reader (files inside inherit)
svc.permissions().create(fileId=FID,
                         body={"type": "anyone", "role": "reader"},
                         fields="id").execute()
```

**The native `gws_drive_permissions_add` tool cannot do link-sharing** — it only
takes `perm_type: user | domain`. Anyone-with-link requires `build_service` as
above. Sharing the *folder* rather than each file keeps it revocable in one call
and limits exposure to exactly the documents in question.

**Verify anonymously before handing the link over** — do not assume the
permission took:

```bash
for u in "$LINK1" "$LINK2"; do
  code=$(curl -s -o /tmp/o.html -w "%{http_code}" -L "$u")
  echo "$code $(grep -c -o -E 'You need access|Request access' /tmp/o.html)  $u"
done
# want: 200 and 0
```

Strip `?usp=drivesdk` / `&ouid=…` / `&rtpof=true` — the bare
`https://drive.google.com/file/d/<id>/view` is what you paste.

## Give a phone-readable PDF alongside any .docx

A `.docx` on Drive opens in the Docs viewer, which on mobile often bounces the
recipient into the Google Docs app or a sign-in. Render a PDF copy and send that
link as the primary, keeping the `.docx` in the folder for the advocate.

Container reality: no libreoffice / weasyprint / chromium. **reportlab is
present.** Rebuild a clean PDF straight from the docx text:

```python
import zipfile, re, html
x = zipfile.ZipFile(SRC).read("word/document.xml").decode("utf8")
x = re.sub(r"</w:p>", "\n", x)
t = html.unescape(re.sub(r"<[^>]+>", "", x))
paras = [p.strip() for p in t.split("\n") if p.strip()]
# then ParagraphStyle Times-Roman 10/13.6, TA_JUSTIFY; treat lines whose
# alpha chars are >85% uppercase (and len<=90) as headings
```

8 pages, ~27 KB, reads correctly on any phone.

## Check the folder's existing sharing before you circulate it

```python
svc.permissions().list(fileId=FID, fields="permissions(type,role,emailAddress)").execute()
```

Real find: the Ranka Oasis `RENDER` folder was already **`anyone` / `writer`** —
anyone with the link could *edit* a design folder. Flag that to the user and name
the owner (you may only hold `commenter` yourself, so you cannot fix it). Same
check applies to any folder you are about to hand out.

## Message shape that worked (external professional, query, at NDR's instruction)

1. Who it is for, one line of context (project, plot, vendees).
2. **"This is at Nishant sir's instruction"** — establishes the mandate.
3. Numbered list: each document, who prepared it, and what it contains — with its
   link immediately under it. Never a bare "see attached".
4. The other party's position, stated fairly: *"Sampath's position is that…"*
5. Our concern, concretely: name the clauses that got dropped, and say plainly
   which fields are placeholders that will be filled later (so the reader does
   not litigate the placeholders).
6. Numbered questions — (a) is this correct, (b) what minimum must be typed back.
7. "Happy to get on a call if that is easier."
8. Signed in the user's own name.

## Pitfalls

- `contact_resolver` fails with *"User '<id>' has no gws_service configured"* for
  non-default session users — pass `account_email="sales1.blr@draas.com"`.
- There are **14** "Kamal" rows in the contacts sheet. State the number you used
  and make the user confirm before they tap, rather than silently picking one.
- Never report a generated deep link as "sent". It is a link; the user taps it.
- Offer to revoke the anyone-with-link permission once the recipient has the
  files — the user cares, and it is one call.
