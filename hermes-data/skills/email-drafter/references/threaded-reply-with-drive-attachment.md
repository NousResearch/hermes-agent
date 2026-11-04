# Threaded Gmail reply draft carrying a Drive-sourced attachment

Validated end-to-end **2026-10-16** on Sarthak Sharma's mailbox (`admin3.blr@draas.com`,
session service_name `google-draas`) — two consecutive drafts built this way:

1. **Sudheer Desu** `<sudheer@vgrofinserv.com>` (VGRO Finserv, lender-side due diligence)
   — thread "Re: reg:- information required" → attached the Mortgage Deed, document
   no. 7049/2025, 17-page registered scan, 946,590 B.
2. **Shivani Salunke** `<shivani.salunke@naukri.com>` — thread "Naukri.com :- DRA REALTY
   PRIVATE LIMITED" → attached the DRA Realty GST REG-06 certificate, 3 pages, 108,106 B.

Both ended as `DRAFT` only, verified present in the Drafts folder, and were **never sent**.
Works identically for any DRA mailbox — resolve the account first, don't hardcode.

---

## Step 0 — The decisions to settle BEFORE building

Show the user, and get an answer on, these three — they are the only real variables:

- **Attach the file, or share a Drive link?** (For a lender/DD counterparty the answer is
  usually attach; for checklist/tracker work the house preference is a fresh folder copy.)
- **Recipient set** — reply-all participants, or the sender alone?
- **Does this mail also answer the counterparty's OTHER open questions?** Do not silently
  pad the draft. In the VGRO case the counterparty had listed 3 asks and the user said to
  drop two ("just drop this … just send it to Sudhir") — the mail carried exactly one item.

Do not create the draft until these are settled. A draft is not a send, but a draft built on
guessed recipients wastes a review cycle.

## Step 1 — Locate the thread

```
gws_gmail_search(service_name=<resolved>, query="from:<domain> OR from:<firstname>", max_results=25)
gws_gmail_thread_get(service_name=<resolved>, thread_id=<threadId>)
```

The search summary gives you `threadId` but **not** the threading headers. Fetch them:

```python
g.users().messages().get(userId="me", id=<mid>, format="metadata",
    metadataHeaders=["Message-ID", "References", "In-Reply-To",
                     "From", "To", "Cc", "Subject", "Date"]).execute()
```

Do this for **every** message in the thread, not just the last one — you need the full
References chain and you need the participant union (Step 3).

## Step 2 — Choose the reply target

Reply to the **last message in the thread** (that is what Gmail does when you hit Reply):

- `In-Reply-To` = that message's `Message-ID`
- `References`  = that message's existing `References` **+** that message's `Message-ID`
- `threadId` in the `drafts().create` body = the master thread id

Print the assembled chain and eyeball it before building — it runs to 5–6 message-ids and is
trivially easy to truncate. A truncated `References` drops the draft out of the conversation.

## Step 3 — Derive the recipient set (THE PITFALL)

**The user's description of the recipient set may not match the latest message's own
headers.** In the Naukri case the latest message carried
`To: admin3.blr@draas.com` and **no `Cc` header at all** — yet Gmail's thread-list line read
*"to me, Saurabh, Pooja"*, and the user quoted exactly that back. Gmail's thread list shows
the **union of thread participants**; the Cc set that actually mattered came from the user's
**own earlier reply on the same thread** (he had Cc'd `saurabh.sud@naukri.com` and
`pooja.mandloi@naukri.com`).

Rule:

1. If the latest message's To/Cc is **narrower** than the participant set the user names,
   walk every message in the thread and collect all `From`/`To`/`Cc` addresses.
2. Reply-all to that union. Treat the user's own earlier reply as authoritative evidence of
   who he wants on the thread.
3. Never invent an address. Never silently drop a participant the user named.
4. State the resolved set back to the user in the hand-off ("Cc: … — say the word if you want
   just the sender") so a wrong set is caught at review, not after sending.

## Step 4 — Source the attachment from Drive

```python
d.files().get(fileId=<id>,
    fields="id,name,mimeType,size,modifiedTime,owners(emailAddress),parents,"
           "permissions(type,role,emailAddress)").execute()
d.files().get_media(fileId=<id>).execute()
```

- **Shortcuts.** DRA Drive is full of `application/vnd.google-apps.shortcut` rows that come
  back with `size: null`. Read `shortcutDetails.targetId` and fetch **that** id — calling
  `files().get` / `get_media` on the *shortcut* id returns `404 File not found`.
- **Verify the document is the right one before attaching.** Download it and read it. When
  the mail names a document number, a GSTIN, or an entity, confirm those strings appear in
  the file itself — never attach on filename alone.
  (For registered Indian property documents see `ocr-and-documents` →
  `references/registered-property-document-ocr.md`.)
- **Prefer the native PDF over a scan of the same document.** Three copies of the DRA Realty
  GST certificate existed (two byte-identical native PDFs + one 5× larger scan). Ship the
  small, digitally-signed native PDF.
- **Rename on the way out.** Source names are machine-generated (`... (4).pdf`, double
  spaces, trailing noise). Give the outbound attachment a clean, human filename; leave the
  Drive original untouched. Example: `20251310 Mortgage Deed 7049-2025 Between SLP &
  Investors (2.5 Acres Sevaganapalli)  (4).pdf` →
  `Mortgage Deed 7049-2025 - Sevaganapalli - 13.10.2025.pdf`.

## Step 5 — Build the raw MIME and create the draft

Write the script to `/tmp/<name>.py` with `write_file` (a heredoc containing `&` is rejected
as backgrounding), then run it with `/opt/hermes/.venv/bin/python3`.

```python
import sys, base64
sys.path.insert(0, "/opt/hermes")
from email.message import EmailMessage
from tools.gws_auth import build_service

THREAD_ID = "<master thread id>"
LAST_MID  = "<Message-ID of the last message in the thread>"
REF_CHAIN = " ".join([...])          # existing References + LAST_MID

msg = EmailMessage()
msg["To"] = "..."
msg["Cc"] = "..."                    # omit the header entirely if no Cc
msg["Subject"] = "Re: <original subject>"
msg["In-Reply-To"] = LAST_MID
msg["References"] = REF_CHAIN
msg.set_content(PLAIN, subtype="plain", charset="utf-8")
msg.add_alternative(HTML, subtype="html", charset="utf-8")

with open(src_pdf, "rb") as f:
    data = f.read()
msg.add_attachment(data, maintype="application", subtype="pdf",
                   filename="<clean human filename>.pdf")

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
g = build_service("gmail", "v1", service_name="<resolved>")
created = g.users().drafts().create(userId="me", body={
    "message": {"raw": raw, "threadId": THREAD_ID}
}).execute()
```

Do **not** set `Message-ID` yourself on a fresh draft — Gmail mints it. (Re-supplying it only
matters when *re-creating* a draft you already made; see the update-in-place reference.)

## Step 6 — Verify before reporting (always)

```python
lst = g.users().drafts().list(userId="me", maxResults=20).execute()
print([(d["id"], d["message"]["threadId"]) for d in lst.get("drafts", [])])

d = g.users().drafts().get(userId="me", id=DID, format="full").execute()
m = d["message"]
h = {x["name"]: x["value"] for x in m["payload"]["headers"]}
print(m["threadId"], m.get("labelIds"))
for k in ["From", "To", "Cc", "Subject", "In-Reply-To", "References"]:
    print(k, str(h.get(k))[:260])

# walk the MIME tree; confirm multipart/mixed -> multipart/alternative -> plain+html, + pdf
# then byte-verify the attachment against the source file:
for p in m["payload"]["parts"]:
    if p.get("mimeType") == "application/pdf":
        a = g.users().messages().attachments().get(
            userId="me", messageId=m["id"], id=p["body"]["attachmentId"]).execute()
        print(p["filename"], p["body"]["size"], len(a["data"]) * 3 // 4)

# and print each text part's DECLARED type next to its opening characters
# text/plain | '<div style=...'  == the HTML-demotion bug
```

Confirm, in order: draft appears in `drafts().list()`; `threadId` is the master thread (not a
fresh one); To/Cc match the agreed set; `In-Reply-To`/`References` present; MIME tree shape;
attachment declared size == source byte count; text/html part really opens with `<div`.

## Pitfalls

- `drafts().get()` does **not** accept `metadataHeaders` — use `format="full"` and parse
  `payload.headers`.
- A returned draft-id string proves nothing on its own. List the folder **and** read the
  draft back.
- `body.size` on the attachment is authoritative; `len(base64)*3//4` can read one or two
  bytes high because of padding. Compare against the source file's exact byte count.
- The draft's `Date` header comes back as API creation time with a UTC offset — harmless,
  do not "fix" it.
- Report the outcome honestly: **"draft created, open Gmail → Drafts → review → Send."**
  Never say "sent".
