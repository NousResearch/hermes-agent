# Revising a live draft + how to phrase a REQUEST to an internal colleague

Two lessons from Bharat's 16-Oct Gowri Singh hoarding draft (13 hoardings, Ranka Udaya).

---

## 1. Tone rule: never write "this was not our scope" in an internal request

Bharat's verbatim correction (17-Oct, voice):

> "whatever you have prepared … it was not under our scope but you cannot keep the
> same thing in the draft, **it looks rude**. … so we need to get the 13 hoardings
> prepared from Gowri ma'am, **so we are requesting her** — can't be so rude."

The draft opened with:

> "To be clear on the sequence – preparing these posters was not our scope, but we
> have prepared them anyway… Please treat the attached posters as a reference only."

It was factually defensible (we did extra work outside scope) and still wrong:
when the SAME email asks the recipient to do the work, a scope complaint reads as
"you owe us" framing and sours the request. Bharat's rule generalises:

**Banned in any internal mail that contains a request to the recipient:**
- "not our scope", "outside our scope", "not in scope", "this is your job"
- "we did this anyway", "we had to step in", "we are not responsible for"
- "as already told / as I said earlier", "for your information and action"
- "kindly note that this was pending from your side"
- "please treat X as reference only" when X is work we did to help them — it
  devalues the help. Say "working reference" / "starting point" instead.

**Replace with ownership-transfer framing** — the same facts, granted as a gift:

> "A quick note before the details. We have prepared a direction poster for each of
> the 13 locations ourselves, purely so that the distance, the drive time and the
> arrow direction are settled before design begins. Please treat these as a working
> reference – **the design itself is entirely in your hands**."

Three moves that make it a request rather than a rebuke:
1. State what *we* did as service ("we have prepared…ourselves, purely so that…"),
   never as a burden or a scope violation.
2. Explicitly hand the recipient their authority — "the design itself is entirely
   in your hands", "final call is yours".
3. Add one line offering to help: "Happy to walk you through any of the locations on
   a call before you begin." (Bharat kept this addition.)

Keep the ask, the data table, and the two confirmation points unchanged — the fix is
the framing paragraph, not the substance. Do NOT pad with thanks or apology either;
the same draft must not swing from rude to obsequious.

Applies to: Gowri Singh (creative head), Sinchana, Anbu, Prakash, Roshini, and any
internal @draas.com colleague from whom we are requesting a deliverable.

---

## 2. Revising an EXISTING Gmail draft that carries an attachment

`gws_gmail_draft_create` has **no attachment parameter**, and there is no
draft-update tool. Editing the stored raw MIME in place means re-encoding a
quoted-printable/base64 part by hand — fragile.

Reliable pattern (used successfully here, account `google-draas` = sales1.blr@draas.com):

```python
from tools.gws_auth import build_service
svc = build_service("gmail", "v1", service_name="google-draas")

# 1. draft -> message id -> FULL message (see pitfall below)
d   = svc.users().drafts().get(userId="me", id=OLD_DRAFT_ID).execute()
mid = d["message"]["id"]
msg = svc.users().messages().get(userId="me", id=mid, format="full").execute()

# 2. pull plain body + every attachment's bytes back down
def flat(part, out):
    out.append(part)
    for p in (part.get("parts") or []):
        flat(p, out)
    return out                       # RETURN the list; don't iterate the None return

for p in flat(msg["payload"], []):
    if p.get("filename"):            # attachment
        pid = p["body"]["attachmentId"]
        data = svc.users().messages().attachments().get(
                   userId="me", messageId=mid, id=pid).execute()["data"]
        blob = base64.urlsafe_b64decode(data)
    elif p["mimeType"] == "text/plain" and body is None:
        body = base64.urlsafe_b64decode(p["body"]["data"]).decode("utf-8")

# 3. edit the BODY TEXT, rebuild MIME from scratch (attachment re-attached)
m = MIMEMultipart()
m["To"] = "Gowri Singh <gsingh@draas.com>"
m["Subject"] = <unchanged original subject>
m.attach(MIMEText(new_body, "plain", "utf-8"))
part = MIMEApplication(blob, _subtype="pdf")
part.add_header("Content-Disposition", "attachment", filename=<same filename>)
m.attach(part)
nd = svc.users().drafts().create(
        userId="me", body={"message": {"raw": base64.urlsafe_b64encode(m.as_bytes()).decode()}}
     ).execute()

# 4. VERIFY: read the new draft back (body text + attachment filename present)
# 5. delete the old draft so exactly ONE draft exists for that mail
svc.users().drafts().delete(userId="me", id=OLD_DRAFT_ID).execute()
```

Pitfalls learned:
- **`users().messages().get()` must be called with `format="full"` explicitly.** Without
  it the response can come back with `payload: None` and the part walk blows up with
  `'NoneType' object is not iterable`.
- A recursive `walk(part, out)` helper that appends to `out` returns `None` — call it
  as a statement (`walk(x, out)`), never `for p in walk(x, out)`.
- Keep the **original subject** unless the user asks to change it — the subject here
  already said "Request: Design of 13 Hoardings (Distance, Drive Time & Direction)".
- Always report to the user: which draft was replaced, the new draft message id, what
  the removed sentence said, and what it now says. Show the before/after wording in
  chat — Bharat reviews wording, not just delivery.
- **Never send.** Deleted-and-recreated is still draft-only; tell the user it is in
  Drafts for them to send.
- File the attachment bytes under a stable local path (`/tmp/att_<filename>`) so the
  new draft is built from the exact same PDF.

---

## 3. Voice-dictation decode notes (Bharat, 17-Oct)

Garbled nouns in that voice note, resolved from the mailbox rather than by guessing:
- "13 forces" → **13 hoardings**
- "Gauri map" → **Gowri ma'am** (Gowri Singh, gsingh@draas.com, creative head)
- "Anjali you have prepared" → **"the [draft] you have prepared"**

Method that worked: `gws_gmail_draft_list` on the requesting user's account, then
`gws_gmail_search` for the likely name (Gowri OR Gauri), then read the top draft. The
draft subject itself ("Request: Design of 13 Hoardings") confirmed every garbled
number in one shot. Do this BEFORE asking the user to repeat themselves.
