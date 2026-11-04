# Multi-photo review mail — inline HTML images + real attachments, and how short to make it

**Trigger:** NDR sends 8–12 site photographs and asks for "a detailed email ... attaching the
photos", usually as a quality/issue review to the site team, with an ask for feedback.
Verified 2026-10-24 (Ranka Oasis, Sevaganapalli — underground services and chambers, 10 photos,
to Anbarasan cc Kanthesh).

---

## 1. LENGTH — the correction that governs this whole class of mail

The first version was thorough and correct and NDR threw it out:

> *"The email is too elaborate, make it shorter, talk about just the issues, the suggested fixes, no
> need to give any background things, say that I analyzed the 10 photos through AI. This is what I
> found. I want an explanation. I want us to discuss it point by point ... So let's please discuss it
> in the morning."*

It had been roughly 11 screens: what the photos show, what is being done right, 12 issues at
120 words each, a rating table, 12 recommendations, a photo-by-photo catalogue, and a six-item
list of documents and method statements to produce. All of that content was *useful*; none of it
was *wanted*. The mail he actually sent is ~2 screens.

**Default shape for any issue / quality-review / snag mail to the site team:**

1. One line of context, including the method: *"I ran the 10 photographs posted in the Oasis daily
   update group between 13 and 23 October through an AI analysis, as a quality check rather than a
   progress check."* Attribute the analysis to AI — he asks for this explicitly and it pre-empts
   "who says so".
2. Numbered issues. **Two lines each: what was seen, then a single `Fix:` line.** No elaborate
   justification, no IS-code lecturing, no "why it matters" paragraph per item.
3. One short rating box, if he asked for a rating — overall score, grade, and the one sentence that
   explains why it is not higher.
4. The ask, as one short paragraph: explanation point by point, on specification and methodology,
   and the meeting time he named.
5. Photo captions: **one line each** — date, location, what it shows. Not a paragraph per photo.

**Cut entirely** unless he asks: background and project description, "what is being done right",
a separate recommendations section (fold each fix into its issue instead), the photo-by-photo
essay, and the list of documents/method statements to be produced. Those were the bulk of the
rejected draft.

Keep: the safety item, even though it is not a "quality" issue — but as one numbered item in the
same list, not a separate red callout with its own heading.

## 2. MECHANICS — images must be BOTH inline and attached

`gws_gmail_draft_create` (and the DWD variant) cannot attach files. Build the MIME by hand and
`drafts().create` it through the bridge:

```python
import sys; sys.path.insert(0, "/opt/hermes")
from tools.gws_auth import build_service

root = MIMEMultipart("related")        # root = related
root["From"] = "Nishant Ranka <ndr@draas.com>"
root["To"], root["Cc"], root["Subject"] = TO, CC, SUBJECT
root["Date"] = formatdate(localtime=True)
root["Message-Id"] = "<%s@mail.gmail.com>" % uuid.uuid4().hex.upper()

alt = MIMEMultipart("alternative")     # plain first, html second
alt.attach(MIMEText(PLAIN, "plain", "utf-8"))
alt.attach(MIMEText(HTML, "html", "utf-8"))
root.attach(alt)

for i, path in photos:                 # images at ROOT level, not inside alt
    part = MIMEImage(open(path, "rb").read(), "jpeg")
    part.add_header("Content-ID", "<photo%02d>" % i)
    part.add_header("Content-Disposition", "inline", filename="Photo-%02d.jpg" % i)
    root.attach(part)

raw = base64.urlsafe_b64encode(root.as_bytes()).decode()
svc.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
```

Reference it in the HTML as `<img src="cid:photo01" width="520" style="width:520px;max-width:100%;
height:auto;display:block;">`. Gmail then shows each photo inline **and** lists it as a real
attachment on the message — one set of bytes serves both, which is exactly what NDR means by
"attach the 10 images".

- Phone portrait shots: cap the display width at 520px and let `height:auto` run. Do not fight the
  aspect ratio.
- 10 photos of ~350 KB = ~3.4 MB MIME / ~4.7 MB encoded. Fine for Gmail; mentions the size to NDR
  if it gets near 10 MB.
- Always ship a `text/plain` alternative that carries the full content (issues, fixes, ask). NDR's
  recipients often read on a phone client that strips HTML.
- Working script to copy: `/data/hermes/tmp/oasis-inspection/build_short.py`. It separates
  `ISSUES` (`(title, what_was_seen, fix)` tuples), `CAPTIONS`, `HTML` and `PLAIN`, so a rewrite is
  an edit to two lists, not to markup.

## 3. VERIFY BY READ-BACK before telling him it is ready

```python
d = svc.users().drafts().get(userId="me", id=draft_id, format="full").execute()
p = d["message"]["payload"]        # walk: related -> alternative -> text/plain,text/html
                                   # then 10 x image/jpeg with Content-ID <photoNN>
```

Check To, Cc, Subject verbatim, the plain-text body, and the count of `image/jpeg` parts with
Content-IDs. Print the plain body and read it — that is the fastest way to catch a missing `Fix:`
line or a numbering break.

## 4. Pitfalls

- **`cid:` images do not render in a local `file://` browser preview** — they show as blank grey
  boxes with the alt text. That is expected, not a bug. Do not "fix" it by hosting the images
  somewhere or switching to remote URLs; check the layout, and say plainly when showing NDR the
  screenshot that the grey boxes are the local preview only and will render in Gmail.
- **Delete the superseded draft** whenever he asks for a rewrite (`drafts().delete()`). Drafts
  otherwise fill with near-duplicates of the same mail and he may send the wrong one. Report that
  the old one was deleted.
- **One draft on this topic may already exist** from an earlier turn — `drafts().get(format="full")`
  on it is also the cheapest way to recover the previous HTML, captions and photo paths instead of
  rebuilding from scratch.
- **Recipient address comes from Gmail headers, not the contacts sheet** — Anbu mails as
  `anbarasan@draas.com`, not `pm2.blr@`. Where a colleague has no `@draas.com` identity, use the
  address that appears on the project's mail threads (Kanthesh: `kantheshbgme@gmail.com`) and tell
  NDR you did, so he can override.
- **Run the build with `env -u PYTHONPATH /opt/hermes/.venv/bin/python3`.**
