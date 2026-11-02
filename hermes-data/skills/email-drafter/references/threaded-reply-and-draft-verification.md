# "Respond to the same email" — threaded reply, then PROVE the draft exists

> Companion file for this class of task: **`references/followup-on-the-lines-of-the-last-mail.md`**
> — NDR's *"send X a follow-up on the lines of the one we sent"* instructions: find the prior
> mail, mirror its structure, carry the thread's figures, normalise dictated names, and flag
> dictation ambiguity instead of smoothing it.
>
> **Reply WITH an attachment (snapshot PNG, PDF, report)?** Neither native draft
> tool has an attachment parameter — use the raw-MIME recipe, the four
> read-back assertions and the "quote the source verbatim in the body" rule in
> **`references/attachment-in-threaded-draft.md`**.
>
> **`SKILL.md` is at its size ceiling (~100k) — `skill_manage(action="patch")` on it is refused.**
> Put all new material in `references/` and point to it from a sibling reference like this one.
>
> **Body FORMATTING is a hard rule, not a preference.** NDR rejects word-wrapped "poem format"
> bodies outright: paragraphs with no hard newlines, multipart `text/plain` + `text/html`, the
> `<strong>` section-head skeleton, the read-back assertions (part list, max plain-line length,
> corrected-name count) and the cross-channel time-consistency rule are in
> **`references/email-body-formatting-no-hard-wraps.md`**.

Two failure modes this file exists to prevent, both hit on 2026-10-22 (Ayana Woods):

1. A standalone mail was drafted when NDR/Bharat wanted a **reply in his thread**.
2. The previous turn reported *"it's in your Drafts"* for a draft that **was no longer in
   the mailbox** — no draft, no sent copy, nothing in Trash.

## Rule 1 — "respond to the same email" means reply IN-THREAD, never a new mail

Bharat's own words: *"we need to respond to the same email where he has sent about X… we'll
respond to the same email. But I want you to put it in draft before you go ahead and send."*
So: reply in-thread, **draft only**.

Recipe (see also the frontmatter's CREATE note — payload dict/bytes = 400):

```python
m = svc.users().messages().get(userId="me", id=MSG_ID, format="metadata").execute()
mid = {h["name"]:h["value"] for h in m["payload"]["headers"]}["Message-ID"]
# build the MIME, then:
new["In-Reply-To"] = mid
new["References"]  = mid            # parent's References + parent's Message-ID if deeper
svc.users().drafts().create(userId="me",
    body={"message": {"raw": b64url, "threadId": m["threadId"]}}).execute()
```

**Two mails from the same sender with the same subject can sit in SEPARATE Gmail threads.**
On this thread NDR's 00:17 brief and his 10:40 "Re: …" checklist had different `threadId`s.
Reply to the one that **carries the questions**, and tell the user which thread it landed in
so he isn't hunting in the other one.

## Rule 2 — never say "it's in your Drafts" without looking, in the same turn

After creating, verify and report the *evidence*:

```python
# a) the draft exists and is attached to the intended thread
d = svc.users().drafts().get(userId="me", id=DRAFT_ID, format="raw").execute()
t = svc.users().threads().get(userId="me", id=THREAD_ID, format="metadata").execute()
print([msg["id"] for msg in t["messages"]])       # expect parent + draft

# b) attachments actually carried over
[p.get_filename() for p in msg.walk() if (p.get("Content-Disposition") or "").startswith("attachment")]

# c) the content edits landed
for s in ["Note on provenance", "Sagar", "<the added section heading>"]:
    print(s, s in html)     # must be False / True as intended
```

A draft can vanish between turns (deleted by the user, or a creation that silently did not
persist). `drafts().list()` + a per-draft subject scan, or a Gmail search for a unique phrase
in the subject, settles it in one call. If it is gone, **say so plainly** and rebuild — do not
repeat the earlier claim and do not let the user discover it at send time.

### An empty Drafts folder usually means NDR already pressed send

Verified 2026-10-23: `gws_gmail_draft_list` returned `count: 0` on google-draas when six drafts
were expected, and the reflex conclusion — "the drafts were lost" — was wrong. NDR had opened
and sent **all six himself** over the preceding four hours (10:58 → 14:22). One `in:sent`
search showed every one with its send timestamp.

So when Drafts comes back empty and you expected drafts there, **do not report a loss** —
check Sent first:

```
gws_gmail_search(query="in:sent newer_than:1d", service_name="google-draas")
```

One call lists everything he sent today with To / Subject / date. Then **compare the send time
against the time NDR's "send" message arrived**: a mail sent a minute *before* his message means
he pressed it himself and there is nothing left for you to do but confirm. This also settles the
"which of the pending drafts did he already action?" question without touching Drafts at all.

### Native draft ids (`r-…`) are not Gmail message ids

Drafts created through the native tools come back with an `r`/`r-` prefixed id
(`r-5433421391425712667`, `r3230969868209831585`). Passing one to `gws_gmail_get` fails with
`HttpError 400 … "Invalid id value"` — that is **not** evidence the draft is gone, and it must
not be reported as such. Draft-scoped calls take that id (`gws_gmail_draft_delete`,
`users().drafts().get(id=DRAFT_ID)`); message-scoped calls need the message id.

### `gws_gmail_reply_draft` has NO `to=` — and the bridge REFLOWS the body (2026-10-24)

Two traps on a plain reply-all to Dharmesh + Manish on "Re: Call for a meeting"
(`thread 1a1178935be29c75`), both found only by reading the draft back:

- **The native tool has no `to` parameter** (schema: `message_id, body, cc, bcc, html,
  service_name). Anchored on NDR's own latest mail in the thread it threaded perfectly and
  minted **`To: Nishant Ranka <ndr@draas.com>`** — a reply addressed to himself. With no `to` to
  override, the fix is to **re-anchor on the counterparty's most recent message** (same
  `threadId`, correct `To`) or to use the bridge / raw MIME with an explicit `To`.

  **This recurs silently, so run the pre-flight every time** (2nd occurrence 2026-10-27, Ranka
  Iris → Bhavik: a follow-up "on the lines of the one we sent" was anchored on NDR's 19:36 sent
  mail, and the draft came back `to: Nishant Ranka <ndr@draas.com>` with no error at all — the
  tool reported success, only reading the draft back caught it). In a live back-and-forth NDR is
  usually the **last** person to write in the thread, so the newest message is the wrong anchor
  more often than not. Before calling `gws_gmail_reply_draft`:

  ```
  # pick the anchor = newest message in the thread whose From is NOT ndr@draas.com
  ```

  Search the counterparty's own mail (`from:bhavik@draas.com newer_than:7d`) and anchor on that
  message id. If they genuinely never replied (NDR chasing a silent party), there is no inbound
  anchor — then use the bridge / raw MIME with an explicit `To`, never the native tool.
  Either way, `gws_gmail_get` the new message id and confirm `to:` names the counterparty before
  reporting anything to the user. Deleting the wrong draft first
  (`gws_gmail_draft_delete`) and creating the replacement is fine — it leaves one draft and the
  same `thread_id`.
- **The bridge (`tools.gws_skill_bridge.call("draft_reply_create", …, to=…, cc=…)`) honours
  `to`/`cc` but re-wraps the body at ~72 columns with hard CRLFs** — the dictated text came back
  as `"…for whichever\r\nday we fix, so we come in with a list\r\nand leave with decisions."`
  Keep every line under ~60 characters when using it (break paragraphs and list items yourself),
  or the recipient sees a broken text body.
- **A draft message id 404s on `gws_gmail_get`** (`Requested entity was not found`). That is not
  a missing draft — verify with `drafts().get(id=DRAFT_ID, format="raw")` and decode the MIME,
  which is also the only read that proves `Cc`.
- **Run any draft-building script exactly ONCE.** In `execute_code`, capturing stdout and stderr
  via two separate `subprocess.run(...)` calls re-executes every side effect — it left two live
  reply drafts in the folder. Use `r = subprocess.run(cmd, …)` once, print `r.stdout` and
  `r.stderr`, then sweep `drafts().list()` for strays.

## Rule 3 — rebuild from the saved build artifacts, not from memory

Previous turns leave the exact MIME on disk. Rebuild from those so wording, numbering and
attachments stay consistent with what the user already reviewed:

- `/tmp/draft_raw.eml` — the full `format="raw"` draft, **including every attachment**
- `/tmp/draft.html` / `/tmp/ayana_v1.html` — the HTML body as sent
- `/tmp/build_*.py`, `/tmp/redraft*.py` — the script that produced it (records the exact
  string edits and the removal asserts)

Carry attachments by walking the saved `.eml`:

```python
for p in cur.walk():
    if (p.get("Content-Disposition") or "").startswith("attachment"):
        att = MIMEApplication(p.get_payload(decode=True),
                              _subtype="pdf" if p.get_filename().lower().endswith(".pdf") else "jpeg")
        att.add_header("Content-Disposition", "attachment", filename=p.get_filename())
        new.attach(att)
```

The original files may be long gone from the upload cache; the `.eml` still holds them.
Edit the HTML with **asserted** replacements (`assert old in html`) so a stale build fails
loudly instead of silently dropping the wrong paragraph.

## Rule 4 — replying to a NUMBERED CHECKLIST: open with a status block

When the mail being answered is a numbered set of questions, the reply's first section must
be a point-by-point block — one paragraph per question, in his numbering, each tagged
**answered / partly answered / still to be obtained** — placed **before** the detail
sections. NDR writes these checklists and reads them back in order; making him hunt for
answers across a report reads as a non-answer. Then keep the evidence sections below,
unchanged, so the detail is there for anyone who wants it.

Format that landed well:

```
<h3>Your seven questions — where we stand</h3>

<p><strong>1. Infrastructure — answered from the ground.</strong>
<em>Sidewalks:</em> … <em>Drainage:</em> … <em>Still pending:</em> …</p>
…
<p><strong>4. Registration and GST — answered, and it does not hold together.</strong> …</p>
```

Lead each with a verdict phrase, not a neutral heading — "answered", "answered, and it does
not hold together", "still to be obtained". That is what he scans for.

## Rule 5 — standing preferences that came out of this thread

- **Draft only, always.** Never send, never offer to send — "open it and send when you're happy".
- Attachments: number the site photos in the same order as the sections that reference them,
  and name the files so the order is self-evident
  (`1_<project>_<date>_<subject>.jpg`, `2_…`, `3_…`).
- When the user accepts a caveat you flagged earlier (e.g. quotations addressed to a third
  party), **remove the caveat from the draft and the matching "still to collect" line** — do
  not keep re-raising it in the mail or in the chat summary.
- Report the draft state with the concrete facts he can check: thread it sits in, To line,
  attachment count, and the two or three content changes made since the last version.

## Rule 5b — "put the link in the email, formatted in HTML" (2026-10-28, Ranka Oasis invoice gaps)

When a review/escalation mail itemises records, NDR's follow-up is *"for each item where we have found
gaps, include the link to the corresponding invoice in the email itself. Format the links in HTML so
they are easy to read."* What "easy to read" means in practice:

- **An at-a-glance table first**, one row per item — `Invoice | Vendor | Amount | Main gap | Open` —
  then the full detail blocks below it. The table does the triage, the prose stays long. Do not
  shorten the analysis to compensate for adding links.
- **A link on every item, twice**: once in the table's `Open` column, once as a
  `Link to the invoice in Kelsa →` line under each detail heading. Inline styles only (`border`,
  `font-size`, `color` on the `td`/`a`), table-based layout, no external CSS, no `<style>` block —
  Gmail strips it.
- Boilerplate that landed: a one-line preamble saying *why* the links are there ("so you can open the
  bill, the receipt and the PO together while you fix it"), then the reference-numbers box, then the
  table, then the detail, then the systemic section.
- **Count the links when you verify**: expect `2 × items`. Walk the `text/html` part of the draft and
  assert every id in the body is the one you resolved for that row — a 13-row table is exactly where a
  transcription slip happens.

### `body: ""` from `gws_gmail_get` is an HTML-only draft, not an empty one

For a draft built with `html=True` the message is `multipart/alternative` carrying **only** a
`text/html` part. `gws_gmail_get` reads the plain part, so it returns `body: ""` while the `snippet`
quotes real text — that combination means "look at the HTML", not "the body was lost". Verify To / Cc /
Subject and the HTML content from one read:

```python
msg = svc.users().drafts().get(userId="me", id=DRAFT_ID, format="full").execute()["message"]
hdrs = {h["name"].lower(): h["value"] for h in msg["payload"]["headers"]}   # to / cc / subject
# walk payload.parts, urlsafe_b64decode each body.data, keep the text/html one
```

### A recipient resolved from Kelsa is NOT the same as the address he mails from

`kelsa_call_tool("list_users", {"pipeline_id": 516})` is the fastest authoritative name → address map
for DRA staff (it gave `Anbarasan <pm2.blr@draas.com>`, `Kantesh B G <kanteshbg@draas.com>`,
`Aravindan Jyothi <eng5.blr@draas.com>` for a mail NDR described as *"Arvind's email is eng5.blr or
something"*). But a **Kelsa login address is a `role.function` mailbox, not necessarily where the
person sends mail from** — this skill's own frontmatter records that Anbu SENDS as
`anbarasan@draas.com`. So before addressing, cross-check the name against the Gmail headers of any
existing mail from that person; if the two differ, say which one you used and why in the hand-over
rather than picking silently. Same rule for a Cc the user only half-remembers: resolve it, quote the
resolved name back (`eng5.blr@draas.com` → Aravindan Jyothi) so he can veto.

## Rule 6 — correcting a draft: edit it IN PLACE with `drafts().update`

When the user comes back with a correction to a draft you already made ("for unit details you
must mention the RERA data"), **update the same draft**. Creating a second one leaves a stale
copy in Drafts that he may send by mistake.

```python
d   = svc.users().drafts().get(userId="me", id=DRAFT_ID, format="raw").execute()
cur = email.message_from_bytes(base64.urlsafe_b64decode(d["message"]["raw"]))
html = [p.get_payload(decode=True).decode() for p in cur.walk()
        if p.get_content_type() == "text/html"][0]
# …edit html, rebuild multipart/mixed -> alternative(plain+html) + re-attach from cur…
svc.users().drafts().update(userId="me", id=DRAFT_ID,
    body={"message": {"raw": b64url, "threadId": THREAD_ID}}).execute()
```

The draft's own raw is the base — body **and** every attachment. Re-derive the plain-text part
from the edited HTML with a strip function; never reuse the previous plain part, it will
disagree with the new HTML.

⚠️ **The stored raw is not byte-identical to what you sent.** Gmail returns it with `\r\n`
line endings and entities normalised (`&#8377;` comes back as a literal `₹`, `&#215;` as `×`).
So an `assert old in html` copied from your earlier build script **fails on the second edit**.
Locate blocks with regex / index instead:

```python
old = re.search(r"<li><strong>Density contradiction\.</strong>.*?</li>", html, re.S).group(0)
i, j = html.index("<p><strong>5. Unit mix"), html.index("<p><strong>6. Clubhouse")
html = html[:i] + new_block + html[j:]
```

After updating, verify from the API again, never from your own variables: `threadId`,
`labelIds == ['DRAFT']`, attachment count + filenames, edited strings present, removed strings
absent. The message id changes on every update — quote the latest one.

**When you are using the native `gws_gmail_reply_draft` tool instead of raw `drafts()` calls**,
there is no update action — only create. The in-place rule still holds in spirit: **delete the
superseded draft first** (`gws_gmail_draft_delete` with its `draft_id`), then create the
replacement, then quote the NEW `draft_id` back to the user. Never leave both. Verified
2026-10-23 on the VGRO thread (`reg:- information required`): the first draft carried three bare
file links; the pack was then filed into a Drive folder and the draft was rewritten to lead with
the folder link — old draft deleted, new one created in the same `thread_id`, one draft in
Drafts.

## Rule 7 — a status block must never say "pending" for data already in hand

Before writing **"still to be obtained"** against a numbered question, search your own earlier
work: `/tmp/<project>_research.md` and other saved exports, the previous draft, and the
project's reference documents. Bharat's correction: *"The mixture of the units has already been
obtained… for unit details you have to mention what data we have taken from the RERA website."*
Re-asking for data you already pulled is the worst kind of non-answer.

So every item in the status block carries a **source tag** — `K-RERA plot schedule`,
`developer website`, `sales team, verbal`, `quotation Q20261022-90` — and where two sources
describe the same thing, the reconciliation between them (e.g. RERA carpet ÷ plot count = plot
area, therefore the registration carries no BUA inventory).

And when a source **resolves** an earlier "issue", correct that issue in the same edit. On this
thread the "45 vs 75 acres" contradiction was replaced by the accurate statement (Phase 1 =
47.57 acres / 456 plots registered; 75 acres is the master plan) plus the narrower question
that survives. Leaving both statements in the mail is worse than either.

## Rule 8 — "update the draft" when that draft has ALREADY BEEN SENT (2026-10-27, Ranka Udaya P31)

First establish what is actually in the mailbox, before editing anything: list the thread's
messages **and** the drafts. If the draft you are asked to "update" is no longer in Drafts and a
message from `ndr@draas.com` sits in that thread stamped after it was built, NDR pressed send
(Rule 2). The task is then no longer *update the draft* — it is **supersede one instruction in a
mail that is already out**.

- Do **not** recreate the old draft as though it were still pending, and do not reword it so it
  reads as if nothing was sent. Bharat has already acted on the sent version.
- Create a **further reply-all in the same thread**. The opening paragraph names the correction
  explicitly: *"I have re-read your email, and I misread it the first time… so my earlier note
  needs one correction, and only one: drop the direction to execute the Sale Deed. Everything
  else in it stands."* Then carry every unchanged section forward in substance — a correction
  mail must not quietly drop approved content.
- Keep the numbering so he can see what still stands, and say explicitly which earlier direction
  is cancelled. Where the corrected item was a condition (KYC, blank cheque, balance), restate it
  and state that the change does not relax it — otherwise the other side reads the correction as
  a general softening.
- In the chat summary, tell him plainly that the earlier mail has **already gone out** with the
  superseded instruction and that this new draft is the correction. He often does not know it was
  sent.

The previous turn's working files may not survive (a prior job id's directory can be gone). Recover
state from Gmail — `users().messages().list(q='subject:"…"')`, read headers + bodies — and rebuild
the HTML from that, rather than from memory of the earlier version. Keep ledger/agreement figures
identical to what was already quoted in the thread.

### Never name a helper script after a stdlib module

`inspect.py` in a directory where `sys.path.insert(0, "/opt/hermes")` runs first shadows the
stdlib `inspect`, and `tools.gws_auth` then dies with a misleading
`ImportError: cannot import name 'build_service' from partially initialized module
'tools.gws_auth' (most likely due to a circular import)` — `importlib.metadata` imports stdlib
`inspect` and picks up your script instead. It is not a broken venv and not a circular import in
the tool; renaming the file fixes it. Avoid `inspect.py`, `email.py`, `token.py`, `code.py`,
`types.py`, `copy.py`, `json.py` for scratch scripts — use `look.py`, `read_msgs.py`,
`mk4.py`, `verify4.py`.
