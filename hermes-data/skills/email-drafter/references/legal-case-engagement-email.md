# Legal Case Engagement Email — Pattern & Pitfalls

Use when NDR asks to brief an external lawyer / advocate about a RERA or civil case, attach supporting documents, and ask whether they'll take the case.

## Required Sections

1. **Background** — client name, property, project, the disputed fact (e.g. "Dinesh Ranka's landowner share allocation"). One paragraph.
2. **Why unattended** — essential credibility note. State the reason no one contested (e.g. believed co-developer was handling it; key person expired; legal heirs not on record).
3. **Case chronology** — numbered events with dates: complaint filing, hearing dates, order/judgement, execution notice, recovery notice. Be precise.
4. **Current status** — what the last notice says (demand amount, attachment/execution action, deadline).
5. **What you want from the lawyer** — bullet list: review docs, opinion on appeal + stay, willingness to handle E2E, timeline/cost.

## Attachments from Mixed Sources

Common pattern: documents arrive from different channels.

| Source | How to collect | Attachment handling |
|--------|---------------|---------------------|
| Gmail PDF attachment | `gmail.users().messages().attachments().get(userId='me', messageId=id, id=attachmentId)` | Save to /tmp/ with meaningful filename, attach to draft |
| Gmail inline HTML body (no PDF) | Extract body, save as `.html` file | Save to /tmp/ as `.html`, attach as-is. The lawyer can open in browser. |
| Google Drive file | `drive.files().get_media(fileId=...)` or download via webViewLink | Save to /tmp/ with original filename |

## Draft Creation via Gmail API (not gws_skill_bridge)

When `gws_skill_bridge.draft_create` fails (e.g. credential resolution issue), fall back to direct Gmail API:

```python
from tools.gws_auth import load_credentials
from googleapiclient import discovery
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import base64

creds = load_credentials('google-draas')
gmail = discovery.build('gmail', 'v1', credentials=creds)

msg = MIMEMultipart('mixed')
msg['To'] = 'recipient@example.com'
msg['Cc'] = 'cc@example.com'
msg['Subject'] = 'Clear subject line'
msg['From'] = 'ndr@draas.com'

# Attach body
msg.attach(MIMEText(body_text, 'plain'))

# Attach files
for local_path, display_name in attachments:
    with open(local_path, 'rb') as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment', filename=display_name)
    msg.attach(part)

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode('utf-8')
draft = gmail.users().drafts().create(
    userId='me',
    body={'message': {'raw': raw}}
).execute()
```

## Counsels' PROGNOSIS email — the follow-up round (verified 2026-09-18, Behera/Jasleen)

Once external counsel has been engaged and has the pack, NDR's next ask is a **"what are our
chances" email** — he wants a prognosis and a strategy view, not a document handover. This is a
distinct pattern from the initial briefing above.

**Trigger:** *"Jasleen has responded to our email asking a strategy, what's our thought process —
send a nice polite email back, thank her for the prompt revert, and ask…"*

**Shape that works (reply-all on the existing thread, 3 numbered questions + NDR's own read):**

1. **Thank for the prompt revert** — one line, genuine, not boilerplate.
2. **Q1 — chances of securing a STAY.** Ask specifically: probability on the material as it
   stands; whether it holds against the attachment alone or the recovery proceedings as a whole;
   realistic timeframe; what conditions/deposits they'd likely be asked to meet. Conditions and
   time are as valuable as the probability — do not ask only "will we win".
3. **Q2 — CONDONATION OF DELAY / limitation.** Frame the ADVERSE narrative fairly and then ask
   how it holds up: the respondents' explanation (demise of the key person, estate confusion,
   the co-developer who ought to have been in the frame never being impleaded). Ask how much of
   the case turns on it. NDR's own framing: *"setting aside the question of who is at fault."*
4. **Q3 — overall case strength on the appeal**, taking the sequence of events as a whole
   (outstanding interest owed by the customer, what has already been delivered, the
   co-developer's handling of comparable claims, facts the Authority was never told).
5. **State NDR's OWN instinct candidly and ask for the lawyer's independent view on it** — he
   wants his hypothesis stress-tested, not validated. He will say something like *"my own
   reading is the case should be played out up to a stay and a counter-claim, and from that
   position we settle — I don't see another route working, and egos may have to be set aside."*
   Put this in the email as **his** view and explicitly invite disagreement (*"or whether you
   see a materially better path"*).
6. **Offer a call** at the recipient's convenience, and invite the answer by email beforehand
   *or* on the call — NDR said *"if you can send it by email before the call, or as part of the
   call, it would really help."* Carry both options; don't force one channel.

**Tone:** courteous and businesslike, but direct about the commercial objective (obtain a stay;
contesting/ dropping the attachment is secondary). NDR names the endpoint (settlement) openly to
his own counsel — do not sanitise that out, and do not editorialise about "egos" beyond his own
phrasing.

**Addressing:** "Adv. Jasleen" / "Dear Adv. Jasleen" in the salutation; sign "Nishant Ranka".
Reply-all must carry **every** participant from her latest message — she writes to 7 people
(law firm admin + the two colleagues + Vinay T + Suhail + Pradyumna). Build the To/Cc from her
message headers, not from an older message in the thread.

**Thread discipline:** verify which message is genuinely the LAST in the thread before replying —
the newest entry can be NDR's own prior email (a soft nudge he sent the evening before), in which
case the reply-all still threads onto the counsel's newest message. Fetch thread messages with
`format='metadata'` and read `Date` + `From`, skipping any `labelIds: ['DRAFT']`.

### Behera / Mirabilis RERA-6093 — live case state (as of 2026-09-18)

- Counsel: **Adv. Jasleen Kaur**, Smriti Legal, jasleen@smritilegal.com, +91 9740012005 — plus
  Vinay T (vinay@vnv.ca / VNV) as the referrer. Case: Sabyasachi Behera, RERA
  CMP/200628/0006093, Mirabilis flat B1-203.
- Thread subject: *"Sabyasachi Behera - RERA Case CMP/200628/0006093: Evidence Pack & Defense
  Strategy"*, threadId `1a07b0f2821d61f6`.
- Counsel's position (18-Sep): **documents reviewed; drafting the appeal with the IA; draft to be
  shared the following Tuesday.** Plan is stay first, then the REAT appeal.
- Court/authority action: attachment order 3-Sep-2026 under KLRA 1964 for ₹26,63,278 against KPDL.
  Objective: stay against attachment (minimum), ideally have it dropped.
- NDR's stated endgame: stay + counter-claim, then settle with Behera.
- **Fact corrections that must survive into every future email** — the sale deed for B1-203 WAS
  registered (10-Apr-2023, Registration 236/2023-24) and possession delivered; the Authority was
  never told, which makes the possession relief in the judgment moot. An earlier questionnaire
  assumed the deed was still pending registration — that assumption is wrong and NDR corrected it
  explicitly. Never reintroduce "sale deed not registered" language.
- The ₹2,00,000 was **received FROM** Behera (Feb-2020, cheque handed to Bharat, receipt never
  issued) — it is not a sum paid to him. Do not invert this.
- Companion Mirabilis delay cases cited as the pattern that KPDL handled all delay claims
  independently: D-607 (Abhishek Kumar) and C-207 (Sumil Patel).

## Common Pitfalls

- **Execution Notice may be inline HTML, not PDF** — check the Gmail message payload parts tree; if no attachment is found, save the HTML body as a file.
- **Multiple Gmail accounts** — documents may live across ndr@draas.com, ndr@drahomes.in, or nishantranka@gmail.com. Search all accounts via `gws_resolve_account` first.
- **Contact resolution for lawyers** — "Vinay Reda" may resolve to a RERA consultant/CA (Vinay T at Venu and Vinay, vinay@vnv.ca) rather than "Vinay Reddy" which matches a different person. Search contacts sheet + People API across all accounts.
- **Tahsildar notices on Drive** — these may be in the "RERA Complaints & Orders" folder or under the project name (e.g. "Mirabilis"). Use Drive file search with project/client name fragments.
- **Draft link** — always return the Gmail draft link so NDR can review before sending.