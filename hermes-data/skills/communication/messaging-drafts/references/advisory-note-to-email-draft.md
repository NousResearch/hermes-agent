# Advisory Note → Email Draft (NDR's review-first workflow)

## When this fires
NDR asks you to get an opinion/advice from an EXTERNAL advisor (advocate, CA,
transaction-structuring counsel) on a live deal, and wants the substance
recorded "on record" for that advisor. Typical phrasing:

- "prepare a one-pager note for him that he can fully understand"
- "prepare it using HTML, CSS only so I can read it"
- "if I'm not happy, I'll ask you to update that note in an email and finally
  we'll send the draft email"

Known advisors this pattern has been used for:
- **Adv. Vishwas Rao** — vishwas@vantagepointadvisors.in, Vantage Point Advisors.
  Transaction structuring / capital-gains opinions (e.g. Ranka North Star).

## The workflow NDR expects (do NOT skip the review step)

1. **Gather the facts from the actual documents**, not from recollection. Read
   the JDA, addendums, assignment deed, compromise decree/settlement, share
   statements, and the landowner KYC/Aadhaar records for ages. NDR's standing
   rule: never assert figures, versions or trends you did not re-read this turn.
2. **Build the note as an HTML artifact** and publish it via `content_publish`
   so NDR can read it rendered in the browser. Return the `content.ahfl.in/a/...`
   URL — that is what he reviews.
3. **Wait for his corrections.** He will respond with factual fixes ("I read X
   as Y — correct?") and possibly a reframing of the ask. Do not jump to the
   email.
4. **On approval, convert the note into an HTML email draft** (see below).
   He may change the ASK between the note and the email — e.g. the note proposed
   a DRA-Realty + NCD structure, but the email was to lead with "your suggestion
   of an LLP". Rework the closing section to the NEW ask and keep the old
   structure as the alternative being compared against. Say so explicitly when
   you report back.
5. **Never send.** Draft only, always. Tell him it is in his Drafts folder and
   that he must send it himself.

## Creating the HTML draft from a file (avoid re-emitting the HTML inline)

The note HTML runs 12–14 KB. Do NOT paste it inline as a tool argument — write
it to disk and have the bridge read it. Verified pattern:

```bash
cd /opt/hermes && env -u PYTHONPATH /opt/hermes/.venv/bin/python3 - << 'PYEOF'
import sys
sys.path.insert(0, '/opt/hermes')
from tools.gws_skill_bridge import call

body = open('/tmp/<note_dir>/email_body.html').read()
res = call("draft_create",
    service_name="google-draas",
    to="vishwas@vantagepointadvisors.in",
    subject="Ranka North Star - Transaction Structuring: Briefing Note & Request for Your Opinion on an LLP Structure",
    body=body,
    html=True)
print(res)
PYEOF
```

Notes:
- `env -u PYTHONPATH` matters — `$HOME/.local` shadows the venv packages otherwise.
- Keep the subject ASCII (`-` not `—`) to avoid MIME-header mangling.
- `html=True` makes the bridge use `MIMEText(..., 'html')`.
- The bridge blocks `gmail_send` / `gmail_reply` by design; `draft_create` is the
  only compose path. Never call `.messages().send()`.

## ALWAYS verify the draft by reading it back

A `draft_created` status can be a ghost. Confirm the draft actually exists with
the right recipient before telling NDR it is ready:

```bash
cd /opt/hermes && env -u PYTHONPATH /opt/hermes/.venv/bin/python3 - << 'PYEOF'
import sys, json, urllib.request
sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import load_credentials
from google.auth.transport import requests as google_requests

creds = load_credentials("google-draas")
creds.refresh(google_requests.Request())
url = f"https://gmail.googleapis.com/gmail/v1/users/me/drafts/<DRAFT_ID>?format=full"
req = urllib.request.Request(url, headers={'Authorization': f'Bearer {creds.token}'})
with urllib.request.urlopen(req, timeout=20) as r:
    d = json.loads(r.read())
hdrs = {h["name"]: h["value"] for h in d["message"]["payload"]["headers"]}
print(hdrs.get("To"), hdrs.get("From"), hdrs.get("Subject"))
PYEOF
```

Then report To / From / Subject / draft-id to NDR — he treats the read-back as
the proof, not the creation status.

## HTML email body conventions
See `references/email-drafter-full.md` for the Gmail API recipe, and the
`email` skill's `references/html-email-templates.md` for NDR's CSS skeleton
(container, `h2` section rules, `.highlight`, `.consent-box`, `.footer`).
Professional salutation "Dear <Name>," and sign-off "Regards, Nishant Ranka".
For a note-as-email, reuse the note's section numbering (§1…§6) verbatim so the
recipient can refer to "your §3" — NDR works this way.

## Pitfalls
- Do NOT deliver a note and an email in one step. NDR explicitly wants the note
  reviewed first.
- Do NOT silently carry the note's original ask into the email — re-read his
  instruction; the ask often shifts (LLP vs NCD, etc.).
- Do NOT describe a draft as "sent". Drafts sit in the Drafts folder.
- Landowner ages must come from the ID/KYC records or the settlement deed, not
  inferred from names.
