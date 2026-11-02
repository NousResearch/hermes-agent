# Threaded drafts, internal-only replies, sender-identity checks & the Rakuten-Kelsa AI POC thread

Session-verified patterns (Sep 2026) — see `email-drafter` SKILL.md for the top-level GWS bridge rules.

## 1. Bridge return type is a JSON STRING, not a dict (hit 2026-09-14)

`gcall("draft_reply_create", ...)` / `gcall("draft_create", ...)` return a JSON-encoded
**string** like:

```
{"status": "draft_created", "draft_id": "r4994596242592065112", "message_id": "...", "threadId": "1a05b876f957cf9b"}
```

Calling `.get()` directly on the raw result crashes with
`AttributeError: 'str' object has no attribute 'get'` — and the crash can look like the
draft failed when it actually SUCCEEDED.

Recovery:
```python
if isinstance(res, str):
    res = json.loads(res)
draft_id = res.get("draft_id")
```
Then verify with `drafts().get(userId="me", id=draft_id, format="metadata")` — read headers
from `d["message"]["payload"]["headers"]`. Remember `drafts().get` has NO `metadataHeaders`
kwarg (that is `messages().get` only). For body verification, read top-level
`payload.body.data` BEFORE walking `parts[]` (single-part drafts carry all text in body.data).

## 2. "Reply to this email but ONLY to internal team" — use draft_create, not draft_reply_create

When the thread carries external counterparties but NDR's response is an internal directive
(hit 2026-09-14: Rakuten/Kelsa AI POC — reply to Manohar + Aagney + Kelsa team only, Cc Ashwin,
absolutely no Rakuten):
- `draft_reply_create` would thread into the external conversation and inherit headers;
  the default To would leak the external party back in.
- Use `gcall("draft_create", service_name="google-draas", to=..., cc=..., subject=..., body=...)`
  with a fresh internal subject like `"<Customer> - <Topic>: Review & Next Steps"`.
- Recipient set mirrors the LAST sent mail minus the external parties (for Kelsa internal:
  To Manohar msingh@o3infotech.com, Aagney aagney@kelsa.io, Pavan pavan@kelsa.io, Guna
  gunarka@kelsa.io, Yogesh yogesh@kelsa.io; Cc Ashwin ashwin@kelsa.io).

## 3. "Send from my other account" requests — verify before promising

NDR may believe a chain marks his alternate identity (e.g. ndr@o3infotech.com for
Kelsa/O3 business). NEVER promise an alternate From until you check BOTH:
1. **Chain marking**: walk To/Cc/From headers of the actual thread messages. Often the
   chain marks only the default account (ndr@draas.com, or ndr@kelsa.io on the original
   Rakuten mail) and the premise is wrong. State the finding factually.
2. **Send-as aliases**: `gmail.users().settings().sendAs().list(userId='me')` on EVERY
   connected account (google-draas / google-ahfl / google-gmail). Only an
   `verificationStatus == accepted` alias enables drafting with that From identity.
3. If the mailbox has no vault token and no alias → say so plainly, offer OAuth setup.
   (ndr@o3infotech.com: not in vault, no alias — as of Sep 2026. NDR's preference that
   Kelsa/O3 business route to O3 is recorded in memory.)

## 4. Rakuten → Kelsa AI POC thread map (live deal, Sep 2026)

- **Where it lives**: ndr@draas.com inbox, forwarded by Manohar Singh from
  msingh@o3infotech.com (his O3 address — the O3 mailbox itself is NOT vault-linked,
  so "the O3 email" always surfaces in draas).
- **Original**: 24-Aug-2026 "Kelsa ( AI capability Demo )" from Satish Raju Raelekar
  <satishraju.raelekar@rakuten.com> (Rakuten Symphony India, HR Transformation, Bangalore);
  To included ndr@kelsa.io, Cc ndr@draas.com.
- **POC scope** (Anantha Prasanna <anantha.prasanna@rakuten.com>, 28-Aug "Kelsa - AI - POC"):
  - AI Agent on R-Social: queries on stage/status/pending/budget; raise a request
    (new process item); approve a request; upload bills; budget reports for admin team.
  - AI Wrapper dashboards: Query Management KPI/SLA ticket metrics; attendance regularisation.
  - Explicit ask for cost/commercial model to extend services.
- **Kelsa-side status**: dashboard wrapper LIVE at https://rakuten-qms-dashboard.onrender.com/
  (delivered 2-Sep by Aagney); agentic POC discussion still pending; 8-Sep Andrea Leena
  <andrea.leena@rakuten.com> requested an AI Wrapper demo call (proposed 9-Sep 1:30 PM);
  no confirmed outcome reached NDR.
- **Contact set**: Rakuten — Andrea Leena, Anantha Prasanna, Prajakta Kanaglekar,
  Sunil T V, Satish Raju Raelekar. Kelsa/O3 — Manohar (msingh@o3infotech.com /
  msingh@kelsa.io), Aagney aagney@kelsa.io, S Pavan Kumar pavan@kelsa.io,
  Guna gunarka@kelsa.io, Yogesh yogesh@kelsa.io, **Ashwin Hegde ashwin@kelsa.io**
  (also hashwin@o3infotech.com; contacts sheet lists ashwin.hegde12@gmail.com — use
  ashwin@kelsa.io for Kelsa-team mails).
- **NDR's internal-review positioning** (14-Sep reply draft): most agentic items are
  "already possible" (viewer-scoped queries, create process item, complete task, upload
  docs); budget reports need the specific use case (template-fetch vs manual action);
  Ashwin must confirm whether the MCP exposes dashboard tabular data for query/response
  and HTML/CSS chart rendering in chat channels (Slack/Viber); Aagney to pin exact
  query-against-pipeline examples with the customer.