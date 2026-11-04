---
name: email
description: "Complete Gmail email workflow: inbox triage and analysis, drafting new emails, threaded replies with custom To/Cc, forwarding, attachments, and account management. Use for any email operation."
version: 2.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [email, gmail, draft, analysis, triage, communication, reply]
    related_skills: [google-workspace]
---

# Email — Gmail Workflow

> ## ⚠️ BEFORE DRAFTING ANY EMAIL
> Load `email-writing` — it is the single authority on body format (rich HTML, headings,
> bullets, never hard-wrapped). Use its renderer
> `/data/hermes/skills/communication/email-writing/scripts/email_draft.py`. This skill covers
> triage, search, threading and account mechanics only.

Comprehensive class-level skill covering all Gmail operations: inbox analysis/triage, drafting new emails, threaded replies, forwarding, attachments, and multi-account management.

## When to Use

Load this skill for any email operation:
- **Analysis/Triage** — "analyze my inbox", "check my email", "triage last N hours"
- **Drafting** — "email [name]", "draft an email to [name] about [topic]"
- **Replies** — "reply to [name]'s email", "reply all except [person]"
- **Forwards** — "forward that email to [new recipients]"
- **Attachments** — "reply with these PDFs attached"
- **Multi-account** — "check my ahfl.in email", "use my Gmail account"

## Account Resolution

Default: `ndr@draas.com` (service: `google-draas`). Always resolve first:

```python
gws_resolve_account(account="ndr@draas.com")
```

Other accounts: `nishantranka@gmail.com` (service: `google-gmail`), `ndr@ahfl.in` (service: `google-ahfl`). Use only when explicitly asked.

**Domain-derived services (any Google Workspace domain):** the vault resolves ANY
`user@domain.com` to service `google-<domain>` (verified `ndr@o3infotech.com` →
`google-o3infotech`; probe with `tools.gws_auth._service_for_email(...)`, no token
needed to check the mapping). So emails from non-listed company domains (O3
Infotech, Kelsa, etc.) are readable once that account is OAuth-authorized —
confirm the domain is actually Google Workspace first via DNS-over-HTTPS MX
lookup (`curl -s "https://dns.google/resolve?name=<dom>&type=MX"` →
`aspmx.l.google.com`). To GET an authorization link delivered to the user's chat
for such a service, call `send_oauth_url` via a delegated leaf subagent with
`toolsets=["oauth"]` (the oauth toolset owns it; it is NOT in the base
execute_code registry — direct `registry.dispatch("send_oauth_url", ...)` raises
"Unknown tool"). Pass only `service_name` + `label`, never a user/telegram id.

Other users on this Hermes instance (per-user vault tokens, each user resolves their OWN email to a vault service):
- Anbarasan M / Anbu (pm2.blr@draas.com) → service `google-draas` (same key as ndr's default, but a SEPARATE per-user token in the vault). Always `gws_resolve_account(account="pm2.blr@draas.com")` first — do not assume the ndr token covers him.

**Sending-identity rule (NDR, Sep 2026):** every draft to colleagues/team is created in `google-draas` (ndr@draas.com). NEVER leave a colleague-facing draft in google-gmail (nishantranka@gmail.com) or any other identity — NDR's communication to colleagues is exclusively from ndr@draas.com. After `draft_create`, verify From via `drafts().get(format='full')`; if a draft landed in the wrong account, capture the body (text/plain or HTML), recreate in google-draas, verify, then delete the wrong-account draft.

## Skill Sections

### 1. Email Analysis

Triage inbox, identify pending actions, detect bounces, find sent mail awaiting replies, flag drafts.

**Core flow:**
1. Resolve account → build Gmail service
2. Fetch messages from last N hours (default 48)
3. Classify: project/business, replies, bounces, financial, marketing
4. Check sent mail, pending inbox replies, drafts
5. Compile structured report with 🔴🟡 sections

Key pattern for bounce detection — Gmail delivery failures are SEPARATE threads, not replies:
```python
# Search bounces separately
bounces = gmail.users().messages().list(userId='me',
    q='from:mailer-daemon@googlemail.com after:{timestamp}').execute()
```

Detailed reference: `skill_view(name="email", file_path="references/email-triage-code-patterns.md")`
Credential/password recovery search ("did vendor X ever send me a password?", "find my login for service Y in mail") across all accounts: `skill_view(name="email", file_path="references/credential-search-patterns.md")`
Sent mail verification ("did I send X to Y with that PDF?"): `skill_view(name="email", file_path="references/sent-mail-verification.md")`
Legal case thread tracing: `skill_view(name="email", file_path="references/legal-case-thread-tracing.md")`
Forensic "who was marked on that email" lookup (find a specific past email from a fuzzy/voice description, resolve mis-transcribed names/amounts, report exact To/Cc of original vs forward vs reply; sender-domain sweeps when only the sender is known; financial-artifact tracing and threads.get-404 fallback; the **"I sent an email to X" → X was actually only Cc'd** trap and how to handle the reply-all decision it creates; building the **Gmail thread deep-link** for "give me a link to that email"): `skill_view(name="email", file_path="references/email-forensic-thread-lookup.md")`
"Give me the subject line / link to that email" where the email is a *published DRA Content briefing* NdR authored ("there is an email I shared with Roshni and possibly Aamir regarding the Riverstone architecture briefing"): search by document CONTENT not recipient, expect a `content.ahfl.in/a/DRA-ART-…` URL in the body, disambiguate near-duplicate briefs of the same project across months, and check WhatsApp when the named person's "follow-up email" does not exist: `skill_view(name="email", file_path="references/deliverable-recall-cross-channel.md")`
Vendor / partner email hunt from a half-remembered description ("there is an email from one of our <category> partners, Brillo or something, about outstanding payments — find it"), INCLUDING the honest negative: the cross-account `in:anywhere` query matrix, the no-date-filter rule, the "the name he gives is wrong" rule, mailboxes that are NOT in the vault, and the anti-pattern of manufacturing a reply-all draft when the source thread cannot be found: `skill_view(name="email", file_path="references/vendor-email-forensic-sweep.md")`
Attachment content verification (verify JPG/PDF attachment content type via vision analysis when filenames don't match reality — e.g. signature pages labelled as floor plans): `skill_view(name="email", file_path="references/attachment-content-verification.md")`
Sender-sweep thread analysis (review all threads from one person — find real sender via domain sweep, group by threadId, read Google-Sheets-links-in-body budget models tab by tab, verify facts before analysis): `skill_view(name="email", file_path="references/sender-sweep-thread-analysis.md")`
Automated "declined / rejected / failed" status mail (DSC, eKYC, KYC vendors, registrars, banks — *"there's an email saying my DSC was declined and it's got some image I can't understand"*): enumerate MIME parts and grep the raw source for `cid:`/`<img src>` BEFORE any vision call, because the "image" is usually just a hosted logo and the reason is plain text in the `<html>` part; then harvest the application number/identifiers and pull the neighbouring messages from the same sender, since these pipelines emit contradictory sequences (completed → declined → completed): `skill_view(name="email", file_path="references/automated-status-email-triage.md")`

### 2. Email Drafting

Draft new emails or threaded replies. NEVER send autonomously — always create Gmail drafts.

**New email:**
```python
from tools.gws_skill_bridge import call
call("draft_create", service_name="google-draas",
     to="name@example.com", subject="Subject", body="Body")
```

**Threaded reply:**
```python
call("draft_reply_create", service_name="google-draas",
     message_id="...", body="Reply body")  # NOT thread_id!
```

**Reply-All (with CC):**
```python
call("draft_reply_create", service_name="google-draas",
     message_id="...",
     body="Reply body",
     cc="recipient1@domain.com, Recipient2 <r2@domain.com>")
```

The `cc=` parameter adds a Cc header to the draft. Extract the CC list from the original message's CC headers. Use comma-separated email addresses (with or without display names). The To field is auto-populated from the original sender — no need to pass it.

Key patterns:
- Always use `message_id` (not `thread_id`) for `draft_reply_create`
- Verify draft landed in Drafts, NOT Sent
- Contact resolution via `contact_resolver` tool, never guess emails
- Reply-From-Same-Inbox rule: match the `From:` to the inbox where the original was received
- **Colleague-draft account rule (NDR, Sep 2026): ALL communication to DRAAS colleagues is from ndr@draas.com (google-draas).** If a colleague draft lands in any other account (e.g. nishantranka@gmail.com / google-gmail), recreate it in google-draas with identical To/Subject/Body (thread it when the original was threaded), verify the `From:` header on the new draft, then delete the wrong-account draft. Never assume the draft you found is in the right account — check which account it lives in first.
- For attachments: use `MIMEMultipart` with `MIMEBase` parts
- **HTML formatting for complex emails**: For business-critical or family emails covering multiple commercial terms (property transactions, legal agreements, partnership terms), the user prefers **clean HTML+CSS formatting** — section headers, highlighted callout boxes, consent boxes, bullet lists, and organised layout. Use `call("draft_create", ..., html=True)` with inline CSS. Keep the tone conversational for family (first names, "Hi All", "Warmly") but structured enough that each commercial term is visually scannable. See `references/html-email-templates.md` for patterns and reusable CSS snippets.
- **Packaging and PROVING that HTML draft** — `html=True` yields a single `text/html` part (no text fallback); rebuild the same draft id as `multipart/alternative` via `drafts().update`, generate the text part with `scripts/html_to_text_email.py`, and verify the render by writing the HTML to disk + `browser_navigate("file://...")` + two `browser_vision` calls (top and bottom). Also carries the approval / buy-in email shape (terms table, "my vote for the record", per-person ask table, explicit limit of the authority, sequence table) and recipient-set discipline: `skill_view(name="email", file_path="references/html-email-mime-and-render-verification.md")`
- **Job / API-server sessions**: `HERMES_SESSION_USER_ID` is present but **empty**, so native `gws_*` tools answer *"No session user context (HERMES_SESSION_USER_ID not set)"* — that is not a vault outage and not a missing token. Bootstrap in-process BEFORE importing the auth layer: `sys.path.insert(0, '/opt/hermes')` → `from gateway.session_context import set_session_vars; set_session_vars(platform="api_server", user_id="ndr", session_key="<job key>")` → then `build_service` / `gws_skill_bridge`. Confirm with `users().getProfile()` and only ever for the job owner's own account.

Special workflows:
- Medical PAP OTP followup: `skill_view(name="email", file_path="references/medical-pap-otp-followup.md")`
- **Mass personalized draft campaign from a lead sheet** ("draft this offer email individually to every company in this spreadsheet tab, CC ndr@draas.com, embed this image") — 1,000+ individual Gmail DRAFTS, inline-image MIME (`cid:`), resume-capable background loop, drafts.list/update quirks, md5 verification of embedded images, and the cold-blast deliverability play (DMARC/DKIM/Postmaster ordering, 50-100/day cadence, complaint thresholds): `skill_view(name="email", file_path="references/mass-draft-mailmerge-deliverability.md")`
- Visiting card → Contact → Draft: `skill_view(name="email", file_path="references/visiting-card-to-draft.md")`
- Regulatory research for drafting: `skill_view(name="email", file_path="references/bangalore-regulatory-sources.md")`
- Data-heavy approval verification: `skill_view(name="email", file_path="references/refund-approval-verification.md")`
- Legal case follow-up to counsel (check reply, draft threaded follow-up with case developments, strategy changes, new attachments): `skill_view(name="email", file_path="references/legal-case-followup-drafting.md")`
- Investment holding forensics — recover exact share count, cost basis, acquisition date, and expected gain from registrar/depository/deal emails (dividend intimations, CDSL alerts, SPA PDFs): `skill_view(name="email", file_path="references/investment-holding-forensics.md")`

### 3. Attachments & PDF Handling

For threaded replies with file attachments, build MIMEMultipart:
```python
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from tools.gws_auth import build_service

gmail = build_service('gmail', 'v1', service_name='google-draas')
msg = MIMEMultipart()
msg['To'] = 'recipient@example.com'
msg['Subject'] = 'Re: ...'
msg['In-Reply-To'] = orig_hdrs['Message-ID']
msg.attach(MIMEText(body_text))
# Attach PDFs...
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
draft = gmail.users().drafts().create(
    userId='me', body={'message': {'raw': raw, 'threadId': thread_id}}
).execute()
```

For PDF highlighting/annotation before attaching:
`skill_view(name="email", file_path="references/pdf-highlighting.md")`

**Email attachment → Drive shareable links (when the user says "give me the links"):** download the attachment (Gmail API `attachments().get()` → base64.urlsafe_b64decode; verify `len(decoded) == size`), then upload binary to Drive with `MediaFileUpload` — the `files().create(body=..., media_body=<mimetype-dict>, media_path=<path>)` shortcut raises `TypeError: Got an unexpected keyword argument media_path`; the correct form is `MediaFileUpload(path, mimetype='application/pdf', resumable=False)` passed as `media_body`. Then `permissions().create(fileId=..., body={'type':'anyone','role':'reader'})` for anyone-with-link, and return `webViewLink`s (plus a folder link if you group them). Full recipe: `skill_view(name="email", file_path="references/email-attachments-to-drive.md")`. Also note: **"latest email" ≠ "latest email with plan attachments"** — later messages in a plan thread are often discussion-only (no attachments); check each candidate's attachment metadata before telling the user which set is current.

### 4. Gmail Scope Fallback

If `gws_skill_bridge.call("draft_create")` throws `invalid_scope` (the Gmail scope is broken in the vault but other Google APIs work), fall back to:
1. Compose the email as a structured Google Doc in TMP folder
2. User copies body from Doc into Gmail compose

```python
docs = build_service('docs', 'v1', service_name='google-draas')
created = drive.files().create(body={'name': '...', 'parents': [TMP]}, ...)
```

### 5. Sender Blacklisting (Auto-Cleaner)

When the user wants to blacklist senders so future mail lands in spam, there are two approaches:

**A. Native Gmail Filters (requires `gmail.settings.sharing` scope):**
- Create a filter via the Gmail API: `gmail.users().settings().filters().create(...)`
- This needs the user to re-authorize their Google account with the additional scope
- Run `send_oauth_url` with a label explaining the new scope after adding it to `HERMES_GWS_SCOPES` in `tools/gws_auth.py`
- The scope to add: `https://www.googleapis.com/auth/gmail.settings.sharing`
- **CORRECTION (verified 2026-09-22 on google-o3infotech): a never-send-to-spam whitelist filter does NOT need the sharing scope.** `filters().create(userId='me', body={'criteria': {'from': 'nvidia.com'}, 'action': {'removeLabelIds': ['SPAM']}})` succeeded with a token that only carries `gmail.settings.basic` (which the standard gws OAuth URL already includes — no re-auth prompt). Use the plain filter for domain whitelisting; reserve the sharing scope for settings beyond simple filters.

**B. Auto-Cleaner (fallback, no extra scopes needed):**
When native filter creation isn't available, use a cron-scheduled Python script that:
1. Queries the inbox for messages from target senders (via Reply-To header or From address)
2. Moves matching messages to SPAM using `gmail.users().messages().modify(addLabelIds=['SPAM'])`
3. Runs every 15 minutes

See the full reference script at: `skill_view(name="email", file_path="references/gmail-sender-blacklist.md")`

**Sender matching strategy:**
- For emails received via group forwards (e.g. `ndr@draas.com`), the `Reply-To` header often carries the actual sender address — search by that
- For direct emails, use the `From` header
- Always log which senders were matched and which messages were moved for audit

**Corner cases:**
- Gmail's `modify` with `addLabelIds=['SPAM']` silently skips messages already in SPAM
- Rate limits / bulk deletes: **do NOT use `new_batch_http_request()` for destructive ops (verified Sep 2026)** — a batch of ~100 `messages().trash()` calls silently trashed only ~40/100 while `execute()` returned without raising; per-item HTTP errors inside batch responses are swallowed, so "success" is not success. For bulk delete/trash use a per-message loop: `gmail.users().messages().trash(userId='me', id=...).execute()` inside try/except (~0.5–1s each; 150 msgs ≈ 2–3 min; idempotent so re-trashing is harmless), then VERIFY with `in:inbox <sender> "<distinctive phrase>"` search that zero matching messages remain. Full find→classify→trash→verify recipe for Drive ownership-transfer notifications: `references/drive-ownership-notification-cleanup.md`.
- **Drive share-REQUEST cleanup (verified 2026-10-09, ndr@draas.com).** Every Drive/Sheets/Docs share notification arrives from ONE sender, `drive-shares-dm-noreply@google.com`, carrying TWO opposite message types: `Share request for '…'` (someone is asking for access — this is what "delete all share requests" means) and `Item/Spreadsheet/Document shared with you: '…'` (someone shared something WITH the user — do NOT trash these). Always classify on the subject string before trashing; a bare `from:drive-shares-dm-noreply@google.com` sweep trashes both. Two further traps: (a) `messages().list` WITHOUT `includeSpamTrash=True` silently omits messages already sitting in Trash, so a second pass can return a *larger* set than the first and reclassify things you never saw; (b) `messages().untrash()` does NOT return a message to its previous folder — it lands in the INBOX, so an accidental untrash resurrects mail the user had deliberately deleted. If you over-trash, re-trash rather than leaving it in the inbox. Trashing these notification mails also does NOT clear the underlying pending access requests in Drive — say so explicitly, since the user will otherwise assume the requests are answered.
- The auto-cleaner cannot retroactively un-spam messages — the **daily not-spam whitelist cron** does the opposite: it reads rules from the DRAAS whitelist Google Sheet (tab `Whitelist`), scans the SPAM folder (max 200) and moves matches to INBOX via `modify(addLabelIds=['INBOX'], removeLabelIds=['SPAM'])`. Full sheet schema, matching semantics, pitfalls, and the working script: `references/not-spam-whitelist.md`.
- **Bulk ARCHIVE a sender's CC/marketing blast — keep searchable, clear the inbox** (verified 2026-10-12, 237 msgs): enumerate with paginated `messages().list` (maxResults=500 + nextPageToken; the API caps at 100 per query), `batchModify` removing `INBOX`+`UNREAD` in ≤900-id batches, then verify `in:inbox` → 0, exact total unchanged, samples show only category labels. Differs from trash sweeps: messages stay in All Mail. Recipe + pitfalls: `references/bulk-archive-inbox-cleanup.md`.

### Attachment Content Verification

See `references/attachment-content-verification.md` for the workflow of verifying whether email attachments match their claimed content type (e.g., "floor plan" vs actual "signature sheet"), cross-referencing with other threads for the real documents, and organizing on Drive.

## Repository Layout

```
/data/hermes/skills/email/email/
├── SKILL.md
└── references/
    ├── email-triage-code-patterns.md
    ├── legal-case-thread-tracing.md
    ├── pdf-highlighting.md
    ├── email-forensic-thread-lookup.md
    ├── attachment-content-verification.md
    ├── medical-pap-otp-followup.md
    ├── sent-mail-verification.md
    ├── credential-search-patterns.md
    ├── refund-approval-verification.md
    ├── html-email-templates.md
    ├── bangalore-regulatory-sources.md
    ├── visiting-card-to-draft.md
    ├── legal-case-followup-drafting.md
    ├── gmail-sender-blacklist.md
    ├── gmail-bulk-delete.md
    ├── html-email-mime-and-render-verification.md
    └── not-spam-whitelist.md
└── scripts/
    └── html_to_text_email.py
```

## Common Pitfalls

0. **First-time OAuth for a new user is async + throttled** — when `gws_resolve_account` shows `has_token: false` for the user's own email:
   - Send ONE `send_oauth_url` (login_hint = their email, label states the email).
   - After the user says done, RE-CHECK `gws_resolve_account` before building the service — token registration can lag the user's "successful" confirmation by tens of seconds. Wait ~45s and re-check rather than assuming failure.
   - Do NOT resend `send_oauth_url` repeatedly: Telegram flood control blocks re-sends for ~9 minutes (`Flood control exceeded. Retry in 553 seconds`). If the first send failed AND the vault still has no token, ask the user to re-tap the original button, or wait out the cooldown — don't hammer.
   - A fetch like `gws_fetch_token(service_name=...)` returning "No <service> token for user <uid>" confirms the user hasn't authorized yet — it is NOT a vault outage. Distinguish via `gws_resolve_account` first.
0a. **`draft_reply_create` auto-populates To from the message's SENDER — NOT the intended reply-all audience.** Confirmed Aug 2026: replying to a message whose last sender was an internal colleague (Eshwari → cc chain) produced To: Eshwari, not the external party the user wanted. The bridge is only safe when the original message's From IS the target recipient (e.g. replying to an insurer's own email). For any reply-all with a custom recipient set — especially when the last message in a thread came from a colleague or was addressed to a third party — use the **direct MIME pattern**: `gmail.users().drafts().create()` with an `email.mime.multipart.MIMEMultipart` that sets explicit `To`/`Cc` headers AND `In-Reply-To`/`References` from the target message's `Message-ID`, plus `threadId` in the draft body, so threading is preserved but recipients are exact. Verify the draft's To/Cc via `drafts().get(format='full')` before presenting. **Anchor selection (verified Sep 2026):** when the user says "reply all to that email from <person>" but the thread has moved past their message, still use the **newest message in the thread** as the In-Reply-To/References anchor (threading then lands the reply at the end of the conversation) while building the To/Cc from the full intended audience — the anchor and the recipient set are independent choices. Attach files (e.g. a quotation PDF) with `MIMEBase` parts; uppercase-extensions PDF filenames survive fine. **Threading needs THREE things, and it fails SILENTLY without any one of them (verified 2026-10-23):** the `In-Reply-To` header, the `References` header (`orig References` + `' '` + `orig Message-ID`), AND `threadId` inside the draft body — `drafts().create(userId='me', body={'message': {'raw': raw, 'threadId': <orig threadId>}})`. Omit them and Google accepts the draft with `In-Reply-To: None` / `References: None` and files it in a **brand-new thread**; it reads as a successful reply and is not. Always read the created draft's headers back and assert `threadId == orig threadId`; if it missed, **`drafts().delete()` the orphan first** and rebuild, or the user finds two copies of the same reply in Drafts. Source the anchor `Message-ID` from the message being replied to, not the thread — for Outlook senders it looks like `<MAUP287MB632229F709…@MAUP287MB6322.INDP287.PROD.OUTLOOK.COM>`.
0b. **`drafts().get()` does NOT accept `metadataHeaders`** — `gmail.users().drafts().get(userId='me', id=..., format='metadata', metadataHeaders=[...])` raises `TypeError: Got an unexpected keyword argument metadataHeaders`. Use `format='full'` and parse `payload.headers` yourself.
1. **`message_id` vs `thread_id`** — `draft_reply_create` needs `message_id` (Gmail message ID), NOT the thread ID
2. **Account mismatch** — never silently use the default account when user names another; surface expired tokens
3. **Draft promotion to Sent** — Gmail may move drafts to SENT; always verify after creating
4. **Bridge returns strings** — `gws_skill_bridge.call()` returns JSON strings, not dicts; use `json.loads()`
5. **Bounces are separate threads** — delivery failure notifications are not threaded with the original sent message
6. **Voice/chat names are often wrong** — verify from the actual email thread before drafting. This cuts BOTH ways: (a) before *drafting* to a person, confirm their real address from the thread; (b) when *locating* a specific past email from a voice-memo description, do NOT search the verbatim spoken entity name / amount / person name — voice transcription mangles them (e.g. user said "Reformers Collective" → real entity "Red Soul Farmers Collective"). Triangulate with phonetic/substring variants. And when asked "who was marked on that email," the **original, the forward, and the reply each have their own To/Cc** — check each variant's own top-level headers (a forward to a third party is frequently `Cc: None`, even when the embedded original body shows a Cc). See `references/email-forensic-thread-lookup.md`.
7. **A thread that "acknowledges receipt" of an attachment may not carry the attachment in your mailbox** (verified 2026-10-13, Sales1/Ranka NorthStar). NDR's 28 Sep reply *thanked* the architect for "the area statement Excel" and Arvind's mail said "the area statement is sent on 21st September" — but the attachment was only in NDR's mailbox / WhatsApp, NOT in Bharat's (sales1.blr@draas.com). Before promising a file to the user: check every message's `attachmentId`s in the thread, and if empty, search that person's OWN mailbox (via DWD only if the user explicitly asks to read that person's account) or state plainly the file isn't in reachable mail. Do NOT fabricate a Drive location for it; use the area-statement messaging-drafts pattern (draft a reply to the requester saying the link is needed / ask the architect directly).
7. **Financial/broker statement emails are often portal notifications with NO attachment** (IBKR "Monthly Activity Statement", Fidelity "Disclosure About Your Retirement Plan" — verified 2026-08-30 on IBKR U10942986). The mail body shows only a masked account number (e.g. `U***42986`) and a portal link; the real numbers live behind login (IBKR Portal → Performance & Reports → Statements/Tax Docs; Fidelity NetBenefits). Before telling the user "I found the statement data in email", check parts for `filename` — if none, the data must be exported from the portal by the user (or requested as CSV), and the notification merely proves the account + which mailbox it is registered to. Cross-account search pattern: IBKR notifications for NDR land on *both* ndr@ahfl.in (older months) and nishantranka@gmail.com (recent) — search all accounts, don't assume one mailbox.
8. **HTML-only emails have an EMPTY `text/plain` part** (verified 2026-09-09 on iMeridian/eSSL vendor mail): many B2B/vendor mailers are multipart with only `text/html` populated — the `text/plain` part is missing, so a naive body extractor returns "". When `payload.parts` yields no `text/plain`, decode the `text/html` part (base64url) and strip tags: `<br>` → `\n`, block-closing tags (`</p|div|tr|li|h1|h2|h3>`) → `\n`, strip remaining `<[^>]+>`, then `html.unescape` and collapse whitespace. Look for `text/plain` and `text/html` via a recursive `find_part(payload, mime)` walker — nested parts are common.
9. **Gmail API batch() destructive ops SILENTLY PARTIAL-FAIL** (verified 2026-09-10 on a 154-message trash sweep): `gmail.new_batch_http_request()` + `messages().trash()` executed without raising, but only ~40/154 actually landed in TRASH; a per-message `trash().execute()` loop trashed all 154 cleanly. Never trust batch `execute()` for delete/trash/modify sweeps — loop individually (add progress prints + per-item error capture) and verify afterwards with an explicit `in:trash` count, because default Gmail search EXCLUDES Trash/Spam and will show only the non-trashed remainder.
10. **Gmail message IDs are 16 hex chars — never truncate in prints/logs.** Printing `id[:14]` into diagnostics then re-using that string later raises `HttpError 404 "Requested entity was not found."` on `messages().get()`. Always carry full IDs end-to-end (search returns full IDs; only your own prints truncate them).
11. **Drive share / ownership-notification cleanup ("delete all drive notification emails"):** Drive notifications arrive from **TWO** addresses — `drive-shares-dm-noreply@google.com` AND `drive-shares-noreply@google.com` — never assume just one (verified Sep 2026 sweep of ndr@draas.com, 626 msgs, 0 failures). NDR treats the whole category as ONE cleanup: "delete all drive share notifications period … and all share request notifications as well" — do NOT bucket by sharer (Bharat/Sarthak it doesn't matter); sweep both senders with `from:drive-shares-dm-noreply@google.com OR from:drive-shares-noreply@google.com` paginated. Subject families:
- `drive-shares-dm-noreply@google.com` → "Item/Spreadsheet/Document shared with you", "Share request for ..." (someone requesting access)
- `drive-shares-noreply@google.com` → "now the owner of ...", "Invitation to edit/collaborate/view" (subject often = the file title), "Request for access"
- Match on the FROM ADDRESS, not the display name: the From string is `"NAME (via Google Drive)" <drive-shares-*-noreply@google.com>`, and the sharer's account login (e.g. sales1.blr) is NOT in the header — voice search on the display name gives zero hits.
- Bulk sweep: search each sender paginated (maxResults=500 loop on nextPageToken), trash PER-MESSAGE in a loop (batch silently partial-fails — see pitfall 9), run as a background terminal script with notify_on_complete for >~400 msgs (~0.5-0.6s per trash; 626 msgs ≈ 6-7 min — foreground 600s cap is tight; progress prints every 100), then VERIFY each sender search returns zero (Gmail search excludes Trash by default, so zero hits = all trashed). Trash is reversible — state that to the user. Full recipe: `references/gmail-bulk-delete.md`.
12. **`from tools.gws_auth import build_service` needs `sys.path.insert(0, '/opt/hermes')` first.** The `tools` package lives at `/opt/hermes/tools/`, but the default cwd for `execute_code`/`terminal` is `/opt/data`, so the bare import raises `ModuleNotFoundError: No module named 'tools'` — even though the system prompt's examples show it unqualified. Prepend the path insert to every Gmail/Workspace script. This is a path-resolution issue, NOT broken Gmail access — never report the account as unreachable because of it.
13. **The vault's `canonical_uid: vault has no identity mapping for '<uid>' -- using raw id as fallback key` line on stderr is harmless noise.** Calls still succeed against the correct mailbox. Verify the mailbox with `users().getProfile()` (`MAILBOX: ndr@draas.com`) rather than treating the warning as an auth failure.
14. **"The email I sent to <colleague>" frequently means the colleague was CC'd, not To'd.** NDR's colleague-facing coordination mail is often addressed to the external party with the colleague copied. A `to:<colleague>` search alone will miss it. Search `in:sent <name>` as well, then read the real `To`/`Cc`. Before drafting or offering a **reply-all**, always print the actual audience — a reply-all to such a thread goes to the *external* party, which is a materially different action. See `references/email-forensic-thread-lookup.md` (Step 2c).
15. **`gws_gmail_attachment_get` returns URL-safe base64 — decode with `base64.urlsafe_b64decode()`, NOT plain `b64decode()`.** The payload uses the URL-safe alphabet (`-` and `_` instead of `+` and `/`). Plain `b64decode()` silently skips those characters, returning a **shorter, corrupt file** that `pdftotext`/`pymupdf` fail on with `Illegal character in hex string` or zero pages, while the base64 string length still looks "complete" (e.g. 5,894,480 chars → only 4,276,086 bytes instead of 4,420,858). Verified 2026-09-23 on NSDL invoice PDFs (4.4 MB and 84 KB attachments). Sanity-check: `len(decoded)` must equal the `size` field from the tool result; if shorter, you used the wrong decoder.
16. **`gmail.users().messages().attachments().get()` takes `messageId=` (camelCase), NOT `message_id=`** — passing `message_id=` raises `TypeError: Got an unexpected keyword argument message_id`. When looping thread attachments, call `attachments().get(userId='me', id=<attachmentId>, messageId=<msg_id>)`.
17. **Preliminary deal emails ≠ final reality for unlisted/pre-IPO purchases.** The thread subject and first confirmation can quote a pre-corporate-action share count ("NSE Sale of 450 Shares"); the final signed SPA restates quantity × price proportionately at a CONSTANT total (450 @ ₹5,650 → 2,250 @ ₹1,130, both ₹25,42,500). Trust only: final SPA Schedule 1 + RTA dividend intimation + CDSL credit alert. Full recipe: `references/investment-holding-forensics.md`.
18. **A "find that email" hunt must survive the four traps that make a PRESENT email look ABSENT — and must end in an honest negative rather than an invented one** (verified 2026-10-09 hunting a Google Workspace partner billing mail): (a) a date bound (`newer_than:120d`) hides the target — the live vendor thread sat ~580 days back, so ALWAYS run one pass with **no date filter**; (b) the name the user half-remembers is usually wrong ("Brillo" = **Brio** Technologies) — search the **domain, the topic and the display-name family**, never the spoken name; (c) an address used only in `To:` never appears as `From:` (`BK@fnw.homes` is a recipient-only address; Bhuvanesh *sends* from `bk@findingform.design`) so a `from:`-only sweep misses the person entirely; (d) some of the user's mailboxes are **not in the vault** (no `@drahomes.in` service — Brio's billing chasers and NSDL notices land there), so report "not in the mailboxes I can reach", never "it doesn't exist" — and note the converse: mail *addressed* to `@drahomes.in` (BCC'd or group-expanded) can still be sitting in `ndr@draas.com`, and a Gmail result shows the message's original `To:` header, so never eliminate a mailbox on the strength of a domain in the address; (e) if ONE account's sweep errors with `invalid_grant: Token has been expired or revoked.` (seen on `google-o3infotech`), that account alone needs re-auth — send the button, keep the other accounts' results, and don't call the hunt failed. `in:anywhere` is mandatory in every query (plain queries silently exclude Trash AND Spam). When the result is genuinely empty, deliver the swept-mailbox list + query axes + the best 2–3 real candidates + what the topic maps to today + ONE precise unblocking question. **If the ask was "reply all", the source thread is a hard prerequisite — never create a speculative draft to a guessed vendor on a stale thread.** Full method: `references/vendor-email-forensic-sweep.md`.
19. **Large Gmail attachments truncate in the native `gws_gmail_attachment_get` tool output — download big files via execute_code instead** (verified Oct 2026 on a 1.1 MB PDF: the tool returned a 1.5M-char base64 string that got cut with `[Truncated: tool response was 1,513,350 chars...]`, so the bytes never reached disk). For files larger than a few hundred KB, or whenever the tool response truncates, use: `gmail = build_service('gmail','v1',service_name='google-draas')` then `gmail.users().messages().attachments().get(userId='me', messageId=<msg_id>, id=<att_id>).execute()` (camelCase `messageId`, NOT `message_id`), decode `data` with `base64.urlsafe_b64decode(data + '=' * (-len(data) % 4))`, write to disk, verify `len(decoded) == size` from the message metadata, then confirm with `pdfinfo`/`pdftotext` (expect `Pages:` present) before delivering.
20. **Drive uploads & anyone-links from execute_code:** `drive.files().create(..., media_body={'mimeType': ...}, media_path=<path>)` raises `TypeError: Got an unexpected keyword argument media_path` — use `MediaFileUpload(path, mimetype='application/pdf', resumable=False)` from `googleapiclient.http` instead. The native `gws_drive_permissions_add` requires an `email` argument and fails for `perm_type='anyone'`; to make a file/folder viewable by anyone-with-link (so the user can forward directly), call `drive.permissions().create(fileId=..., body={'type':'anyone','role':'reader'})` directly.
19. **Never pull large Gmail attachments into context via the native attachment tool — write them to disk with `build_service` instead (verified 2026-10-13).** `gws_gmail_attachment_get` returns the FULL base64 inline: a 1.1 MB PDF came back as ~1.5M chars and was truncated before it was usable. For any attachment you need to actually deliver or re-file: `execute_code` → `sys.path.insert(0, '/opt/hermes')` → `gmail = build_service('gmail','v1', service_name=...)` → `gmail.users().messages().attachments().get(userId='me', messageId=<msg_id>, id=<attachmentId>)` (note the camelCase `messageId=`), decode with `base64.urlsafe_b64decode(data + '=' * (-len(data) % 4))`, write to disk. Then VERIFY: `len(decoded)` must equal the attachment `size` from search metadata and the first bytes must be the expected magic (e.g. `%PDF`). Deliver via `MEDIA:/path` or upload to Drive for links.
20. **Making Drive files link-shareable — native `gws_drive_permissions_add` requires a user `email` and cannot do an 'anyone' public-view link.** For 'anyone with the link' sharing (needed when handing file links to external parties/landowners): `drive.permissions().create(fileId=..., body={'type':'anyone','role':'reader'})` via `build_service('drive','v3',...)` in execute_code. When uploading a local file, use `googleapiclient.http.MediaFileUpload(path, mimetype=...)` as `media_body` — passing `media_path=` raises `TypeError: Got an unexpected keyword argument media_path`, and a bare dict as `media_body` silently does not upload. Verified 2026-10-13 (3 A.J. Architects sanction PDFs + meeting-kit doc).
21. **Audit EVERY recipient of a multi-recipient send separately — one bounce does not mean all bounced** (verified 2026-10-13: a two-recipient authority send had `rpo.bangalore@mea.gov.in` fail with `550 5.1.1 User does not exist` while `passport.pg@mea.gov.in` was accepted). Query `from:mailer-daemon@googlemail.com OR from:postmaster@googlemail.com` and read the DSN's `Final-Recipient` / `Status` / `Diagnostic-Code` lines — they name exactly which address died. `550 5.1.1 User does not exist` means the mailbox is genuinely retired; greylisting and spam filtering produce NO DSN, so a missing bounce is not proof of delivery, and a present bounce is not proof that the whole send failed.
22. **To re-send an already-sent email to a CORRECTED address, recover the sent body instead of rewriting it** (verified 2026-10-13). Find the sent copy with `in:sent to:<dead-address>`, walk its `payload.parts` for the `text/html` part, decode base64url, and reuse that exact string as the body of the new draft — then re-attach the evidence files with `MIMEMultipart` + `MIMEBase`. This preserves the wording the user already reviewed (and any edits they made in Gmail) and guarantees the replacement is a true copy. Note that `drafts().list()` will show NOTHING once the user has actually sent the drafts — an empty Drafts folder after a bounce means "already sent", not "never created".
23. **A reference number that appears in NO mailbox is itself a finding: the external system has a different address on file.** When a portal/authority says "send from the email registered on your application", search every reachable mailbox for that reference (`from:<portal-domain>` AND the literal ARN/ID, with `includeSpamTrash=True`). If the user's OWN older applications produced notifications but this reference produced zero, the registered address is not one of the user's mailboxes — report it as a gap and have them read it off the portal profile. Never guess it.
24. **Pulling a handful of images out of a message: use the native `gws_gmail_attachment_get` per attachment and decode from the PERSISTED tool output — do not try to replay the metadata token against the raw API** (verified 2026-10-16, 5 JPGs on Bhavik's Ranka Iris mail). Two working facts and one trap:
   - When the tool result is large it is **persisted**, not truncated: the response becomes `/tmp/hermes-results/chatcmpl-tool-<hash>.txt` with only the first 1500 chars inlined, and the `dataBase64` field is complete inside that file. Decode straight from it — no second API call needed:
     ```python
     import json, base64, os
     raw = open('/tmp/hermes-results/chatcmpl-tool-<hash>.txt').read()
     o = json.loads(raw[raw.find('{"service_name"'):raw.rfind('}')+1])['result']
     b = o['dataBase64'].replace('_', '/').replace('-', '+')       # URL-safe alphabet
     open(out, 'wb').write(base64.b64decode(b + '=' * (-len(b) % 4)))
     assert os.path.getsize(out) == o['size']                       # ALWAYS assert
     ```
     Name the output from `attachment_id[:8]` so several decoded files stay distinguishable.
   - The trap: the `attachmentId` shown in `gmail_thread_get` metadata is **not reliably replayable** against `users().messages().attachments().get()`. Copying it verbatim into a `build_service` script raised `HttpError 400 … "Invalid attachment token"` for tokens that the native tool had just used successfully. So for a small set of images, loop the **native tool** and decode from the persisted files; reach for `build_service` only for the large-file case in pitfall 19, where you are re-requesting via the same message and can afford to re-resolve.
   - One attachment in the set can fail on a stale token while the rest succeed. Decode what came back, **name the one you could not retrieve**, and say what it was supposed to show — never report the set as unreadable or quietly drop it.
   - `file` is not installed on the host; verify a decoded attachment with its magic bytes, or with `pdfinfo` / `pdftoppm -r 200 -png` for PDFs (poppler is present in `/usr/bin`).
25. **"There is an email in which <person> followed up" is frequently a WhatsApp message, not mail** (verified 2026-10-21: NdR said Arvind had emailed a follow-up asking for the Riverstone brief; no Arvind mail existed in 20 days, but WhatsApp held *"Regarding riverstone, waiting for the brief"* of 13 Oct, answered *"Sending today w/o fail"*). When a named person's follow-up is absent from mail, run `wa_list_chats(query="<First name>")` → `wa_read_messages` BEFORE reporting a negative, then state plainly which channel it was on, with the quote, the date, and NdR's own reply. Do not let the correction imply the email exists anywhere.
26. **The same project accumulates near-duplicate briefs months apart — name the one you chose AND the one you rejected.** Riverstone has both *"Terra Green - Estate Home Architectural Briefing (Riverstone Farms & Portfolio)"* (8 Oct 2026, artifact `DRA-ART-2026-000058`, carries the process note + image study + an "OPENING NOTE — FORWARD THIS TEXT TO ANY ARCHITECT" block) and the much older *"River Stone Farms — Villa Design Briefing"* (4 May 2026, 4,000 sft G+1 villa typology, material palette, Abelion reference). Search one bare project keyword first to enumerate them, pick on what the description actually mentions (process/compositions/forms/portfolio → Terra Green; villa typology/material palette → the villa brief), deliver `Subject · Sent · To · artifact link · thread link` as plain URLs, and recommend the draft-only next step. Full recipe: `references/deliverable-recall-cross-channel.md`.
27. **Gmail links you hand over must be `rfc822msgid:` search permalinks — `#all/<hex>` and `/mail/u/<email>/` are both dead on a cold click** (NDR, 22-Oct-2026: *"the links don't seem to be working"* — all 16 were). The API's 16-hex message ID is not a web-UI ID, and the web UI's `QgrcJ…`-style IDs are per-account server-side ciphertext the API never returns and that cannot be derived. Use `https://mail.google.com/mail/u/0/#search/rfc822msgid%3A%3C<msgid>%3E` (account **index**, not address), `urllib.parse.quote(..., safe='')` for `+`/`=`/`<`/`>`, and assert each query returns exactly one message before delivering. Also: `build_service('gmail','v3')` raises `UnknownApiNameOrVersion` — Gmail is **v1** only. Full recipe: `references/gmail-permalink-construction.md`.
28. **A message on ANOTHER channel that asserts "I have just sent you the email" must be gated on the email actually being in SENT, not Drafts** (hit twice on 22-Oct-2026 with the NorthStar landowners). A WhatsApp staged for Swaminathan and then Vaidyanathan both opened "I have just sent you the complete workings by email" while the Gmail draft had only been *created* — NDR had not sent it. Had the WhatsApp gone first, a counterparty already disputing ₹2.4 crore would have gone looking for mail that did not exist. Protocol: (a) before delivering any cross-channel message that references an email, read the real state — `drafts().list()` for a live draft id **and** an `in:sent to:<addr>` search for proof it went out, quoting the sent timestamp; (b) if it is still a draft, say so plainly and offer either to reword to "I am sending you…" or to hold until he sends; (c) do not silently reword a claim that has become true. **Inverse trap:** an empty Drafts folder does NOT mean "never created" — it usually means NDR already sent it (it happened here at 12:27, which is why the draft had vanished). Confirm via SENT before concluding anything, and never re-draft an email that is already sent. Same discipline as pitfall 22.
29. **"Forward these to <X>, cc <Y>" has NO forward tool — the draft that substitutes for one silently drops attachments** (verified 2026-10-23: Kotak balance confirmation forwarded to Eshwari, cc Roshni + Sarthak). The sanctioned surface is draft-only: native `gws_gmail_draft_create` (or bridge `draft_create`) with explicit `to`/`cc` and a hand-built block — `---------- Forwarded message ---------` then `From:` / `Date:` / `Subject:` / `To:` and the original body. Pitfalls: (a) **this path carries no attachments** — read the source message's `attachmentId`s first; if any exist, rebuild with `MIMEMultipart` + `MIMEBase` (section 3 above), otherwise state in the hand-off that the forward is text-only; (b) `draft_reply_create` is the WRONG tool for a forward — it threads onto the source and auto-addresses the original sender; (c) when the forward target is already the original recipient, say so — it is then only a record copy, and offering the alternative reading before he approves is cheaper than a pointless send. "All the emails individually" means one draft per source email — do not merge them into one, and do not invent extra emails to match an expected count (see pitfall 30).
30. **"There are a bunch of emails about X" is a hypothesis, not a fact — establish the real volume BEFORE drafting** (verified 2026-10-23: *"a bunch of emails yesterday on balance confirmations from Kotak… Nilesh, I think, has sent it"* = exactly **one** email, in a 3-message thread). Run the full sweep first: `in:anywhere` (never a bare query — Trash/Spam excluded) on the topic, a sender-domain sweep across the sibling domains (`from:kotak.com`, `from:kotak.bank.in`), `in:spam`, and the other vault accounts for that user. Then report the **actual** count, the thread's message-by-message structure, and name the plausible alternatives (a second mailbox not in the vault, a second person sharing that first name, a different date) as offers in one line. Never manufacture per-email forwards to satisfy the expected volume. **Class fact — bank balance confirmations:** the bank's reply is frequently a **dispatch notice with no attachment** (Kotak: *"Sent on registered mail ID with request ID 50536257"*, nothing attached; the certificate went to the registered email on the bank's side). Quote the request ID in the forward, and say plainly that the certificate itself is not in the mailbox — never imply the forward carries it.
31. **Bank statements have their OWN sender and subject — and "the whitelist didn't catch it" is usually the wrong hypothesis** (verified 2026-10-23, Kotak). Statements do **not** come from the relationship manager's address; they arrive from the bank's alert robot. Kotak: sender `bankalerts@kotak.bank.in`, subject `Bank statement for Kotak A/c X#### from dd-mm-yyyy to dd-mm-yyyy`, **one mail per account, one PDF each** named `<masked a/c>_<from>_<to>.pdf`. 27 such mails sat in INBOX, none ever in SPAM. `Nilesh.Prasar@kotak.com` (DVP & Branch Manager) sends **cover notes only, never an attachment**, so a hunt keyed on his address finds nothing and reads as "the bank never sent them". Before reporting statements as missing or mis-filtered: (a) search `in:anywhere subject:("Bank statement for")` — not the manager's name; (b) query the exact bank domain, never a bare `from:kotak`, because Gmail's `from:` also matches DISPLAY NAMES and body text and will return `kotakmutualfund@camsonline.com` as a false hit; (c) check **SENT** (`to:<colleague> (statement OR <bank>)`) — the set is often already forwarded weeks earlier (these six went out 9-Oct 06:45); (d) only then discuss rules. Whitelist coverage diagnosis for the DRAAS sheet (which rows cover Kotak, `kotakbank.in` gap, `@drahomes.in` mail landing in the draas mailbox) lives in `productivity/not-spam-whitelist` → `references/kotak-mail-coverage-and-statement-senders.md`.
32. **In an automated "declined / completed" pipeline, a later "completed" does NOT reverse a "declined".** Both the decline and the completion notice are emitted by the same robot on the same application ID, minutes apart, and in either order — Pantasign DSC 2026-10-23: video completed 10:46:19 → **DECLINED 10:48:22** → *"video recording completed"* again 10:50:10, which NDR then forwarded to Eshwari believing the job was done. Report **both events with timestamps**, state that the decline is the last written status until an approval mail or the status portal says otherwise, and hand over the application number plus the portal/status URL. Never let a post-decline "completed" notice read as a reversal, and never assume the colleague who received the forward knows about the decline that preceded it.
33. **A dictated recipient who exists in NO record must be left off the send, never approximated to the nearest contact** (verified 2026-10-23: NDR dictated *"…and Mark, Sartak and Roshni in the CC"*). "Sartak" resolved to Sarthak Sharma (`admin3.blr@draas.com`) and "Roshni" to Roshini Ranka (`rnr@draas.com`) — both real, both confirmable from headers. **"Mark" was nobody**: `contact_resolver(query="Mark", context="Ranka Northstar landowners")` returned only Mark Hocker (South Africa / TBR), Mark UAE and Mark Whitehead, none with any connection to the project, and no Mark appears in any landowner thread or in the Cc of NDR's own earlier mails on the same subject. Correct handling is to build the draft WITHOUT him and report in one line that Mark is unresolved + ask who he means. Do not add a similarly-spelled real contact to an external email to make the list look complete — a wrong external recipient on a landowner dispute mail is unrecoverable, and it silently exposes one party's commercial position to a stranger. Resolve every voice-dictated name independently and treat "resolves to nobody" as a finding to report, not a gap to paper over. Where the same briefing also names legitimate parties the user omitted (here Sreeraman and Kankshita, both registered landowners, both on NDR's own 14-Oct distribution for that exact subject), include them and SAY you did, with the one-line offer to remove.
34. **When the dictation contradicts what the counterparty actually wrote, follow the written record and flag the deviation** (2026-10-23: NDR dictated *"…tomorrow, Vaidyanathan sir if you are available, Swaminathan sir is happy to come in"* — but Swaminathan's last WhatsApp that same morning read *"Call won't be sufficient, we can meet in your office and sort it out once for all. Fix a meeting in your office sometime next week."*). Drafting "tomorrow" would have cut straight against the counterparty's stated position and read as careless to a man already disputing ₹2.4 crore. The draft instead asked him to *"confirm the earliest date we can meet at my office"* — satisfies both readings — and the deviation was surfaced in the report. Rule: before writing any scheduling or commitment line, re-read the counterparty's most recent inbound on EVERY channel (`wa_read_messages` + the mail thread) and reconcile the dictation against it. If they conflict, phrase to the record and name the difference; never silently obey either one.
35. **A DRA staff mailbox is 90% machine noise — exclude the sign-in/digest families or every `kelsa` / `from:ndr` search drowns** (verified 2026-10-26 on sales1.blr@draas.com). `from:ndr@draas.com kelsa` returned 25 rows of *"Please sign in for the day"* / *"Please sign out"* (auto-generated by Kelsa) plus `Kelsa Notifier <no-reply@kelsa.io>` dailies; the real email never surfaced. Always append `-subject:"Please sign" -subject:"Please sign out" -subject:"Sign In" -subject:"Daily Summary" -from:no-reply@kelsa.io`. Run the full multi-axis ladder (sender+topic, `to:`/`in:sent`, subject fragment, body terms, outside-party domain, one pass with NO date filter, `in:anywhere`) BEFORE reporting a negative — and when a "find that email" sweep on the user's own mailbox comes up empty, do not assume it exists elsewhere: report the mailbox swept + the axes + the only real hits, then offer exactly two named options (a subject-topic hint, or explicit go-ahead to search the named person's mailbox via DWD). A user who is certain a mail is in his own inbox is often remembering one he was *shown*, or one addressed to a colleague. Full query matrix + noise table: `references/dra-mailbox-search-noise.md`.
36. **"I sent you a request / did you get it?" — the mail is usually in SPAM, and a plain search will not see it** (verified 2026-10-28: NDR's cousin had chased twice about a *Common App* recommendation request from his son; `Semlani newer_than:60d` on google-draas returned **zero**, but `in:anywhere (… ) newer_than:20d` surfaced `Jay Semlani requests your recommendation` from `noreply@commonapp.org`, to the `ndr@drahomes.in` alias, sitting in **SPAM**). Portal / institution / vendor robots are the worst offenders: Common App, university admissions portals, DSC-eKYC pipelines, KYC vendors — Gmail files them as spam because they are bulk-ish, single-purpose, link-heavy sends. Protocol:
   - A bare `query` in Gmail **excludes Spam and Trash**. Any "but I definitely was sent this" hunt must carry `in:anywhere` (and, when still empty, an explicit `in:spam <term>`).
   - Search the **portal domain and the product name**, not only the person's surname — the subject line may be the only place the student's name appears, and the sender is a robot.
   - Remember the alias: mail addressed to `ndr@drahomes.in` lands in the `google-draas` mailbox, so search there even though there is no `@drahomes.in` vault service.
   - Once found, **rescue it**: `gws_gmail_labels_modify(add_label_ids=["INBOX","UNREAD"], remove_label_ids=["SPAM"])` on the message id, and tell NDR it was in spam — that fact is why he thought he had missed it, and he needs to know the actual receipt time versus the counterparty's chasers.
   - Report the precise original timestamp (convert the header's UTC to IST) alongside the counterparty's follow-up dates; a request that "never came" is usually a request that arrived at 03:51 and was never seen.
   - When the ask is a third party's action on a document the user must sign/submit, draft the apology + commitment message on the channel the counterparty actually used (here WhatsApp, via `wa_send_message`), attach a rendered snapshot of the email, and quote the portal's own deadline/link-validity line so the recipient can act.
