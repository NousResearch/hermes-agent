---
name: email-drafter
description: |
  Drafts new/threaded emails for ndr@draas.com. Threaded: In-Reply-To/References + threadId on
  drafts().create. CREATE: EmailMessage -> base64url raw (payload dict/bytes = 400).
  Refs: new-email-html-attachment-nonascii.md, forward-existing-sent-mail.md,
  threaded-reply-all-routing.md, sheet-range-as-html-table-email.md,
  threaded-reply-recipient-pitfall.md (native reply_draft mis-addresses To when the thread's
  last message is your own - build raw MIME instead)
  Escalation & accountability: references/internal-escalation-email-stern-not-offensive.md,
  references/accountability-and-escalation-emails.md
  Addresses from GMAIL HEADERS, not the sheet - Anbu SENDS as anbarasan@draas.com. Other mailbox:
  HERMES_SESSION_USER_ID=<slug>. Bridge: sys.path.insert(0,"/opt/hermes") before tools.gws_auth.
  Use nishantranka@gmail.com/ndr@ahfl.in only if asked. Reply+verify: references/threaded-reply-and-draft-verification.md Voice dictation: read thread tail first -
  NDR may be apologising for HIS OWN mail. Never send; draft only.
---

> ## ⚠️ FORMAT AUTHORITY — read this before writing a single body line
> **`email-writing` is the ONE skill that decides how every email body looks** (rich formatting,
> HTML part, bullets/headings, no hard wraps, plain-text only on explicit instruction).
> Load it first: `skill_view(name='email-writing')`. Build every draft with its renderer
> `/data/hermes/skills/communication/email-writing/scripts/email_draft.py` (markdown in →
> multipart draft out). THIS file only supplies Gmail mechanics on top of that standard.
> A draft created with a hand-written plain-text body is a defect, not a style choice.

> **Voice dictation -> numbered briefing** (incl. reply-all-only-to-<Person> + colleague map): `references/ndr-dictation-to-briefing-email.md`. NO internal attribution: `references/relayed-position-no-internal-attribution.md`. Threaded-reply script: `scripts/create_threaded_reply_draft.py`.

> **NEW email (no thread) + attachment:** see `references/new-email-html-attachment-nonascii.md` —
> ASCII-only Subject, base64 text parts, byte-verify the attachment, list drafts first to avoid duplicates.
> RECIPIENT ADDRESSES COME FROM GMAIL HEADERS, NOT THE CONTACTS SHEET: Anbu's sheet address is
> pm2.blr@draas.com but he SENDS as anbarasan@draas.com — mail him there. Import path for the bridge is
> `sys.path.insert(0, "/opt/hermes")` before `from tools.gws_auth import build_service`
> (`/opt/data` raises `ModuleNotFoundError: No module named 'tools'`).
>
> **Primary path: native GWS tools** — see `references/native-gws-tools-workflow.md` for the
> modern `gws_gmail_search` → `gws_gmail_get` → `gws_gmail_reply_draft` workflow. The native tools handle
> ~80% of email tasks without terminal/python. The legacy `google_workspace_manager` sections below
> apply to the remaining 20% (attachments, raw MIME, bulk ops).

## Voice-dictation names
A dictated name is not a fact — resolve it (ladder + garble table: `references/voice-dictation-name-resolution.md`). If it resolves nowhere: build the draft without it and flag it in the final reply (`clarify` is absent in API sessions). Avoid `&` in subjects — it arrives as `&amp;`.

## Template — HTML email structure

NDR prefers HTML-formatted emails (found plain text "not very visually easy to read" and
"one continuous running text"). Always use `html=True` on draft_create / reply_draft. Build inline-styled HTML:

**Plain text with `*text*` markdown = "one continuous running text" (NDR, 2026-10-03).** Markdown-style emphasis (`*bold*`) in a plain-text email is NOT acceptable for multi-point coordination. Convert to true HTML (`<strong>text</strong>`) before creating the draft. Applies to ALL architect/engineering/project-team emails with 3+ sections.

**Already-sent plain-text fix: reply on the SAME thread** (2026-10-03). When NDR says a sent email is "one continuous running text", create a **reply-all draft on the same thread** with the SAME content reformatted as HTML/CSS. Keep recipients and content identical; change only formatting. Delete any earlier drafts on that thread before recreating.

**Accepted multi-point pattern (validated 2026-10-03, North Star 2nd Floor Modification):**
- Numbered sections with blue `border-left: 4px solid #2a6496; background: #f5f5f5` header blocks
- Section title in `<strong style="color:#2a6496;">`
- Body content with normal `<p>`, sub-items in `<ul><li>`
- ✅ resolution callout: `<p style="font-weight:bold;color:#060;">✓ resolved ...</p>`
- Annotation style: `<p style="font-size:12px;color:#888;"><em>Note: ...</em></p>`
- `<hr style="border:none;border-top:1px solid #ddd;">` between sections
- Reply-all must include ALL participants from the original thread (To + Cc preserved)

- Blue `<hr>` dividers (`border-top:2px solid #1a73e8`) between sections
- `<h3>` blue (`#1a73e8`), `<h4>` red (`#cc0000`) for headings
- Tables with alternating `#f0f4ff` row backgrounds for structured comparisons
- Font: Calibri/Arial/sans-serif `#333`, clean spacing
- Inline styles only (Gmail strips `<style>` blocks)
REPLY-ON-THE-SAME-THREAD DISCIPLINE (user-corrected 2026-09-29): when NDR says "reply on the same thread", "put it back on email", or references a chain he already sent ("we sent a series of emails about X"), the deliverable is a reply-all THREADED draft on the ORIGINAL thread — never a standalone draft. Find the thread first (he often mails these from ndr@drahomes.in, NOT in the vault — locate via a participant's reply in draas). See references/reply-on-same-thread-discipline.md — full search recipe, reply-all recipient rule, sensitive-content flag, build+verify, orphan-draft deletion.

metadata:
  hermes:
    tags: [email, gmail, draft, communication, reply, thread]
category: communication
version: 2.4.0
author: ndr@draas.com
---

# Email Drafter

> `references/threaded-draft-relayed-attachments.md` — threaded Gmail reply draft (raw
> API: downloads().attachments() from a DIFFERENT email, EmailMessage with
> In-Reply-To/References from the original Message-ID, drafts().create with
> threadId), the transient-404-after-create retry, and metadata-format
> attachment-count verification via format='full'. Load when a reply draft must
> carry attachments that came from another message/thread (e.g. Sanjay's docs →
> reply on Dharmesh's thread).
> `references/forward-draft-relayed-attachments.md` — FORWARD draft (fresh recipient,
> no In-Reply-To) carrying attachments relayed from MULTIPLE source emails (latest
> vendor reply + earlier quote/brochure + original RFP), plus the Gmail 25 MB cap
> split: attach the small files, put oversized photos/videos in a shared Drive
> folder and link them in the body. Load when NDR says "forward that email to X"
> with attachments from several messages.
> **Forwarding a vendor/counterparty email with a reference pack that EXCEEDS Gmail's 25 MB cap** → `references/oversized-forward-attachment-split.md` — enumerate attachments across the thread, download everything (incl. videos embedded as Drive links — fetch via `drive.files().get_media()`), split into a small attach-set (docs + small photos) + a mirrored Drive folder with role-scoped permissions (doer=writer, cc'd=reader), keep the counterparty's real email body as a quoted block, verify DRAFT label + attachment count. Validated 2026-09-22 (AR Portable Cabin executed-work refs → Sinchana, Cc Neha+Gowri).
> **Forwarding a colleague's mail because the ORIGINAL went to a WRONG recipient address** → draft To = the correct recipients (resolved from headers/contacts — NEVER the bounced address), Cc = the original sender, with an explicit note that the wrong address was used and that NDR's own reply-all also went there; THREAD onto the original thread (In-Reply-To = original Message-ID + `threadId` in the `drafts().create` body) even though it is a Fwd, and quote the original under a "---------- Forwarded message ----------" block. Verified 2026-09-29 (Prakash's "Ranka Amber/Oasis/Udaya - Required - Estimated Construction Cost" → Anbu `pm2.blr@draas.com` + `anbarasan@draas.com` + Kantesh `kanteshbg@draas.com`, Cc Prakash; the original had gone to dead `kantesh@draas.com`). Keep both of Anbu's live addresses when Prakash used both — only Kantej's was wrong.
> **Handing a vendor's submissions + quotation to a COLLEAGUE to coordinate** (\"send that email to <colleague> and ask them to take it forward\") → `references/colleague-coordination-handoff-reply.md` — REPLY on the colleague's own thread (not a forward), answer their pending question first, vendor contact block, the reuse-the-existing-mirror-folder check (link it, don't re-upload), the ⚠️ **`permissions().list()` a shared folder BEFORE linking it into any outbound mail** rule (it surfaced three unrelated external vgrofinserv.com reader grants that had to be flagged, not silently shipped), the 4-block colour-coded body, and the `%`-in-body-text / two-body-script patching pitfalls. Validated 2026-09-25 (AR Portable → Gowri, Ranka Udaya site office).
> **Checking a vendor quotation against OUR render + spec** (\"is anything missing from their quote, highlight the gaps\") → `references/vendor-quotation-gap-analysis.md` — pdftotext the quote, read our own spec section by section, bucket into covered / missing-or-excluded / contradicts-the-design (citing the vendor's OWN T&C lines as evidence), hunt internal contradictions (transport included vs \"extra at actual\"; toilet in the standard spec vs none in our design), then the numbered HTML gap-closure reply ending in our 48-hour commitment. Validated 2026-09-15 (AR Portable `ARPC/PMO/7042-1/26-27`).

> Excluded-participant / non-bridge replies → `messaging-drafts` → `references/thread-reply-exclude-participant-2026-09.md`. Flameback AI portfolio updates → `references/flameback-portfolio-update-emails.md`

> **Unit/asset photo follow-up blasts ("give everyone the cleaned-up photos + share the folder 90/180-day")** → `references/hermitage-2201-followups-2026-09.md` — file the photo set into the project photos tree first, enumerate prior recipients from the consolidation mail, grant reader+expiry BEFORE drafting, draft-only threaded replies (message id ≠ thread id 404 pitfall), share ONLY the images folder. **Inbound share-requests on that folder** (grant 30-day viewer → confirm draft with link → trash the two notification mails; "20201" voice garble; requester address from the notification, NOT contact_resolver) are in the same reference.
> **Financier / lender onboarding packs — "consolidate docs for each firm/entity"** → `references/entity-wise-financier-document-pack.md`. When an NBFC/bank BD person asks for the standard onboarding bundle (audited financials, company & promoter profile, debt profile, project details, sanction letter) across SEVERAL entities: build ONE parent Drive folder + one subfolder per entity + per-promoter folders, share the tree as `reader` (never attach 30-50 files), verify each source doc's entity attribution before it goes in, and surface gaps (no sanction letter, unaudited firms, ambiguous debt-profile scope) as questions in the email rather than omitting them. Carries the correction that a firm document can sit in the WRONG entity's folder — read the deed's own party list, not its folder. **Also carries:** do-not-link the internal investor/portfolio workbook (scope-wider than the ask) + the two-sided revert when the user says "remove this" (trash the shortcut AND revoke the grants); the My Drive `403 "Increasing the number of parents is not allowed"` → use a `shortcutDetails` shortcut instead of a move; and the follow-up-to-a-DIFFERENT-participant rewrite rules (stale "I am sharing…" sentences, third-person self-reference, `References` = chain + replier's Message-ID). **And the "reply with ONLY what is asked for" rule** — once the pack is on the thread, a later reply to a participant carries ZERO Drive links and no attachments: cut the folder table, the per-entity listings and every sentence describing the pack, keeping just the answer box (`html.count("drive.google.com") == 0` is the proof you did not re-send). Plus the user-supplied financing position to **quote, not infer**: project-wise financing, entities and parent hold **no existing sanctions** (fresh credit is the ask), **APF held for named projects only**.

> **Structural-consultant / architect briefing emails — what we expect from a design partner** → `references/architect-design-partner-briefing.md` — the approved two-phase engagement model (Creative Phase run JOINTLY with the design team incl. joint reference harvesting; Detailing Phase by in-house/local teams with senior partners doing checks and balances), the recipients (Roshini `rnr@draas.com` To, Gowri `gsingh@draas.com` Cc), the 150–250-word cap, and the 2026-09-25 correction that killed the "we hand you references, you generate" framing. Cross-links the Tapovana DNA programme as the sample run to offer.
> **Handing a published DRA Content artifact to an EXTERNAL reviewer** ("share it with him as well", "give Aamir access please") → `references/content-artifact-external-reviewer-share.md` — grant BEFORE drafting, read the ACL back, and the ⚠️ **sign-in-address rule**: an artifact is viewed through Google sign-in, so the draft must ASK the recipient to confirm their Google account matches the address granted. Validated 2026-09-28 — the architect's contact address was a Yahoo, but his Google sign-in was a gmail.com he had to tell us about; the Yahoo grant was live and useless for sign-in. Grant BOTH when in doubt.

> > **Samudra / ORR forward-purchase mail to Godrej Fund (Saurabh Vashishth)** → `real-estate-financial-modeling` → `references/samudra-orr-forward-purchase.md`. Carries the ⚠️ **two-seats trap** — two spreadsheets on the same 7-acre parcel produce ₹47.85 Cr/acre (purchaser seat) vs ₹128.97 Cr/acre (developer seat), so match the number to the ROLE the recipient is offered, or send bare facts only. Also: the sketch lives as an EMAIL attachment on `google-draas` (not in Drive), the spreadsheets live on `google-gmail`, the cap rate was instructed 9%→8% but never rebuilt, and the prior drafts are gone.

**Drafting from ANOTHER user's mailbox** (user says "reply to the email *I* received" — e.g.
> Prakash/psingh, Anbu/pm2.blr) → `references/cross-mailbox-thread-resolution.md`. The thread
> usually also exists in NDR's mailbox with DIFFERENT thread/message IDs; reusing NDR's ID under
> the user's env prefix 404s "Requested entity was not found". Set `HERMES_SESSION_USER_ID` FIRST,
> search INSIDE that mailbox, thread against the ORIGINAL inbound mail (not the user's forward of
> it), then re-check NDR's mailbox has no stray draft.

## CRITICAL: How to call google_workspace_manager

`google_workspace_manager` is a **registered Hermes tool** — call it via the tool API, exactly like `terminal` or `memory`.
**NEVER run it as a shell command. NEVER try to import it as a Python module.**

Call it like this (tool_use API):
```
tool: google_workspace_manager
input:
  command: "gmail messages list --params '{\"maxResults\":20,\"q\":\"is:unread after:2026/05/01\"}'"
  account_email: "ndr@draas.com"
```

If `google_workspace_manager` is not exposed in the current session's tool list, fall back to `terminal` with the Hermes venv Python:

```
# Example: create a threaded reply draft
cd /opt/hermes && /opt/hermes/.venv/bin/python3 -c "
from tools.gws_skill_bridge import call
result = call('draft_reply_create',
    service_name='google-draas',
    message_id='...',
    to='Recipient <email@example.com>',
    cc='Other <email2@example.com>',
    body='Email body text',
    html=False)
print(result)
"
```

**The bridge returns a JSON STRING, not a dict (hit 2026-09-23).** `call('draft_create', ...)` returns
`"{...}"` — calling `result.get('draft_id')` raises `AttributeError: 'str' object has no attribute
'get'` AFTER the draft was already created. Parse exactly once: `res = json.loads(raw) if isinstance(raw, str) else raw`,
then read `res['draft_id']`. The draft itself succeeded; only the verification code crashed — fix the
parse and verify with `drafts().get(id=..., format='full')` (top-level `payload.body.data` walker,
per the earlier pitfall), do NOT recreate the draft. Same wrapper behaviour is recorded for the
`whatsapp_link_tool` in personal-task-tracker.
```

**USER CORRECTION (2026-09-22, Behera appeal reply):** NDR explicitly said **"You need to use
the GWS client only to create draft and nothing else."** For counsel-legal threads he wants
drafts made ONLY through the raw GWS client path (`tools.gws_auth.build_service('gmail','v1',
service_name='google-draas')` + `drafts().create()` with a base64url EmailMessage and explicit
threadId), NOT through the bridge helpers — he could not see drafts produced via bridge calls
(the bridge reported phantom success and the drafts did not exist). When he reports "I can't
see the draft", the authoritative check is `users().drafts().list()` — a draft resource fetched
by ID can exist while the folder list shows nothing, and a returned draft-ID string alone proves
nothing. Also: the reply-all Cc set must be taken VERBATIM from the counterparty's latest
message headers (Suhail's address moved from e.suhail.ahmed@gmail.com to suhail@trialbase.in on
the 22-Sep-2026 counsel thread), and `drafts().get()` does NOT accept `metadataHeaders` (use
format='full' and parse payload headers).

**USER CORRECTION (2026-09-23, counsel review email — "too elaborate"):** When NDR asks for a
reply-all to counsel (Jasleen/VNV) containing feedback on their draft, the email MUST BE SHORT.
His words: "that email is too elaborate, there's way too much content… it's going to be too
overbearing for the lawyer to read." Rules that survived his correction:
- Lead with appreciation of their work; frame ALL feedback as suggestions, never "not filing
  ready" or questioning counsel's authority ("these are areas we could improve… your professional
  judgment comes first").
- Keep it to 3-5 tight sections max (confirmed facts / suggested refinements / stay+relief
  thoughts / asks) — bullets, no essay paragraphs, no repeated evidence-pack narration.
- Do NOT re-narrate every fact already shared; link the pack folder once and move on.
- Confirm facts BEFORE asserting them; if the record contradicts the desired framing (e.g. a
  condonation claim of total ignorance when RERA emails sit in NDR's own mailbox), flag it to NDR
  and frame the honest version — do not let a false premise reach counsel.
- End with specific asks (signatory name, IA drafts, their view on admission/stay/settlement).

**Emergency fallback for phantom-success drafts (2026-09-22):** when a previous run "created"
a draft that the user cannot see, and the bridge is suspect, create the draft fresh via raw
Gmail API with the verified message-ID/thread headers (see `references/threaded-reply-raw-api.md`),
then prove it exists by listing drafts AND reading the draft object back (threadId, To, Cc,
In-Reply-To) before reporting success.

The bridge mirrors these operations: `call("draft_create", service_name="google-draas", ...)`, `call("draft_reply_create", ...)`, `call("gmail_thread_get", ...)`, etc.

**IMPORTANT — do NOT use `execute_code` for gws bridge calls.** The `execute_code` sandbox lacks the `gws_fetch_token` import needed by the vault credential loader. Gmail/Calendar/Drive operations through the bridge will fail with `ImportError: cannot import name 'gws_fetch_token'`. Always use `terminal` with `/opt/hermes/.venv/bin/python3` instead.

**Pitfall — '&' in heredoc trips backgrounding guard (2026-09-01):** heredocs containing '&' are rejected as backgrounding; write the script to `/tmp/<name>.py` via write_file then run it instead. Same for contact-creation scripts.

**DRA family:** Manish=mdr@draas.com; Dharmesh=dharmesh.ranka@gmail.com. Wind-up map: `references/dra-family-windup-matters.md`
> **BRDPL / Godrej Venture / BNA recipient map + voice garble fixes (verified 2026-09-24):** `references/bux-ranka-godrej-venture-bna-2026-09.md` — "NSTL"→NSDL dues for Bux-Ranka Developers (BRDPL), "Bucks Ranka"→BRDPL, "Sharaaf"→Amit Saraf, "Saurav Vashisht"→Saurabh Vashishth, "Harish/Bhaksadi"→Haresh Buxani, "Balan Nambisar"→balan+nambisan architects. Forward shape NDR wants: To = Harsimran+Viraj, Cc = Amit/Saurabh/Tina (+ seller group Haresh/Pradeep/Vishwas). NSDL reminders land at ndr@drahomes.in. — carries the asset/owner map, the Step 1 (distribute work) / Step 2 (wind down) close-out framing, the June 2027 timeline, NDR's "too long — shorten it, use HTML bullets, group DRA Projects assets into one block" formatting corrections, the polite counter to Manish's "there is a process" pushback, and the **BCC-the-wives pattern** (Roshni `rnr@draas.com`, Shruti Chhajer `Shrutichhajer@gmail.com`).
**Draft updates that orphan the thread:** `references/gmail-draft-update-in-place.md` — re-pin via explicit `threadId` in the update body (no delete needed), plus the full `References` chain rule and the `/opt/hermes/.venv/bin/python` + `whatsapp_link_tool` invocation notes. **Also carries the body-text-only edit path** for "change one sentence in the draft you already made" (keep recipients/thread/attachments intact): decode each MIME part and edit the DECODED text (a raw-bytes regex matches only the un-encoded part and half-applies the edit — `text/plain` may be base64 while `text/html` is quoted-printable in the SAME message). **⚠️ THE ONE RULE TO CARRY IN YOUR HEAD: when editing a MIME part in place, strip ONLY `Content-Transfer-Encoding` — NEVER `del part['Content-Type']`.** `set_payload(txt, charset='utf-8')` already encodes exactly once (following it with `encode_quopri` double-encodes → `UnicodeDecodeError: ... byte 0xb4`), and deleting the `Content-Type` header silently DEMOTES an HTML part to `text/plain` so the recipient sees literal markup. **That second failure is the one NDR actually reports: "the email body consists of HTML code rather than rendered HTML. So something is wrong in the way the email was generated."** It is NOT a missing `html=True` flag (that is only the bridge/new-draft pitfall). Detect by printing each text part's declared type *next to* its opening characters (`text/plain | '<div style=...'` = the bug); recover by rebuilding from the pristine snapshot with explicit `MIMEText(txt, 'plain'|'html', 'utf-8')` subtypes, verifying the stored structure is `multipart/mixed → multipart/alternative → text/plain + text/html` with each attachment's decoded byte length unchanged, then deleting the broken draft (validated end-to-end 2026-09-25).
**Threaded-draft RE-CREATE must preserve Message-ID (confirmed 2026-09-24, Behera counsel reply):** when re-creating a threaded reply draft (delete + `drafts().create` after editing the body), a raw MIME rebuilt WITHOUT the original `Message-ID` header makes Gmail mint a BRAND-NEW threadId — the draft silently drops out of the master conversation (verified: draft landed in thread `1a0d201ef470c633` instead of master `1a07b0f2821d61f6`). Fix: snapshot the ORIGINAL draft's `Message-ID` + full `References` + `In-Reply-To` headers BEFORE editing, then re-create with those headers AND `threadId=<master>` in the create body; verify `drafts().get()` shows the master threadId, not a fresh one. Do NOT attempt in-place `drafts().update(raw=...)` for body edits on a threaded draft unless you re-supply Message-ID — the update path is what drops it.

**Sent-but-unthreaded diagnosis + resend-in-same-thread pattern (confirmed 2026-09-24, Behera counsel reply):** when NDR reports "I sent it but it was not a reply-all in the same thread", diagnose on the SENT copy BEFORE rebuilding: `messages().get(id=<sent id>, format='raw')` and check the raw headers `In-Reply-To` / `References`. If BOTH are None/absent, Gmail had nothing to nest under and minted a brand-new conversation — that is the whole root cause, not a recipient/account issue. **DO NOT edit the threaded draft in place via `drafts().update(raw=...)`** — verified 2026-09-24: the raw-MIME update ORPHANED the draft into a fresh thread (`1a0d30b499c79876`) even though it had been sitting in the master thread, and the resend-note insert failed because the HTML body inside raw MIME is base64-encoded (string-searching raw text for `<p>Dear M` finds nothing). The reliable sequence is: snapshot the draft's To/Cc/In-Reply-To/References headers + download the attachment bytes via `messages().attachments().get()`, DELETE the draft, re-CREATE with the full body (note prepended BEFORE base64), In-Reply-To = anchor's OWN Message-ID, and `threadId=<master>` explicitly in the create body — then VERIFY by `drafts().get()` that the stored threadId equals the master thread (`1a07b0f2821d61f6`) and the attachment part is present with the right size. Fix per NDR's explicit instruction in this case:
1. Rebuild as a proper reply-all: SAME To/Cc/subject/body/attachment as the sent email (extract the sent message's body+attachment from its raw MIME so the resend is byte-identical where it matters).
2. PREPEND a highlighted amber HTML box at the very top of the body — "Resending in the same thread… ignore the earlier copy of this email if you receive it separately" — because NDR wants all communication in one thread for tracking.
3. **CRITICAL: `In-Reply-To` must be the ANCHOR message's OWN Message-ID — never that message's `In-Reply-To` header value.** Verified: setting In-Reply-To to the anchor's In-Reply-To mints a fresh threadId (`1a0d2f1ddaeffbad`); setting it to the anchor's own Message-ID (the message inside the master thread you are replying to) threads correctly. `References` = anchor's full reference chain + anchor's Message-ID appended.
4. Gmail reports a PROVISIONAL threadId on `drafts().create` — it may differ from the master thread and is NOT the failure signal; the definitive thread is resolved at SEND time from In-Reply-To/References. Verify the stored draft's HEADERS instead, then re-list the master thread to confirm the anchor's Message-ID is in it.
**Anbu/Oasis:** anbarasan@draas.com in threads (headers win). Reply template: `templates/confirmation-request-email.html`.
**Millers Road office / India Chai building structural-test briefing (2026-09)** → `references/millers-road-structural-briefing-2026-09.md`
> **Millers Road office internal-measurements & flow-plan email for the BNA redesign (2026-09-29)** → `references/millers-road-measurements-bna-2026-09.md` — laser-distance quick measurements (floors/lobby/lift core/staircase/1-2 columns) → basic flow plan for the BNA (Balan+Nambisan) architects meeting 30-Sep; recipient map Anbu pm2.blr / Kantesh kanteshbg / Sinchana sgowda; "Makin" = UNRESOLVED, never guessed; Option B fallback (Google Earth floor plate); draft r1820996147210042155 verified; tracker P-354. — property identity (70/3 Millers Rd, DRA Realty lessee), the "Tushar Giri" = tushar patil identity, ACT NDT report results + Drive filing (DRA Realty > Millers Road Office Building > Architectural and Engineering, file id 1vUlNQBGOTmmDTuZ5XHl-g35dNO8lKFMV), the consultant-briefing email structure (context box → plan → results table → highlighted terrace ask), and the pymupdf technique for extracting building photo + facade render out of the Nova Spaces proposal PDF for attachments.
**DRA–Jiraaf thread / Vishwas follow-up letters WITH the land map (2026-09-21)** → `references/dra-jiraaf-followup-land-numbers-2026-09.md` — the corrected approved-layout survey table from Sinchana's map (SPL own ≈5.46 acc. NDR / 5.67 by row-sum; DRA Realty ≈0.90 with road-relinquishment point; JV ≈0.95; total 7.52), the no-percentage plot-area + built-up-area framing, the avoid-DRA→SPL-contribution-deed rationale, the 6 PM interim-loan call ask, the wrong-Vinay (Vinay T/VNV ≠ TN liaisoning Vinay) and unverifiable-Mark contact corrections, plus the RERA-villa comparable registration numbers (Sobha Gardenia 0043/2018, DAC trio, Palm Beach, Aurum, House of Palisade).
> **Voice garble + team map (re-verified 2026-09-29 while building the 5–7 PM transaction meeting):** "Giraffe Capital" / "Jiraffe Capital" (NDR voice) = **Jiraaf** — never a separate fund; Vineet Agrawal <vineet@jiraaf.com> is "Vineet". Ranka Oasis structuring/funding team (21-Sep call attendees, use these when NDR dictates the group): Vineet Agrawal <vineet@jiraaf.com>, Prakash M <prakash.m@jiraaf.com> (Jiraaf), **Nishant Prakash <nishantprakash1@gmail.com>** for the transaction group (NOTE: nishantprakash@theyelloweye.com is his OTHER identity from older Yellow Eye meetings — the gmail.com one is what the Oasis transaction calls use), Vishwas Rao <vishwas@vantagepointadvisors.in>. NDR's own name is Nishant Ranka — a dictated "Nishant" in this team means Nishant Prakash, not the user.
**Scheduling an EXTERNAL professional (Deloitte/tax advisor/consultant) into a brothers' thread** → `references/external-advisor-scheduling-thread.md` — the two-channel split (email to brothers only, WhatsApp to the outsider; never CC him), Amit Jain's verified contacts-sheet details, the "after 1 PM + 3–4 days' notice" constraint, the short HTML draft shape NDR accepted, and the delete-the-superseded-draft rule for each date shift.
> Bridge str-return + silent-counterparty anchor → references/gws-bridge-str-silent-followups.md

**To search Gmail or read thread content when `google_workspace_manager` is unavailable**, use the raw Gmail API directly from terminal:

```bash
cd /opt/hermes && /opt/hermes/.venv/bin/python3 << 'PYEOF'
from tools.gws_auth import build_service
import base64, json

service = build_service("gmail", "v1", service_name="google-draas")

# Search for messages
results = service.users().messages().list(
    userId='me', q='from:someone subject:"topic"', maxResults=5
).execute()

for msg in results.get('messages', []):
    detail = service.users().messages().get(
        userId='me', id=msg['id'], format='metadata',
        metadataHeaders=['Subject','From','To','Cc','Date']
    ).execute()
    h = {x['name']: x['value'] for x in detail['payload']['headers']}
    print(f"Thread: {detail['threadId']} | Subject: {h.get('Subject')}")

# Read full body from a message
full = service.users().messages().get(userId='me', id=MSG_ID, format='full').execute()
def get_body(payload):
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain' and part['body'].get('data'):
                return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
            if 'parts' in part:
                nested = get_body(part)
                if nested: return nested
        if payload['body'].get('data'):
            return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
    elif payload['body'].get('data'):
        return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
    return None
body = get_body(full['payload'])
PYEOF
```

Use `build_service` from `tools.gws_auth` (NOT the bridge's internal `_build_service`) when you need the full Gmail API surface — searching, reading bodies, fetching threads with format='full', etc. The `gws_skill_bridge` is sufficient for creating drafts once you have the thread context.

## Trigger Conditions

Activate when the user says anything like:
- "Email Raghu about the land valuation"
- "Reply to Nishant Prakash's email about Oasis"
- "Draft an email to Manohar about the project update"
- "Reply all to the email from Bhavesh"
- "File an insurance claim", "reimbursement claim", "pre-operative expenses" → load `references/insurance-claim-reimbursement.md` for the document inventory checklist and completeness validation
- **TPA asks for originals / portal re-submission** → load `references/irdai-original-documents-guidelines.md` for IRDAI regulatory citations to push back

**Default account:** `ndr@draas.com` for ALL emails unless the user explicitly says:
- "use my Gmail account" → `nishantranka@gmail.com`
- "use AHFL account" → `ndr@ahfl.in`

**Bharat's lead-outreach emails — sender is `sales1.blr@draas.com` (verified 2026-08-25):** When the session user is Bharat and the email targets a Ranka Udaya / portal lead customer (booking-amount account details, project handoff, title-deed follow-up), draft from his SALES mailbox, NOT ndr@draas.com. Verified recipe:

```bash
HERMES_SESSION_USER_ID=sales1_blr GWS_VAULT_SOCKET=/run/gws-vault/vault.sock \
  /opt/hermes/.venv/bin/python3 -c "
from tools.gws_skill_bridge import call
call('draft_create', service_name='google-draas',
     to='customer@example.com',
     subject='Ranka Udaya - Block 5 - ...',
     body='...', html=False)"
```

Lands as `BHARAT H <sales1.blr@draas.com>` with label DRAFT (verify via `drafts().list()`). Before drafting to a lead, look the customer up in **Kelsa Pipeline 10 by name** to address them properly — e.g. `roy.pragyajyoti@gmail.com` resolved to Pragya Joythi (lead #54917102, Hot, assigned to Bharat). The customer's email supplied by the user in chat is authoritative for `To`; create as DRAFT only, never auto-send.

**Bharat → NDR internal approval drafts (Ranka Amber SSA / landowner-doc workflow, validated 2026-09-22):** When Bharat says "draft an email stating the draft is ready and require your approval so I can upload tomorrow" in a project-document context, the recipient is **Nishant Ranka <ndr@draas.com>** (internal approval), NOT the landowners/counterparty — even though the document is ultimately for them. After drafting, the user may correct with "I am not sending this to Iyer" → recipient is internal. Thread it on the existing project thread (e.g. NDR's "Executed Supplementary Sharing Agreement (Signed PDF + Source Document)" message) via `draft_reply_create(message_id=...)`, sender stays `sales1.blr@draas.com` (Bharat's mailbox, same env as the lead-outreach recipe: `HERMES_SESSION_USER_ID=sales1_blr GWS_VAULT_SOCKET=/run/gws-vault/vault.sock`). Confirm the mailbox with `getProfile()` before creating, then verify `drafts().get()` shows labels `['DRAFT']` + correct From/To/threadId.

**"i need only draft now" / "just the draft" (user preference):** When Bharat (or NDR) says this, skip the analysis/verification narrative entirely — create the Gmail draft immediately from the verified thread context already at hand, then confirm in one short line (To / Subject / threaded-on-what / Drafts folder). Long pre-draft recaps of what was cross-checked are not wanted at this stage.

**Plot booking-confirmation emails (congrats + booking-amount receipt + legal pack + receipt-follows-EOD):** full proven workflow in `references/plot-booking-confirmation-email.md` — pull the booking amount from the Kelsa lead notes (never invent it), attach the compressed legal pack via raw Gmail API (bridge has no attachment support), draft-only from the sales mailbox.

**Booking-confirmation / congrats emails — quote the EXACT amount from the lead's notes (2026-08-25):** When Bharat dictates a congrats email ("congratulate him for booking, confirm receipt of the amount transferred"), the authoritative figure is the Kelsa lead's notes, NOT memory or the voice dictation: `list_lead_notes(pipeline_id=10, lead_id=...)` — Bharat logs settlements like "The client has transferred the ₹50,000 booking amount, and the same has been received and confirmed with the client." Quote that exact figure in the draft ("We confirm the receipt of your booking amount of ₹50,000"). If the notes contain no received amount, ask Bharat — never guess or leave a placeholder. Booking-congrats emails follow the dictated structure: congratulations → amount-received confirmation → attached legal docs (per commitment) → receipt follows by stated EOD.

**NDR uses ONLY ndr@draas.com as his work account (user-stated 2026-08-17).** If the user asks to find a work email and it's not in draas.com, do NOT keep re-attempting ahfl.in or ask them to re-auth it — he does not use ndr@ahfl.in for work at all. Skip it.

**"Did I email X about Y?" search pattern (validated 2026-08-17, Siddhapura/Glenmore case):**
1. Search draas.com FIRST with ALL spelling variants (people search on "Siddhapura" returned 0 while "Siddapura" returned hits; place names garble in memory/voice).
2. Check sent, inbox AND drafts (a draft is a legit "I wrote it" artifact).
3. If it's genuinely absent, say so plainly and pivot — do NOT just stop. Check the other stores where the context may live (Drive fullText, gbrain, Kelsa pipelines, session history), collect the artifacts, and offer to draft the follow-up email (To the colleague, CC the user-named others) since the assignment was clearly made verbally.
4. Verify every recipient via find_contact.py before drafting.

**Sender must match recipient family — AND topic:** Family emails split on content, not just relationship:
- **Personal/family matters** (kids' school, health, social) → use `nishantranka@gmail.com` (confirmed 2026-07-13)
- **DRA business / company / legal matters** (succession petitions, shareholder agreements, family settlements, court filings) → use `ndr@draas.com` even when the recipient is extended family (brother-in-law, sister, sister-in-law). The content is DRA-governance, not personal, so the work account is correct (validated 2026-08-27, Ranjeeth Rathod/Mamata Rathod/Roshni Ranka succession petition email)
- **When ambiguous:** confirm with the user before drafting.

**DRA-group / director work always uses `ndr@draas.com`** (user correction 2026-07-13: "No use ndr@draas.com for all work"): even if the user is a director of a DRA-group entity (DRA Aadithya / DRAAPPL, DRA Aadithya South City / DRAAS, Truliv, DRA Homes, etc.) AND is reading the notice in personal Gmail, the draft + Drive filing + Calendar event must all go in `ndr@draas.com`. The personal Gmail is just the inbox where the notice arrived; the work artefacts live in the work account. Confirm work account explicitly when the user says "file this" or "draft this" about a director-of notice and the inbox is personal Gmail.

- **IRDAI original-documents pushback:** When drafting an insurance-related reply where the TPA/insurer demands original medical records or portal re-submission, load `references/irdai-original-documents-guidelines.md` for the specific Master Circular citations and rebuttal language. Add these citations to the email body as a factual paragraph before the escalation ask.
- **Drug-alternative request to doctors** (drug out of stock, need alternative generic recommendation) OR **Charitra consult-prep / post-review thank-you replies** (send Sameer+Annie updates, prep Q&A after a PET, thank both after a review): load `references/charitra-medical-communication.md` for the correct account (google-draas, NOT ahfl), doctor emails (samdoc_mamc@yahoo.com for Dr. Sameer, anniekbaa@gmail.com for Dr. Annie), Charitra's ACTIVE email (charitrakamath@gmail.com — murjanicharitra@gmail.com is obsolete), the active thread + its branches, the drug-alternative email pattern, the post-review thank-you shape (short + nice, anchor on latest INBOUND message), the plain-English consult Q&A style, the PET necrosis-vs-active differentiation research, the future-therapy shortlist (B7-H3 ADC/CAR-T, utidelone+anlotinib), and the HTML→Drive TMP public-link delivery. Verify every recipient from thread headers — never guess medical email addresses.

- **DEFER** to the `regulatory-complaint-escalation` skill if the email is:
- A formal complaint to a regulated counterparty (insurer, bank, NBFC, telco)
- Escalating to a GRO / Principal Nodal Officer / Internal Ombudsman
- Citing IRDAI / RBI / TRAI / SEBI regulations with a regulator-portal threat
- Part of a multi-month dispute where normal channels have been exhausted
The complaint skill produces a structured 5-section email + a Google Doc plan, which is wrong shape for the normal business reply path here.

## 2. Stage 1 — Context Gathering

### For a NEW email

**ALWAYS use the contact lookup tool FIRST — HARD RULE (corrected 2026-07-31).** Any email-address lookup starts with `scripts/find_contact.py` from the `messaging-drafts` → `references/personal-messaging-workflow.md`, in the same session, immediately:

```bash
/opt/hermes/.venv/bin/python "/data/hermes/skills/communication/messaging-drafts/scripts/find_contact.py" "<name or number>"
```

The script queries Google People API AND both DRA contacts sheets in one shot (with automatic vault-socket recovery) and prints labelled phones + emails. If it finds the person, confirm the address with the user before drafting (matches may be ambiguous or outdated). Never guess an email address, never rely on memory alone, never fall back to a raw sheet read.

Only if the tool returns nothing: escalate down the chain (memory spelling cross-check → Gmail thread search → ask the user). Full priority chain lives in the `messaging-drafts` → `references/personal-messaging-workflow.md`.

When a match is found, confirm with user:
> Found: **Raghu Iyer** — Director, [Company]
> Drafting to raghu@example.com (work). Say if you want a different address.

When no match: report clearly and ask for the address — do not invent one.

**Pitfall — contact-sheet address can be stale/dead (2026-08-12):** find_contact.py may return an old address that the person no longer uses (e.g. Anbu/Anbarasan → sheet says anbarasandraass@gmail.com, last used 2017 and defunct; his ACTIVE address is anbarasan@draas.com from Jul–Aug 2026 threads). Before drafting, cross-check the person's most recently used address by searching Gmail (`q=<name>` via `build_service('gmail', 'v1')`, inspect latest To/From). A dead address also fails Drive permission grants with HttpError 400 `invalidSharingRequest` — same fix applies. When you use a non-sheet address, tell the user why (e.g. "the gmail.com address in the sheet is stale — using the active draas.com one").

**HARD RULE — recipient verification before ANY outbound address (user-stated 2026-09-28, after the AJ Architects bounce):** NDR's rule, verbatim in substance: *every* recipient address must be (a) looked up, (b) present in contacts, AND (c) previously used in a sent email that did NOT bounce. If an address bounces, you must go back to the contact lookup and take the address from there — never re-use the bounced one, and never hand-derive a plausible-looking variant. A 550/`Address not found` from Gmail is treated as a dead contact locally; substitute the verified one and TELL the user you swapped it and why.

**AJ Architects (Arvind Jain) — verified addresses, one DEAD (confirmed 2026-09-28):**
- **Arvind Jain** — `arch_arvind2000@yahoo.co.in` (contacts-sheet row 417, "Arvind Jain Architect"; also arch.arvind2000@gmail.com on the row). Live: his own replies come from it.
- **Shrinidhi GV** — `shrinidhi.a.j.architects@gmail.com` (contacts-sheet row 4249, org AJ Architects; she is the one who sends sanction files — 21-Sep "ranka northstar @sanction"). THIS is the correct AJ Architects team address.
- ⚠️ **`arvindsarchitects@gmail.com` is DEAD** — Gmail hard-bounced it 550 5.1.1 "account that you tried to reach does not exist" against the 28-Sep area-statement mail. It is not in the contacts sheet. It was previously attached to a person addressed as **"Sridevi"** — and there IS a Sridevi row (3519, SysN Consultant, +91 99801 20060) with **no email at all**, which is where the invented address came from. Do not put "Sridevi"/arvindsarchitects@gmail.com on any AJ Architects mail; use Shrinidhi GV. If NDR's dictation says "his Sinchana or Srin", it means **Shrinidhi** (transcribed also as "Sriniti"/"Shinchen").

**Pitfall — in-house staff and long-standing partners have NO contacts-sheet row at all (2026-08-25):** find_contact.py (and contact_resolver) return zero matches for internal DRAAS people and frequent external partners — e.g. Sinchana Gowda (sgowda@draas.com, in-house architect/design covering Palya, Serenity Hill View, Oasis) and Arvind Jain (arch_arvind2000@yahoo.co.in, A.J. Architects). These people don't live in the contacts sheet, so "tool returns nothing" must NOT stop the draft. Resolve them from Gmail: `q='"Full Name" OR from:<domain>'` and read From/To/Cc headers across recent threads (`q='sinchana'`, `q='from:arch_arvind2000'`). Headers are the authoritative address for both the To field and Drive permission grants. Only after a Gmail search also fails should you ask the user for the address.

**Core DRAAS land-project team — canonical resolution (validated 2026-08-30 via Gmail headers):**
- **Prakash Singh** — psingh@draas.com (in-house land/project lead; assignee on Land Proposal pipeline). NOT the same as "Prakash (Asset Properties)" (external broker, no sheet row) nor "Nishant Prakash" (external consultant) — voice dictation "Prakash" alone is ambiguous, resolve by thread context.
- **Rahul = Vinod Kumar Das** — vkdas@draas.com, +91 99000 93813 (contacts sheet row 4091). Feature in Gmail as `Vinod Kumar Das (Rahul)`; user's voice says just "Rahul". Phone extraction trick: contacts sheet has 3 numbers for him — cross-check which is WhatsApp-active before generating links, else the link opens the wrong chat.
- **Bhavesh = bhavesh@draas.com** (user says "Bhavesh Bafna" or "Bhavesh Mehta" interchangeably — same person; DRA account has no vault identity, DWD-only; personal Gmail bvbafna@gmail.com seen in NDR sent mail). Handles AHFL / Ranka Stelo approvals docs (2026-09-22).
- **Anbu = Anbarasan** — TWO live identities: `anbarasan@draas.com` (his From/identity since Sep-2026 threads) AND `pm2.blr@draas.com` (what NDR targets in To/Cc — Aug-2026 Millers Road facade thread Cc, Sep-2026 "Please sign out" To). The sheet's anbarasandraass@gmail.com is defunct. Use `pm2.blr@draas.com` when NDR says "mark Anbu" in a Cc. +91 81500 29900. Voice-garble aliases (verified 2026-09-22, Ranka Stelo approvals handover): "Anbar Hassan" and "Anudha" (= "Anbu da") both resolve to Anbarasan.
- **Anbu = Anbarasan** — anbarasan@draas.com (ACTIVE; the sheet's anbarasandraass@gmail.com is defunct per 2026-08 pitfall), +91 81500 29900.
- **Neha = nvaddadi@draas.com (2026-09-22):** DRA in-house creative/render person ("Neha help" on renders, AI content videos). No contacts-sheet row; verified via Gmail headers on the "Request: 2 short videos on how we use AI for content" thread (To: Gowri Singh + Neha). NOT "Neha Ryan"/"Neha Kiran MSD"/"Neha Kewalramani" (external, different people).
- **Gowri = Gowri Singh <gsingh@draas.com> (2026-09-22):** DRA marketing. Contacts sheet has gowrisingh1341@gmail.com (work) / gowrisingh72@yahoo.com (other); GMAIL HEADERS use gsingh@draas.com — use the header-verified draas.com address when CC'ing. NOT "Gowri Deepak" or "G Raghunath".
When NDR names just first names ("email Prakash marking Rahul and Anbu"), resolve all three via Gmail headers before drafting — the sheet rows are ambiguous (51 Prakashs, 19 Rahuls).

**AI / content / render team — canonical resolution (validated 2026-09-22 via Gmail headers):**
- **Neha** — `nvaddadi@draas.com` (renders, AI content videos; NDR addresses her simply as "Neha"). The contacts sheet has NO matching Neha with an email (Neha Kewalramani Federal Bank, Neha Kiran MSD, Neha Ryan are phone-only, unrelated). Pair with Gowri for render/AI-content workstreams.
- **Gowri = Gowri Singh** — `gsingh@draas.com` (marketing/content; the work identity seen in headers). The contacts-sheet "Gowri Singh" row (row 1156) lists gowrisingh1341@gmail.com (Work) and gowrisingh72@yahoo.com — those are NOT the draas.com work address; headers win.
- **Sinchana (in-house architect/design)** — `sgowda@draas.com` (also covers Palya, Serenity Hill View, Oasis; her attendance/sign-in mails land at sinchanalgowda16@gmail.com but work identity is sgowda@draas.com from her own sent headers).
- **Kantesh = Kantesh BG** — `kanteshbg@draas.com` (internal DRA; appears with Anbu + Bharat in "call out the overdue feedback" dictations). Resolve from NDR's OWN sent Kelsa attendance mails — search `q='kantesh'` in google-draas, subject "Please sign in for the day" (verified 2026-09-24). Contacts sheet row 1554 ("Kantesh Dra") has phone +91 95918 01389 but NO email; `contact_resolver` returns nothing usable. ⚠️ **`kantesh@draas.com` is DEAD — hard bounce (Mail Delivery Subsystem failure notice, 28-Sep-2026).** Prakash addresses Kantej as `kantesh@draas.com` (28-Sep estimate-cost mail), so the original AND NDR's reply-all never reached Kantej (worked 2026-09-29: forward rebuilt to `kanteshbg@draas.com` and the swap stated in the draft). Always swap to `kanteshbg@draas.com` and tell the user you swapped it, never reuse the bounced variant.

**Structural consultant & Millers Road refurbishment circle (validated 2026-09-21 via Gmail headers + certificate OCR):**
- **"Tushar Giri" (NDR dictation) = Tushar Patil** <structure.pune@gmail.com> — Designage Consultants Pvt Ltd, Office 403, 4th Fl Wing-E, Audumber Apartment, Canal Road, Warje, Pune-411062; M.Tech Civil (Structures), Chartered Engineer M-145727-0, +91 90110 76607. There is NO person named "Tushar Giri" in his ecosystem — the dictation is a permanent garble (confirmed June 2026 + Sep 2026). He signs the Ranka Amber structural stability certificates (Drive `20260702_RankaAmber_Structural_Stability_Certificate_Signed_TusharGiri.pdf`).
- **"Amir" in the Millers Road context = Aamir Khan** <khan.hussain.aamir@gmail.com> (also aamirkhan@icloud.com) — broker/conduit on the Millers Road lease and the Nova Spaces façade proposal. Not to be confused with anyone else named Aamir.
- **"Anwar Asan" = voice error for "Anbu Asan" = Anbarasan** (pm2.blr@draas.com / anbarasan@draas.com). Do NOT invent an "Anwar" contact unless NDR explicitly names a firm.
- Millers Road building (70/3 Miller's Road, Vasanth Nagar, leased by DRA Realty): lessors M.Akber Hussain 30%, M.Akther Hussain 30%, Mrs M.Asma Banu 20%, Mrs M.Sara Banu 20% (Kilpauk, Chennai); GPA holder Avoor Mohammed Muzzammil; AH India contacts Atheeq <padirector@ahindia.com>, Akber <akber@ahindia.com>.
- Full thread context, NDT numbers, draft id and attachment sourcing: `references/millers-road-office-structural-review-2026-09.md`.

**Advisors & professionals — canonical resolution (verified 2026-09-22 via contacts sheet + People API):**
- **"Nitin Oswal" / "Nitin Ostwal" / "Nitin Osthwal" (NDR voice) = Nitin Osthwal Kalpesh** — "Nitin CA", DRA's official Chartered Accountant, partner/cousin of Saurabh T Jain (TSJ & Associates LLP); +91 80889 79777, Kalpeshnitin@gmail.com, firm ph 080-26705995. Contacts sheet row 2333 (canonical "Nitin Kalpesh", nickname "Nitin CA"); People API shows full name "Nitin Osthwal Kalpesh". His outreach channel is WhatsApp; use him for company + personal financial matters. NOT "Nitin Agarwal" (SNN Builders) or "Niti Narang" (Vikram Hospital).

**Phone-from-signature technique (2026-08-30):** When a team member has no phone in the contacts sheet, regex their OWN sent emails' signature blocks for `(\+?91[\s\-]?[6-9]\d{4}[\s\-]?\d{5}|\b[6-9]\d{9}\b)` over the last 5-8 messages from their address — Prakash Singh's mobile +91 81230 28716 was recovered this way from his signature. Always sanity-check the recovered number against the pattern (valid 10-digit Indian mobile) before putting it in a WhatsApp link.

### For a REPLY (existing thread)

**Pitfall — the email to reply to is a FORWARD where the original thread is NOT in the user's mailbox (2026-08-27, Ruhaan school absence):** When a family member (Roshini) forwards a school/third-party email to NDR saying "please handle this," the original thread (teacher ↔ Roshini) is NOT in NDR's mailbox — only the single forwarded message exists there. This breaks the standard reply flow:

1. There is no threadId to use for a threaded reply from NDR's account. `draft_reply_create` cannot work because the source message_id belongs to Roshini's mailbox.
2. Recipients must be extracted from the forwarded body text — parse the "Begin forwarded message" block for From, To, Cc, and Subject. Do NOT search for a thread that doesn't exist.
3. The draft must be composed as a FRESH message (`draft_create`, not `draft_reply_create`), with:
   - To = the teacher (From of the forwarded original)
   - Cc = original Cc (minus self), plus the family member who forwarded it (Roshini), plus any additional people the user names (e.g. Ruhaan, Joel)
   - Bcc = any person the user wants blind-copied
   - Subject = `Re: <original subject from the forwarded block>`
4. There is no In-Reply-To/References to set — this is a new message on a new thread. The subject prefix `Re:` is a courtesy convention only.
5. Verify recipients with the user: the person who forwarded the email often knows who should be included that isn't obvious from the forwarded headers (e.g. Joel the administrator, Ruhaan himself).
6. When the user names an ADDITIONAL recipient who is NOT visible in the forwarded headers (e.g. "BCC Joel, he's also an administrator"), check the contacts sheet first via contact_resolver. If the contact exists but has NO email (common — phone-only entries), search Gmail history across ALL accounts (google-draas, google-ahfl, google-gmail) for the person's name. An old CC'd email from years ago is often the only source for their email. Pattern that worked (Aug 2026, Joel Kribairaj at Aditi school):
    - contact_resolver confirmed the contact existed: phone +919972072401, no email
    - Gmail search across all accounts with query `q="joel" AND (gsuite.aditi.edu OR aditi.edu.in)` and `q="kiribayaraj OR kiruba"`
    - Found in a March 2024 Science Club email from Jayashree KG: `Joel Kribairaj <joel@aditi.edu.in>` (CC'd)
    - Update the contacts sheet row with the found email via contact_learner/update
    - Use it for BCC in the draft
    - If Gmail search across all accounts returns nothing, THEN ask the user.
7. Before finalising, verify the person's email via a real message header (not the contacts sheet value) — confirm the spelling matches the mailbox (Kribairaj vs Kiribayaraj). The user's voice transcription of the name may differ from the sheet's canonical form.

Search pattern for forwarded emails: the user says "there's an email from Ruhan's school" or "reply to this email." Check if it arrived as a forwarded message (Subject: "Fwd: ...") or a direct email. If forwarded, the original may be in another mailbox — work with the forwarded body, don't hunt for a non-existent thread.

**Pitfall — reply to the ORIGINAL email, not your forward of it (2026-08-27, Viraj Godrej 4.5 FAR Drawings):** When the user receives an email from someone, forwards a copy of it to a colleague for action, and then later asks you to "reply all" to the original sender — there are now TWO copies of that email in the mailbox: the original thread and the user's forward. The user will explicitly clarify: "reply all to the email received from [sender], not the one I forwarded to [person]."

1. Search by the **sender's email address**, not by subject keywords — the forward has the same subject but different sender (the user themself).
2. Verify the thread belongs to the original sender: check the `From` header of the first message in the thread.
3. If the user mentions having forwarded the email to someone earlier in the same session, proactively ask which thread they mean before drafting — "The original from Viraj, or the copy you forwarded to Anbu?" This saves a correction.
4. When the user says "reply all to the email received from [name]" — that unambiguously means the original thread, not the forward.

**Related: SharePoint/OneDrive link access problem (2026-08-27, Viraj Godrej 4.5 FAR Drawings):** When the user can't download a file from a SharePoint/OneDrive link (requires Microsoft sign-in, and the user's domain — e.g. ndra.dra.homes.in — has no Microsoft login), the email draft will:
- State the problem factually: "I tried signing in but [domain] is just a domain name — I don't have a Microsoft login there."
- Request the file via an alternative: "please attach the file and send it, or share it via VTransfer / any other modality from which I can download it."
- Explain what it's for: "I will share it with the [team] so they can give us a quotation to get the process started."
- Add urgency: "This is a bit urgent… we have already lost enough time on this matter."

**Pitfall — the counterparty's reply may NOT be in the thread you sent from (confirmed 2026-08-13, Sanjay Sethia succession-certificate matter):** the recipient replied from the same domain (sanjay@lawsquare.in → nishanth@lawsquare.in) but as a NEW email — different subject ("Succession Certificate (Dinesh Ranka)" vs our "Priority: … Review of Draft Petition"), `In-Reply-To: None`, its own thread ID. Listing only the sent thread misses their response entirely. When the user says "we got a reply", search the whole account by domain/name (`q='lawsquare.in newer_than:7d'`) across ALL vault accounts (google-draas, google-gmail, google-ahfl) before concluding. Draft the reply against the counterparty's message ID so it threads on THEIR side.

**Pitfall — email body attachment lists can lie (2026-08-13):** the sent email claimed "Attached: petition, death certificate, family tree, Aadhaar cards" but only 3 files were actually attached (Aadhaar was listed, never sent). When the user asks "did the attachment actually go?", do not trust the body — walk the message payload via `messages().get(format='full')` and enumerate `payload.parts[].filename` for each message in the thread.

Search Gmail for the thread using the `google_workspace_manager` tool:
```
tool: google_workspace_manager
input:
  command: "gmail messages list --params '{\"maxResults\":5,\"q\":\"from:raghu subject:land valuation\"}'"
  account_email: "ndr@draas.com"
```

**Pitfall — `q='threadId:...'` silently returns nothing (confirmed 2026-08-01):** the Gmail API
`messages().list(q='threadId:XXX')` search operator is unreliable — it returns an EMPTY result even
when the thread exists. Do not use it to locate thread messages. Instead fetch the thread directly:

```python
from tools.gws_auth import build_service
service = build_service("gmail", "v1", service_name="google-draas")
t = service.users().threads().get(userId='me', id=THREAD_ID, format='metadata').execute()
for msg in t['messages']:
    h = {x['name']: x['value'] for x in msg['payload']['headers']}
    print(msg['id'], h.get('Date'), h.get('From'))
```

This also gives you the message IDs needed for `draft_reply_create(message_id=...)` when the last
message in the thread is from the user themselves (see "Threaded reply — last message is from you").

Then fetch the full thread:
```
tool: google_workspace_manager
input:
  command: "gmail threads get --id THREAD_ID"
  account_email: "ndr@draas.com"
```

Extract from thread: `threadId`, sender, all To/CC participants, subject line.

Present context:
> Found thread: **"Land Valuation — Allalsandra Survey"**
> Last message: from Raghu Iyer on [date]
> Participants: Raghu Iyer, Nishant Ranka, CC: Bhavesh Bafna
> Drafting a reply. Reply-all? (yes/no)

## 3. Stage 2 — Draft

**Full detail: `references/stage2-draft-detail.md`** — read it before building any draft. It carries the
body-construction rules, HTML/formatting preferences (`references/ndr-email-html-preferences.md`,
`references/ndr-email-format-preferences.md`), attachment handling, threading via MIME headers
(`references/threaded-reply-raw-api.md`), and the worked per-case examples.

Core invariants that stay inline:
- A hand-made `payload` 400s with "Missing draft message" — always build `EmailMessage` → base64url `raw`.
- `POST /gmail/v1/users/me/messages.send` returns 404 on this account — always use the draft workflow.
- `threadId` in the draft payload body: **bridge** `draft_create` rejects it — thread via MIME headers there. The **raw API** `drafts().create(userId='me', body={'message': {'raw': ..., 'threadId': THREAD}})` DOES accept it, and passing it is the reliable way to pin a reply draft to the master thread (re-confirmed 2026-09-25: In-Reply-To + References + explicit `threadId` → the draft stored under the master threadId, no orphan). Use both belts: correct In-Reply-To/References headers AND explicit threadId.
- Only `messages.send` is blocked; the draft workflow is the only path.
- Resolve invitees/CC via `contact_resolver`; never hand-type an @draas.com address.

## 5. Rules

- **HTML reply with attachment, on the same thread:** use `references/html-reply-with-attachments-threaded.md` + `templates/html-reply-with-attachment.py` (hand-built raw MIME: multipart/mixed → multipart/alternative (text/plain + text/html) → base64 image; In-Reply-To/References + threadId; verify with drafts().get(format='full')). The bundled create-draft-with-attachments.py script is text/plain only — for NDR's styled HTML follow-ups with an attached image, use the template. Verified Oasis 50-cents critical reply, 2026-09-29.
- Use `tools.gws_auth.build_service("gmail", "v1", service_name=...)` directly (see Pitfalls for account resolution)
- Build a `MIMEMultipart("mixed")` message with MIMEText (body) + MIMEBase (attachments)
- Call `service.users().drafts().create()` with the full MIME message
- **Template:** `templates/draft-with-attachments.py` in this skill — copy and configure
**Full workflow reference:** `references/drive-to-draft-pipeline.md` — covers the entire search → download → organize → build MIME → create draft pipeline, including deduplication and completeness checking.

**Statutory/corporate form filling (MGT-11 proxy, consent forms, "fill the proxy form", "prepare a form for signature"):** `references/statutory-form-pdf-pipeline.md` — download the template from Gmail, read legacy .doc via `strings` (no LibreOffice on VPS), extract member/shareholder data from the notice PDF, rebuild the filled form as a typed PDF with reportlab, attach to a forward draft via raw Gmail API. Validated 2026-08-13 (DRA Aadithya AGM proxy for Roshni).

**Architect/consultant fee proposal comparison (2026-08-21):** `references/fee-proposal-comparison.md` — workflow for comparing a new/revised fee proposal against the original work order. Covers: finding the latest email, extracting the signed WO PDF from email attachments, comparing rates AND basis (SBUA vs Built-Up Area), identifying double-dipping, assessing justification, recommending a negotiation position.

**Architect fee COUNTER-OFFER reply (2026-09-05):** `references/architect-fee-counter-offer-reply.md` — rejecting a fresh per-sft quote: benchmark vs the architect's own prior harder-scope rate, argue complexity contrast, propose lump sum, anchor deliverables, invite counter. Validated Arvind Jain, Palya row houses.
- **Google Doc text cleanup (force all text black) before attaching:** `references/google-doc-text-cleanup.md` — Docs API pitfalls: `rgbColor {}` = black not white (default-channel check), `suggestionsViewMode` allowed values, batchUpdate to black, PDF-render verification, confirming the right doc via `lastModifyingUser`

**Threaded reply-all via raw Gmail API** — When the bridge or `google_workspace_manager` tool can't deliver a properly threaded reply-all draft, use the raw API. Full recipe in `references/threaded-reply-raw-api.md`. Key steps: get the most recent message in the thread via `format="raw"`, extract `Message-ID` + `References` headers, build an `EmailMessage` with `In-Reply-To` and `References` set correctly (both required for Gmail nesting), then create the draft via `service.users().drafts().create()`. Always clean header values with `re.sub(r'\s+', ' ', val).strip()` to remove newlines before passing to `EmailMessage.__setitem__`.

**Pitfall — `threads().get()` does NOT accept `format='raw'` (confirmed 2026-08-02):** `service.users().threads().get(userId='me', id=THREAD_ID, format='raw')` raises `TypeError: Parameter "format" value "raw" is not an allowed value in "['full', 'metadata', 'minimal']"`. To read the latest message's raw MIME for Message-ID/References: fetch the thread with `format='metadata'`, take `t['messages'][-1]['id']`, then `messages().get(userId='me', id=that_id, format='raw')` — the raw fetch must go through the **messages** endpoint, not the thread endpoint. Only after this can you build In-Reply-To/References and call `drafts().create()` with `threadId`.

**Pitfall — `drafts().create()` with malformed References can produce SENT label instead of DRAFT (confirmed 2026-07-30):**
When creating a threaded reply draft via the raw Gmail API (`service.users().drafts().create()`), the message can end up with label `['SENT']` instead of `['DRAFT']` if the `References` header contains non-Message-ID content. This happens when extracting References from raw email with an overly broad regex that captures DKIM signature content. The API interprets the malformed header as a send instruction.

Fix: always sanitize References to contain only Message-IDs wrapped in angle brackets, then verify the draft has `DRAFT` label:
```python
import re
clean_refs = ' '.join(re.findall(r'<[^>]+>', existing_refs)) if existing_refs else ''
```
Verify immediately: `verify['message'].get('labelIds', [])` must contain `'DRAFT'`. If SENT, delete and recreate with clean References.

**Extended finding (2026-08-10): even a clean, immediately-verified draft can end up SENT.** Two drafts created this way (MOU reply + insurance reply) each showed `labels: ['DRAFT']` right after creation — sanitized References, verified via `drafts().get()` — yet when the user later said "I can't see the draft", `drafts().list()` returned neither of them and a subject search found both in **SENT** with real Message-IDs and In-Reply-To headers. Causes: (a) the user sent them from the Gmail UI before checking, or (b) delayed auto-send behaviour. Either way the lesson is the same:

**The authoritative "is it still a draft" check is `users().drafts().list()` — NOT the label on a draft resource fetched by ID.** A `drafts().get(id=...)` can return `labelIds: ['DRAFT']` at creation and that same resource can later read back as SENT.

**Diagnostic when user reports a missing draft:**
1. `users().drafts().list(userId='me')` — if the draft isn't there, it is NOT in Drafts, full stop.
2. Search the whole account by subject (`q='subject:"<keyword>"'`) and inspect `labelIds`. If the message is `['SENT']`, the email went out — tell the user it was sent (give time, To, Cc) and do NOT recreate a fresh draft (that creates a duplicate the user then sends twice).
3. Report plainly: "It's not in Drafts — it's in Sent (sent at HH:MM)." This matches the user's own recollection in most cases ("I might hv sent it").

**Pitfall — a presentation PDF can blow Gmail's 25 MB cap once base64-encoded (confirmed 2026-09-25, Ranka Udaya site office).** Gmail's limit applies to the ENCODED message, and base64 inflates bytes by ~33%. A 18,840,915-byte PDF (= ~25.1 MB encoded) sits right on the line, and the user's own chat upload of it failed outright at a 20 MB cap. Compress with ghostscript before attaching — `/ebook` is far too aggressive (18.8 MB → 735 KB, visibly soft on renders), so use a targeted 150-dpi JPEG down-sample instead:

```bash
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dNOPAUSE -dQUIET -dBATCH \
   -dDownsampleColorImages=true -dColorImageResolution=150 \
   -dColorImageDownsampleType=/Bicubic -dJPEGQ=88 \
   -dAutoFilterColorImages=false -dColorImageFilter=/DCTEncode \
   -sOutputFile=out.pdf in.pdf
```
Result on this deck: **18.84 MB → 2.59 MB**, page count and A4 landscape geometry preserved, renders still crisp at 150 dpi. (`gs -dPDFSETTINGS=/printer` gives 3.03 MB; `/ebook` gives 735 KB.) Always state in your reply to NDR that the attached copy is a compressed derivative and offer to swap in the original if he wants full fidelity.

**When the user's own attachment never reached you, say so and substitute explicitly.** NDR tried to send the "updated render and plan" through chat and hit *"The document is too large or its size could not be verified. Maximum: 20 MB."* — so the file he meant to hand over was never available. Rather than silently attaching something else, pull the same-named file from the thread (`RankaUdaya_Site_office.pdf`, Gowri's 24-Sep message) and flag it: "your upload didn't come through; I attached Gowri's copy — if your updated file differs, send it and I'll swap". Also note the pre-existing files already staged in the working dir from earlier sessions (AR Brochure 2025.pdf, Quote for marketing office.pdf, Ranka_Udaya_Interior_Execution_Specification.pdf) before re-deriving anything.

**Pitfall — `html=True` is required for HTML drafts (user correction 2026-07-13):**
The `gws_skill_bridge.draft_create` operation defaults to `MIMEText(body, "plain")` if you don't pass `html=True`. The result: the recipient sees your entire HTML document (`<!DOCTYPE html>`, `<table>`, `<tr>`, `<td>`) as raw literal text in the email body. The user said: "you have taken HTML code and stuck it into the email body rather than creating the email body itself as an HTML using HTML so it looks like a nice rendered email." Fix: pass `html=True` to the bridge call, OR if using `google_workspace_manager`, use the `--bodyHtml` flag (not `--body`) for HTML content. Always verify a draft by calling `draft_get` and inspecting the MIME structure (`payload.mimeType` should be `text/html`) before declaring it done.

Present the draft for confirmation before sending:
> **Subject:** `Ranka Oasis: Site Visit — Confirming Date and Access`
>
> Please confirm your availability for a site visit this week.
>
> 1. Confirm date — by Wednesday 5pm
> 2. Arrange access to the south plot
>
> Ready to send?

## 4. Stage 3 — Send

### New email
```
tool: google_workspace_manager
input:
  command: "gmail messages send --to raghu@example.com --subject 'Ranka Oasis: Site Visit' --body 'Please confirm...'"
  account_email: "ndr@draas.com"
```

For HTML: add `--bodyHtml '<p>...</p>'` flag alongside `--body`.

### Threaded reply — MUST include --threadId
```
tool: google_workspace_manager
input:
  command: "gmail messages send-reply --to raghu@example.com --subject 'Re: Land Valuation' --body 'Thanks...' --threadId THREAD_ID_HERE"
  account_email: "ndr@draas.com"
```

Without `--threadId` Gmail creates a brand-new disconnected thread. Always include it for replies.

After sending:
> Sent! Message ID: `[id]` | Thread: `https://mail.google.com/mail/u/0/#inbox/[threadId]`

### Who am I? — verify the mailbox before ANY draft (confirmed 2026-08-17)

Terminal subprocesses in this deployment can inherit a wrong `HERMES_SESSION_USER_ID` (observed `8502281203`), making `build_service('gmail','v1',service_name='google-draas')` resolve to **psingh@draas.com** instead of ndr@draas.com. A draft created under that state lands in Prakash's mailbox — a data-isolation breach that looks like success. ALWAYS run a whoami check before touching Gmail/Drive via terminal:

```python
svc = build_service('gmail', 'v1', service_name='google-draas')
who = svc.users().getProfile(userId='me').execute()['emailAddress']
assert who == 'ndr@draas.com', f'WRONG MAILBOX: {who}'
# fix if it fails: prefix the shell command with  HERMES_SESSION_USER_ID=7449813913
```
Gateway tools (`gws_resolve_account`, `execute_code` sandbox, `gws_fetch_token`) use the correct session identity — only raw terminal python can flip. Also verify drafts after creation with `drafts().get()`: From must be Nishant Ranka <ndr@draas.com>.

**Pitfall — the SAME thread exists in two mailboxes with DIFFERENT message IDs (confirmed 2026-09-21, V-Gro Finserv/Prakash):** When a colleague forwards a third party's email to NDR, the thread now exists in BOTH mailboxes with different IDs. Searching under NDR's identity finds NDR's copy; the message ID from that copy returns **404 "Requested entity was not found"** when you then re-run the script with `HERMES_SESSION_USER_ID=<colleague>` to draft from THEIR mailbox. This looks like a broken thread but is simply the wrong copy.

1. Decide the mailbox FIRST (who should send?), set `HERMES_SESSION_USER_ID=<colleague-slug>` on the script, and confirm `users().getProfile()` before any lookup.
2. Search for the thread FROM THAT MAILBOX (`q='<sender-domain> OR <sender-name>'`) to get the correct local message ID + threadId.
3. The colleague's forward legitimately has a different threadId than the counterparty's original — reply against the **counterparty's original message** (From = external sender), which is the real thread, and whose ID is valid in the colleague's mailbox.
4. Do NOT carry a message ID across mailboxes. Always re-resolve inside the target account.

**Pitfall — Gmail attachment download base64 padding:** `messages().attachments().get()` returns the attachment data as a base64url-encoded string whose length may not be a multiple of 4 (e.g. `len(data) % 4 == 1`). `base64.urlsafe_b64decode(data)` raises `binascii.Error: Incorrect padding` in this case. Fix with explicit padding before decode:
```python
padding = 4 - (len(data) % 4)
if padding != 4:
    data = data + '=' * padding
file_data = base64.urlsafe_b64decode(data)
```
Always add this guard when downloading attachments from Gmail — it accounts for the trailing-padding variation in Gmail's base64 output.

`messages().attachments().get()` base64 payloads can arrive truncated (observed on a 24 KB docx: header bytes OK, central directory missing → `BadZipFile`; base64 length mod 4 == 1 → decode error). Do NOT retry more than twice — the same file almost always exists in Drive (it was attached to an email that also got saved/exported there). Fetch the Drive copy instead:

```python
raw = drive.files().export(fileId=DOC_ID, mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document').execute()
# for native binary files: drive.files().get_media(fileId=...).execute()
```
Live lesson: Millers Road v6 lease attachment was corrupt via Gmail; the v5 base + edited copy exported cleanly from Drive (Drive's `files().export` / `get_media` don't go through the JSON gateway that truncated the attachment response). Validate any downloaded .docx/.zip by checking the PK header AND opening with zipfile before building a draft attachment.

### HTML email formatting — when to use (user preference, 2026-08-21)

When the email carries more than 3 Drive links, multiple sections (approvals, NOCs, drawings, queries), or is an instruction brief to a colleague — ALWAYS use HTML with inline CSS so hyperlinks render clickable. Plain text with bare URLs forces the recipient to copy-paste each link manually, which the user explicitly flagged as broken.

**Pattern that worked for the Ranka North Star approval-queries email:**
- Use a `<table>` for the summary header (project name, LP number, total land, area under sanction, key contacts)
- Use a coloured `<div>` per query (different background colour per numbered question) to visually separate the 5 items
- Every document reference is a clickable `<a href="...">` link — never a bare URL in parentheses
- Inline CSS only (no `<style>` block — many email clients strip it): `style="color:#...;background:#...;padding:...;margin:..."`
- End with a reference-document link and a "Documents referenced in this email:" section listing everything in one place
- The `gws_skill_bridge.call("draft_create", ..., html=True)` flag MUST be set; without it the HTML source is rendered as literal text
- Verify the draft after creation: `drafts().get()` → `payload.mimeType` should be `text/html`, and the body should render links not literal `<a>` tags

**Fee-revision / negotiation confirmation emails — HTML with color-coded highlighted figures (NDR preference, validated 2026-08-25):** After an in-person fee discussion, NDR wants the follow-up email as a structured HTML draft with each agreed term in its own color-coded box and every key number visually highlighted. Working pattern from the Arvind A.J. Architects fee revision (Ranka North Star):
- One `<div>` per agreed item, each with a distinct left-border + light background color family: rework scope (blue), enhanced base fee math (green), fixed-price scope add-on (red), payment adjustment (amber/yellow), unchanged terms (grey).
- Highlight the key rupee figures with `<span style="background:#fff3cd; font-weight:bold; padding:1px 5px;">` — and strike the superseded original rate (`text-decoration: line-through`).
- Reconstruct the arithmetic explicitly so the vendor can verify: "38 + 5 + 4 = ₹47" with each addend explained (e.g. MEP+structural ₹13/sqft BUA × 1.2 = ₹15.6 − ₹11 already factored ≈ ₹4.6 → rounded ₹5).
- **Payment-adjustment clause — always include when past payments exist:** "All payments made to date will be adjusted while computing the new payable; fees recomputed at the new rate, all earlier payments deducted and adjusted accordingly — only the balance will be payable." NDR explicitly required this highlighted in its own section.
- Plain-text multipart alternative required (same content, no markup); `email.message.EmailMessage` + `set_content` + `add_alternative(html, subtype='html')` produces the multipart/alternative that Gmail drafts need.
- Create as threaded reply against the vendor's ORIGINAL proposal Message-ID (In-Reply-To + References + explicit `threadId`), delete any earlier plain-text draft on the same thread so there is exactly one draft.
- Same agreement goes into the record: post the identical terms as a Kelsa note on the PO-WO lead (pipeline 537) so the PO record and the email match. See `references/fee-proposal-comparison.md` for the negotiation groundwork; the confirmation-email pattern above is the follow-up half of the same workflow.

### Advisor / consultant mandate emails — attach PDF, HTML, highlighted ask (NDR, 2026-08-25, Vishwas Rao/Jiraaf case)

When emailing a briefing to a structuring/tax consultant or advisor with a term sheet / engagement scope:
1. **Attach the exported PDF — do NOT give a Drive link.** NDR explicitly: "download and attach the term sheet for now instead of giving him a link." Export the Google Doc → PDF via Drive `files().export()` and attach as MIMEBase.
2. **Open with the user's exact briefing line** (e.g. "As I had discussed with you some time ago, I am in discussion with closing a commercial arrangement with Jiraaf Capital… I am sharing with you the term sheet that we have signed"). Use his framing, not generic boilerplate.
3. **HTML body with color-coded sections** — one colored `<div>` per scope area (land/structure, key commercial terms, structuring considerations, regulatory/RERA, GST-opinion) and a **final highlighted callout box** (yellow bg `#fff8e1`, bold `<span style="background:#fff3cd">`) carrying the specific ask: review in the next couple of hours → call with the counterparty team today → drive to conclusion → total timeline → priority/fees separate. NDR: "make sure all those points are specifically neatly highlighted… redo the entire email using HTML so that everything is properly highlighted and properly captured and it's easy to read."
4. **If the user says they've made an edit to the doc since last export, re-fetch the doc and adopt the live edit** into the email text and the attached PDF (user edited v1.1's uplift % to 20% in-place; the attachment must carry it).
5. Rebuild flow: `drafts().delete(draft_id)` with the draft resource ID (not message ID), then `drafts().create()` — never leave the superseded draft behind.

### Architect / consultant briefs with an R&D reference collection — share-then-link, don't attach (2026-08-25, Palya row villas → Arvind Jain)

When the brief references a COLLECTION of competitor/R&D material (RERA summaries, sanction plans, brochures, layout plans — dozens of PDFs), attaching everything blows the 25 MB cap and buries the ask. Pattern that worked:

1. **Grant Drive access BEFORE creating the draft** — `drive.permissions().create(fileId=..., body={'type':'user','role':'writer','emailAddress':...})` to the external partner AND the internal coordinator who gets Cc'd. Verify each call returns `role: writer`. Only share the specific subfolders/sheets the brief references — avoid sharing catch-all parent folders that contain unrelated files (the Palya R&D lived in a mixed Aug-2026 folder; the row-villa subfolder alone was shared).
1a. **⚠️ ALWAYS `permissions().list()` the folder you are about to link — before it goes into any outbound mail (confirmed 2026-09-25).** If a folder was shared in a prior session, the recipient may ALREADY be on it — read the ACL first and just link it (don't re-upload or re-grant). More importantly, an ACL can carry grants nobody intended: the Ranka Udaya `Container Site Office (Prefab) — AR Portable Cabin Refs` folder (`1mq2pdtX_sXNeZhgjZoTEAyqjsd2v-BVc`) had **three external `vgrofinserv.com` readers** alongside the internal ones. Linking a folder without reading its ACL means shipping a folder whose audience you have not verified, and silently publishing the counterparty's material to whoever else sits on it. Surface any anomalous external grant to NDR and **ask before revoking** — never act unilaterally on someone else's access.
2. **Write the email as color-coded HTML sections** (project brief / concept anchor / design targets / reference materials / landscape / what-we-need) with the FAR, rate and sq-ft figures highlighted (`<span style="background:#fff3cd;font-weight:bold">`), and every Drive folder/sheet as a clickable link — never bare URLs.
3. **Attach only single decisive documents if any**; the folder links carry the collection.
4. **Flag drift between the brief and existing internal models** in your reply to the user (e.g. Palya IRR model still assumed FAR 1.05 @ ₹12k/sf while the brief asked for 1.8 @ ₹15k/sf) — do not silently align the email to the stale model, and warn that the model needs a refresh.

**User refinement on review — second round (2026-08-25, same Palya brief):** When NDR reviewed the draft he narrowed it down sharply. Encode these in any follow-up/revised brief:
- **One attachment only** — the decisive comparison sheet, exported to xlsx via Drive `files().export()` and attached as MIMEBase. Drop auxiliary summary-sheet LINKS from the body (he removed the RERA summary sheet link entirely); keep the collection folder + working folder as links so the architect's team has the full reference.
- **Embed the comparison as a TABLE in the email body** — columns: Project | Subtype | Achieved FAR | link to the specific plan showing that FAR. The sheet alone is not enough; the data must be visible in the email itself.
- **Quote the FAR measured off the sanctioned plan's title block**, not the RERA registered number — they differ (Sattva La Vita STP layout plan = **1.27** vs RERA 1.35; The Roots Development Plan = **1.93** vs RERA 2.0). For The Roots the FAR lives on the DEVELOPMENT PLAN, not the STP plan — check which plan in the folder actually carries the FAR before linking it.
- **Design-consequence callouts belong in the email as highlighted question boxes** — e.g. The Roots reaches 1.93 with basement + G + 1 + 2 + partly covered terrace ≈ G+3 ⇒ a lift/elevator becomes mandatory; without one G+2 + terrace is difficult. Frame as "we want your professional view on storey count vs FAR" rather than asserting the answer.
- **Long HTML bodies: write the body in part files** (`/tmp/email_part1.html` … part3) via separate write_file calls, then concatenate inside the build script — one giant write_file times out the stream. Verify final draft has exactly one html part + the attachment part list.

### WhatsApp vs Email dual-channel pattern (user-defined 2026-08-21):
When sending project instructions to an employee (Vinod, Prakash, etc.) about approval queries:
1. **WhatsApp** = short, high-level, ask-only. State the 3-5 questions in brief numbered form. Tell them a detailed email is incoming with full references. Use `messaging-drafts` → `references/personal-messaging-workflow.md` / `whatsapp_link` tool.
2. **Email** = full detailed brief, with every fact, figure, approval reference, link to all documents (Drive folders/files), and document access instructions. Use HTML with all hyperlinks rendered. CC relevant colleagues.
The WhatsApp primes the recipient. The email is their working document. Do NOT try to merge the two into one channel.

**Variant — external-meeting follow-up: WhatsApp to the outsider, EMAIL TO THE BROTHERS/TEAM (user correction 2026-09-01, Deloitte M&A partner):** after NDR meets an external professional (consultant, tax/M&A advisor), the standard two-part follow-up is:
1. **WhatsApp to the external person** — thank-you + share NDR's contact + heads-up that a mail to his brothers/colleagues follows to set up a meeting; date confirmed later when the outsider is next in town. Share contact as **+91 98800 55634 and ndr@drahomes.in** (his SHARING contact — never the draas.com work address). There is no NDR visiting-card image on Drive (PAN cards only) — put the contact in the text; NDR attaches the card photo himself when sending.
2. **Email to the brothers/team only** — meeting summary: who the person is (credentials/background), the idea discussed with his view, what was asked of him, and the proposed next step (e.g. presentation in office when he's next in Bangalore).

**Comprehension trap (this session's wrong draft — deleted after user correction):** when NDR dictates "I e-mail to that effect, mark to Dharmesh and Manish", the email is **FOR Dharmesh and Manish — primary To recipients**, NOT an email to the external person with them CC'd. "Mark to X and Y" in his voice = the internal distribution list. Do not draft the external-CC variant from a meeting follow-up; check which audience the dictation describes before building To/Cc. If ambiguous, present the channel split (WhatsApp→outsider, email→brothers) for confirmation rather than guessing.

**Variant — "push all the details into both channels" (2026-08-30, guideline-value confirmation):** When NDR says "build the message with all of these details... and just send a link to the PDF in the email AND the WhatsApp messages", the default short-ping split is REVERSED: WhatsApp carries the SAME full detail as the email (exact row/page reference, the figure, the sub-registrar ask, and the PDF link), with the email being the working record. Example: for the Yelahanka guideline-value confirmation he dictated the exact book reference ("row 79, page 6 — 79 Allalasandra Village 26450 → ₹2,645/sqft for R 14/1") and required it in both channels for all three recipients. Follow his explicit instruction over the default split — when he names what goes into the messages, put it in verbatim.

**Sub-case — stale-thread follow-up (email draft + WhatsApp ping, validated 2026-08-28, 18 & Oak DTCP):** When following up on an email that received no reply for 2+ weeks, the dual-channel split is different:
1. **Email** = reply-all draft on the original thread, with numbered questions asking for a status update (who spoken to, contact details, why each resource can help, current progress).
2. **WhatsApp** = short notification ping: "I've sent a follow-up email on the same thread — please check." Not a separate brief.
Full reference: `references/stale-thread-follow-up-whatsapp-ping.md`

**Variant — same content, different audience split (2026-08-27, Ranka Oasis collaterals):**
When requesting project assets (renders, walkthroughs, master plans) from an architect/designer AND the same information needs to reach colleagues who are CC'd on email but don't need a WhatsApp blast:

1. **WhatsApp to individual contributors** — send the identical brief to each person separately (Sinchana gets her own link, Bharat gets his own link). Keep the content the same: what's needed + urgency framing + who's marked.
2. **Email to the primary contact** — same content as WhatsApp, sent to the person who will do the work, with CCs to the stakeholders. The email is the working document; WhatsApp is the immediate ping.
3. **Do NOT send WhatsApp to people who are only CC'd on email** — they only need the email for visibility.
4. **Key distinction in the request: "basis renders vs marketing renders"** — when asking for architectural renders/collaterals, the user explicitly distinguishes between:
   - **Basis renders** (what you want): the actual architectural output from SketchUp/V-Ray with all materials matching the architect's plan — used for costing and engineering coordination
   - **Marketing renders** (not what you want here): the polished brochure-ready content already developed separately
   Always clarify which one is being requested; mixing them up leads to wrong deliverables.

**Pitfall — a bridge-created HTML draft stores its body BASE64 inside `multipart/alternative`, so raw-MIME string checks all report False (confirmed 2026-09-25).** After `draft_create(..., html=true)`, fetching `format='raw'` and searching the decoded MIME for `width:100%`, `max-width`, or any body phrase returns **False for everything** — because `Content-Transfer-Encoding: base64` means the HTML is not plain text in the raw payload. Do not conclude the body is missing or mangled. Two reliable verifications:
  1. **Preferred:** fetch `format='full'` and walk for the `text/html` part, then `base64.urlsafe_b64decode(part['body']['data'])`. Check `width:100% in body`, `'max-width' not in body`, and the expected phrases.
  2. If hand-splitting the raw MIME: **strip all whitespace before decoding** (`re.sub(r"\s+", "", b64)`), then pad (`+ "=" * (-len(b64) % 4)`). Gmail's line-wrapped base64 fails `b64decode` with `Number of data characters (N) cannot be 1 more than a multiple of 4` when the newlines are left in. Even then, prefer (1) — it is one call and cannot mis-parse.
  Note the same draft may store `Content-Transfer-Encoding: 7bit` (plain) on one creation and `base64` on the next — do not assume either.

**Pitfall — `%` characters in the body break `%`-formatting (confirmed 2026-09-25).** Real business bodies are full of them: `18% GST`, `95% of the work will be completed at our factory`, `60% advance`, `50–55″`. Building the body with `html = """...""" % (LINK_A, LINK_B)` raises `TypeError: must be real number, not str`. Never reach for `%`-formatting on email bodies — use token placeholders plus `.replace()`:

```python
html = "...<a href=\"__FOLDER__\">... 18% GST ..."
html = html.replace("__FOLDER__", FOLDER).replace("__EARTH__", EARTH)
```

Related trap when the same script builds BOTH a plain-text and an HTML body: a bulk `str.replace()`-style edit of that script can point the `text` closing line's replacement at `html`, leaving `text` un-substituted so the plain-text alternative ships literal `__FOLDER__` tokens while the HTML looks perfect. After any programmatic edit of a two-body build script, grep the two assignment lines and confirm each variable name matches its own body.

**Pitfall — headers read back empty via `messages().get(format='metadata', metadataHeaders=[...])` while the draft is fine (confirmed 2026-09-25).** A freshly bridge-created draft can return `Subject: None`, `To: None`, `Cc: None` from the metadata format even though `format='full'` and `format='raw'` both show the correct headers. Do not treat this as a failed create and do not re-create the draft. Confirm via `format='full'` header parse (or the raw MIME header block) before concluding anything.

**Pitfall — `metadataHeaders` kwarg is NOT accepted by `drafts().get()` (confirmed 2026-09-21):** `drafts().get(userId='me', id=..., format='metadata', metadataHeaders=[...])` raises `TypeError: Got an unexpected keyword argument metadataHeaders`. The `format='metadata'` + `metadataHeaders` combo works on `messages().get()` but NOT on drafts. Read drafts with `format='full'` and parse `payload['headers']` (`{h['name']: h['value'] for h in msg['payload']['headers']}`). When listing drafts to verify Cc/To/threads, this is the reliable pattern.

**Pitfall — the assets for one deal routinely SPAN accounts; searching one finds half the deal (confirmed 2026-09-26, Samudra/Godrej mail):** before drafting a mail that attaches or quotes deal material, sweep BOTH `google-draas` and `google-gmail` (and `google-ahfl` if the deal is group-level). On Samudra, the email thread + the site-sketch attachment lived on `google-draas`, while BOTH valuation spreadsheets lived on `google-gmail` (nishantranka@gmail.com) — a single-account sweep would have produced a draft with no numbers or no sketch. Run the sweeps in parallel: Gmail `messages().list(q=...)` per account AND Drive `files().list(q="name contains '<project>'")` per account, then reconcile.

**Pitfall — the attachment you need is often on an EMAIL, not in Drive.** A filename search of Drive is not evidence the asset is missing. On Samudra, Drive search for `Samudra`/`sketch` returned only unrelated Gunjur/Stella sketches, while the actual sketch was an attachment on a colleague's email (find it via `q='<project> has:attachment'` against the colleague's address, then walk `payload.parts[].filename` for the sender's own filename — the wire name can differ from how anyone refers to it). Enumerate thread attachments before telling the user a document can't be found.

**Verifying a sketch/PDF attachment before you rely on its content** — `vision_analyze` rejects PDFs outright (*"Only real image files are supported for vision analysis"*). Render first, then read:

```bash
pdftoppm -r 120 -png /tmp/<name>.pdf /tmp/<name>_pg   # -> /tmp/<name>_pg-1.png
```

then call `vision_analyze` on the PNG. (`pdftotext`/pymupdf are unreliable for CAD-exported sketches — text is often vector outlines, and `fitz` may not be installed.) This is how the Samudra sketch's labels — "EXISTING OUTER RING ROAD", "EXISTING SERVICE ROAD", "EXISTING CANAL", "TOTAL SITE AREA - 7 ACRES" — were confirmed, including the finding that it carries **no dimensions and no survey numbers** (matching the drafter's own "approximate sketch at this stage" note). Never assert what an attached plan shows without reading it.

**Pitfall — a draft from a prior session is very often GONE (confirmed 2026-09-26).** Do not plan a mail around "the draft I made earlier" or offer to "update it". A `drafts().list()` on 2026-09-26 for the Samudra thread returned only an unrelated 2023 Marriott draft — both earlier Samudra drafts (created 09-Sep) had been sent, or cleared, in the interim. Enumerate drafts fresh at the start of the task; if the old one is absent, rebuild from source rather than referencing a draft that no longer exists.

**Pitfall — a draft from a PREVIOUS session may already EXIST for the same request (confirmed 2026-09-21):** the inverse of the missing-draft case. When NDR dictation arrives twice (voice message + repeated instruction, or a parallel/API session ran the same request), the draft can ALREADY be in Drafts with the exact same subject/recipients. Building a fresh one creates a duplicate. ALWAYS `drafts().list(userId='me')` and enumerate subjects FIRST when the request sounds familiar — match by subject keywords and To/Cc set, then verify the existing draft (labels=['DRAFT'], attachment names via payload walk, HTML mimeType) and present it for review instead of recreating. Also check session_search for a parallel session that resolved recipient voice-garble (e.g. the api-* sessions) before re-litigating the same name lookups.

**Pitfall — a draft you created earlier in the SAME session may already be SENT by the time a later turn reports on it (confirmed 2026-09-25).** NDR works fast and sends drafts himself between turns. Mid-session, after creating two drafts, a later `drafts().list()` returned only ONE — the other two had been sent at 18:01 and 18:02. Before repeating "it's sitting in your Drafts folder" in a later turn, **re-verify `drafts().list()` and search the subjects**: if a message now carries `['SENT']`, report it as sent with the time and recipients, and do NOT recreate it. A stale "it's in Drafts" claim sends the user looking for something that is no longer there. (His own sends also answer the "should I send this?" question — check the Sent copy before offering to send.)

**Editing a draft to remove a WRONG recipient (invalid/incorrect Cc or To) without losing HTML body/attachments (2026-09-21, wrong \"Vinay T\" in RERA Cc):** fetch the draft raw (`drafts().get(id=..., format='raw')` → `base64.urlsafe_b64decode`), regex-remove/replace only the offending header line (`re.sub(r'Cc: .*?\\r?\\n', '', raw, count=1)`), re-encode, `drafts().update(userId='me', id=..., body={'id': ..., 'message': {'raw': ...}})`. Much safer than delete+recreate when the body has a large embedded HTML/attachment. Then re-verify with `format='full'` that the header is gone. NEVER keep an unverified person (see Vinay/Mark case below) on a draft's Cc.

**Pitfall — `drafts().get()` body verification: bridge drafts store body at TOP-LEVEL `payload.body.data`, NOT in `parts` (2026-09-01).** After `draft_create` / `draft_reply_create` with `html=False`, the MIME is a bare `text/plain` part — `payload['parts']` is absent and the recursive `get_text(parts...)` walker returns `None`, printing a misleading "(no text)". The text walker must check `payload['body']['data']` at the top level BEFORE recursing into parts:

```python
def walk(payload, indent=0):
    body = payload.get('body', {})
    if body.get('data'):
        print(base64.urlsafe_b64decode(body['data']).decode('utf-8', 'replace')[:500])
    for p in payload.get('parts', []):
        walk(p, indent + 2)
```

**DRA family canonical addresses — Dharmesh/Manish/Mamta (verified via Gmail headers 2026-09-01, NOT the contacts sheet):**
- **Dharmesh Ranka** — `dharmesh.ranka1@gmail.com` (active in Sep-2026 threads; contacts sheet row 870 lists ddr@draas.com / dharmesh.ranka1@gmail.com / ddr@drahomes.in — use the gmail.com one seen in headers)
- **Manish Ranka** — `mdr@draas.com` (active; contacts sheet row 1901 lists mdr@drahomes.in as primary — headers win)
- **Mamta Rathod** (sister, Chennai) — `mamatadr@gmail.com`
- **Roshni Ranka** — `rnr@draas.com`
All four are on the Dinesh Ranka succession / estate threads. Verify via Gmail headers (`q='to:(ddr@draas.com OR dharmesh.ranka1@gmail.com ...)'`) before drafting — the sheet rows carry stale/alternate domains.

**DRA family voice garble map (succession context):** "Hermit"/"Damesh"/"Dharmash" → **Dharmesh**; "Manesh"/"Mantha" → **Manish**; "Mamtha"/"Mantha" → **Mamta Rathod** (sister, signs petition + GPA); "vocala"/"vakalath" → **vakalat**; "GPI" → **GPA**; "Sanjay Setia/Satya" → **Sanjay Sethia** (sanjay@lawsquare.in, Law Square); "Gauravar Bhavan Jain trust" → **Godwad Bhavan Jain Trust** (GBJT — the trust work Sanjay handled). Petition details: 5 petitioners = surviving Class-I heirs of Late Dinesh D Ranka; GPA holder = **Rajendra**; certificate drawn jointly in favour of Petitioners 1-5 (s.370 change validated 2026-09-01). Voice garble for NON-family work: "Anwar Asan" → **Anbu Asan** (Anbarasan); "Tushar Giri" → **tushar patil** (structure.pune@gmail.com).
**Legal counsel voice-garble (Behera / Mirabilis RERA matter):** "Jastin in Cor" → **Adv. Jasleen Kaur of Smriti Legal** (jasleen@smritilegal.com, +91 9740012005, Unity Building Annexe, Mission Road, Bengaluru; firm inbox info.smritilegal@gmail.com; partners Vinay T vinay@vnv.ca / E Suhail Ahmed suhail@trialbase.in / Pradyumna pradyumnakhg@smritilegal.com). Full case document-map + where each requested instrument lives: `references/behera-rera-doc-supply-2026-09.md`.

## 5. Rules

- ALWAYS call `google_workspace_manager` as a Hermes tool, never as a shell command
- ALWAYS include `--threadId` for replies
- NEVER add boilerplate greetings unless asked
- ALWAYS confirm contact/thread before drafting
- NEVER guess email addresses — ALWAYS run the contact lookup tool first (`find_contact.py` from `messaging-drafts` → `references/personal-messaging-workflow.md`, HARD RULE 2026-07-31) for any phone/email lookup
- NEVER send without showing draft and getting confirmation
- For vendor-feedback escalation emails: see `messaging-drafts` skill → `references/vendor-feedback-escalation.md` for the full dual-channel workflow (Google Doc → email draft with attachments → WhatsApp group follow-up). The tone guidance in Stage 2 — Draft (Vendor-feedback / frank escalation tone) applies to the email body.
