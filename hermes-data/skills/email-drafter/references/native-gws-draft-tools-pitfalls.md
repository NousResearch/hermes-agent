# Native `gws_gmail_*` draft tools — pitfalls & verification

Verified 2026-10-16 in **Bharat's session** (sales1.blr@draas.com), drafting a new
(non-threaded) email to Gowri Singh <gsingh@draas.com>, Cc Roshini Ranka <rnr@draas.com>.

Use this file when the session user is a DRA **employee**, not NDR, and you are drafting
through the native tools (`gws_gmail_draft_create`, `gws_gmail_get`, `gws_gmail_search`,
`gws_gmail_draft_delete`) rather than the raw-MIME `drafts().create` recipes in
`draft-create-verify-api.md`. The raw-MIME recipes still apply when you need custom
threading headers, attachments, or an in-place update.

---

## 1. Which mailbox do the native tools write to?

`gws_resolve_account` with **no arguments** lists every vault service with live auth status.
In this session it returned:

```
google-ahfl  -> has_token: false
google-draas -> sales1.blr@draas.com   has_token: true
google-gmail -> has_token: false
```

So `service_name='google-draas'` resolved to the **session user's own mailbox**, not to
`ndr@draas.com`. This is the opposite of the older Anbu-session note in
`draft-create-verify-api.md` (where `google-draas` resolved to ndr@draas.com and needed
`HERMES_SESSION_USER_ID=pm2_blr`). Resolution is deployment- and session-dependent.

**Rule: never assume who `google-draas` is.** Either
`build_service('gmail','v1', service_name=...).users().getProfile(userId='me')` first, or
read the created draft's `From:` back (see §4) before reporting the draft as ready. The
`From:` line in the read-back is the cheapest proof and is required anyway.

## 2. Pitfall — the native tool does NOT HTML-escape the body, so `&amp;` is stored LITERALLY

`gws_gmail_draft_create` passes `body=` through verbatim. It does **not** decode or
HTML-escape entities, and there is no auto-detection of HTML.

Real failure: the body was written with `"Thanks &amp; Regards,"` (HTML entity) while
`html` was left at its default `false`. The stored draft read literally
`Thanks &amp; Regards,` — Gmail renders `&amp;` as the characters `&amp;` in a plain-text
part, so the sign-off would have gone out visibly broken.

**Rules:**

- With `html=false` (plain text): write the real characters — `&`, `—`, `'`, `"`.
  Never `&amp;`, `&mdash;`, `&rsquo;`.
- With `html=true`: the opposite — entities are correct there, and a bare `&` is risky.
- Before reporting the draft, **read the body back and assert `'&amp;' not in body`**
  (plain-text drafts only). `&amp;` in a plain-text body is the signature of this bug.

**Repair — the native tool has no update, so delete and recreate:**

```
gws_gmail_draft_delete(draft_id=<bad draft id>)   # then re-create with corrected body
```

Do not attempt to "edit around" it. If you must patch in place, switch to the raw-MIME
`drafts().update` path (pin `threadId` — see the `drafts().update` re-threading pitfall in
`draft-create-verify-api.md`). Re-verify the new draft and report the **surviving**
`draft_id` so the user cannot send the stale one.

## 3. Pitfall — `gws_gmail_get` output OMITS the `Cc` header

A draft created with `cc="Roshini Ranka <rnr@draas.com>"` came back from `gws_gmail_get`
with `id / threadId / snippet / date / from / to / subject / labelIds / body /
attachments` — **no `cc` field at all**. The Cc was in fact set correctly. Reading the
tool output alone therefore *cannot* confirm or refute a Cc, and a missing field reads
exactly like a dropped recipient.

To verify To/Cc/From/Subject/labels, fetch the raw MIME and read the header block:

```python
import sys, base64, quopri
sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import build_service

svc = build_service('gmail', 'v1', service_name='google-draas')
m = svc.users().messages().get(userId='me', id=MESSAGE_ID, format='raw').execute()
raw = base64.urlsafe_b64decode(m['raw']).decode('utf-8', 'replace')
hdr, _, body = raw.partition('\n\n')
for line in hdr.split('\n'):
    if line.lower().startswith(('to:', 'cc:', 'from:', 'subject:')):
        print(line.strip())
print('LABELS:', m.get('labelIds'))            # must be ['DRAFT']
text = quopri.decodestring(body).decode('utf-8', 'replace')
print('literal &amp; present:', '&amp;' in text)  # must be False for plain-text drafts
```

- Use the `message_id` returned by `gws_gmail_draft_create` (or
  `drafts().get(...)['message']['id']`) — `messages().get` resolves it while it is still a
  draft.
- `labelIds: ['DRAFT']` is the proof it was not sent. Never describe a draft as ready
  without seeing `DRAFT` and not `SENT`.
- Ready to re-run: `scripts/verify_draft_headers_raw.py`.

## 4. Non-ASCII subjects and quoted-printable bodies are NORMAL — do not "fix" them

Reading the raw header block, an em-dash subject appears as
`Subject: Video covering all four projects =?utf-8?b?4oCU?= Ranka Amber / Ranka` and the
body arrives quoted-printable (`=E2=80=94` for `—`, soft line breaks `=\n`). Both are
correct MIME for non-ASCII content and are **not** the malformed-header defect described in
`gmail-draft-integrity-and-attachments.md`. That defect is diagnosed by *missing*
`MIME-Version` / `Content-Type` / `Message-Id`, not by encoded-words per se. Decode with
`quopri.decodestring` before content assertions; don't delete a draft for this.

## 5. `contact_resolver` (and the noun tools) in an employee session need `account_email`

In a session whose user has no `gws_service` in their profile, `contact_resolver` fails:

```
Sheets connection failed: User '8717455402' has no gws_service configured in their
profile. An admin must set 'gws_service', or pass account_email explicitly
(call gws_resolve_account to see valid accounts).
```

**Fix — pass the session user's address explicitly:**
`contact_resolver(query='Gowri', account_email='sales1.blr@draas.com')`.

This is not a broken tool and not a reason to hand-read the contacts sheet.

## 6. Recipient addresses: Gmail headers beat the contacts sheet (worked example)

- **Gowri Singh** — the contacts sheet row (1156) carries only *personal* addresses
  (`gowrisingh72@yahoo.com`, `gowrisingh1341@gmail.com`). Her **work** address
  `gsingh@draas.com` exists only in Gmail headers, confirmed from multiple live threads
  (she sends from `gsingh@draas.com` and is addressed there by Bharat). Use the header
  address.
- **Roshini Ranka** — contacts row 2989, `rnr@draas.com` (work), plus
  `roshiniranka@gmail.com` / `roshini.ranka@gmail.com` and `rnr@drahomes.in`. For DRA staff
  the `@draas.com` address is the only correct one (NDR's 2026-10-09 ruling, see
  `draft-create-verify-api.md`).
- A `gws_gmail_search` for a bare name is **fuzzy** — results for `"roshini"` returned
  messages that do not contain the name at all. Confirm an address from an actual
  `To:`/`From:`/`Cc:` header on a returned message, not from the fact that a message
  appeared in the result list.

## 7. Bharat's (sales1.blr) drafting conventions

- Salutation `Hi ma'am,` for Gowri / Roshini; sign-off `Thanks & Regards,` +
  `Bharat Hawaldar` on its own line (some mails close `Regards,` / `Bharat`).
- A voice note **is** the body. *"Ma'am is there any possibility we can create one minute
  video…"* → that sentence becomes the opening of the draft; it is not a request to research
  video vendors. Do not expand a short dictated ask into a long structured email.
- An instruction like *"keep Roshini ma'am in CC"* **persists to the next draft in the same
  drafting flow** unless restated — but say so explicitly in the handoff note so he can
  correct it.
- Flag, don't silently fix, dictated domain slips. Here the four listed items included
  **Ranka Udaya Rental Guarantee**, which is a *proposition* on Ranka Udaya, not a separate
  project (Gowri's own mails call it "Ranka Udaya RG"). Keep his wording in the draft and
  raise it as a check item.
- Do not invent a use case, deadline, or budget that he did not state. Where the draft is
  silent on something that would sharpen the answer (e.g. where the video will be used),
  list it as an open item in the delivery note instead of writing it into the email.
