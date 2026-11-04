# Forward a document/bank-statement batch to an internal group alias

**Trigger (verified 2026-10-09, KOTAK statements):**
*"There are a bunch of <bank statement> emails that came to me in the last N days. Please
forward all of them. If they are already addressed to <alias>, then don't worry. If they
are all addressed to me, then forward them to <group alias> **after you confirm that such
a group email exists**. And X, Y, Z are part of that group. Can you check on that? First
<alias>… if that exists, forward to it. If it doesn't exist, then forward to <person 1>
and <person 2>, CC <person 3> and <person 4>."*

NDR dictates a **conditional routing tree**. Resolve it in order: (1) confirm the alias,
(2) check whether the mail is already addressed to it, (3) only then forward.

## Step 1 — Establish the source set

Search the requesting account for the doc type, not just the sender domain:

```
from:kotak.bank.in after:YYYY/MM/DD          # broad — includes alerts, balances, offers
subject:"Bank statement for Kotak" after:... # narrow — the actual statements
subject:statement after:...                  # cross-bank sweep
```

Report the **narrow** set as the deliverable, but run the broad query once so you can say
what else came in and why it is excluded (balance alerts, NEFT/RTGS alerts, credit-card
txns, "scheduled maintenance" notices are NOT statements).

**Bank statements arrive from a shared sender and land at MULTIPLE of NDR's addresses.**
The KOTAK batch split 3 → `NDR@draas.com` and 3 → `NDR@drahomes.in` (his old alias,
which forwards in). "Sent to me" is satisfied by EITHER address — his own mail counts.
The `To:` header is the discriminator for the *"already addressed to the alias?"* test.

## Step 2 — Confirm the alias EXISTS (do this BEFORE promising the forward)

**Fastest reliable proof = Gmail history, not the Admin Directory API.**

```
to:<alias>       # if NDR forwards there, and vendors address it, it exists
from:<alias>     # reveals WHAT the alias actually is
```

Verified on `accounts@draas.com`: `to:` returned many NDR forwards (Advance Tax, IDFC
dormant-account, Evolve Back bookings, TDS alerts) **and** `from:` returned replies signed
**"Prathvi Soni <accounts@draas.com>"** — i.e. it is a live shared accounts alias, used as
the From-identity of the accounts person. That single pair of queries answers "does it
exist" and "who is behind it" in two calls.

**Admin Directory is authoritative but often unavailable:**
`build_service('admin','directory_v1')` → `groups().list(domain=...)` +
`members().list(groupKey=...)` requires `admin.directory.group.readonly` +
`admin.directory.group.member.readonly`. If the token has been re-consented without those
scopes it fails `403 "Request had insufficient authentication scopes"` (seen 2026-10-09).
Do **not** block the task on it and do **not** claim the group doesn't exist — say the
membership is unverified, present the Gmail-history evidence, and offer the one-tap
scope-expansion consent (`gws-domain-admin` → `references/admin-directory-group-inventory.md`).
Never fall back to DWD for this — Directory is out of DWD scope.

## Step 3 — Apply the routing tree explicitly

- If the source mail is **already addressed to the alias** → **no forward**; report that.
- If it is addressed only to NDR (any of his addresses) → forward to the alias.
- Only if the alias genuinely does not exist → forward to the named people, honouring the
  To vs CC split exactly as dictated.

State which branch you took and why. The "already-addressed" branch is a real terminal
outcome — don't drop it and forward anyway.

## Step 4 — DUPLICATE GUARD (do this before creating the forward)

NDR often has **already sent the same documents earlier the same day**. Before building
the draft, search Sent:

```
in:sent <subject keyword> after:YYYY/MM/DD
```

Verified 2026-10-09: the 6 KOTAK statements had **already been emailed at 06:45 that
morning** to Eshwari + Sarthak in a thread *"RERA Account Statements - Ranka Amber & Ranka
Oasis (Kotak) - Oct 2026"*. Same attachment sizes, same accounts. Surface the overlap
prominently and let NDR decide (send anyway for filing, or skip) — never silently create a
duplicate-forward that makes his books look double-filed.

## Step 5 — Build the forward draft with attachments (raw MIME)

The bridge's draft ops don't carry attachments; build the MIME yourself and use
`drafts().create()` (draft-only — **never** `messages().send()`):

```python
msg = MIMEMultipart('mixed')
msg['To'] = '<alias@domain>'; msg['From'] = 'Nishant Ranka <ndr@draas.com>'
msg['Subject'] = '<clean subject>'
msg.attach(MIMEText(body, 'plain'))
for path, clean_fn in attachments:
    part = MIMEBase('application', 'pdf'); part.set_payload(open(path,'rb').read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment', filename=clean_fn)
    msg.attach(part)
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
draft = gmail.users().drafts().create(userId='me', body={'message': {'raw': raw}}).execute()
```

Rules:
- **Download each source attachment and verify integrity before attaching** — check the
  bytes start with the right magic (`%PDF`) and `len(bytes) == reported size`. Attachment
  downloads can be silently truncated (see `analyze-work-emails` →
  `references/gmail-attachment-truncation-drive-fallback.md`). All 6 verified exact here.
- **Rename attachments to a self-describing convention** — `YYYYMMDD_<Project>_<AcctNo>_<Role>.pdf`.
  Bank PDFs arrive as `85XXXXX370_01-01-2026_08-10-2026.pdf` (masked number, raw dates);
  unmasked + labelled is far more useful to the accounts desk.
- Order attachments by project/account so the covering email's numbered list matches.
- **Verify the created draft** with `drafts().get(format='full')` — confirm To, Subject,
  and count/walk the attachment parts (metadataHeaders kwargs are NOT accepted on the
  drafts resource).

## Step 6 — Delivery

The draft is inert — it sits in Drafts for NDR to review and send himself. Report the
Gmail draft id, the To, and the attachment list. **Route the report to `origin`** (the
chat that asked), never a hardcoded chat id.

## Voice-garble → identity resolution (this class)

NDR's dictated group-member and alias names are routinely garbled. Resolve via Contacts
sheet / `contact_resolver` / Gmail, and show BOTH the garble and the resolution:

| Dictated | Resolved | Notes |
|---|---|---|
| "accounts at dras.com" | `accounts@draas.com` | draas, not dras |
| "accounts at Graal" | **unresolved** | flag it — do not invent a domain; ask |
| "Sartak" | **Sarthak Sharma** `admin3.blr@draas.com` | DRA accounts |
| "Kishwari" | **Eshwari Chamundeshwari** `echamundeshwari@draas.com` | DRA accounts |
| "Roshni" | **Roshini Ranka** `rnr@draas.com` | |

When a dictated alias can't be resolved anywhere (here "Graal"), say so and ask — never
guess a domain or silently substitute.

## Pitfall — "some are for personal"

Voice "spam are for personal" was actually *"some are for personal"*: of the 6 KOTAK
statements, 3 were NDR's **personal** accounts and 3 were the **RERA project** accounts
(Amber + Oasis). Confirm which subset is being forwarded and name each account by role
(Collection 100% / Designated 70% / Operative 30%) so the accounts desk isn't left to
guess from masked numbers.