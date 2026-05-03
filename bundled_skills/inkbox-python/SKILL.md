---
name: inkbox-python
description: Use when writing Python code that imports from `inkbox`, uses `pip install inkbox`, or when adding email, phone, text/SMS, contacts, notes, contact rules, vault, or agent identity features using the Inkbox Python SDK.
user-invocable: false
---

# Inkbox Python SDK

API-first communication infrastructure for AI agents — email, phone, encrypted vault, and identities.

## Install & Init

```python
pip install inkbox
```

Always use the context manager — it manages the underlying HTTP session:

```python
from inkbox import Inkbox

with Inkbox(api_key="ApiKey_...") as inkbox:
    ...
```

Constructor: `Inkbox(api_key, base_url="https://inkbox.ai", timeout=30.0)`

## Runtime notes for Inkbox-routed Hermes sessions

When answering factual questions about the live Inkbox state (identity channels, contact records, messages, delivery status), use the Inkbox API/SDK as source of truth, not transcript memory. In Hermes tool contexts, `execute_code` may run in a sandbox without `INKBOX_API_KEY` / `INKBOX_IDENTITY`, and the SDK may not be importable unless the repo SDK path is inserted. If that happens, run via `terminal` with the Hermes venv and local SDK path, for example:

```bash
/home/ec2-user/inkbox-powered-hermes-agent/.venv/bin/python3 - <<'PY'
import os, sys
sys.path.insert(0, '/home/ec2-user/inkbox/sdk/python')
from inkbox import Inkbox
with Inkbox(api_key=os.environ['INKBOX_API_KEY']) as ink:
    identity = ink.get_identity(os.environ.get('INKBOX_IDENTITY', 'inkbox-on-call-agent'))
    print(identity.email_address)
    print(identity.phone_number.number if identity.phone_number else None)
    print(ink.contacts.lookup(phone='+15167251294'))
PY
```

The inbound routing marker already contains the resolved Contact fields; it is acceptable immediate context for conversational identity questions, but verify through the SDK when asked for live/current contact or channel facts.

**Voice-call caller identity pitfall:** if a voice turn arrives with `contact=unknown_in_inkbox`, the caller may still exist in the contact book. Use the marker's `call_id` to find the live call via `identity.list_calls(...)`, read `remote_phone_number`, then run `inkbox.contacts.lookup(phone=remote_phone_number)` before saying you do not know who the caller is. This is especially useful for natural questions like “what's my name?”, “do you know my number?”, or “can you look me up?”.

```python
call_id = "59e84d61-..."
with Inkbox(api_key=os.environ["INKBOX_API_KEY"]) as ink:
    identity = ink.get_identity(os.environ.get("INKBOX_IDENTITY", "inkbox-on-call-agent"))
    call = next((c for c in identity.list_calls(limit=50, offset=0) if str(c.id) == call_id), None)
    remote = call.remote_phone_number if call else None
    contact = ink.contacts.lookup(phone=remote) if remote else None
```

## Core Model

```
Inkbox (admin-only client)
├── .create_identity(handle)  → AgentIdentity
├── .get_identity(handle)     → AgentIdentity
├── .list_identities()        → list[AgentIdentitySummary]
├── .mailboxes                → MailboxesResource
├── .phone_numbers            → PhoneNumbersResource
├── .texts                    → TextsResource
├── .mail_contact_rules       → MailContactRulesResource
├── .phone_contact_rules      → PhoneContactRulesResource
├── .contacts                 → ContactsResource  (.access, .vcards)
├── .notes                    → NotesResource     (.access)
├── .vault                    → VaultResource
├── .whoami()                 → WhoamiResponse
└── .create_signing_key()     → SigningKey

AgentIdentity (identity-scoped helper)
├── .mailbox                 → IdentityMailbox | None
├── .phone_number            → IdentityPhoneNumber | None
├── .credentials             → Credentials  (requires vault unlocked)
├── mail methods             (requires assigned mailbox)
├── phone methods            (requires assigned phone number)
└── text methods             (requires assigned phone number)
```

An identity must have a channel assigned before you can use mail/phone methods. If not assigned, an `InkboxError` is raised with a clear message.

## Agent Signup

For the full agent self-signup flow (register, verify, check status, restrictions, and direct API examples), read the shared reference:

> **See:** `skills/inkbox-agent-self-signup/SKILL.md`

Python SDK methods: `Inkbox.signup(...)`, `Inkbox.verify_signup(api_key, ...)`, `Inkbox.resend_signup_verification(api_key)`, `Inkbox.get_signup_status(api_key)`.

## Identities

```python
identity = inkbox.create_identity("sales-agent")
identity = inkbox.get_identity("sales-agent")
identities = inkbox.list_identities()  # → list[AgentIdentitySummary]

identity.update(new_handle="new-name")   # rename
identity.update(status="paused")         # or "active"
identity.refresh()                       # re-fetch from API, updates cached channels
identity.delete()                        # unlinks channels
```

## Channel Management

```python
# Identity is created with a mailbox automatically — provision a phone number
phone = identity.provision_phone_number(type="toll_free")       # or type="local", state="NY"
print(identity.email_address)  # e.g. "sales-agent@inkboxmail.com"
print(phone.number)            # e.g. "+18005551234"

# Link existing channels
identity.assign_mailbox("mailbox-uuid")
identity.assign_phone_number("phone-number-uuid")

# Unlink without deleting
identity.unlink_mailbox()
identity.unlink_phone_number()
```

## Mail

### Send

```python
sent = identity.send_email(
    to=["user@example.com"],
    subject="Hello",
    body_text="Hi there!",          # plain text (optional)
    body_html="<p>Hi there!</p>",   # HTML (optional)
    cc=["cc@example.com"],          # optional
    bcc=["bcc@example.com"],        # optional
    in_reply_to_message_id=sent.id, # for threaded replies
    attachments=[{                  # optional
        "filename": "report.pdf",
        "content_type": "application/pdf",
        "content_base64": "<base64>",
    }],
)
```

### Read

```python
# Iterate all messages — pagination handled automatically (Iterator[Message])
for msg in identity.iter_emails():
    print(msg.subject, msg.from_address, msg.is_read)

# Filter by direction
for msg in identity.iter_emails(direction="inbound"):   # or "outbound"
    ...

# Unread only (client-side filtered)
for msg in identity.iter_unread_emails():
    ...

# Mark as read
ids = [msg.id for msg in identity.iter_unread_emails()]
identity.mark_emails_read(ids)

# Get full thread (oldest-first)
thread = identity.get_thread(msg.thread_id)
for m in thread.messages:
    print(f"[{m.from_address}] {m.subject}")
```

### Thread Folders

Threads carry a `folder` field: `inbox`, `spam`, `archive`, or `blocked` (server-assigned, never client-set).

```python
from inkbox import ThreadFolder
# Thread.folder / ThreadDetail.folder is always one of the four values above.
```

Low-level folder listing / per-thread updates (`list(folder=…)`, `list_folders(email)`, `update(..., folder=…)`) live on `ThreadsResource`. Passing `folder="blocked"` to `update` raises `ValueError` before the HTTP call.

## Phone

```python
# Place outbound call — stream audio via WebSocket
call = identity.place_call(
    to_number="+15167251294",
    client_websocket_url="wss://your-agent.example.com/ws",
)
print(call.status)
print(call.rate_limit.calls_remaining)

# List calls (offset pagination)
calls = identity.list_calls(limit=10, offset=0)
for c in calls:
    print(c.id, c.direction, c.remote_phone_number, c.status)

# Transcript segments (ordered by seq)
for t in identity.list_transcripts(calls[0].id):
    print(f"[{t.party}] {t.text}")   # party: "local" or "remote"
```

## Text Messages (SMS/MMS)

**Outbound SMS limits and gates (current):**

- Allowed only from **local** numbers, not toll-free.
- **15 outbound sends per phone number per rolling 24h.**
- New local numbers need **~10-15 min** for 10DLC carrier propagation. `identity.phone_number.sms_status` is `SmsStatus.PENDING` until ready; sends in this window return `409 sender_sms_pending`.
- Recipient must have texted **`START`** to any number in the org. Unknown → `403 recipient_not_opted_in`. `STOP` → `403 recipient_opted_out`.

**Coming soon:** toll-free SMS sending, customer-managed 10DLC brands/campaigns (drastically higher per-number limits).

```python
# Send an SMS from this identity's phone number.
# Returns a queued TextMessage; final delivery state arrives via the
# incoming_text_webhook_url configured on the sender.
sent = identity.send_text(to="+15167251294", text="Hello from Inkbox")
print(sent.id, sent.delivery_status)   # SmsDeliveryStatus.QUEUED

# List text messages (offset pagination)
texts = identity.list_texts(limit=20, offset=0)
for t in texts:
    print(t.id, t.direction, t.remote_phone_number, t.text, t.is_read)

# Filter by read state
unread = identity.list_texts(is_read=False)

# Get a single text message
text = identity.get_text("text-uuid")
print(text.type)   # "sms" or "mms"
if text.media:     # MMS media attachments (temporary signed URLs)
    for m in text.media:
        print(m.content_type, m.size, m.url)

# List conversation summaries (one row per remote number)
convos = identity.list_text_conversations(limit=20)
for c in convos:
    print(c.remote_phone_number, c.latest_text, c.unread_count, c.total_count)

# Get messages in a specific conversation
msgs = identity.get_text_conversation("+15167251294", limit=50)

# Mark a text as read (identity convenience method)
identity.mark_text_read("text-uuid")

# Mark all messages in a conversation as read
result = identity.mark_text_conversation_read("+15167251294")
print(result["updated_count"])

# Admin-only: search, update, delete
results = inkbox.texts.search(phone.id, q="invoice", limit=20)
inkbox.texts.update(phone.id, "text-uuid", status="deleted")
```

## Vault

Encrypted credential vault with client-side Argon2id key derivation and AES-256-GCM encryption. The server never sees plaintext secrets. Requires `argon2-cffi` and `cryptography` (included as dependencies).

### Initialize

```python
# Initialize a new vault (org ID is fetched automatically from the API key)
result = inkbox.vault.initialize("my-Vault-key-01!")
print(result.vault_id, result.vault_key_id)
for code in result.recovery_codes:
    print(code)  # save these immediately — they cannot be retrieved again
```

### Unlock & Read

```python
from inkbox import LoginPayload, APIKeyPayload, SSHKeyPayload, OtherPayload

# Unlock with a vault key — derives key via Argon2id, decrypts all secrets
unlocked = inkbox.vault.unlock("my-Vault-key-01!")

# Optionally filter to secrets an agent identity has access to
unlocked = inkbox.vault.unlock("my-Vault-key-01!", identity_id="agent-uuid")

# All decrypted secrets from the unlock bundle
for secret in unlocked.secrets:
    print(secret.name, secret.secret_type)
    print(secret.payload)   # LoginPayload, APIKeyPayload, SSHKeyPayload, or OtherPayload

# Fetch and decrypt a single secret by ID
secret = unlocked.get_secret("secret-uuid")
print(secret.payload.username, secret.payload.password)   # for login type
```

### Create & Update

```python
# Create a login secret (secret_type inferred from payload type)
unlocked.create_secret(
    "AWS Production",
    LoginPayload(password="s3cret", username="admin", url="https://aws.amazon.com"),
    description="Production IAM user",
)

# Create an API key secret
unlocked.create_secret(
    "GitHub PAT",
    APIKeyPayload(api_key="ghp_xxx"),
)

# Create an SSH key secret
unlocked.create_secret(
    "Deploy Key",
    SSHKeyPayload(private_key="-----BEGIN OPENSSH PRIVATE KEY-----..."),
)

# Create a freeform secret
unlocked.create_secret("Misc", OtherPayload(data="any freeform content"))

# Update name/description and/or re-encrypt payload
unlocked.update_secret("secret-uuid", name="New Name")
unlocked.update_secret("secret-uuid", payload=LoginPayload(password="new", username="new"))

# Delete
unlocked.delete_secret("secret-uuid")
```

### Metadata (no unlock needed)

```python
info = inkbox.vault.info()                                   # VaultInfo
keys = inkbox.vault.list_keys()                              # list[VaultKey]
keys = inkbox.vault.list_keys(key_type="recovery")           # filter by type
secrets = inkbox.vault.list_secrets()                         # list[VaultSecret] (metadata only)
secrets = inkbox.vault.list_secrets(secret_type="login")     # filter by type
inkbox.vault.delete_secret("secret-uuid")                    # delete without unlocking
```

### Payload Types

| Type | Class | Fields |
|------|-------|--------|
| `login` | `LoginPayload` | `password`, `username?`, `email?`, `url?`, `notes?` |
| `api_key` | `APIKeyPayload` | `api_key`, `endpoint?`, `notes?` |
| `key_pair` | `KeyPairPayload` | `access_key`, `secret_key`, `endpoint?`, `notes?` |
| `ssh_key` | `SSHKeyPayload` | `private_key`, `public_key?`, `fingerprint?`, `passphrase?`, `notes?` |
| `other` | `OtherPayload` | `data` |

`secret_type` is immutable after creation. To change it, delete and recreate.

### Agent Credentials (identity-scoped)

Agent-facing credential access — typed, identity-scoped. The vault stays as the admin surface; `identity.credentials` is the agent runtime surface.

```python
from inkbox import Credentials

# Unlock the vault first (stores state on the client)
inkbox.vault.unlock("my-Vault-key-01!")

identity = inkbox.get_identity("support-bot")

# Discovery — returns list[DecryptedVaultSecret] with name/metadata
all_creds = identity.credentials.list()
logins    = identity.credentials.list_logins()
api_keys  = identity.credentials.list_api_keys()
ssh_keys  = identity.credentials.list_ssh_keys()
key_pairs = identity.credentials.list_key_pairs()

# Access by UUID — returns typed payload directly
login    = identity.credentials.get_login("secret-uuid")      # → LoginPayload
api_key  = identity.credentials.get_api_key("secret-uuid")    # → APIKeyPayload
ssh_key  = identity.credentials.get_ssh_key("secret-uuid")    # → SSHKeyPayload
key_pair = identity.credentials.get_key_pair("secret-uuid")   # → KeyPairPayload

# Generic access — returns DecryptedVaultSecret
secret = identity.credentials.get("secret-uuid")
```

- Requires `inkbox.vault.unlock()` first — raises `InkboxError` if vault is not unlocked
- Results are filtered to secrets the identity has access to (via access rules)
- Cached after first access; call `identity.refresh()` to clear the cache
- `get_*` raises `KeyError` if not found, `TypeError` if wrong secret type

## One-Time Passwords (TOTP)

TOTP secrets are stored inside `LoginPayload.totp` in the encrypted vault. Codes are generated client-side — no server call needed.

### From an agent identity (recommended)

```python
from inkbox.vault.totp import parse_totp_uri
from inkbox.vault.types import LoginPayload

# Create a login with TOTP
secret = identity.create_secret(
    name="GitHub",
    payload=LoginPayload(
        username="user@example.com",
        password="s3cret",
        totp=parse_totp_uri("otpauth://totp/GitHub:user@example.com?secret=JBSWY3DPEHPK3PXP&issuer=GitHub"),
    ),
)

# Generate TOTP code
code = identity.get_totp_code(str(secret.id))
print(code.code)              # e.g. "482901"
print(code.seconds_remaining) # e.g. 17

# Add/replace TOTP on existing login
identity.set_totp(secret_id, "otpauth://totp/...?secret=...")

# Remove TOTP
identity.remove_totp(secret_id)
```

### From the unlocked vault (admin-only)

```python
unlocked = inkbox.vault.unlock("my-Vault-key-01!")

# Same methods available on UnlockedVault
unlocked.set_totp(secret_id, totp_config_or_uri)
unlocked.remove_totp(secret_id)
code = unlocked.get_totp_code(secret_id)
```

### TOTPCode fields

| Field | Type | Description |
|---|---|---|
| `code` | `str` | The OTP code (e.g. `"482901"`) |
| `period_start` | `int` | Unix timestamp when the code became valid |
| `period_end` | `int` | Unix timestamp when the code expires |
| `seconds_remaining` | `int` | Seconds until expiry |

## Admin-only Resources

### Mailboxes (`inkbox.mailboxes`)

```python
mailboxes = inkbox.mailboxes.list()
mailbox   = inkbox.mailboxes.get("abc@inkboxmail.com")

inkbox.mailboxes.update(mailbox.email_address, display_name="New Name")
inkbox.mailboxes.update(mailbox.email_address, webhook_url="https://example.com/hook")
inkbox.mailboxes.update(mailbox.email_address, webhook_url=None)   # remove webhook

# Switch contact-rule filter mode (admin-only — agent-scoped keys get 403)
updated = inkbox.mailboxes.update(mailbox.email_address, filter_mode="whitelist")
if updated.filter_mode_change_notice:
    # Populated when filter_mode actually changed — tells you how many
    # rules are now redundant under the new mode.
    n = updated.filter_mode_change_notice
    print(n.redundant_rule_count, n.redundant_rule_action, n.new_filter_mode)

# Mailbox responses now also carry mailbox.agent_identity_id when the
# mailbox is linked to an identity.

results = inkbox.mailboxes.search(mailbox.email_address, q="invoice", limit=20)
inkbox.mailboxes.delete(mailbox.email_address)
```

### Phone Numbers (`inkbox.phone_numbers`)

```python
numbers = inkbox.phone_numbers.list()
number  = inkbox.phone_numbers.get("phone-number-uuid")
number  = inkbox.phone_numbers.provision(agent_handle="my-agent", type="toll_free")
local   = inkbox.phone_numbers.provision(agent_handle="my-agent", type="local", state="NY")

inkbox.phone_numbers.update(
    number.id,
    incoming_call_action="webhook",            # "webhook", "auto_accept", or "auto_reject"
    incoming_call_webhook_url="https://...",
)
inkbox.phone_numbers.update(
    number.id,
    incoming_call_action="auto_accept",
    client_websocket_url="wss://...",
)

hits = inkbox.phone_numbers.search_transcripts(number.id, q="refund", party="remote", limit=50)
inkbox.phone_numbers.release(number.id)
```

Phone numbers carry the same `filter_mode` / `agent_identity_id` / `filter_mode_change_notice` fields as mailboxes; flipping `filter_mode` is admin-only and returns a change-notice when the value actually changed.

## Contact Rules

Per-mailbox or per-phone-number allow/block lists, enforced server-side. The active `filter_mode` on the owning resource controls whether the rules are interpreted as a whitelist or blacklist. Mail matches by exact email or domain; phone matches by exact E.164 number.

```python
from inkbox import (
    MailRuleAction, MailRuleMatchType, PhoneRuleAction, PhoneRuleMatchType,
    DuplicateContactRuleError,
)

# Mail rules — scoped to a single mailbox. New rules always start active;
# call `update(..., status="paused")` afterwards to pause one.
rule = inkbox.mail_contact_rules.create(
    mailbox.email_address,
    action=MailRuleAction.ALLOW,         # or BLOCK
    match_type=MailRuleMatchType.DOMAIN, # or EXACT_EMAIL
    match_target="example.com",
)
inkbox.mail_contact_rules.list(mailbox.email_address)
inkbox.mail_contact_rules.get(mailbox.email_address, rule.id)
inkbox.mail_contact_rules.update(mailbox.email_address, rule.id, status="paused")  # admin-only
inkbox.mail_contact_rules.delete(mailbox.email_address, rule.id)                   # admin-only

# Admin-only list; optionally narrow to a single mailbox_id
all_rules = inkbox.mail_contact_rules.list_all(mailbox_id=str(mailbox.id))

# Duplicate (match_type, match_target) on the same mailbox raises 409:
try:
    inkbox.mail_contact_rules.create(
        mailbox.email_address,
        action="allow", match_type="domain", match_target="example.com",
    )
except DuplicateContactRuleError as e:
    print(e.existing_rule_id)   # UUID of the rule that already matched

# Phone rules — same shape, only match_type="exact_number" is supported.
inkbox.phone_contact_rules.create(
    number.id,
    action=PhoneRuleAction.BLOCK,
    match_type=PhoneRuleMatchType.EXACT_NUMBER,
    match_target="+15551234567",
)
inkbox.phone_contact_rules.list(number.id)
inkbox.phone_contact_rules.list_all(phone_number_id=str(number.id))
```

## Contacts

Admin-only address book with per-identity access grants and vCard import/export. For routed-session contact-book upsert patterns, transport-sender-vs-described-person pitfalls, and user-facing distinctions between the Inkbox contact book and Hermes memory/user profile, see `references/contact-book-upserts.md`.

```python
from inkbox import (
    Contact, ContactEmail, ContactPhone, ContactAddress,
    RedundantContactAccessGrantError,
)

# CRUD
contact = inkbox.contacts.create(
    given_name="Ada",
    family_name="Lovelace",
    emails=[ContactEmail(label="work", value="ada@example.com")],
    phones=[ContactPhone(label="mobile", value="+15551234567")],
    # access_identity_ids defaults to "wildcard" (every active identity);
    # pass [] for admin-only, or a list of identity UUIDs for explicit grants.
)
inkbox.contacts.get(str(contact.id))
inkbox.contacts.list(q="ada", order="recent", limit=50, offset=0)
inkbox.contacts.update(str(contact.id), job_title="Analyst")       # JSON-merge-patch via kwargs
inkbox.contacts.delete(str(contact.id))

# Reverse-lookup — exactly one filter required (else ValueError before HTTP)
inkbox.contacts.lookup(email="ada@example.com")
inkbox.contacts.lookup(email_domain="example.com")
inkbox.contacts.lookup(phone="+15551234567")
inkbox.contacts.lookup(email_contains="ada")
inkbox.contacts.lookup(phone_contains="555")

# Contact-book updates in Inkbox-routed sessions:
# - If the inbound marker resolves to a contact and the user gives details for that same person,
#   update that contact_id directly and verify the returned record.
# - If the user asks the agent to “remember” a role/title for a known or lookupable third party,
#   save durable memory if appropriate AND update that person's contact record; do not treat
#   Hermes memory as a substitute for the Inkbox contact book.
# - If the user says “add Ray too” (or otherwise names a different person) while texting from
#   someone else's channel, do NOT overwrite the sender's contact. Require/use the new person's
#   email or phone, lookup by email then phone, and update-or-create a separate contact.
# - Normalize phones to E.164 before lookup/create/update. Dict payloads work for email/phone
#   collections when importing ContactEmail/ContactPhone is inconvenient.
# Example upsert for a distinct person:
email = "ray@example.com"
phone = "+18573008599"
existing = None
for kwargs in ({"email": email}, {"phone": phone}):
    try:
        match = inkbox.contacts.lookup(**kwargs)
        existing = match[0] if isinstance(match, list) and match else match
        if existing:
            break
    except Exception:
        pass
payload = dict(
    preferred_name="Ray",
    given_name="Ray",
    company_name="Inkbox",
    job_title="Cofounder",
    emails=[{"label": "work", "value": email, "is_primary": True}],
    phones=[{"label": "mobile", "value_e164": phone, "is_primary": True}],
)
contact = inkbox.contacts.update(str(existing.id), **payload) if existing else inkbox.contacts.create(**payload)

# Access grants (admin + JWT only; agents can self-revoke)
inkbox.contacts.access.list(str(contact.id))
inkbox.contacts.access.grant(str(contact.id), identity_id="agent-uuid")
inkbox.contacts.access.grant(str(contact.id), wildcard=True)       # every active identity
inkbox.contacts.access.revoke(str(contact.id), "agent-uuid")

# Redundant grants (e.g. per-identity on top of wildcard) raise 409
try:
    inkbox.contacts.access.grant(str(contact.id), identity_id="agent-uuid")
except RedundantContactAccessGrantError as e:
    print(e.error, e.detail_message)

# vCards
result = inkbox.contacts.vcards.import_vcards(vcf_text)   # bulk, ≤5 MiB, ≤1000 cards
print(result.created_ids)     # list[UUID]
for item in result.errors:    # list[ContactImportResultItem]
    print(item.index, item.error)

vcf = inkbox.contacts.vcards.export_vcard(str(contact.id))  # vCard 4.0 string
```

## Notes

Admin-only free-form notes with per-identity access grants. Identities must be granted access explicitly — there is no wildcard for notes.

```python
note = inkbox.notes.create(body="Customer prefers email follow-up.", title="Ada")
inkbox.notes.get(str(note.id))
inkbox.notes.list(q="email", identity_id="agent-uuid", order="recent", limit=50)
inkbox.notes.update(str(note.id), body="Updated body")
inkbox.notes.update(str(note.id), title=None)   # clear title (body cannot be null)
inkbox.notes.delete(str(note.id))

# Access grants (admin + JWT only)
inkbox.notes.access.list(str(note.id))
inkbox.notes.access.grant(str(note.id), identity_id="agent-uuid")
inkbox.notes.access.revoke(str(note.id), "agent-uuid")
```

## Whoami

```python
# Check the authenticated caller's identity
info = inkbox.whoami()
print(info.auth_type)        # "api_key" or "jwt"
print(info.organization_id)
```

Returns `WhoamiApiKeyResponse` (with `key_id`, `label`, `creator_type`, `auth_subtype`, etc.) or `WhoamiJwtResponse` (with `email`, `org_role`, etc.) based on `auth_type`.

For branching on API-key scope, compare against the exported constants:

```python
from inkbox import (
    AUTH_SUBTYPE_API_KEY_ADMIN_SCOPED,
    AUTH_SUBTYPE_API_KEY_AGENT_SCOPED_CLAIMED,
    AUTH_SUBTYPE_API_KEY_AGENT_SCOPED_UNCLAIMED,
)

if info.auth_type == "api_key" and info.auth_subtype == AUTH_SUBTYPE_API_KEY_ADMIN_SCOPED:
    ...   # admin-only operations (filter_mode flips, rule updates/deletes, etc.)
```

## Webhooks & Signature Verification

Webhooks are configured directly on the mailbox or phone number — no separate registration.

```python
from inkbox import verify_webhook

# Rotate signing key (plaintext returned once — save it)
key = inkbox.create_signing_key()

# Verify an incoming webhook request
is_valid = verify_webhook(
    payload=raw_body,                                    # bytes
    headers=request.headers,
    secret="whsec_...",
)
```

Algorithm: HMAC-SHA256 over `"{request_id}.{timestamp}.{body}"`.

## Error Handling

```python
from inkbox import (
    InkboxAPIError,
    DuplicateContactRuleError,
    RedundantContactAccessGrantError,
)

try:
    identity = inkbox.get_identity("unknown")
except InkboxAPIError as e:
    print(e.status_code)   # HTTP status (e.g. 404)
    print(e.detail)        # str for legacy errors, dict for structured ones
```

`InkboxAPIError.detail` can now be a `dict` for structured responses (e.g. contact-rule / access conflicts). Catch the narrower subclasses when you need the parsed fields:

- `DuplicateContactRuleError` — 409 when creating a contact rule with an already-taken `(match_type, match_target)` on the same resource. Exposes `.existing_rule_id: UUID`.
- `RedundantContactAccessGrantError` — 409 when a contact-access grant is redundant (e.g. per-identity grant on top of an active wildcard). Exposes `.error` and `.detail_message`.

## Key Conventions

- All method and property names are **snake_case**
- `iter_emails()` / `iter_unread_emails()` return `Iterator[Message]` — auto-paginated, lazy
- `list_calls()` returns `list[PhoneCall]` — offset pagination, not an iterator
- To clear a nullable field (e.g. webhook URL), pass `field=None`
- The `Inkbox` client **must** be used as a context manager (`with` statement) or `.close()` called manually
- Mail/phone methods on `AgentIdentity` raise `InkboxError` if the relevant channel isn't assigned
