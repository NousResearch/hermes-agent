---
name: inkbox
description: "Inkbox — runtime guidance for agents routed via the Inkbox platform plugin: email, SMS, live voice. Contact-keyed sessions, common pitfalls, minimal SDK surface."
version: 1.0.0
platforms: [linux, macos, windows]
---

# Inkbox

API-first communication infrastructure for AI agents — email, phone, encrypted vault, and identities. When this skill is loaded, Hermes is running with the Inkbox platform plugin: inbound email, inbound SMS, and inbound voice calls route to one session per remote party (`chat_id = contact_id`).

For the full SDK reference, see [inkbox.ai/docs](https://inkbox.ai/docs). What follows is the subset you need at agent runtime.

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

When answering factual questions about live Inkbox state (identity channels, contact records, messages, delivery status), use the SDK as source of truth — not transcript memory. The inbound routing marker contains resolved Contact fields and is acceptable immediate context for conversational identity questions, but verify through the SDK when asked for live / current facts.

**Live voice visibility pitfall:** during `[inkbox:voice_call ...]` turns, the prompt-visible marker does not include precise audio latency, transcript timestamps, or any initial adapter greeting that may have happened before the visible transcript context. If the caller asks about latency, timestamps, or "the first thing you said," answer only from the visible transcript unless you perform a post-call SDK / log inspection. Explicitly say when you cannot see the underlying timing in the live turn.

**Third-party "text X" pitfall:** if the user explicitly asks for an SMS to a contact, prefer `identity.send_text(to="+E164", text=...)` after resolving the contact phone. Do not assume `send_message(target="inkbox:Contact", ...)` will text — contact targets can default to email when the adapter lacks recent SMS modality for that contact. Verify delivery in `identity.get_text_conversation(phone, ...)` before reporting success.

**Voice-call caller identity pitfall:** if a voice turn arrives with `contact=unknown_in_inkbox`, the caller may still exist in the contact book. Use the marker's `call_id` to find the live call via `identity.list_calls(...)`, read `remote_phone_number`, then run `inkbox.contacts.lookup(phone=remote_phone_number)` before saying you don't know who the caller is. This handles natural questions like "what's my name?", "what's my email?", "do you know my number?". If latency makes a live lookup inappropriate, say you don't see the detail in the visible live-call context and would need to look it up after the call.

**Voice-call contact creation pitfall:** when an unknown voice caller asks to be saved as a contact ("my name is Dima; add my phone number"), do not guess or reuse a remembered phone number. Resolve the live call's `remote_phone_number` from the `call_id` first, then lookup/upsert the contact with that E.164 number. If the caller corrects an email spelling, update the contact record immediately and preserve the exact spelling — even for Gmail addresses where dots are delivery-equivalent.

```python
call_id = "59e84d61-..."
with Inkbox(api_key=os.environ["INKBOX_API_KEY"]) as ink:
    identity = ink.get_identity(os.environ["INKBOX_IDENTITY"])
    call = next((c for c in identity.list_calls(limit=50, offset=0) if str(c.id) == call_id), None)
    remote = call.remote_phone_number if call else None
    contact = ink.contacts.lookup(phone=remote) if remote else None
```

## Core Model

```
Inkbox (admin-only client)
├── .get_identity(handle)     → AgentIdentity
├── .list_identities()        → list[AgentIdentitySummary]
├── .contacts                 → ContactsResource  (.lookup, .access, .vcards)
├── .texts                    → TextsResource
├── .whoami()                 → WhoamiResponse
└── ...

AgentIdentity (identity-scoped helper)
├── .mailbox                 → IdentityMailbox | None
├── .phone_number            → IdentityPhoneNumber | None
├── send_email / iter_emails / get_thread / ...
├── send_text  / list_texts  / get_text_conversation / ...
└── place_call / list_calls  / get_call_transcript / ...
```

An identity must have a channel assigned before mail / phone / text methods work. If not assigned, an `InkboxError` is raised with a clear message.

## Contacts (the routing primitive)

The plugin keys every session on `chat_id = contact_id`. The single most important lookup at runtime:

```python
contact = inkbox.contacts.lookup(email="alice@example.com")
contact = inkbox.contacts.lookup(phone="+15167251294")
# Returns Contact | None. Multiple matches → ambiguous; the plugin
# falls back to raw email/phone as chat_id rather than guessing.
```

A `Contact` carries `.id`, `.name`, `.emails` (list with `is_primary`), `.phones` (list with `is_primary`), `.notes`. Mutations are documented in inkbox.ai's contact-book reference.

## Mail

### Send

```python
sent = identity.send_email(
    to=["user@example.com"],
    subject="Hello",
    body_text="Hi there!",
    body_html="<p>Hi there!</p>",        # optional
    cc=["cc@example.com"],               # optional
    bcc=["bcc@example.com"],             # optional
    in_reply_to_message_id=prior.id,     # threaded reply
    attachments=[{
        "filename": "report.pdf",
        "content_type": "application/pdf",
        "content_base64": "<base64>",
    }],
)
```

The plugin's `send()` calls this when `metadata['mode'] == 'email'` or when `chat_id` resolves to an email-only contact.

### Read

```python
for msg in identity.iter_emails():           # auto-paginates
    print(msg.subject, msg.from_address, msg.is_read)

for msg in identity.iter_unread_emails():
    ...

# Full thread, oldest-first
thread = identity.get_thread(msg.thread_id)
for m in thread.messages:
    print(f"[{m.from_address}] {m.subject}")

identity.mark_emails_read([m.id for m in unread])
```

### Thread folders

Threads carry a `folder`: `inbox`, `spam`, `archive`, or `blocked` (server-assigned, never client-set).

## Phone (live voice calls)

```python
# Place outbound call — stream audio via WebSocket
call = identity.place_call(
    to_number="+15167251294",
    client_websocket_url="wss://your-agent.example.com/ws",
)
print(call.status)

# List calls (offset pagination)
for c in identity.list_calls(limit=20, offset=0):
    print(c.id, c.direction, c.remote_phone_number, c.status)

# Transcript segments (ordered by seq)
for seg in identity.get_call_transcript(call.id):
    print(f"[{seg.speaker}] {seg.text}")
```

### Outbound calls from a Hermes-routed identity

When the gateway plugin places an outbound call from the active session, it joins the contact's main session (not a separate call thread) so the agent inherits prior context. Voice replies stream as `text` frames on the call WebSocket; Inkbox handles TTS playback.

Write call-purpose context the in-call agent will see on its first turn via `identity.set_call_context(call_id, ...)` — see [inkbox.ai/docs/routed-session-operations](https://inkbox.ai/docs/routed-session-operations) for the full pattern.

## Text Messages (SMS / MMS)

**Outbound SMS limits and gates (current):**

- Allowed only from **local** numbers, not toll-free (toll-free SMS in progress).
- **15 outbound sends per phone number per rolling 24h.**
- New local numbers need **~10–15 min** for 10DLC carrier propagation. `identity.phone_number.sms_status` is `SmsStatus.PENDING` until ready; sends in this window return `409 sender_sms_pending`.
- Recipient must have texted **`START`** to any number in the org. Unknown → `403 recipient_not_opted_in`. `STOP` → `403 recipient_opted_out`.

```python
sent = identity.send_text(to="+15167251294", text="Hello from Inkbox")
print(sent.id, sent.delivery_status)   # SmsDeliveryStatus.QUEUED

unread = identity.list_texts(is_read=False)

# Get messages in a specific conversation
msgs = identity.get_text_conversation("+15167251294", limit=50)

# MMS media (temporary signed URLs)
text = identity.get_text("text-uuid")
if text.media:
    for m in text.media:
        print(m.content_type, m.size, m.url)

identity.mark_text_read("text-uuid")
identity.mark_text_conversation_read("+15167251294")
```

## When to reach for the SDK vs the routing marker

| Question type | Source |
|---|---|
| "Who is this?" during a voice call | Marker first; SDK if marker says `unknown_in_inkbox` |
| "What's the delivery status of my last text?" | SDK — `identity.list_texts()` |
| "Read me my unread emails" | SDK — `identity.iter_unread_emails()` |
| "Save my new phone number as a contact" | SDK — resolve `call.remote_phone_number` first, then upsert |
| Anything time-sensitive on a live call | Marker only; SDK calls add latency |

## Further reading

- [Inkbox docs](https://inkbox.ai/docs) — full SDK reference
- [Contact book operations](https://inkbox.ai/docs/contact-book) — upsert / merge / vCard patterns
- [Routed session operations](https://inkbox.ai/docs/routed-session-operations) — Hermes-specific recipes for forcing SMS, scheduling calls, checking bounces
- [Vault](https://inkbox.ai/docs/vault) — encrypted credential storage with client-side Argon2id key derivation
