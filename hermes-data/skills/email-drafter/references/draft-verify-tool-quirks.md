# Draft verification & contact-lookup tool quirks (verified 2026-09-14)

## contact_resolver — `account_email` fallback

The tool fails when the session user has no `gws_service` configured:

```
Sheets connection failed: User 'ndr-7449813913' has no gws_service configured in their profile.
An admin must set 'gws_service', or pass account_email explicitly (call gws_resolve_account to see valid accounts).
```

Fix: retry with `account_email` set explicitly, e.g. `account_email='ndr@draas.com'` (short label 'draas' also works). This resolved "Ruhan" → Ruhaan Ranka (`pebblyshark69@gmail.com`, +91 98804 78352) on 2026-09-14. Ruhaan is NDR's son and the sole 'ruhaan' contact; when multiple candidates come back, disambiguate with context (family vs company) rather than picking blindly.

## Verify drafts — no `metadataHeaders` kwarg

`gmail.users().drafts().get(userId='me', id=..., format='metadata', metadataHeaders=['Subject','To','Date'])` raises:

```
TypeError: Got an unexpected keyword argument metadataHeaders
```

The current googleapiclient (v3, Python 3.13 venv at /opt/hermes) does not accept `metadataHeaders` on `drafts().get` (it IS accepted on `messages().get`). Fix: fetch `format='metadata'` (no metadataHeaders), then filter `message['payload']['headers']` client-side for Subject/To/Date.

## Draft-creation verification loop

`gws_skill_bridge draft_create` returns `{status: draft_created, draft_id, message_id, threadId}` — treat that as success but still confirm via `drafts().list(maxResults=5)` + `drafts().get(id, format='metadata')` when reporting to NDR, quoting the resolved recipient and subject line back.