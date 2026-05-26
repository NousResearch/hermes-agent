# Inkbox contact-book upserts from routed sessions

Session lesson: when a user gives contact-book facts over an Inkbox-routed SMS/email, distinguish the transport sender from the person being described.

## Patterns

- If the inbound marker resolves to `contact_id=...` and the user says “I am …” or “my title is …”, update that exact contact via `ink.contacts.update(contact_id, ...)` and verify the returned record.
- If the user says “add Ray/Alex too” while texting from another person's number, do not overwrite the sender's contact. Ask for or use the new person's email/phone, lookup by email then phone, and create/update a separate contact.
- Normalize phone numbers to E.164 before lookup/create/update.
- After a contact-book update, reply with the concrete fields saved: name, company, title, email, phone.
- In Inkbox-routed sessions, distinguish three stores in user-facing language: the live Inkbox contact book, Hermes persistent memory/user profile, and transcript/session search. If the user asks what profile was updated, say explicitly whether you updated the Inkbox contact record, saved a durable memory, or both.
- When the user says “remember that X is the CEO/role/title” and X is a known contact or includes lookupable email/phone, treat this as BOTH a durable-memory update and an Inkbox contact-book update unless they explicitly limit it to memory. Update the contact's `job_title`/company fields through the SDK, then verify the returned record. On live voice calls, acknowledge briefly first if needed for latency, but do not leave the session without performing the contact-book mutation when tools are available.
- Do not use durable memory as the source of truth for live contact-book state; use it only as a conversational hint, then verify or mutate through the SDK.

## Known-good command shape

```bash
/home/ec2-user/inkbox-powered-hermes-agent/.venv/bin/python3 - <<'PY'
import os, sys, json
sys.path.insert(0, '/home/ec2-user/inkbox/sdk/python')
from inkbox import Inkbox

email = 'ray@vectorly.app'
phone = '+18573008599'

with Inkbox(api_key=os.environ['INKBOX_API_KEY']) as ink:
    existing = None
    for kwargs in ({'email': email}, {'phone': phone}):
        try:
            match = ink.contacts.lookup(**kwargs)
            existing = match[0] if match else None  # lookup always returns list[Contact]
            if existing:
                break
        except Exception:
            pass

    payload = dict(
        preferred_name='Ray',
        given_name='Ray',
        company_name='Inkbox',
        job_title='Cofounder',
        emails=[{'label': 'work', 'value': email, 'is_primary': True}],
        phones=[{'label': 'mobile', 'value_e164': phone, 'is_primary': True}],
    )
    contact = ink.contacts.update(str(existing.id), **payload) if existing else ink.contacts.create(**payload)
    print(json.dumps({
        'id': str(contact.id),
        'preferred_name': getattr(contact, 'preferred_name', None),
        'company_name': getattr(contact, 'company_name', None),
        'job_title': getattr(contact, 'job_title', None),
        'emails': [getattr(e, 'value', None) for e in getattr(contact, 'emails', [])],
        'phones': [getattr(p, 'value_e164', getattr(p, 'value', None)) for p in getattr(contact, 'phones', [])],
    }, indent=2))
PY
```
