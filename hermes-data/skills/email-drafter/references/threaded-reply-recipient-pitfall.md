# Threaded reply: the To-recipient trap, and how to verify a draft

## The trap

`gws_gmail_reply_draft(message_id=...)` derives the To address from the **From header of the
message you point it at**. In a thread where the *last* message is your own follow-up (very common
— you mailed, the counterparty never replied, you mailed again), that means the draft comes out
addressed **to yourself**.

Observed: a reply pointed at Bharat's own sent message produced
`To: BHARAT H <sales1.blr@draas.com>` instead of `To: ndr@drahomes.in`, with the Cc silently
dropped. The draft looks fine in the body and fails only on the header — so always read the
headers back.

## The reliable recipe

Build the raw MIME yourself. Explicit To/Cc, real threading headers, `threadId` on the create call
so it lands in the right conversation.

```python
from tools.gws_auth import build_service
from email.message import EmailMessage
import base64

g = build_service('gmail','v1', service_name='<service_name>')

# 1. threading headers off the LAST message in the thread
last = g.users().messages().get(userId='me', id=LAST_MSG_ID, format='metadata',
        metadataHeaders=['Message-ID','References','Subject']).execute()
h = {x['name'].lower(): x['value'] for x in last['payload']['headers']}
mid, refs = h.get('message-id',''), h.get('references','')

m = EmailMessage()
m['To'] = 'counterparty@example.com'          # from the thread's own headers, not the sheet
m['Cc'] = 'a@x.com, b@y.com'
m['From'] = '<your sending identity>'
m['Subject'] = 'Re: <original subject>'
if mid:
    m['In-Reply-To'] = mid
    m['References'] = (refs + ' ' + mid).strip() if refs else mid
m.set_content(body)                            # plain text; use set_content(html, subtype='html') for HTML

raw = base64.urlsafe_b64encode(m.as_bytes()).decode()
d = g.users().drafts().create(userId='me', body={'message': {'raw': raw, 'threadId': THREAD_ID}}).execute()
```

## Verify every draft before reporting it

```python
g.users().messages().get(userId='me', id=d['message']['id'], format='metadata',
    metadataHeaders=['To','Cc','From','Subject','In-Reply-To']).execute()
```
Check: To is the counterparty (not you), Cc survived, In-Reply-To is set, `threadId` matches,
`labelIds == ['DRAFT']`.

## Clean up a wrong draft

`g.users().drafts().delete(userId='me', id=draft_id).execute()` — do this before creating the
replacement, so the user is not left with two versions of the same mail in Drafts.

## Why not just patch the To afterwards

`drafts().update()` with a new raw replaces the whole message and is easy to get wrong on
threading. Deleting and re-creating from a correct raw is simpler and leaves a clean audit trail.

## But patching IS valid when the threading headers are already right

Re-verified 2026-10-29 (succession follow-up to Dharmesh + Manish). The native
`gws_gmail_reply_draft` had already generated **correct** `In-Reply-To`, `References`, `Message-Id`
and the right `threadId` — only the `To` line was wrong (addressed to NDR himself, because the last
message in the thread was his own). Rebuilding raw MIME from scratch would have thrown that away.
Patching one header line preserved all of it and was verified end-to-end.

The rule: **rewrite ONLY the header line that is wrong, leave every other byte identical**, and
re-supply `threadId` on the update. Never re-serialise the whole message.

```python
d  = g.users().drafts().get(userId='me', id=DRAFT_ID, format='raw').execute()
raw = base64.urlsafe_b64decode(d['message']['raw'].encode()).decode('utf-8','replace')

new_to = 'To: A <a@x.com>, B <b@y.com>'
out, done = [], False
for ln in raw.split('\r\n'):
    if ln.lower().startswith('to:') and not done:
        out.append(new_to); done = True
    else:
        out.append(ln)

enc = base64.urlsafe_b64encode('\r\n'.join(out).encode()).decode()
res = g.users().drafts().update(userId='me', id=DRAFT_ID,
        body={'message': {'raw': enc, 'threadId': d['message']['threadId']}}).execute()
```

Then read the headers back (`To`, `Cc`, `From`, `Subject`, `In-Reply-To`) — that verification step is
not optional either way. Note `drafts().get(format='raw')` returns headers as **`\r\n`-separated**;
splitting on `\n` alone leaves stray `\r` and corrupts the re-encode.

Choose: patch when only recipients are wrong and threading is already correct; rebuild raw MIME when
the subject, body structure or threading headers also need changing.

## Empty References — a thread whose only message is your own

The most common shape of a *follow-up* mail (you wrote, nobody replied, you write again): the last
message has a `Message-ID` but **no `References` header at all**. Handle it explicitly:

```python
m['In-Reply-To'] = mid
m['References']  = (refs + ' ' + mid).strip() if refs else mid   # refs == ''
```

Rebuild-from-scratch is the right choice here (nothing to preserve, and the native reply_draft would
address it to you). Verified 2026-10-29: landed in the original thread, `labelIds: ['DRAFT']`, both
counterparties on `To`.

For the whole workflow around this — decoding NDR's dictated subject line, proving the counterparty
really never replied, and restating the decision from the meeting note rather than from his dictation
— see `references/follow-up-on-silence.md`.

