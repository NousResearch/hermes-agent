# Pitfall: reply-draft addressed to YOURSELF

## Symptom

`gws_gmail_reply_draft(message_id=<latest message in thread>, ...)` produces a
draft with `To: <your own address>` instead of the counterparty.

Real case: Bharat (sales1.blr@draas.com) replying in a thread where the most
recent message was **his own sent mail** to `ndr@drahomes.in`. The tool derived
the recipient from that message's `From` header — which was Bharat. Draft was
created `To: BHARAT H <sales1.blr@draas.com>`. It also silently dropped the Cc.

This is easy to miss: the draft *looks* correct in the body and sits in the
right thread. It only fails at the moment someone hits send.

## Rule

Before calling `gws_gmail_reply_draft`, check who authored the last message in
the thread (`gws_gmail_thread_get` → `messages[-1].from`).

- Last message is from the counterparty → the native tool is fine.
- Last message is from **you** (a sent mail) → the native tool will address the
  reply to you. Build the raw MIME yourself.

## Fix — raw MIME with explicit To/Cc and threading headers

```python
from email.message import EmailMessage
import base64

# 1. get the Message-ID you are replying to
last = g.users().messages().get(
    userId='me', id=LAST_MSG_ID, format='metadata',
    metadataHeaders=['Message-ID', 'References', 'Subject']).execute()
h = {x['name'].lower(): x['value'] for x in last['payload']['headers']}
mid, refs = h.get('message-id', ''), h.get('references', '')

# 2. build the reply
m = EmailMessage()
m['To']      = 'ndr@drahomes.in'              # from the THREAD HEADER, never guessed
m['Cc']      = 'a@x.com, b@y.com'
m['From']    = 'BHARAT H <sales1.blr@draas.com>'
m['Subject'] = 'Re: <original subject>'
m['In-Reply-To'] = mid
m['References']  = (refs + ' ' + mid).strip() if refs else mid
m.set_content(body)

# 3. create the draft INSIDE the existing thread
raw = base64.urlsafe_b64encode(m.as_bytes()).decode()
g.users().drafts().create(
    userId='me',
    body={'message': {'raw': raw, 'threadId': THREAD_ID}}).execute()
```

Passing `threadId` is what keeps it in the conversation; `In-Reply-To` +
`References` are what make it render as a reply rather than a new mail.

## Always verify, never assume

After creating any draft, read it back and confirm the four headers:

```python
g.users().messages().get(userId='me', id=NEW_MSG_ID, format='metadata',
    metadataHeaders=['To','Cc','From','Subject','In-Reply-To']).execute()
```

Check `To`, `Cc`, `Subject`, `In-Reply-To`, and `threadId`. If a wrong draft was
already created, delete it with
`g.users().drafts().delete(userId='me', id=DRAFT_ID)` before recreating — do not
leave two drafts in the thread.

## Related

- Draft-only, always. Never `.send()`. Tell the user the draft is in Drafts.
- Reply into the existing thread with the original Cc list intact; do not start a
  fresh mail on the same subject.
