# Drafting from ANOTHER user's mailbox — resolve the thread THERE, not in NDR's

Verified 2026-09-21 (V-Gro Finserv consolidated-docs reply, drafted as `psingh@draas.com`).

## The trap

When the user asks for a reply to *their own* inbound mail (e.g. Prakash: "draft email reply
to subject email received from Vamshi Raj"), the thread usually ALSO exists in NDR's mailbox
— because the sender forwarded it to NDR, or NDR was Cc'd. Both copies look identical by
subject. They do **not** share IDs.

Real case:

| Mailbox | Thread ID | Message ID (Vamshi's original) |
|---|---|---|
| ndr@draas.com | `19fdb9f280c85ceb` (Prakash's *forward*) | — |
| psingh@draas.com | `19fda683c682cd3f` | `19fda683c682cd3f` |

Searching NDR's mailbox first and grabbing that message ID, then running the draft script
under `HERMES_SESSION_USER_ID=psingh`, produced:

```
MAILBOX: psingh@draas.com
HttpError 404 ... users/me/messages/19fdb9f280c85ceb?format=raw
"Requested entity was not found."
```

The 404 is **account-scoping**, not a deleted message. Note the mailbox line was CORRECT —
`psingh@draas.com` — which makes the failure look like a permissions problem when it is
simply the wrong ID for that account.

## The recipe

1. **Set the mailbox FIRST, then search inside it.** Never reuse a message/thread ID
   discovered in a different mailbox.

```bash
cd /opt/hermes && HERMES_SESSION_USER_ID=psingh /opt/hermes/.venv/bin/python3 -c "
import sys; sys.path.insert(0,'/opt/hermes')
from tools.gws_auth import build_service
svc = build_service('gmail','v1',service_name='google-draas')
print('MAILBOX:', svc.users().getProfile(userId='me').execute()['emailAddress'])
res = svc.users().messages().list(userId='me', q='<sender-domain or name>', maxResults=25).execute()
for m in res.get('messages',[]):
    d = svc.users().messages().get(userId='me', id=m['id'], format='metadata',
        metadataHeaders=['Subject','From','To','Cc','Date']).execute()
    h = {x['name']:x['value'] for x in d['payload']['headers']}
    print('---'); print('id:', m['id'], '| thread:', d['threadId'])
    print('From:', h.get('From'), '| To:', h.get('To'), '| Cc:', h.get('Cc'))
    print('Subj:', h.get('Subject'))
"
```

2. **Pick the ORIGINAL inbound message, not the user's forward of it.** In Prakash's
   mailbox both appeared: his forward to NDR (`From: Prakash Singh`, 15:17) and Vamshi's
   original (`From: Vamshi Raj`, 09:37). Thread against the original — `To` = Vamshi, and
   the `Message-ID`/`threadId` come from that row.

3. **Build the draft with that mailbox's IDs**, keeping the same `HERMES_SESSION_USER_ID`
   prefix on the creating script (it must match the mailbox you searched).

4. **Verify From after creation** — the draft must show the *requesting user*, not NDR:

```python
v = svc.users().drafts().get(userId='me', id=dr['id'], format='full').execute()
h = {x['name']: x['value'] for x in v['message']['payload']['headers']}
print(h.get('From'))       # e.g. Prakash Singh <psingh@draas.com>
print(v['message'].get('labelIds'))   # ['DRAFT']
print(v['message'].get('threadId'))   # must equal the thread from step 2
```

5. **Confirm the OTHER mailbox stayed clean.** A mis-scoped create can silently land a
   draft in the vault-owner's (NDR's) mailbox — a data-isolation breach that looks like
   success. Check explicitly and report it:

```python
# run WITHOUT the env prefix -> ndr@draas.com
r = svc.users().drafts().list(userId='me', q='<matter keyword>', maxResults=10).execute()
print('drafts in NDR mailbox:', r.get('resultSizeEstimate'))
```

## Related hard rules that apply here

- NDR's **email-recipient rule**: only addresses verifiable in contacts (online contact
  sheet or Google People API) may go on a draft. Here Prakash supplied no new addresses —
  every recipient was taken verbatim from the thread headers (`vamshiraj@`, `praveen@`,
  `sudheer@vgrofinserv.com`), which is the correct source.
- **Colleague-facing drafts default to ndr@draas.com**; this case is the exception the user
  explicitly directed (a reply to Prakash's own correspondence, in his own mailbox). When
  the user says "reply to the email I received", the mailbox is *theirs* — but confirm if
  ambiguous, because the general rule points the other way.
- Draft only. Never `messages().send()`.
