# Draft threading traps (verified Sep-Oct 2026, Behera RERA pack sessions)

The user was repeatedly unable to see reply-all drafts in the same thread. All four
root causes below were hit in one week of sessions; each has a stable fix.

## 1. threadId goes INSIDE the `message` object

`drafts().create(body={'message': {'raw': raw, 'threadId': MASTER_THREAD}})` — NOT
`body={'message': {'raw': raw}, 'threadId': MASTER_THREAD}`.
When passed at draft top level the field is IGNORED and Gmail mints a brand-new thread
(even though In-Reply-To/References look correct). Verify with
`drafts().get(...)` → `message.threadId == MASTER_THREAD` — never trust the create
response alone.

## 2. Skill-bridge draft_reply_create is for simple replies only

For a true reply-all WITH attachments to a specific recipient (To: lawyer, Cc: the crew):
- bridge sets To = anchor message's From (WRONG for reply-all — you want the original thread recipient)
- bridge drops the Cc list
- bridge cannot attach files
Build the raw MIME manually instead:
```python
msg = MIMEMultipart('mixed')            # + ('alternative') for plain+HTML
msg['To'], msg['Cc'], msg['Subject']
msg['In-Reply-To'] = anchor_msg_id
msg['References']  = anchor_references + ' ' + anchor_msg_id   # full chain, not truncated
# attach via MIMEBase('application','pdf') + encode_base64 + Content-Disposition
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
svc.users().drafts().create(userId='me', body={'message': {'raw': raw, 'threadId': MASTER}})
```

## 3. Anchor = latest message ALREADY in the master thread

Find via `threads().get(userId='me', id=MASTER_THREAD, format='metadata')`, use the last
message's Message-ID as In-Reply-To and its References chain (+ own ID) as References.
A sent email that went out WITHOUT In-Reply-To/References becomes a standalone thread —
recreate, don't try to re-parent it.

## 4. Updating a draft in place can DETACH its thread

Partial-MIME updates regenerate the Message-ID and Gmail fragments the thread. Prefer
delete + recreate with the full correct header set, then re-verify.

## 5. User's hard rule after this saga

"Use the GWS client only to create drafts, nothing else." For ndr@draas.com that means
build_service('gmail','v1') raw-MIME drafts with threading headers, then verify by
reading the draft back. The user checks the Drafts folder / thread UI himself.