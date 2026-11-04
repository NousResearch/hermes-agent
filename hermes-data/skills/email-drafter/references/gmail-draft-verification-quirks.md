# Gmail Draft Verification — API Quirks

Verified Sep 2026 while verifying threaded Raghu Iyer drafts (Ranka Amber SSA).

## Quirk: `drafts().list()` may return 0 even when the draft exists (2026-09-21)

Verifying a freshly created `draft_create` (bridge) with `drafts().list()` can return an
EMPTY list (`"total drafts listed: 0"`) even though the draft was created successfully
and is visible in Gmail. This session: bridge returned `draft_id r-8897618152847737016`,
`drafts().list(q='subject:...')` AND `drafts().list(maxResults=25)` both → 0, yet
`messages().list(q='subject:"Reversal" subject:"Prakash"')` returned the message with
`labelIds: ['DRAFT']` — a live, correctly-threaded draft.

**Verification fallback when `drafts().list` looks empty:** do NOT conclude the draft
failed. Search messages by subject (`messages().list(userId='me', q='subject:<keyword>')`)
and check the returned message's `labelIds` contains `'DRAFT'`. That is just as
authoritative as listing drafts and survives the drafts-list quirk. Then pull
`format='full'` to confirm From/To/Cc/Subject before reporting.

## Quirk: `drafts().get()` does NOT accept `metadataHeaders`

When verifying a draft's To/Cc/Subject/thread/In-Reply-To headers:

```python
# BROKEN — TypeError: Got an unexpected keyword argument metadataHeaders
d = svc.users().drafts().get(userId='me', id=draft_id, format='metadata',
    metadataHeaders=['To','Cc','Subject','In-Reply-To','References']).execute()
```

`metadataHeaders` is only valid on **`messages().get()`**, NOT on `drafts().get()`.
For drafts, use `format='full'` and read headers from `d['message']['payload']['headers']`:

```python
d = svc.users().drafts().get(userId='me', id=draft_id, format='full').execute()
h = {x['name']: x['value'] for x in d['message']['payload']['headers']}
print('thread:', d['message']['threadId'], '| To:', h.get('To'), '| Cc:', h.get('Cc'),
      '| Subject:', h.get('Subject'), '| In-Reply-To:', h.get('In-Reply-To'))
assert 'DRAFT' in d['message'].get('labelIds', [])
```

## Verification checklist for threaded reply drafts

- `threadId` must equal the source thread (e.g. `19ed0777ef8c10f9`).
- `In-Reply-To` must equal the source message's `Message-ID` header (proves threading; see skill's "Threaded reply — last message is from you" section for the anchor rule).
- `References` header present on the draft.
- To/Cc match the expected recipients; subject carries the `Re:` prefix.

## Draft storage note

Drafts created via `gws_skill_bridge` (`draft_reply_create` / `draft_create`) return `{"status": "draft_created", "draft_id": ..., "message_id": ..., "threadId": ...}` — capture `draft_id` (or `message_id`) for the verification call above, then report the draft_id to the user so they can find it in Gmail Drafts.