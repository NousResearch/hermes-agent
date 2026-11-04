# Drive links in outbound messages — verify the RECIPIENT can open them

Verified 2026-10-05 (Ranka Iris letters → Piyush). Extends the
"WhatsApp Message + Document Attachment Pattern" section in SKILL.md.

## The gap

A deep link carries the **URL, not access**. Files created in the Drive **`TMP`** folder — the
default destination for every new artifact under NDR's "no new folders" rule — are owned by
`ndr@draas.com` with **no other permissions**. The recipient taps the link and hits
*"You need access"*. The message is then worse than useless: NDR has to send a second one to fix it.

The link resolving fine *for NDR* proves nothing. He is the owner.

## Check before shipping

```python
gws_drive_get(file_id)   # read the `permissions` array
```

If the only entry is `ndr@draas.com` as `owner`, the recipient **cannot** open it.

## Then what

1. **Say so in the reply**, as an explicit blocker on the message you just delivered.
2. **Offer the one-word fix** — `gws_drive_permissions_add(file_id, email, role)` with reader or
   writer. NDR's call which.
3. **Do NOT grant it unasked.** This is the line that separates the two artifact classes:
   - **Inert, ship immediately:** the WhatsApp deep link, the Gmail draft. These transmit nothing
     on their own (`confirm-before-actions`).
   - **Real side effect, needs go-ahead:** a Drive permission grant. It exposes a document to a
     third party, is visible in his Drive, and is not undone by him simply not tapping a link.

## Worked case — Ranka Iris letters to Piyush (2026-10-05)

Two letters were staged in TMP, renamed, and their links were to go into a WhatsApp message to
Piyush. `gws_drive_get` on both returned:

```
"permissions": [{"type": "user", "role": "owner", "emailAddress": "ndr@draas.com"}]
```

The link was delivered, with the blocker stated separately at the end:

> *"Both letter files currently carry no permission other than you. They are owned by
> ndr@draas.com with no other grants, so the two Drive links will show Piyush 'You need access'.
> Say the word and I'll grant him access to both — reader or writer, your call."*

## Note on split messages

When the message body makes the deep link exceed 4096 chars, `whatsapp_link` returns `split: true`
with a `parts` array — **one complete link per part**. Deliver each part as its own separate
message, in order. Never concatenate them (the platform splitter cuts a link in half) and never
paste a single combined body expecting one WhatsApp message.
