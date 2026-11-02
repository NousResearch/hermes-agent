# Gmail message links that actually open

**Symptom:** NDR says "the links don't seem to be working … those don't appear to be the
links". You hand him `https://mail.google.com/mail/u/ndr@draas.com/#all/19039a2af183dadc`
and he lands on nothing, or on the wrong message.

He is right. Both halves of that URL are wrong for a cold click.

## The two defects

**1. The account path.** `/mail/u/<email>/` does not resolve. The account segment is a
**numeric index**, `/mail/u/0/`. Whatever account the email belongs to, the clickable form
is the digit.

**2. `#all/<hex>` is an unofficial alias.** It resolves only inside a Gmail tab that is
already open and already knows the message — which is why it "works" when you test it in a
live session and dies for the user. Never ship it.

## The ID you cannot reproduce

When NDR pastes the link he copied from his address bar, it looks like:

```
https://mail.google.com/mail/u/0/#search/Re%3A+Addendum+to+JDA+%E2%80%93+Allasandra+Northstar/QgrcJHsBqLKxJQWcVCMXckXcwVdnbNSHBXb
```

That trailing `QgrcJ…` token is a **per-account, server-side ciphertext** of the thread.
The Gmail API never returns it and it cannot be derived offline. Do not promise to rebuild
"the same link he has". Say plainly that the long ID is server-side and offer the
API-derivable form instead.

## The one durable permalink you CAN build

An `rfc822msgid:` search on the message's exact `Message-ID` header. It is stable, requires
no opaque ID, and works cold.

```python
hdr = {h['name'].lower(): h['value'] for h in msg['payload']['headers']}
mid = hdr['message-id']                       # <CAPDkZRa...@mail.gmail.com>
import urllib.parse
q = urllib.parse.quote(f'rfc822msgid:{mid}', safe='')
url = f'https://mail.google.com/mail/u/0/#search/{q}'
```

`#search/` also tolerates a plain subject search — `/mail/u/0/#search/<url-encoded subject>`
— which is a good fallback when a message has no usable `Message-ID` (rare).

## Verify every link before sending it

A search-based link is only a permalink if the query is unique. Check each one:

```
gmail_search(query=f'rfc822msgid:{mid}')  →  must return exactly 1 result
```

If a query returns 2+, qualify it (`rfc822msgid:<mid> from:...`) or fall back to the subject
form with a date qualifier. Never hand over an unverified search link — a multi-hit search
looks identical to a correct one until the user clicks it.

Also confirm the message actually exists in the mailbox you are linking to before citing it.
Cross-mailbox confusion is common here: the June 2024 ledger mail went out from
`ndr@drahomes.in`, not `ndr@draas.com`, but it is visible in the `ndr@draas.com` mailbox.

## Delivery shape

Send **plain URLs, one per line, each on its own line** — not markdown link syntax and not
bare URLs inside a sentence, which Telegram re-wraps and truncates mid-token. Put the
date + subject on the line above each URL so a failure is self-diagnosing ("the 17 Oct one
didn't open" rather than "number 7").

Lead with the fix, not the apology: state what was wrong (account path, unofficial alias),
what you now use (rfc822msgid search), and that each was verified to return a single
message.
