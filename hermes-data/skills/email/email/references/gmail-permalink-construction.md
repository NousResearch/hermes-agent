# Building Gmail permalinks that actually open

Trigger: NDR asks "give me links to those emails", or reports **"the links don't seem to be
working"** after you supplied a list. Verified 22-Oct-2026 on a 16-email Ranka Northstar series
where every link in the first delivery was dead.

## The three hard facts

1. **Gmail API message IDs are 16 hex chars** (`1a1366e2cec0cb1a`). They are **not** web-UI IDs
   and must never be pasted into a `#all/...` or `#inbox/...` fragment as if they were.
2. **Web-UI message IDs are opaque per-account ciphertext** — e.g.
   `QgrcJHsBqLKxJQWcVCMXckXcwVdnbNSHBXb`. The Gmail API **never returns them** and they
   **cannot be derived** from the API ID, the RFC Message-ID, or any local computation. They are
   minted server-side per account per message. If the user pastes one, do not pretend you can
   reproduce that form — say plainly that it is browser-minted and produce the equivalent search
   permalink instead. (The user is fine with this when you explain it; what he is not fine with is
   a link that 404s.)
3. **`#all/<any-id>` is an unofficial alias.** It resolves only when injected into an
   already-open Gmail tab (client-side routing). From a cold click — a Telegram message, a fresh
   tab, a phone — it always fails. This is the exact defect NDR reported.

## The account path

- ❌ `/mail/u/ndr@draas.com/#...` — the `u/` segment takes an **account INDEX**, not an address.
- ✅ `/mail/u/0/#...` — `0` is the first signed-in account.
- If the link opens the wrong mailbox, the index is the variable to change (`1`, `2`). Say this
  when you deliver, so a wrong-mailbox result is a one-character fix, not another round trip.

## The permalink that works

Search on the exact `Message-ID` header, URL-encoded:

```
https://mail.google.com/mail/u/0/#search/rfc822msgid%3A%3C<MESSAGE_ID>%3E
```

Encoding of the Message-ID (it is wrapped in angle brackets):

| char | encode as |
|------|-----------|
| `<`  | `%3C` |
| `>`  | `%3E` |
| `+`  | `%2B` |
| `=`  | `%3D` |
| `@`  | `%40` (optional; bare `@` also works) |

Example (Google-originated):

```
https://mail.google.com/mail/u/0/#search/rfc822msgid%3A%3CCAPDkZRaLen%2Bk7iGO9jFJDZD4_72nRu%3DxxoteReWh2Hbt5Sguow%40mail.gmail.com%3E
```

Example (Outlook-originated — note the `+` and `=` that *must* be encoded):

```
https://mail.google.com/mail/u/0/#search/rfc822msgid%3A%3CKL1PR04MB7014AA269FDD724C3BF9748CFB9E2%40KL1PR04MB7014.apcprd04.prod.outlook.com%3E
```

Encode with `urllib.parse.quote('rfc822msgid:<'+mid+'>', safe='')` — do not hand-type the
escapes.

## Mandatory verification before delivery

A Message-ID search can legitimately return 0 or 2+ hits. Assert exactly one:

```python
q = f'rfc822msgid:<{mid}>'
r = gmail.users().messages().list(userId='me', q=q).execute()
assert r.get('resultSizeEstimate') == 1, (mid, r.get('resultSizeEstimate'))
```

Run that over the whole list in one script and report the count: *"all 16 verified — each query
returns exactly that one message."* That sentence is what makes the second delivery credible after
the first one failed.

## Getting the Message-ID

`gws_gmail_get` does not surface it in its summary. Pull it explicitly:

```python
m = gmail.users().messages().get(userId='me', id=<hex_id>,
        format='metadata', metadataHeaders=['Message-ID']).execute()
mid = [h['value'] for h in m['payload']['headers'] if h['name'] == 'Message-ID'][0]
```

## Also — Gmail is API `v1`, never `v3`

`build_service('gmail', 'v3', service_name=...)` raises
`googleapiclient.errors.UnknownApiNameOrVersion: name: gmail version: v3`. The error reads like
the API does not exist; it is simply the wrong version. **Gmail has only `v1`**
(`drive` is `v3`, `sheets` `v4`, `docs` `v1`, `calendar` `v3`). The static discovery cache ships
`gmail.v1.json`, so `v1` resolves offline while anything else fails.

## Related

- Thread-level (not message-level) link construction, and resolving *which* email the user means:
  `email` → `references/email-forensic-thread-lookup.md`.
- Never claim a link "works" without the `resultSizeEstimate == 1` assertion. A hex ID resolving
  in the API proves the message exists; it proves nothing about the URL.
