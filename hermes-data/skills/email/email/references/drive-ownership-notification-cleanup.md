# Drive Ownership-Transfer Notification Cleanup (Gmail)

When a Google Drive file is transferred to NDR's account, Gmail sends
"You're now the owner of '…'" notifications. They pile up fast when a
colleague (e.g. BHARAT H / sales1.blr@draas.com) bulk-transfers project
files. NDR wants these notifications deleted (Decided Sep 2026: 154
messages trashed in ndr@draas.com).

## How to find them
- Sender address is ALWAYS `drive-shares-noreply@google.com` with a
  display name like `"<Person> (via Google Drive)"`,
  `"<Person> (via Google Sheets)"`, `"<Person> (via Google Docs)"`,
  `"<Person> (via Google My Maps)"`. The person's name lives ONLY in the
  display name, never in the address.
- Searching the person's alias returns 0 hits: `BharatSales` /
  `BharatSales1.blr` → nothing. (Voice-transcribed "BharatSales1.blr" =
  the colleague's mailbox slug `sales1.blr@draas.com`.)
- Reliable query: `from:drive-shares-noreply@google.com "now the owner of"`
- Count with pagination: `messages().list(maxResults=500)` + `nextPageToken` loop.

## Classify by sender
- Metadata-fetch each id with `metadataHeaders=['From']` (~0.5s each),
  bucket by display name: `'BHARAT H' in from_header`.
- Confirm scope before touching senders the user did not name — e.g.
  ownership notifs from other team members (Prakash Singh, Gowri Singh,
  Vinod Kumar Das, Bhagya) may still be wanted as records. Ask.

## Delete
- Use Trash, not permanent delete:
  `users().messages().trash(userId='me', id=...).execute()` — matches
  Gmail UI "delete", recoverable ~30 days.
- Loop per-message, NOT `new_batch_http_request()` — batch silently
  drops per-item failures (see SKILL.md pitfall). ~0.5–1s per call.
- Verify: re-run `in:inbox from:drive-shares-noreply@google.com "now the owner of"`
  and confirm only allowed senders remain.

## Edge cases
- These notifications only announce the transfer — files stay in Drive;
  deleting notifications does not affect ownership/access.
- Default search (API list without `in:anywhere`) excludes Trash, so
  post-trash counts drop immediately — do not mistake that for data loss.