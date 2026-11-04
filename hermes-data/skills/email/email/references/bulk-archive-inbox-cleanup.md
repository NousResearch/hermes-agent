# Bulk Archive — clear the inbox, keep messages searchable (Gmail)

For "these emails flood my inbox but I want them available when I search" — typically a
colleague's/partner's marketing or CC blast (worked example 2026-10-12: 237 messages of
Prakash Singh's "Ranka Udaya, Hosur - Rental Income Homes | Phase 1 Units Now Open" campaign
CC/BCC'd to NDR on the same morning). Differs from the trash sweeps in `gmail-bulk-delete.md`:
**archive = remove the INBOX label, keep the message in All Mail** (still searchable), and remove
UNREAD so the badge clears. Never trash unless the user says delete.

## The pattern (verified 2026-10-12)

```python
QUERY = 'from:psingh@draas.com subject:"Ranka Udaya, Hosur - Rental Income Homes"'
svc = build_service("gmail", "v1", service_name="google-draas")

# 1. Enumerate EVERY match — the search UI/API caps at 100 and client summaries lie.
ids = []
tok = None
while True:
    r = svc.users().messages().list(
        userId="me", q=QUERY, maxResults=500, pageToken=tok,
        fields="messages(id),nextPageToken,resultSizeEstimate",
    ).execute()
    ids += [m["id"] for m in r.get("messages", [])]
    tok = r.get("nextPageToken")
    if not tok:
        break
# worked example: 237 real ids vs 100 shown per query

# 2. Archive + mark read in batches (batchModify accepts up to 1000 ids).
for i in range(0, len(ids), 900):
    svc.users().messages().batchModify(
        userId="me", body={"ids": ids[i:i+900], "removeLabelIds": ["INBOX", "UNREAD"]}
    ).execute()

# 3. Verify three ways.
left = svc.users().messages().list(userId="me", q=QUERY + " in:inbox",
                                   maxResults=100, fields="resultSizeEstimate").execute()
#   → 0 remaining in inbox
#   re-run the full enumeration → same exact total (237) — nothing lost
#   sample messages().get(id, fields="labelIds") → e.g. ['CATEGORY_PERSONAL'] only
#   plain query resultSizeEstimate nonzero → still searchable in All Mail
```

## Pitfalls
- `messages().batchGet` does NOT exist on the Resource — enumerate labels with per-id
  `messages().get(fields="id,labelIds")` if you need a label pre-check (usually unnecessary:
  `removeLabelIds` on a message not carrying the label is a no-op).
- The `fields` parameter must list only real fields: `nextPageResultToken` is NOT valid for
  Gmail `messages().list` (400 `Invalid field selection`) — use `nextPageToken`.
- `resultSizeEstimate` is an ESTIMATE and can be stale/imprecise on composed queries
  (`in:all` variants read 201 vs the 237 the full enumeration returns). Trust the paginated
  id count, not the estimate.
- Scope discipline: search by `from:` + distinctive subject phrase; do NOT archive everything
  from a sender who also sends real work mail.
- batchModify is a label operation — no per-item API error swallowing like batch() HTTP
  requests have; still verify by search afterwards.
- Removing UNREAD changes read-state. The user's intent ("clear my inbox", "not flooding") is
  satisfied by doing both; call it out in the reply so it's visible.