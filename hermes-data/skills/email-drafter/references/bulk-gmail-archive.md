# Bulk-Archive Gmail by Sender/Subject (archived but searchable)

Class: a sender (or a CC/BCC blast) floods NDR's inbox; he wants the emails OUT of the primary inbox but still findable in All Mail. Archive = remove INBOX (+ UNREAD) — never trash/deletes.

Validated 2026-10-12: 237 emails from Prakash Singh's "Ranka Udaya, Hosur - Rental Income Homes | Phase 1 Units Now Open" campaign, archived in one pass, 0 left in inbox, all still searchable.

## Recipe
```python
svc = build_service("gmail", "v1", service_name="google-draas")

# 1. enumerate ALL matching ids (native gws_gmail_search caps at 100; API paginates)
ids = []
tok = None
while True:
    r = svc.users().messages().list(
        userId="me", q='from:psingh@draas.com subject:"Ranka Udaya, Hosur - Rental Income Homes"',
        maxResults=500, pageToken=tok, fields="messages(id),nextPageToken,resultSizeEstimate").execute()
    ids += [m["id"] for m in r.get("messages", [])]
    tok = r.get("nextPageToken")
    if not tok: break

# 2. archive + mark read (one batchModify; max ~1000 ids)
for i in range(0, len(ids), 900):
    svc.users().messages().batchModify(userId="me",
        body={"ids": ids[i:i+900], "removeLabelIds": ["INBOX", "UNREAD"]}).execute()
```

## Pitfalls (all hit 2026-10-12)
- **No `messages().batchGet`** in this googleapiclient build → don't try to pre-inspect labels in bulk. Removing a label a message doesn't have is a silent no-op, so archive blindly and verify after.
- **`fields="messages(id),nextPageResultToken,..."` → HttpError 400 "Invalid field selection nextPageResultToken".** Use only `nextPageToken` (+ `messages(id)`, `resultSizeEstimate`).
- **`resultSizeEstimate` is an ESTIMATE** — re-paginate for an exact count (same query reported 237 exact vs 201 estimate for `in:all`). Never report estimate as the count.
- **Verify properly:** (a) recount by paginating the same query → exact total; (b) re-run query + ` in:inbox` → must be 0; (c) sample `messages().get(fields="labelIds")` → expect only `CATEGORY_*` labels left, no INBOX/UNREAD.
- Searchability: archived = All Mail; any future gmail search finds them. Do NOT trash, or the user loses them from search.
- Scope discipline: filter as precisely as possible (from + subject). Don't archive the sender's other mail unless asked. Report that the batch was sender/subject-scoped and offer a repeat or an auto-skip rule for future blasts.