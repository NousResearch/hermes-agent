# Gmail Bulk Deletion / Cleanup — working recipe

Verified 2026-09-10 on a 154-message Drive ownership-notification sweep in ndr@draas.com; extended 2026-09-23 with the two-sender matrix + 626-message full sweep.

## Use case
User wants "delete all those emails" of a category (Drive share/ownership notifications, vendor alerts, etc.) in a specific account (resolve via `gws_resolve_account` first; e.g. google-draas = ndr@draas.com).

## The TWO Drive notification senders — sweep BOTH (NDR preference, 2026-09-23)

Google Drive notifications arrive from TWO distinct sender addresses, and NDR's standing instruction is to treat them as ONE category:

| Sender | Subject patterns |
|---|---|
| `drive-shares-dm-noreply@google.com` | "Item shared with you: '…'", "Spreadsheet/Document shared with you", "Share request for '…'" |
| `drive-shares-noreply@google.com` | "Invitation to edit / collaborate / view", "Request for access", "You're now the owner of '…'" |

- **NDR rule (verbatim 2026-09-23):** "we can delete all drive share notifications period. Whether from Bharat, Sarthak, doesn't matter. And all share request notifications as well. Wherever notification has come, someone is requesting some share requests from me. Delete those as well." → when he asks to clear drive notifications, do NOT bucket by sender and keep any; delete BOTH senders, including share-request/invitation emails. No confirmation needed per sender once he's said "delete all."
- **Account scope:** he names the account (e.g. "Draas.com account"). Sweep only that account; explicitly REPORT the counts still sitting in the other vault accounts (google-gmail / google-ahfl) and offer to sweep those too — do not silently cross account boundaries.
- Classification before deleting is still worth one sample pass (bucket subjects into SHARE_REQUEST / SHARED_WITH_YOU / OWNERSHIP / INVITATION to say what was removed with confidence), but it is diagnostic, not a filter.

## Search pattern for Drive notifications
- Combined query: `from:drive-shares-dm-noreply@google.com OR from:drive-shares-noreply@google.com` — paginate with `nextPageToken` (do NOT cap at one page).
- Per-sender queries for verification after the sweep: run each separately and assert 0 remaining in each.
- Display-name search does NOT work reliably: the 'BharatSales1.blr' voice reference returned 0 hits; the real From header is `"BHARAT H (via Google Drive)" <drive-shares-dm-noreply@google.com>` — `sales1.blr@draas.com` is the sender's account. Always classify by reading the `From` header per message.

## ⚠️ Batch API silently partial-fails — use per-item calls
```python
# ❌ UNRELIABLE — execute() can succeed while most items fail silently
b = gmail.new_batch_http_request()
for mid in ids: b.add(gmail.users().messages().trash(userId='me', id=mid))
b.execute()   # reported OK but only ~40/154 landed in TRASH

# ✅ Reliable
ok, failed = 0, []
for mid in ids:
    try:
        gmail.users().messages().trash(userId='me', id=mid).execute()
        ok += 1
    except Exception as e:
        failed.append((mid, str(e)))
```
This applies to any destructive/modify sweep (trash, delete, modify labels). Batch is fine for reads.

## Scale: 600+ message sweeps need background execution (~7 min)
Per-item `trash()` runs ~0.5–1 s per message; a 626-message sweep ≈ 6–7 min — too long for a foreground terminal call. Pattern that worked 2026-09-23:
1. Write the sweep to `/tmp/drive_share_sweep.py` with `write_file` (assert `profile['emailAddress'] == <intended>`, collect paginated ids, per-item trash loop with progress every 100).
2. Run via `terminal(background=True, notify_on_complete=True)` — the completion notification arrives with the final `DONE trashed: N failed: 0` line.
3. IMPORTANT: the pipe `| grep -v canonical_uid` delays/buffers output — the background log shows nothing until the process exits; don't interpret empty `poll()` as a hang.
4. The sweep is idempotent: a foreground run interrupted mid-way (SIGINT) leaves some messages trashed; re-running the same script re-lists only the remainder and finishes the job — no need to count or repair.

## Delete vs trash
- Gmail UI "delete" = move to TRASH (auto-purges after 30 days, recoverable). Use `messages().trash()` unless the user explicitly wants permanent `delete()`.
- Default Gmail search (`users().messages().list` with q) EXCLUDES Trash and Spam. To verify the sweep: query `in:trash <same q>` and `in:inbox <same q>` separately.

## Full sweep skeleton
1. `gws_resolve_account()` → pick service_name.
2. Paginated `list(q=...)` collecting FULL message ids (never truncate; ids are 16 hex chars — truncated ids 404 later).
3. `get(format='metadata')` per id, parse `From` header, bucket by sender.
4. Per-item `trash()` loop on the intended bucket with progress prints every 25.
5. Verify: `in:trash` count == expected, `in:inbox` count == unintended remainder.
6. Report honestly: trashed count vs remaining-from-other-senders (offer, don't assume, when the sweep crosses sender boundaries).

## Voice-transcription mappings encountered (NDR)
- "BharatSales1.blr" → BHARAT H, sales1.blr@draas.com (DRAAS sales colleague)
- "Goodridge team" → Godrej Ventures (godrejventure.com) team on BuxRanka Hudson Circle project ("4.5 FAR" / "Pakistanka 4.5 Fer" → Hudson Circle 4.5 FAR drawings)
- "Simran" → Harsimran Singh (GFM), harsimran.singh@godrejventure.com — NOT in the contacts sheet's email field; verify from email threads when needed