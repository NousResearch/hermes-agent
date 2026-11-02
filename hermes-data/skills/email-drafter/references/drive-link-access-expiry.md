# Drive-link access expiry when included in drafts

When a draft email (or WhatsApp message) shares Google Drive links with an
EXTERNAL recipient (architect, vendor, counsel, colleague outside @draas.com),
NDR's standing default (2026-09-10, Palya/BIAAPA email to architect Arvind Jain)
is: **30-day viewer access on every shared file/folder.**

Mechanics (verified 2026-09-10):
- Google Drive does NOT allow `expirationTime` on "anyone with link" shares —
  always grant a **per-user reader permission** to the recipient's email with
  `expirationTime` ~30 days out:
  ```python
  perm = {'type': 'user', 'role': 'reader', 'emailAddress': 'recipient@example.com',
          'expirationTime': '2026-10-07T00:00:00Z'}
  svc.permissions().create(fileId=FID, body=perm, sendNotificationEmail=False).execute()
  ```
- Verify with `permissions().list` (`expirationTime` present, role reader) — the
  create response sometimes shows `exp=None` even when stored (see
  draas-drive-organization `references/drive-share-request-grant.md`).
- Applies to folders AND files (folder share propagates to contents for viewers).
- State the expiry date in the reply ("viewer access till 7 Oct") so the sender
  knows it will lapse.
- **Non-default durations are fine** — NDR asked for 360 days on the Ranka
  Stelo approvals handover (2026-09-22): `expirationTime` = today + 360 →
  2027-09-17T00:00:00Z, granted on BOTH the folder and the index sheet. Use
  today+360 when he says "360 days" and state the exact date back to him.
- **Index-sheet-inside-shared-folder pattern (2026-09-22):** create the Google
  Sheets index WITH `parents=[folder]` so every folder viewer inherits it;
  fill rows with `=HYPERLINK("https://drive.google.com/file/d/<id>/view","Open")`
  via Sheets `values().update` with valueInputOption=USER_ENTERED (RAW stores
  the formula as literal text); read the values back to verify. If a second
  named recipient only gets the index sheet, still grant an explicit
  `reader` permission on the sheet (the folder grant already covers that
  person if they're the same as the main recipient).
- Full sharing/expiry mechanics live in the `draas-drive-organization` skill;
  this note just connects the email workflow to that default.