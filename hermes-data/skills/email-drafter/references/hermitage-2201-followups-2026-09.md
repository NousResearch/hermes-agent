# Multi-recipient follow-up with Drive photo link (Prestige Hermitage 2201)

Validated 2026-09-21: NDR uploaded a 32-photo WhatsApp export of unit 2201
after cleanup, wanted it filed under the Prestige Hermitage photos tree, then
wanted follow-up emails sent to everyone who previously received Hermitage
2201 / 22nd-floor rental data — plus 90/180-day viewer shares of the photo
folder for each recipient.

## The flow that worked

1. File the photos first (see draas-drive-organization →
   `references/project-photos-tree-and-permission-expiry.md`): moved
   TMP/"2201 Prestige Hermitage" → Project Photos/Prestige Hermitage, renamed
   `2201 After Cleanup (Sep 2026)` (folder id 1DqjviXTY155vd4NulTAAzcyTOkYJUlKZ).
2. Enumerate who previously received Hermitage/2201 data — NDR asked "give me
   the list and I'll tell you whom all to send this follow-up". Searches that
   surfaced the full set (all inside ndr@draas.com; empty in gmail/ahfl):
   - `Hermitage`, `2201`, `22nd floor`, `"Available Properties"`, `portfolio`,
     `in:sent (hermitage OR 2201 OR prestige) after:2026/07/01`
   - Read the 19-Sep-2026 "DRA Group - Complete Liquidation Inventory" mail:
     its own body names the prior distribution ("separate emails to Century
     Real Estate, Vinod Kumar Das, Rajesh Talreja and Infimax — plus the March
     2026 Available Properties portfolio note"). A consolidation mail is the
     FASTEST map of prior recipients.
   - Resulting recipient map: Vinod Kumar Das (Rahul) vkdas@draas.com (Mar-2026
     portfolio, lists 2201 3BHK 7,481 sft @ INR 8 L/mo), Rajesh Talreja
     rajeshtalreja74@gmail.com (8-Aug units mail, but Century Regalia + Ranka
     Iris only — no Hermitage), Ankush Musaddi ankush@infimaxadvisors.com +
     amusaddi@hotmail.com (Aug-2026 Century Regalia threads), Prakash Singh
     psingh@draas.com (19-Sep liquidation), Bharat Hawaldar sales1.blr@draas.com
     (CC everywhere), Sunny Sadhwani rsadhwani13@gmail.com (NO Hermitage mail
     found in any mailbox → fresh draft instead of reply-all).
3. Grant viewer + expiry BEFORE drafting (share-then-link): reader +
   expirationTime = now + 90 days for rsadhwani13@gmail.com, psingh@draas.com,
   ankush@infimaxadvisors.com, amusaddi@hotmail.com. Internal staff had
   pre-existing writer grants from the upload chain — leave those, or the
   permissions().update to reader 403s (cannotModifyInheritedPermission).
4. Drafts (draft-only, never send):
   - Sunny: NEW email (no existing thread) — "Prestige Hermitage 2201 — Photos
     after Cleanup": cleanup done, bird net being installed on balcony pigeon
     nets, water/all cleaned, one marble-polishing round left, photos link.
   - Prakash: threaded reply-all on the 19-Sep liquidation thread. NDR wording:
     "2201 rentals — here are the latest cleaned-up images, for your reference
     for anybody you are sharing it with from a rental perspective."
   - Ankush: threaded reply on his last active chain (28-Aug Century Regalia
     "Individual Floor Plans & Crissa 401+404 Combined Units"): "just keeping
     you updated, got the unit cleaned, latest photos for your reference in
     case you're looking for a rental for 2201 at Prestige Hermitage."
   - Vinod (Rahul) DONE 2026-09-21: threaded reply on the 26-Mar-2026
     "Available Properties for Sale & Rent - Exclusive DRA Portfolio | Action
     Required" thread (thread 19d2869fa9a7a46c / msg 19d28a08d70045d8) — To
     vkdas@draas.com, Cc Bharat Hawaldar (mirrors the original), reader+90-day
     share granted first. Body mirrors Prakash's: cleanup done, bird net on
     balcony pigeon nets, water/all cleaned, one marble-polishing round left,
     photos link, "share with anyone in your network from a rental
     perspective". Century Real Estate personnel skip.

## Pitfall — message id ≠ thread id (404)

Scanning `messages().list()` gives MESSAGE ids. Passing one directly to
`threads().get(id=<message_id>)` raises
`404 Requested entity was not found` — misleading, looks like a broken thread
but simply is the wrong id type. Correct sequence:
```python
d = svc.users().messages().get(userId='me', id=MSG_ID,
    format='metadata', metadataHeaders=['Subject']).execute()
tid = d['threadId']
t = svc.users().threads().get(userId='me', id=tid, format='metadata').execute()
```
Same for creating the threaded draft: pass the resolved threadId to
`drafts().create(threadId=...)` and thread via MIME headers
(In-Reply-To/References from the last message's raw Message-ID).

## Email-share scope rule

"Give them 90-day viewer access … or 180-day … on this particular images
folder only" — share ONLY the specific photo folder, never its ancestors
(which contain unrelated project docs). 90 days was taken as the default;
180 is acceptable when NDR says so.

## Inbound share-requests on this folder (validated 2026-09-22)

NDR dictation: "share requests for 20201 after cleanup — grant Yogesh S
30-day viewer access, email him the folder link confirming, then delete the
two share request emails."

- **Voice garble: "20201" = the "2201 After Cleanup (Sep 2026)" folder**
  (Hermitage 2201 photos, id 1DqjviXTY155vd4NulTAAzcyTOkYJUlKZ), not a
  new/20201-named folder.
- The request arrives as Google Drive notification mails
  (`drive-shares-dm-noreply@google.com`, subject `Share request for '2201
  After Cleanup (Sep 2026)'`, each its OWN thread). Users who click the
  request twice can emit TWO notifications with different roles — observed
  one asking **reader** (06:32) and one asking **writer** (06:33).
- **Requester address is authoritative from the notification itself —
  NOT contact_resolver.** "Yogesh Infimax" resolves to *Yogesh EH*
  (Yogeshbk96@gmail.com, sheet row 4171) — a different person. The actual
  requester was **Konjengbam Yogesh Singh <yogesh@infimaxadvisors.com>**
  (Infimax Advisors — same firm as Ankush Musaddi). Email a stranger if you
  trust the sheet match; the Google-verified `userstoinvite` address in the
  notification body is the correct recipient even though it has no
  contacts-sheet row.
- Flow that worked (ndr@draas.com, google-draas, raw API):
  1. Grant `drive.permissions().create(fileId=FOLDER, body={"type":"user",
     "role":"reader","emailAddress":YOGESH,
     "expirationTime": now_utc+30d}, sendNotificationEmail=False)` — NDR
     said 30 days here, not the 90-day default used for the original blast.
  2. Draft a NEW email to the requester (draft-only, never send):
     confirm viewer access granted for 30 days + folder link
     `https://drive.google.com/drive/folders/...` + "access will expire
     automatically after 30 days".
  3. Trash the share-request notification messages
     (`users().messages().trash()`) — NDR asks for this deletion explicitly;
     Trash keeps them recoverable.
  4. Verify: re-list folder permissions (grant must show
     emailAddress/role/expirationTime) + `drafts().list()` shows the draft
     with label DRAFT.