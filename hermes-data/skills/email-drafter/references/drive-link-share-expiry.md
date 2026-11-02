# Drive link sharing with 30-day expiry (worked: Palya/BIAAPA email to Arvind, Sep-2026)

NDR's standing rule when an email references Drive files/folders: **every shared
link must get 30-day viewer access** before the draft is created.

## The technique — user-type permission with expirationTime

`type: 'anyone'` (public link) DOES NOT support `expirationTime` — the Drive API
rejects it with:

```
403 Expiration dates cannot be set on domain or anyone type permissions.
```

The working form is a **user-type permission on the recipient's email** with
`expirationTime` set 30 days out:

```python
from datetime import datetime, timedelta, timezone
from tools.gws_auth import build_service

service = build_service("drive", "v3", service_name="google-draas")
expiry = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()

for fid in ["<file_or_folder_id>", ...]:
    service.permissions().create(
        fileId=fid,
        body={
            "type": "user",
            "role": "reader",
            "emailAddress": "<recipient@example.com>",
            "expirationTime": expiry,
        },
        fields="id,type,role,emailAddress,expirationTime",
        sendNotificationEmail=False,   # avoid spamming recipient before email is sent
    ).execute()
```

Recipient emails resolve fine even for non-Google accounts (Yahoo etc.); Drive
sends them a share notification with viewer rights that expire on the date.

## Must do

1. **Share the FOLDER, not just individual files** when the email links a whole
   comp folder — grant on the folder id and every linked file id (folders made
   public to a user propagate viewer access implicitly, but explicit is safer
   for each file link actually pasted in the email).
2. **Run BEFORE creating the draft** — the draft's links must be live when NDR
   reviews it.
3. **Verify** by re-listing permissions and printing
   `emailAddress / role / expirationTime` in the reply as proof.
4. Same service_name as the email account (google-draas for ndr@draas.com).

## Email-drafter integration

The full email flow for a research/competitor email with Drive links:
1. Gather the file/folder ids that appear in the email body.
2. Grant user-reader + expiry (recipe above).
3. Create the draft via `gws_skill_bridge.call("draft_reply_create", ...)` with
   the links in the HTML body.
4. Reply to NDR with the draft link
   (`https://mail.google.com/mail/u/0/#drafts?compose=<draft_id>` or the
   thread link) + the permission-proof table.

## Handling a Drive "share request" email (when NDR says "grant him access per his request")

Google sends **drive-shares-dm-noreply@google.com** notifications when an
outsider clicks "request access" on a folder NDR owns:

- Subject: `Share request for '<Folder Name>'`, and the body carries the
  requester's real email + the folder link + a `role=reader`/`role=writer`
  hint — the requester can submit BOTH a reader and a writer request, so
  there can be **two emails for one person** (e.g. Yogesh, 06:32 reader +
  06:33 writer). Search `from:<name or domain> newer_than:Nd` to find them
  all; the requester's email in the notification is authoritative — verify
  against contacts when possible, but the Google notification wins when the
  sheet has only a homonym (different person).
- Grant what NDR's wording says regardless of the requested role: "30-day
  viewer access" → `type:user, role:reader, expirationTime=+30 days` (recipe
  above; `sendNotificationEmail=False` since NDR emails him the link next).
- Then draft the confirmation email (per his request, link included, expiry
  stated) and **trash both request notifications** via
  `messages().trash()` — recoverable, matches "delete those two emails".
- Voice-garble pitfall: folder numbers mangle in dictation — "20201 after
  cleanup" was really **2201 After Cleanup (Sep 2026)**. Match share-request
  subjects against actual Drive folder names (`files().list(q="name contains
  'NNN'")`) before assuming a folder name.

## Drive share-request notifications — grant + confirm + trash (validated 2026-09-22, "2201 After Cleanup" / Yogesh Singh)

When a third party requests access to a folder ("Share request for 'X'" from
`drive-shares-dm-noreply@google.com`), the flow that worked:

1. **Find all notifications** — one folder often produces TWO share-request
   emails (a reader request and a writer request minutes apart). Search
   `q='from:<name>'` or `q='subject:(share OR shared OR sharing)'`; the folder
   name + link are in the notification body. The requester email INSIDE the
   notification is authoritative — do NOT substitute a same-name contact from
   the sheet (contacts "Yogesh EH" ≠ Konjengbam Yogesh Singh
   yogesh@infimaxadvisors.com, Infimax — same firm as Ankush Musaddi).
2. **Resolve the folder from the notification**, not from the dictated name:
   NDR's "20201" was actually folder **"2201 After Cleanup (Sep 2026)"**
   (Prestige Hermitage unit-2201 photos, id 1DqjviXTY155vd4NulTAAzcyTOkYJUlKZ)
   — search Drive `name contains '2201'` when the dictated number has a digit
   that rings false, and confirm ownership via `files().get(owners)`.
3. **Grant user-reader + 30-day expirationTime on the folder**
   (`type: "user"` — expiry is rejected on `anyone`/domain), with
   `sendNotificationEmail=False`, then draft the confirmation email to the
   requester with the folder link (draft-only, never send). Message: access
   granted per request, valid 30 days, link, auto-expiry note.
4. **Delete = trash the notification emails** — `messages().trash()` on each
   notification id (recoverable; user said "delete"). Verify the grant by
   re-listing permissions and print emailAddress/role/expirationTime.
5. Check existing grants first: internal staff already hold writer on the
   folder from the upload chain — never downgrade/duplicate; just leave them
   and add the new external viewer.