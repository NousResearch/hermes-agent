# Work Deliverable Assignment — Detailed Email + WhatsApp Summary + Drive Access

**Trigger:** User assigns 2-3 deliverables to a team member (architect, designer, engineer) and wants a detailed Gmail draft + a short WhatsApp group summary + Drive permissions set up.

**Pattern (verified Jun 2026 — Shinchana/Sinchana Gowda, Riverstone project):**

## Workflow

### 1. Voice Brief → Structured Understanding

The user dictates a complex assignment via voice. Before drafting anything, confirm you understand:
- Who is the primary assignee? (email address)
- Who should be CC'd? (supervisor, related team members)
- What are the 2-3 deliverables — each with specific, actionable instructions?
- Where should they get data/reference from? (specific people, existing docs)
- What Drive collateral do they need access to?
- Access permission durations (viewer / editor, 1 week / 1 month)

### 2. Research Before Drafting

- Search Gmail for the assignee's email address (may not have been used yet — sgouda@draas.com returned 0 Gmail results because it was a new account)
- Search Drive for all project-related folders and files
- List all relevant Drive items with their IDs, names, and owners
- Note which files are owned by you (you can share) vs owned by third parties (sharing may fail with 403)

### 3. Detailed Email (Gmail Draft)

Structure the email as follows:

**Subject:** `[Project] — [Deliverable 1] & [Deliverable 2]`

**Body:**
```
Hi [Name],

Re: [Project] — [Brief context, 1-2 sentences]

Two clear deliverables for you:

1️⃣ DELIVERABLE 1 — [Title]
- Specific instructions (what, where, how)
- Reference people to coordinate with
- Design constraints / requirements
- Expected output format

2️⃣ DELIVERABLE 2 — [Title]
- Specific instructions
- Reference materials
- Constraints

—

ACCESS TO PROJECT COLLATERALS

I have granted you the following access:

📁 [Folder Name] — [Link]
   → Description of contents
📁 [Folder Name] — [Link]
   → Description of contents
📊 [Sheet Name] — [Link]
📄 [Document Name] — [Link]

👉 Access: Viewer (1wk) on reference folders, Editor (1mo) on working folders.

Please prioritise the deliverables above. [Any follow-up steps: hand-drawn sketch to follow, etc.]

Thanks,
[Name]
```

Save as Draft (do NOT send) so the user can review and correct.

### 4. Drive Permissions

- Try sharing with `sendNotificationEmail=True` first (required when the email is not a Google Account)
- If sharing fails with 400 "no Google Account associated", the email may not be set up as a Google Workspace user yet. Notify the user and add the links in the email body instead of relying on Drive notification
- Viewer (role='reader') for reference materials
- Writer (role='writer') for working folders where they need to upload deliverables
- Use expirationTime for time-limited access

Common failure: Sharing a folder/file owned by a **third party** (e.g. findingform.design) raises 403 — you can only set permissions on files/folders you own. Note this in the email and tell the user to manually share those with the assignee.

### 5. WhatsApp Group Summary

After the email is drafted, create a SHORT WhatsApp message for the group where the supervisor is also present.

Structure:
```
[Name] — [number] deliverables for [Project]:

1️⃣ DELIVERABLE 1 — [one-line summary]
[Key action items, 1-2 bullets]

2️⃣ DELIVERABLE 2 — [one-line summary]
[Key action items, 1-2 bullets]

Detailed email sent to [assignee@email.com] (CC [supervisor@email.com]) with all Drive links and access. Check inbox for folder structure and collaterals.
```

Deliver as markdown code block for copy-paste into WhatsApp group.

### 6. Email Attachment → Drive Upload Workflow

**Trigger:** A team member (e.g. Bhuvanesh) sends an email with an attachment (DWG, PDF, etc.) that needs to go into the project Drive folder.

**Workflow:**

1. **Find the email** — Search Gmail for the sender's email + project name. Get latest message by date.
2. **Extract attachment** — Use `format='raw'`, decode with `base64.urlsafe_b64decode`, parse with `email.message_from_bytes(raw, policy=policy.default)`. Walk parts, find the one with `get_filename()` matching the expected extension.
3. **Save locally** — Write bytes to a temp path (e.g. `/data/hermes/hermes-data/users/ndr/`).
4. **Upload to Drive** — Identify the correct folder:
   - For Riverstone DWGs: `Riverstone > Architectural > Autocad` folder (`1hUGJMyI2h8PFzWZrzYkkRUrQguPTKqRF`)
   - Upload using `drive.files().create()` with `MediaFileUpload`
   - Even if the folder is owned by a third party (findingform.design), you CAN upload files into it (create operation works). You just can't set permissions on the folder itself.
5. **Grant access** — Share the uploaded file (not the folder) with the assignee. You own the file you uploaded, so you CAN set permissions on it.
6. **Report back** — Tell the user:
   - Which folder it's filed in (full path)
   - The Drive link
   - That access has been given (do NOT mention expiry)

### 7. WhatsApp Group Summary — No Phone Number, No Expiry

**Trigger:** After drafting a detailed email with Drive access, the user wants a WhatsApp message for a common architectural/team group (not a 1:1 chat).

**Rules:**
- Do NOT include a phone number in the URL — use `api.whatsapp.com/send?text=...` **without** the `phone=` parameter. This lets the user select the group manually when WhatsApp opens.
- Do NOT mention access expiry/duration (e.g. "1 month editor access", "1 week viewer access"). Just say "access has been given".
- Do include the Drive folder path and direct file link.
- Keep it concise — the email has the detail; the WhatsApp is a heads-up.

**Structure:**
```
[Name] — [one-line context: who shared what].

Filed in: 📁 [Full Drive Path]
[Drive Link]

[Action item: what to do with the file]

Access has been given to work from here. Coordinate with [person] if needed.
```

**Delivery format:** Call the `whatsapp_link` TOOL with the phone parameter **omitted** — it returns `api.whatsapp.com/send?text=…` with no `phone=` so WhatsApp opens the contact/group picker. NEVER hand-encode the URL (hard rule: whatsapp_link is the only sanctioned way to produce WhatsApp deep links; manual construction corrupts encoding on mobile). User pushback precedent (2026-09-29): agent initially said "I can't generate a group link" — wrong; the tool supports omitting phone. When delivering on Telegram, send the `display_link` value. The user taps it, selects the group, and sends.

### Pitfalls

- **New email = sharing fails.** If the assignee's DRA email doesn't have a Google Account associated yet, sharing via Drive API with `sendNotificationEmail=False` raises 400. Use `sendNotificationEmail=True` or include Drive links in the email body as fallback.
- **Third-party owned files can't be shared.** A project folder owned by a third party (e.g. findingform.design) — you can't set permissions on it. However, files you upload INTO that folder ARE yours, so you CAN share those individual files. Note folder-level sharing gaps in the email and ask the user to share the folder manually.
- **Empty subfolders.** Some subfolders (Autocad, Revit, Sketchup, Structural) may be empty. Still list the folder so the assignee knows where to find or upload files when they're populated.
- **Email not in Gmail = not a sent email.** If an address returns 0 Gmail results, the user has never emailed it. It's still the right address; mention to the user that this may be a new/unused Google Workspace account.
- **Verify email spelling from user's voice vs actual email.** The user said "sgouda at dras.com" but the actual email used in correspondence was sgowda@draas.com. Always cross-check voice-dictated email addresses against actual correspondence or CC headers before using.
- **WhatsApp group links: no phone, no expiry.** When the user wants a message for a group, generate `api.whatsapp.com/send?text=` WITHOUT phone parameter — via the `whatsapp_link` tool (omit phone), never by hand. Never mention access duration in WhatsApp messages to teammates — just "access has been given".
