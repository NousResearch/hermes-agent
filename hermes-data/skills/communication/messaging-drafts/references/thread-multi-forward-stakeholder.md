# Thread-based multi-forward to a stakeholder (per-message notes + preserved attachment)

Class of request: *"share both my emails / forward that email to <stakeholder>, telling them <status>, and also forward <other party>'s reply and say <status2>."* NDR wants two-or-more messages FROM AN EXISTING THREAD re-expedited to a third party (often a deal stakeholder who is NOT on the thread — e.g. Amit Saraf / Godrej Venture on the BuxRanka Hudson pre-DCR thread), each with its own short status note, with the original attachment preserved.

Verified 2026-09-29 (P-147 BuxRanka Hudson pre-DCR thread `1a08ea566eef0676`): two drafts created to `Amit Saraf <amit.saraf@godrejventure.com>`.

## Steps

1. **Resolve the stakeholder from the deal context, not the contact sheet.** On the Godrej/BuxRanka threads "Amit" = **Amit Saraf** (amit.saraf@godrejventure.com) per the BRDPL Godrej Venture map (confirm-before-actions → references/brdpl-godrejventure-contact-map.md; voice "Sharaaf"). `contact_resolver` alone may return an unrelated "Amit" (Far Vision ERP etc.) — the deal map wins. Flag the resolution in the reply.

2. **Identify each source message by ID** from the thread (`gws_gmail_thread_get`), and pull any attachment bytes the forward must carry: `messages().attachments().get(userId, messageId, id=attachmentId)` then `base64.urlsafe_b64decode`. Save to /tmp and re-attach — attachment_id alone does not survive a new draft.

3. **Build each draft as raw MIME via `build_service("gmail","v1", service_name=...)`** — `gws_gmail_draft_create` has NO attachment parameter, so use `EmailMessage`, `msg.set_content(note + "\n\n---------- Forwarded message ----------\n")`, `msg.add_attachment(...)` for the PDF, then `drafts().create(userId="me", body={"message": {"raw": raw, "threadId": THREAD}})`. Setting `threadId` keeps the forward on the original thread so the trail stays together. (Alternative if the script approach is preferred: email-drafter → scripts/create-draft-with-attachments.py.)

4. **To / Cc shape:** To = the stakeholder; Cc = the parties already on the thread (so they see the status shared upward) — this matched "marking Manju and Harsimran". If NDR only wants the stakeholder to see it, drop the Cc; ask or state the choice made.

5. **Per-message notes carry the STATUS LADDER, not just the send.** NDR's dictation usually contains the status: "I had reminded both — Manju to share proforma, Harsimran to share DWG" (past reminder) vs "Manju replied with the proforma; Harsimran still has to share the drawing" (current state). Weave both the reminder and the current status into the two notes so the stakeholder gets the full picture in one glance.

6. **Never mark these as sent.** Draft-only, same turn. Verify To/Cc/Subject/attachment/threadId by reading back with `drafts().get`.

## Verification
Read back each draft: To, Cc, Subject, threadId == original, `labelIds == ['DRAFT']`, attachment listed with size. Deliver the `https://mail.google.com/mail/u/0/#inbox/<draft_id>` links to NDR.

## Pitfall — say what each party actually delivered BEFORE drafting the forward
The status notes must reflect the thread's true state, not NDR's assumption. Verified case: NDR asked "did Manju and Harsimran share what we asked?" — reading the full thread showed Manju DID reply 20 min later with `bill.pdf` (invoice received), while Harsimran had NOT sent the 4.5 FAR DWG (his last mail was a 22-Sep promise "we will share the drawings today"). The drafts then correctly said "Manju replied with proforma; Harsimran still has to share the drawing". Always re-read the thread tail before writing the notes.