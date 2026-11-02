# Oversized forward / reference-pack drafts — split attachments vs Drive links (Gmail 25 MB cap)

Worked 22-Sep-2026: NDR dictated "forward Sayed's (AR Portable Cabin) email as-is to
Sinchana, CC Neha + Gowri, with ALL attachments: executed-work photos, their quotation +
brochure, plus OUR original render PNG and Interior Execution Specification PDF from the
RFP email we sent them earlier in the same thread." Total content ≈ 33 MB — over the 25 MB
Gmail message cap, so attaching everything to one draft fails at send time.

## Scenario shape
- The counterparty's latest email in a thread carries hi-res JPG attachments (~7.5 MB each).
- Videos are often NOT attachments — they are Drive links embedded in the email body text
  (e.g. `Aseets.mp4 https://drive.google.com/file/d/…`). Gmail lists them as link lines.
- The "original email we sent" (render + spec PDF) lives earlier in the SAME thread.
- The user wants a NEW draft to an internal colleague (not a reply on the vendor thread),
  so the vendor thread's own email bodies get re-quoted as a forwarded block.

## Recipe

1. **Enumerate the whole thread** — `threads().get(userId='me', id=THREAD, format='full')`,
   walk `payload.parts[].filename` + `body.attachmentId` per message. Also parse the bodies
   for Drive links (the video files).
2. **Download everything to /tmp** (before deciding the split):
   - Email attachments: `messages().attachments().get(...)` — remember the base64
     `padding = 4 - len(data) % 4` guard.
   - Drive-linked media: `drive.files().get_media(fileId=...)` — works because the
     counterparty shared those files to the SESSION USER's vault account specifically.
     This sidesteps whether cc'd colleagues can open the counterparty's Drive links.
3. **Compute the budget.** Attach inside the cap: essential docs (render, spec PDF, quote
   PDF, brochure PDF) + small images ≈ 10 MB total is comfortable. Mirror to Drive: the
   3× ~7.7 MB hi-res photos AND all videos (190 MB in this case) — anything that would
   push the draft over ~25 MB.
4. **Create a project Drive folder** under the correct project root (e.g. `Ranka Udaya`
   folder), upload the heavy files. Name it with project + purpose
   (`Container Site Office (Prefab) — AR Portable Cabin Refs`).
5. **Role-scoped permissions**: the person who must act on the files = `writer` (design
   redo / renderer), visibility-only cc'd person = `reader`. Verify each
   `permissions().create` returns the role.
6. **Build the draft** (`templates/`/raw EmailMessage → `drafts().create`):
   - To = the doer, Cc = collaborators.
   - Attach the small set; put the Drive folder link (and individual big-file links) in
     the body so nothing is hard to find.
   - Embed the counterparty's ACTUAL email body as a quoted `<blockquote>` block with the
     original Drive links preserved — this is the "forwarded as-is" part.
7. **Verify**: `drafts().get(format='full')` → count `parts[].filename` = expected count;
   confirm `labelIds == ['DRAFT']` via `drafts().list()` (authoritative).

## Notes
- Gmail API draft creation tolerates the draft, but SEND fails over the cap — the draft
  must already be under 25 MB, not "fixed later".
- Re-uploading counterparty Drive media to our own Drive is the reliability move: their
  sharing scope often covers only the sender's recipient, not the new internal Cc list.
- Same video links from the counterparty stay quoted in the forwarded block — both paths
  (original link + our mirror) go out in one mail.
- Sender identity: run with `HERMES_SESSION_USER_ID=7449813913` and assert
  `getProfile()['emailAddress'] == 'ndr@draas.com'` before creating the draft.