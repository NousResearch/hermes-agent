# Forward draft with attachments relayed from MULTIPLE source emails + Gmail 25 MB cap split

Worked 22-Sep-2026: NDR dictated "forward AR Portable Cabin's latest email as-is to
Sinchana (cc Neha, Gowri), with the original render + spec PDF we sent them, their
quotation + brochure + photos/videos — all of it." Result: ONE new draft (fresh
thread — Sinchana was never on the vendor thread) carrying attachments sourced from
THREE different messages, with the oversized media Drive-linked instead of attached.

## Pattern

1. **Locate the thread** (`q='"AR Portable" OR "portable cabin"'` style search). Walk
   every message with `format='full'` and enumerate `payload.parts[].filename` +
   `body.size` per message — this gives the full attachment inventory AND their sizes
   in one pass. Big videos often arrive as **Drive links in the body text**, not
   attachments — read the plain-text body for `drive.google.com/file/...` links.
2. **Resolve CC recipients from Gmail headers, not the contacts sheet.** In-house
   people (Sinchana, Neha, Gowri) are not in the sheet; search `q='sgowda'` /
   `q='neha'` / `q='Gowri Singh'` and read From/To/Cc headers. (2026-09-22 verified:
   Neha = nvaddadi@draas.com, Gowri = Gowri Singh gsingh@draas.com — see SKILL.md
   canonical-resolution bullet.)
3. **Sum the payload first.** If every intended attachment was attached, total was
   ~33 MB (3× 7-MB WhatsApp JPGs + 6.8-MB brochure + 2.3-MB render + rest) — over
   Gmail's 25 MB send cap. Split rule that worked:
   - **Attach** everything ≤ ~10 MB total that the recipients need inline (render
     PNG, spec PDF, quote PDF, brochure, small/labelled photos).
   - **Upload the oversized media to a Drive folder** (project-parented, e.g. under
     the Ranka Udaya project root), share it with all recipients BEFORE drafting
     (writer for the doer, reader for mere Cc-ees), and reference the folder + file
     links in the body.
   - Videos that were Drive links on the source thread: try downloading them with
     the user's own token (`files().get_media`) — if shared to the user they come
     down (383 MB of mp4s did). Mirror them to the project folder so the internal
     team is not dependent on the vendor's share permissions.
4. **Body = brief on top + faithful forwarded block below.** HTML body with an
   `<hr>` + "---------- Forwarded message ----------" block quoting the vendor's
   latest email verbatim (From/Date/Subject/To/Cc + body incl. the original Drive
   links). Keep the user's dictated requirements as a numbered list.
5. **Build MIME** (EmailMessage, `set_content` plain + `add_alternative(html)`,
   `add_attachment` per file) → `drafts().create` via raw Gmail API. Fresh thread —
   no threadId, no In-Reply-To.
6. **Verify** with `drafts().get(format='full')`: From/To/Cc/subject + walk payload
   counting attachment parts (metadata format reports 0 attachments).

## Pitfalls

- **⚠️ Forward the message that CARRIES the attachment, not the latest message in the thread
  (verified 2026-10-14).** NDR asked to forward *"the latest latest email from Sanjay Sethiya
  regarding the GPA"*. Sanjay's genuinely-latest message in the thread had **zero attachments** —
  the GPA sat on an earlier message of the same day (18:32 carried the docs; 20:17 carried only
  signature instructions). Enumerate `payload.parts[].filename` + `body.size` across **every**
  message in the thread with `format='full'` and pick the carrier. Then forward THAT message and
  quote the later instruction-only message as a body block, so nothing is lost. State the choice
  in your report ("his latest email has no attachment; the GPA came in the 18:32 one") so NDR can
  correct you if he meant a different message.
- **Prove which version of a document is live before attaching it — diff the files, don't trust the
  filename.** On this matter two files both named "…GPA… Final.docx" existed (02-Oct vs 07-Oct) and
  only one was correct. `zipfile.ZipFile(p).read("word/document.xml")` → replace `</w:p>` with
  newlines → `re.sub(r"<[^>]+>", "", xml)` → compare the party/signatory block. Cheap, no
  python-docx, and it converts "I think this is the right one" into a verified statement. When a
  superseded version exists, SAY SO in the body (NDR explicitly wanted the recipient told that the
  earlier printed set was the wrong version).
- **Don't attach everything just because it fits per-file** — the cap is per-message
  (~25 MB raw MIME). Always sum first.
- **sibling-subagent write_file warning**: when a /tmp script is written while
  another session also writes the same path, read the resolved file before trusting
  it; here the only casualty was a stray `PYEOF` heredoc marker appended to a script
  → `NameError: name 'PYEOF' is not defined` AFTER the real work had succeeded.
  The create/move operations had already completed; verify Drive state, don't rerun
  blindly and duplicate.
- **Drive export API is for Docs-Editor files only** — a `.html` research doc cannot
  be exported via `files().export` (403 "Export only supports Docs Editors files");
  use `files().get_media` and strip tags with regex.
- The video-upload of ~383 MB runs long: run in background with `notify_on_complete`
  and continue drafting meanwhile; the draft links are valid even while the upload
  is in flight (fill-in completes shortly after).