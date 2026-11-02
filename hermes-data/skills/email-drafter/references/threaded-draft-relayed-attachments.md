# Threaded reply draft with attachments relayed from another email (raw Gmail API)

Worked 03-Sep-2026: NDR dictated "reply on Dharmesh's thread, address it to Arvind
(cms-induslaw), mark Dharmesh again, attach the succession petition + GPA that
Sanjay (Law Square) sent". Result: a draft on thread `1a0680657add515f`
(Re: Succession Certificate), To Arvind, Cc Dharmesh, with both .docx files
attached — source of the attachments was a DIFFERENT email (Sanjay's), not the
thread being replied to.

## Pattern

1. **Find the reply target** — `threads().get(threadId=..., format='metadata')`
   to confirm participants and subject. When the user says "mark X again", the
   Cc list usually comes from the thread's original headers.
2. **Get the original Message-ID** — `messages().get(id=<msg>, format='metadata',
   metadataHeaders=['Message-ID','References','Subject'])`. Use it for both
   `In-Reply-To` and `References` so the draft threads correctly.
3. **Download attachments from the SOURCE email** (may be a different message):
   `messages().attachments().get(userId='me', messageId=<source_msg>,
   id=<attachmentId>)` → `base64.urlsafe_b64decode(resp['data'])` → save to
   `/tmp/`. (Get attachmentIds by walking `payload.parts[].filename` on a
   `format='full'` fetch of the source.)
4. **Build the MIME** with `email.message.EmailMessage`: set To/Cc/From/Subject,
   `Message-ID=make_msgid()`, `Date=formatdate(localtime=True)`,
   `In-Reply-To=orig`, `References=orig`, `set_content(body)`; then
   `em.add_attachment(bytes, maintype='application',
   subtype='vnd.openxmlformats-officedocument.wordprocessingml.document',
   filename=...)` for each docx.
5. **Create** — `drafts().create(userId='me',
   body={'message': {'raw': base64.urlsafe_b64encode(em.as_bytes()).decode(),
   'threadId': thread_id}})`.
   Pass `HERMES_SESSION_USER_ID` slug override when the session may resolve to
   another vault user; verify with `about().get(fields='user(emailAddress)')`
   first when identity matters.
6. **Verify** — `drafts().get(id=draft_id, format='metadata')`:
   check From/To/Cc/Subject/In-Reply-To/ThreadId. For attachment confirmation use
   `format='full'` and walk the payload for `filename` parts (metadata format
   reports 0 attachments).

## Pitfalls (both hit 03-Sep)

- **`drafts().get` right after `drafts().create` can 404 transiently**
  ("Requested entity was not found"). The create succeeded; the draft just isn't
  visible yet. Fix: `time.sleep(2)` and retry once; if still 404, fall back to
  `drafts().list()` and match by subject. Do NOT recreate the draft — that
  duplicates.
- **Metadata-format drafts report 0 attachments** — attachment count check must
  use `format='full'` payload walking, not metadata.
- **`messages().attachments().get()` IDs are message-scoped** — an attachmentId
  from one message cannot be fetched against a different messageId; keep the
  source message id paired with its attachmentIds.

## Variant: attachment sourced from Google Drive, draft on the SESSION USER'S OWN mailbox

Worked 16-Oct-2026 (Sarthak / `admin3.blr@draas.com`: serve a registered mortgage
deed on the VGRO Finserv DD thread "Re: reg:- information required"). Deltas from
the relayed-email case above:

1. **Attachment source is Drive, not a message.** Download with the sanctioned
   client — never hand-built credentials:
   `build_service("drive","v3", service_name=<resolved>)` →
   `files().get_media(fileId=FID).execute()` → write to `/tmp/`. Resolve the key
   with `gws_resolve_account` first; in Sarthak's session `google-draas` =
   `admin3.blr@draas.com`, and his profile has **no `gws_service` key**, so pass
   `service_name` explicitly on every call.
2. **Reply at the END of a multi-message thread, not to the counterparty's message.**
   Fetch `format='metadata'` with
   `metadataHeaders=['Message-ID','References','In-Reply-To','From','To','Cc']` for
   both the counterparty's latest message *and* the last message in the thread.
   Gmail's own Reply button targets the **last** message; do the same:
   `In-Reply-To = <last message's Message-ID>` and
   `References = <full prior chain> + <last message's Message-ID>`.
   Replying to the counterparty's earlier message instead nests the draft as a
   **sibling mid-thread** — it looks like a second reply to an old mail.
   Always pass `threadId` in `drafts().create`.
3. **Do NOT set `Message-ID` yourself on a fresh draft** — Gmail mints one. The
   must-preserve-Message-ID rule applies only to delete-and-recreate of an
   *existing* draft (see SKILL.md), not to a new one.
4. **Body**: `set_content(PLAIN, subtype='plain', charset='utf-8')` +
   `add_alternative(HTML, subtype='html', charset='utf-8')`, then
   `add_attachment(...)`. Plain-text-first keeps external/lender clients legible;
   keep the HTML minimal and inline-styled.
5. **Verification is THREE layers — do all three, not just one:**
   a. `drafts().list()` — the draft is genuinely in the folder. A returned
      draft-ID string proves nothing on its own (phantom-success drafts).
   b. `drafts().get(id=..., format='full')` — assert `threadId`, `To`, `Cc`
      (expect `None` when the user said requester-only), `In-Reply-To`,
      `References`, and the MIME tree
      `multipart/mixed → multipart/alternative (text/plain + text/html) + application/pdf`.
   c. **Byte-verify the attachment** —
      `messages().attachments().get(userId='me', messageId=<draft's msg id>, id=<attId>)`
      and check `len(resp['data']) * 3 // 4 == os.path.getsize(source_path)`.
      This catches a silently truncated/blank attachment that the MIME walk misses.
6. Run these scripts with **`/opt/hermes/.venv/bin/python3` from `terminal`**, not
   `execute_code` (the sandbox lacks `gws_fetch_token` for bridge calls). Write the
   script with `write_file` first — never a heredoc (an `&` in the body trips the
   backgrounding guard).