# Email Draft — Recipient & Attachment Pre-Draft Checks

Pre-draft verification steps before composing/creating any email draft (observed 2026-09-01, Anbarasan → electrical contractor Rahman for Ranka Iris).

## 1. Attachment verification

- When the user says "I am attaching the relevant document" in a **voice message**, the file does NOT always arrive. Check the chat for an `<attached_files>` block AND `search_files(target='files', path='/mnt/uploads')`.
- If nothing is there: tell the user plainly the attachment didn't come through and ask them to re-send it (or name the document so it can be pulled from Drive). Do NOT silently compose a draft referencing an attachment that doesn't exist — the recipient will get a mail promising a file that was never attached.
- Web-chat uploads mount at `/mnt/uploads/<file_id>_<original_name>` and are deleted after ~24h — if the user references a doc from earlier, it may already be purged; say so and ask for a re-upload.

## 2. Recipient verification (Email Recipients HARD RULE)

- Before putting any recipient on a draft, resolve via `contact_resolver` (see `contact-phone-lookup` skill for account_email mechanics — session users without `gws_service` configured must pass their own full account email from `gws_resolve_account`).
- If the named recipient is NOT verifiable in contacts (sheet + People API), do NOT guess or derive an email. Surface the closest-looking matches found (name + org), note that none matches the role the user described (e.g. "electrical contractor"), and ask the user for the address.
- Session example: "Rahman, electrical contractor for Ranka Iris" → resolver returned Amar Rahman (A S Cargo, amar@ascargo.co.in) and Suhail Rahman (CoEvolve) — neither an electrical contractor → asked Anbarasan for the email before finalising.

## 3. Draft-first stays default for non-listed users

- Per `references/user-delivery-preferences.md`: when the session user is not the intended recipient (and has no listed preference), present the full draft (To / Subject / Body) in Telegram and collect the missing pieces (recipient email, attachment, signature details) BEFORE creating any Gmail draft.
- Never call `draft_create` on the first pass for a client/vendor-facing mail to an unverified recipient; Gmail drafts may still be created once To: + attachment are confirmed and the user says go.

## 4. Project context via session history

- For project-linked drafts, `session_search` is the fastest way to recover the surrounding task context (e.g. Ranka Iris Electrical Inspectorate: Form 1 & Form 3 via consultant, Ayaz at Gandhinagar/Rajajinagar offices, transformer inspection + charging). Use it before asking the user to re-explain.