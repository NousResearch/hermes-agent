# WhatsApp link rules — person vs GROUP (hard rules, user-confirmed 29-Sep-2026)

Consolidated rules for producing WhatsApp deep links. NDR's verbatim correction
when a group message was needed: *"whenever it is for a group we just encode it
without the telephone number, we only encode the message."*

## Person (1:1) links
- ALWAYS call the `whatsapp_link` tool WITH `phone=` — never hand-encode an
  api.whatsapp.com / wa.me URL, never build one via execute_code/urllib.
- The tool handles all percent-encoding, including the fullwidth ampersand
  rule (`&` → `%EF%BC%86`) in message text.
- Deliver via `display_link` when platform='telegram' (MarkdownV2-safe).

## GROUP links (no specific recipient)
- Call `whatsapp_link` with NO `phone=` parameter. The returned URL is
  `api.whatsapp.com/send?text=<url-encoded message>` WITHOUT phone — WhatsApp
  opens the contact/group picker and the user selects the group manually.
- Do NOT use a phone number for a group. Do NOT fabricate a chat.whatsapp.com
  invite link — those are admin-only invite URLs that cannot be constructed.
- This same rule is stated in
  `references/work-deliverable-assignment-drive-access.md` (section 7) — that
  reference covers the deliverable-assignment variant; THIS file is the
  general rule for any group message.

## Tone for group messages
- Use third-person framing ("NDR needs them to share with Vishwas") rather
  than first-person ("I need...") when the message will sit in a shared group
  where NDR is not the only reader.
- Nothing is ever "sent" — the tool generates a deep link only. Report it as a
  generated link the user can tap.

## Long messages
- If the tool returns a `parts` array + `split: true`, deliver EACH part as its
  OWN Telegram message (Telegram caps ~4096 chars and its splitter breaks links).

## Never truncate the text you pass (user correction, 2026-10-07)
- Pass the COMPLETE message text to `whatsapp_link` — do not truncate/summarize it
  while composing. A truncated `text=` produced a link that cut off after "Dear
  Vivek," — NDR flagged it: "the message is not correctly made, it just cuts off
  after Dear Vivek". The tool encodes everything correctly; the agent's job is to
  hand it the full text, then verify the returned URL visibly covers the end of
  the message before reporting it.
- Same rule for a message that needs updating: regenerate with the FULL text; a
  re-run that still passes truncated text reproduces the same cut-off.
- Always state what was NOT done ("nothing has been sent — it's a deep link for
  you to review and send") so NDR knows the artifact is a draft link.