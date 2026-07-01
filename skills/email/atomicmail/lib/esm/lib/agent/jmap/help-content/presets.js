// Help topic: presets (MCP help / AgentSkill help).
export const helpTopicPresets = `\
# JMAP presets

Save a method-call array or a full \`{ "using", "methodCalls" }\` envelope
as JSON, then pass \`ops_file\` (MCP) or \`--ops-file\` (skill).

Relative paths first resolve against the credential directory (MCP) or current
\`--credentials-dir\` (skill). If not found, the runtime falls back to bundled
presets that ship in both npm packages.

## Bundled presets

- \`send_mail.json\` — sends one email using \`$TO\`, \`$SUBJECT\`, \`$BODY\`.
- \`list_inbox.json\` — latest 50 inbox messages (uses \`$INBOX_MAILBOX_ID\`).
  **Use this preset for hourly inbox polling** (see **cron** topic).
- \`reply.json\` — replies in-thread using \`$MAIL_ID\` and \`$BODY\`.
- \`send_mail_attachment.json\` — \`Blob/upload\` + send; \`vars\`: \`TO\`,
  \`SUBJECT\`, \`BODY\`, \`ATTACHMENT_BASE64\`, \`ATTACHMENT_TYPE\`,
  \`ATTACHMENT_NAME\`. Fine for modest sizes; large files should use RFC 8620
  upload instead (see \`send_mail_blob_attachment.json\`).
- \`send_mail_blob_attachment.json\` — one attachment whose \`blobId\` comes from
  \`$ATTACHMENT_0_BLOB_ID\` (etc.). Use with MCP \`attachments\` or skill
  \`--attachment PATH\` so the client uploads files to \`uploadUrl\` before the
  batch; \`vars\`: \`TO\`, \`SUBJECT\`, \`BODY\`. For \`text/*\` parts referenced
  by \`blobId\`, the client adds \`charset\` (default \`utf-8\`) when omitted (RFC
  8621). For several files in one \`Email/set\`, write normal JMAP JSON
  referencing \`$ATTACHMENT_1_BLOB_ID\`, …

## Placeholders

Syntax: \`$VAR_NAME\` where \`VAR_NAME\` matches \`/^[A-Z][A-Z0-9_]*$/\` (so JMAP
keywords like \`$draft\` stay untouched).

- \`$ACCOUNT_ID\` — primary mail account id (from \`GET /.well-known/jmap\` when
  referenced).
- \`$INBOX\` — inbox email address from credentials.
- \`$INBOX_MAILBOX_ID\` — JMAP mailbox id for the inbox (extra \`Mailbox/query\`;
  use for \`Email/query\` / \`Email/set\` where the API wants a mailbox id).
- \`$UPLOAD_URL\` — RFC 8620 upload URL template from JMAP session.
- \`$DOWNLOAD_URL\` — RFC 8620 download URL template from JMAP session.
- Any other \`$FOO\` — must appear in MCP \`vars\` or skill \`--vars\` as
  \`"FOO": "..."\` (string values only; JSON escaping in the preset body is your
  responsibility).
- \`$ATTACHMENT_N_BLOB_ID\`, \`$ATTACHMENT_N_NAME\`, \`$ATTACHMENT_N_TYPE\`,
  \`$ATTACHMENT_N_SIZE\` (N = 0, 1, …) and \`$ATTACHMENT_COUNT\` — injected when
  you pass MCP \`attachments\` or skill \`--attachment\`; you can still override
  them in \`vars\` / \`--vars\` if needed.

You may override \`ACCOUNT_ID\` / \`INBOX\` / \`INBOX_MAILBOX_ID\` /
\`UPLOAD_URL\` / \`DOWNLOAD_URL\` via \`vars\` / \`--vars\` if needed.`;
