// Help topic: troubleshooting (MCP help / AgentSkill help).
export const helpTopicTroubleshooting = `\
# Troubleshooting

## Custom endpoint configuration issues

Defaults are production endpoints. Set env vars only for custom deployments.

## No API key / register first

Run \`register\`, or set \`ATOMIC_MAIL_API_KEY\`, or copy an existing
\`credentials.json\` into the credential directory.

## auth-service /api/v1/session returned 401

Invalid \`apiKey\` or wrong \`ATOMIC_MAIL_SCRYPT_SALT\` for this deployment.

## Capability JWT missing inboxId

Server/version mismatch — verify \`ATOMIC_MAIL_AUTH_URL\`.

## Could not read ops file

Check the path; use an absolute path if unsure.

## Missing values for variables (\`$TO\`, etc.)

Pass every custom placeholder in MCP \`vars\` or \`--vars\` as a JSON object of
strings. Ensure \`register\` completed so \`$ACCOUNT_ID\` / \`$INBOX\` can resolve.

## \`invalidArguments\` on \`Email/query\` / \`filter/inMailbox\`

\`inMailbox\` must be a **mailbox id**, not your inbox email. Use the built-in
\`$INBOX_MAILBOX_ID\` placeholder (or run \`Mailbox/get\` / \`Mailbox/query\` and
paste the id into \`vars\`).

## \`invalidProperties\` / \`notCreated\` on \`Blob/upload\`

RFC 9404 requires \`data\` as an **array** of objects, each with **exactly one**
of \`data:asText\`, \`data:asBase64\`, or a \`blobId\` slice. Typical mistakes:
\`data\` as a raw string; \`data:asBase64\` on the upload object instead of
inside an array element; mixing two forms in one object. See topic
\`jmap_cheatsheet\` (\`Blob/upload\` shape) and preset \`send_mail_attachment.json\`.

## RFC 8620 binary \`POST\` to \`uploadUrl\` returns 404

The session lists an \`uploadUrl\` template, but your deployment must expose
that HTTP resource. If \`POST\` returns 404, out-of-band upload is not wired
on the server — use \`Blob/upload\` in JMAP instead, or fix the API gateway.

## \`Blob/upload\` succeeds but \`size\` is 0 (or \`Email/set\` rejects the blob)

The server accepted the method but did not persist octets (broken or
incomplete \`Blob/upload\`). Verify with a tiny \`data: [{ "data:asBase64": "QQ==" }]\`
payload; if \`size\`
stays 0, fix the JMAP/blob implementation on the host before sending
attachments.

## Installed package vs other documentation

The version you get from \`npx -y @atomicmail/mcp\` or
\`npx --package=@atomicmail/agent-skill …\` may lag behind other published docs.
If something disagrees, trust **your installed package**: run \`help\` and use
the bundled presets that ship with that version. In MCP runtimes, \`help\` with
topic \`readme\` returns package README.md.`;
