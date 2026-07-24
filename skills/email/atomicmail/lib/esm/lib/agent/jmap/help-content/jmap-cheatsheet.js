// Help topic: jmap_cheatsheet (MCP help / AgentSkill help).
export const helpTopicJmapCheatsheet = `\
# JMAP cheatsheet

## Capabilities (\`using\`)

Common URNs:

- urn:ietf:params:jmap:core
- urn:ietf:params:jmap:mail
- urn:ietf:params:jmap:submission — required for \`EmailSubmission/set\`
- urn:ietf:params:jmap:blob — required for \`Blob/upload\`, \`Blob/get\`, and
  \`Blob/lookup\` (see RFC 9404 §4.3 for reverse blob references).

## Session blob limits

Per-account limits live under
\`accounts[accountId].accountCapabilities["urn:ietf:params:jmap:blob"]\` (see
[RFC 9404 §3.1](https://www.rfc-editor.org/rfc/rfc9404#section-3.1)):
\`maxSizeBlobSet\`, \`maxDataSources\`, etc. MCP and AgentSkill **reject before
POST** when a computable \`Blob/upload\` payload or an \`attachments\` file would
exceed advertised \`maxSizeBlobSet\` or \`maxDataSources\` (\`maxSizeBlobSet:
null\` means no client octet cap). Literal (non-\`#\`) \`blobId\` slices are not
pre-sized on the client.

## \`Blob/upload\` shape (RFC 9404)

Each \`Blob/upload\` \`create\` value is an **UploadObject**: required \`data\` is
an **array** of **DataSourceObject**; each array element uses **exactly one** of
\`data:asText\`, \`data:asBase64\`, or \`blobId\` (+ optional \`offset\` /
\`length\`). Optional \`type\` is a media-type hint. In one batch, reference a
created blob as \`"#b1"\` when the create key was \`b1\`.

**Invalid shapes** (do not expect servers to fix these): \`data\` as a plain
string; \`data:asBase64\` / \`data:asText\` on the upload object instead of
inside an element of the \`data\` array; more than one of the allowed forms inside
a single array element.

**Further reading:** [RFC 9404 §4.1](https://www.rfc-editor.org/rfc/rfc9404#section-4.1).

**Email parts:** in \`Email/set\`, \`attachments[]\` references the blob with
\`blobId\` (e.g. \`"#b1"\` for create key \`b1\`), plus \`type\` / \`name\` per
RFC 8621.

**Out-of-band:** RFC 8620 \`POST\` to \`uploadUrl\` (MCP \`attachments\` / skill
\`--attachment\`) then use \`$ATTACHMENT_N_BLOB_ID\` in the same JMAP JSON.

## Placeholders

- \`$ACCOUNT_ID\`, \`$INBOX\` (full mailbox **email** for \`From\` / envelope; from
  \`inboxId\`, appending \`@atomicmail.ai\` or \`ATOMIC_MAIL_INBOX_DOMAIN\` when
  needed), \`$INBOX_MAILBOX_ID\` (JMAP mailbox id — use for \`Email/query\` →
  \`inMailbox\` and \`Email/set\` → \`mailboxIds\`), \`$UPLOAD_URL\`,
  \`$DOWNLOAD_URL\` resolve from the session.
- Pass \`$TO\`, \`$SUBJECT\`, \`$BODY\`, etc. via MCP \`vars\` or skill \`--vars\`
  (object of strings).

## Bare methodCalls vs full envelope

If \`ops\` is **only** a methodCalls array, the default \`using\` is **core + mail**
only. For submission or blob methods, pass a full \`{ "using", "methodCalls" }\`
object (or use bundled presets, which include the right \`using\`).

## Mailboxes

\`\`\`json
["Mailbox/get", {"accountId": "$ACCOUNT_ID"}, "m0"]
\`\`\`

## Query + fetch latest inbox mail

\`inMailbox\` must be a **mailbox id**, not the email address — use
\`$INBOX_MAILBOX_ID\`.

\`\`\`json
{
  "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
  "methodCalls": [
    ["Email/query", {
      "accountId": "$ACCOUNT_ID",
      "filter": {"inMailbox": "$INBOX_MAILBOX_ID"},
      "sort": [{"property": "receivedAt", "isAscending": false}],
      "limit": 25
    }, "q0"],
    ["Email/get", {
      "accountId": "$ACCOUNT_ID",
      "#ids": {"resultOf": "q0", "name": "Email/query", "path": "/ids"},
      "properties": ["id", "threadId", "receivedAt", "from", "to", "subject", "preview"]
    }, "g0"]
  ]
}
\`\`\`

## Send one email (draft + submit)

Same pattern as bundled \`send_mail.json\`: \`Email/set\` includes
\`mailboxIds\` with \`$INBOX_MAILBOX_ID\` as the mailbox id key, then
\`EmailSubmission/set\` with \`envelope\`.

\`\`\`json
{
  "using": [
    "urn:ietf:params:jmap:core",
    "urn:ietf:params:jmap:mail",
    "urn:ietf:params:jmap:submission"
  ],
  "methodCalls": [
    ["Email/set", {
      "accountId": "$ACCOUNT_ID",
      "create": {
        "d1": {
          "mailboxIds": {"$INBOX_MAILBOX_ID": true},
          "from": [{"email": "$INBOX"}],
          "to": [{"email": "$TO"}],
          "subject": "$SUBJECT",
          "textBody": [{"partId": "b", "type": "text/plain"}],
          "bodyValues": {"b": {"value": "$BODY"}},
          "keywords": {"$draft": true}
        }
      }
    }, "c0"],
    ["EmailSubmission/set", {
      "accountId": "$ACCOUNT_ID",
      "create": {
        "s1": {
          "emailId": "#d1",
          "envelope": {
            "mailFrom": {"email": "$INBOX"},
            "rcptTo": [{"email": "$TO"}]
          }
        }
      }
    }, "c1"]
  ]
}
\`\`\`

## Attachment in one batch (\`Blob/upload\` + send)

\`Blob/upload\` must follow RFC 9404 (see **\`Blob/upload\` shape** above). The
bundled preset \`send_mail_attachment.json\` uses base64 parts:
\`"data": [{ "data:asBase64": "$ATTACHMENT_BASE64" }]\` plus \`type\`. Vars:
\`TO\`, \`SUBJECT\`, \`BODY\`, \`ATTACHMENT_BASE64\`, \`ATTACHMENT_TYPE\`,
\`ATTACHMENT_NAME\`.

Minimal inline example (base64 for UTF-8 \`Hello\`; replace addresses):

\`\`\`json
{
  "using": [
    "urn:ietf:params:jmap:core",
    "urn:ietf:params:jmap:mail",
    "urn:ietf:params:jmap:submission",
    "urn:ietf:params:jmap:blob"
  ],
  "methodCalls": [
    ["Blob/upload", {
      "accountId": "$ACCOUNT_ID",
      "create": {"b1": {"data": [{ "data:asBase64": "SGVsbG8=" }], "type": "text/plain"}}
    }, "b0"],
    ["Email/set", {
      "accountId": "$ACCOUNT_ID",
      "create": {
        "m1": {
          "mailboxIds": {"$INBOX_MAILBOX_ID": true},
          "from": [{"email": "$INBOX"}],
          "to": [{"email": "$TO"}],
          "subject": "With attachment",
          "bodyValues": {"body1": {"value": "See attachment."}},
          "textBody": [{"partId": "body1", "type": "text/plain"}],
          "attachments": [{"blobId": "#b1", "type": "text/plain", "name": "note.txt"}]
        }
      }
    }, "m0"],
    ["EmailSubmission/set", {
      "accountId": "$ACCOUNT_ID",
      "create": {
        "s1": {
          "emailId": "#m1",
          "envelope": {
            "mailFrom": {"email": "$INBOX"},
            "rcptTo": [{"email": "$TO"}]
          }
        }
      }
    }, "s0"]
  ]
}
\`\`\`

## Attachment via RFC 8620 (\`uploadUrl\`) — still standard JMAP

Keep \`Email/set\` / \`EmailSubmission/set\` exactly as in RFC 8621; only the blob
bytes go out-of-band: pass MCP \`attachments\` or skill \`--attachment PATH\`
(repeatable). The client \`POST\`s each file to the session \`uploadUrl\`, then
substitutes \`$ATTACHMENT_0_BLOB_ID\`, \`$ATTACHMENT_0_NAME\`, \`$ATTACHMENT_0_TYPE\`
into your \`ops\` / preset before the \`/jmap/\` batch. Bundled
\`send_mail_blob_attachment.json\` is a minimal one-file example; for several
parts, add more objects under \`attachments\` referencing \`$ATTACHMENT_1_BLOB_ID\`, etc.
Before POST, the client adds \`charset\` (default \`utf-8\`) to any \`Email/set\`
\`text/*\` body part that uses \`blobId\` without \`charset\`, per RFC 8621.

## Blob/get

Use only property names allowed by [RFC 9404 §4.2](https://www.rfc-editor.org/rfc/rfc9404#section-4.2)
(for example \`data:asBase64\`, \`size\`). Do not list \`id\` or \`type\` in
\`properties\` — \`id\` is still returned on each result object.

\`\`\`json
{
  "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:blob"],
  "methodCalls": [
    ["Blob/get", {
      "accountId": "$ACCOUNT_ID",
      "ids": ["$BLOB_ID"],
      "properties": ["data:asBase64", "size"]
    }, "g0"]
  ]
}
\`\`\`

## Blob/lookup

Reverse lookup: which mail objects reference a blob. Parameters \`typeNames\`,
\`ids\`; errors include \`unknownDataType\`. See RFC 9404 §4.3.

## Tips

- Back-references (\`#b1\`, \`#m1\`, \`#draft\`) chain calls in one batch.
- Save reusable JSON as preset files and pass \`ops_file\`.`;
