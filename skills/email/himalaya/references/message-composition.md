# Composition, drafts and sending

Examples target v2.1.0. Contents: composition; replies/forwards; drafts; submission; external MIME tools.

## Build a reviewable MIME artifact

V2 compose/reply/forward are noninteractive. They emit RFC 5322 bytes unless `--save` or `--send` requests a side effect. They do not compile embedded MML.

```bash
himalaya --account work message compose \
  --to recipient@example.org --cc colleague@example.org \
  --subject 'Project update' --body-file body.txt \
  --attach report.pdf --attach schedule.csv > draft.eml
```

The sender defaults to configured `email` and `display-name`; override with `--from` when requested. Recipient flags accept repeated or comma-separated values, but display names containing commas require extra care with CLI splitting; verify parsed addresses in the resulting MIME or use a proper MIME composer.

Use one body source: `--body`, `--body-file`, or stdin fallback. Prefer files for long/untrusted content. Signatures come from config, `--signature` or `--signature-file`; the last two are mutually exclusive. The built-in composer supports ordinary attachments via repeated `--attach`.

Check command success before using the output file: shell redirection creates it even if composition fails. Parse the resulting MIME and verify From/To/Cc/Bcc, subject, body, filenames and attachment contents. Do not expose Bcc recipients in user-facing copies meant for other recipients.

## Reply and forward

```bash
himalaya --account work message reply --mailbox inbox 42 \
  --body-file reply.txt > reply.eml
himalaya --account work message forward --mailbox inbox 42 \
  --to recipient@example.org --body-file introduction.txt > forward.eml
```

Reply derives threading headers and subject, quotes the source body, and derives recipients from Reply-To/From when no To is supplied. Check those recipients against the user's intended audience.

V2.1 has no shared `message reply --all`. For reply-all, inspect original From/Reply-To/To/Cc, remove the sending user's own addresses, deduplicate and explicitly pass the intended `--to`/`--cc`. Do not infer hidden Bcc recipients.

`--posting-style top|bottom` controls quote placement. `--quote-headline` is literal text; tokens such as `{date}` are not interpolated. Preserve In-Reply-To and References; use MIME/header-aware editing if changes are needed. A blanket `sed` replacement on blank lines can corrupt a multipart message or repeat text.

The built-in forward quotes source text. Do not assume original attachments are forwarded: inspect the output and explicitly download/attach the requested parts or use an external composer for full attached-message forwarding. Retain only the content the user intends to share.

## Store a draft

Appending and sending are separate actions. The following shared append requires a backend that supports it. **Gmail REST and Graph do not implement shared `message add` in v2.1.0**, including through `--save`; use native draft creation below:

```bash
himalaya --account work --json message add --mailbox drafts --flag draft < draft.eml
```

Using stdin avoids the variadic `--flag` argument swallowing a following file path. To supply a positional path after options, separate it with `--`.

`compose/reply/forward --save drafts` appends a copy, but in v2.1 the shared routing helper saves it with the **seen** flag, not automatically the draft flag. Prefer `message add --flag draft` when actual draft state matters. Gmail REST requires native draft APIs; see below.

Capture the returned backend ID and account/mailbox. Saving twice can create duplicate drafts. A local `draft.eml` alone is not a server-side draft.

### Gmail REST and Graph drafts

After inspecting the local MIME and verifying native command help:

```bash
himalaya --account work gmail drafts create < draft.eml
himalaya --account work msgraph message create --folder drafts < draft.eml
```

Choose the command matching the account backend, not both. Both store a draft without sending. Native draft creation reports a success message containing its ID; do not assume the shared `{id, sent}` output schema. Keep Gmail draft IDs distinct from message IDs and thread IDs. For an existing Gmail draft, use `gmail drafts update DRAFT_ID < draft.eml` after verifying its identity; repeated `create` makes additional drafts. See [Gmail workflows](gmail-workflows.md).

## Submit an authorized message

```bash
himalaya --account work message send < draft.eml
```

Use this only when the user has authorized sending to the resolved audience with this content. If that authorization is missing, finish the draft and ask for the missing decision. Do not require another confirmation when the user already supplied sufficient authorization.

Shared send uses configured SMTP or the selected JMAP/Gmail/Graph backend. It reads MIME from a path, raw argument or stdin. It does not interpret MML markup. `--send` on compose/reply/forward/add also delivers; it is not a preview switch.

V2.1 `message send --save sent` and combined `--save`/`--send` operations **append first, then send**. If saving fails, sending has not started through this handler. If sending fails, the saved copy can remain. Consequently, a Sent copy is not proof of successful submission. Provider-native sent storage can also make an extra manual copy unnecessary; inspect actual behavior before adding `--save`.

For a failure after submission might have begun, do not automatically resend. Preserve the MIME artifact and error, inspect available provider status/log evidence, and report uncertainty if it cannot be resolved. A Message-ID helps correlation but does not guarantee server deduplication. In v1 the save-after-send sequence differs; do not reuse v1 failure assumptions.

## Rich MIME, MML, PGP and interactive editing

For custom HTML/plain alternatives, inline content IDs, signed/encrypted messages, or editor-driven workflows, use an installed external MIME composer such as [Pimalaya MML](https://github.com/pimalaya/mml). These are companion-tool capabilities, not native Himalaya v2 template/PGP commands.

Inspect that tool's installed version/help and its own documentation before using its syntax. Render/compile to a local `.eml`, inspect MIME structure and recipients, then pass the resulting bytes to Himalaya. Feed source mail to interpreters as `message read --raw`, not the human summary or a JSON wrapper.

Do not copy old MML snippets into a v2 `message send` call: literal directives may reach recipients. Do not claim that exiting an editor always sends; the external composer's workflow controls its output and Himalaya only sends when a send operation is invoked.

