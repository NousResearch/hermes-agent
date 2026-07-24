# Atomic Mail

Atomic Mail exposes a programmable inbox over JMAP with PoW signup and JWT
rotation. This skill ships a single CLI entrypoint with three commands:
**`register`**, **`jmap_request`**, and **`help`** — matching the MCP server.

## When to use this skill

- Register a new inbox or log in with an existing API key.
- Send JMAP batches (inline JSON or preset files).
- Read built-in documentation (JMAP cheatsheet, presets, troubleshooting) or the
  package README (`atomicmail help --topic readme`).

**Call `atomicmail help` early and often** — before guessing
placeholders, `using` URNs, or cron setup. Start with `help --topic overview`,
then `presets` before custom `jmap_request` calls and `cron` after `register`.
If installed behavior disagrees with docs elsewhere, trust help from the running
package.

## Commands

```bash
{{ATOMICMAIL_CLI}} register --username "myagent"

{{ATOMICMAIL_CLI}} jmap_request --ops-file list_inbox.json
```

Run **`atomicmail --help`** or **`atomicmail <command> --help`** for flags.

## Defaults

- `authUrl`: `https://auth.atomicmail.ai`
- `apiUrl`: `https://api.atomicmail.ai`
- credentials directory: `{{CREDENTIALS_DIR_DEFAULT}}`

## Workflow

### 1. Register (new account)

```bash
{{ATOMICMAIL_CLI}} register \
  --username "alice"
```

Writes `credentials.json`, `session.jwt`, `capability.jwt`. Prints JSON
including `inbox` and `accountId`.

**Required next step:** after register, arrange hourly inbox polling per your
runtime (see [Hourly inbox polling](#hourly-inbox-polling-after-register)).
Native cron hosts schedule an **agent** job with `list_inbox.json`; hosts
without native cron should ask the operator or remind manual fetch. Do not cron
`atomicmail jmap_request` alone.

Usernames must be 5–21 characters (local-part of your `@atomicmail.ai`
address).

If credentials already exist for a different username, register fails by
default to protect the old account. To add another inbox without replacing the
current one, pass a separate `--credentials-dir` (MCP: `credentials_dir` on
`register` / `jmap_request`). Use `--forced` only when you intend to replace
credentials in the **same** directory (after backing it up).

### 2. Register (existing API key, in case losing the credentials file)

```bash
{{ATOMICMAIL_CLI}} register \
  --api-key "..."
```

### 3. JMAP request

```bash
{{ATOMICMAIL_CLI}} jmap_request \
  --ops '[["Mailbox/get", {"accountId": "$ACCOUNT_ID"}, "m0"]]'
```

`$ACCOUNT_ID`, `$INBOX`, `$INBOX_MAILBOX_ID`, `$UPLOAD_URL`, and `$DOWNLOAD_URL`
resolve from the session/credentials. Other placeholders such as `$TO` or
`$SUBJECT` require `--vars` with a JSON object of strings (same substitution
applies to `--ops` and `--ops-file`).

Preset file:

```bash
{{ATOMICMAIL_CLI}} jmap_request \
  --ops-file list_inbox.json
```

With custom placeholders:

```bash
{{ATOMICMAIL_CLI}} jmap_request \
  --ops-file send_mail.json \
  --vars '{"TO":"alice@example.com","SUBJECT":"Hello","BODY":"Hi there"}'
```

Bundled presets (no local file creation required):

- `send_mail.json` (`$TO`, `$SUBJECT`, `$BODY`)
- `send_mail_attachment.json` (`$TO`, `$SUBJECT`, `$BODY`, `$ATTACHMENT_BASE64`,
  `$ATTACHMENT_TYPE`, `$ATTACHMENT_NAME`)
- `send_mail_blob_attachment.json` (`$TO`, `$SUBJECT`, `$BODY`; pair with
  repeatable **`--attachment PATH`** for RFC 8620 upload →
  `$ATTACHMENT_0_BLOB_ID`, …)
- `list_inbox.json` (latest 50; uses `$INBOX_MAILBOX_ID`) — **used for hourly inbox polling**
- `reply.json` (`$MAIL_ID`, `$BODY`)

## Hourly inbox polling (after register)

Registration only creates credentials. Inbound mail should be fetched and
triaged about once per hour between interactive sessions.

**Choose the path that matches your runtime — do not mix platforms.**

### If your host supports scheduled jobs

OpenClaw, Hermes, atomic-agent, and similar hosts with native cron: set up an
hourly **agent** job after the first successful `register`. Wire Atomic Mail MCP
or AgentSkill into that agent. The scheduled turn should call `list_inbox.json`
inside the agent prompt.

### If your host does **not** support scheduled jobs

Claude, Pi, Cursor, and other chat agents without a cron API: **do not spend
tokens trying to work around this** — no wrapper scripts, OS crontab/LaunchAgent
setup, or scheduling on a different platform.

After register, either **ask your operator** to set up hourly polling on a
capable host (share the examples below), or **remind your operator** to fetch
mail manually when needed.

### Forbidden: cross-platform scheduling

Do not register in one runtime and schedule the cron job on another (e.g. Pi
runs `register`, then creates an OpenClaw cron job).

**Do not** cron `atomicmail jmap_request` alone (no agent). **Do not** use
headless one-shot CLIs (`claude -p`, `pi -p`, `codex exec`) if you want to
continue the thread.

### Scheduling examples (capable hosts or operators)

| Your setup | Approach |
| --- | --- |
| OpenClaw | `openclaw cron add` with `--announce` |
| Hermes | `hermes cron create` or `/cron` with `--deliver` |
| Atomic Bot | Same as OpenClaw or Hermes |
| atomic-agent | `atomic-agent task create --cron` |
| No native cron (Claude, Pi, Cursor, …) | Ask operator to schedule on a capable host, or remind manual fetch |

Full options, agent prompt, and operator OS-scheduling notes: `atomicmail help
--topic cron` or MCP `help` topic `cron`.

### Agent prompt (all workflows)

```text
Use Atomic Mail to fetch my inbox (MCP jmap_request with ops_file list_inbox.json, or atomicmail jmap_request --ops-file list_inbox.json). Summarize new messages, highlight what needs a reply, and stay available — I may ask you to reply, forward, search, or dig into something important.
```

### Built-in cron examples

**OpenClaw** — [cron docs](https://docs.openclaw.ai/automation/cron-jobs): isolated
session, `--announce` for delivery.

**Hermes** — [cron docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron):
`--deliver origin` (or `telegram`, `discord`, `email`, …); not `--no-agent`.

**atomic-agent** — `atomic-agent task create --cron "0 * * * *" --message "<prompt>"`

For operator OS-scheduling patterns on terminal hosts, see `help --topic cron`.

### 4. Help

```bash
{{ATOMICMAIL_CLI}} help
{{ATOMICMAIL_CLI}} help --topic jmap_cheatsheet
```

## Security

- `credentials.json` holds the API key (mode `0600`). Do not commit it.
- JWT files are bearer secrets — do not log them.

## Attachments and blobs

Use **`send_mail_attachment.json`** (in-band base64) or **`send_mail_blob_attachment.json`**
with repeatable **`--attachment PATH`** (RFC 8620 upload — same flow as MCP
**`attachments`**). Rules, limits, and `Blob/upload` JSON shape:
**`atomicmail help --topic jmap_cheatsheet`**.

```bash
{{ATOMICMAIL_CLI}} jmap_request \
  --ops-file send_mail_attachment.json \
  --vars '{"TO":"you@example.com","SUBJECT":"Hi","BODY":"See file","ATTACHMENT_BASE64":"SGVsbG8=","ATTACHMENT_TYPE":"text/plain","ATTACHMENT_NAME":"note.txt"}'
```

## Overriding defaults

- Endpoints: `--auth-url`, `--api-url` or `ATOMIC_MAIL_AUTH_URL`,
  `ATOMIC_MAIL_API_URL`
- Credentials path: `--credentials-dir` or `ATOMIC_MAIL_CREDENTIALS_DIR`
- PoW salt: `--scrypt-salt` or `ATOMIC_MAIL_SCRYPT_SALT`
