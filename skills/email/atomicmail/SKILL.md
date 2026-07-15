---
name: atomicmail
description: Register and operate an autonomous agent email inbox.
version: 0.3.23
author: Dominic Dalton (dom-dalty)
license: MIT
platforms: [macos, linux, windows]
prerequisites:
  commands: [node]
metadata:
  hermes:
    tags: [Productivity, Email, Communication, Identity, blueprint]
    category: email
    related_skills: [himalaya]
    homepage: https://atomicmail.ai
    config:
      - key: atomicmail.credentials_dir
        description: Directory for Atomic Mail credentials and JWT files (defaults under HERMES_HOME)
        default: ~/.hermes/atomicmail
        prompt: Atomic Mail credentials directory
      - key: atomicmail.auth_url
        description: Atomic Mail auth-service base URL (pass to the CLI as --auth-url)
        default: https://auth.atomicmail.ai
        prompt: Atomic Mail auth service URL
      - key: atomicmail.api_url
        description: Atomic Mail JMAP API base URL (pass to the CLI as --api-url)
        default: https://api.atomicmail.ai
        prompt: Atomic Mail JMAP API URL
      - key: atomicmail.scrypt_salt
        description: PoW scrypt salt override (pass as --scrypt-salt; only when directed by Atomic Mail support)
        default: ""
        prompt: Atomic Mail PoW scrypt salt override
    blueprint:
      schedule: "0 * * * *"
      deliver: origin
      no_agent: false
      prompt: |
        Use ${HERMES_SKILL_DIR}/scripts/atomicmail jmap_request --ops-file list_inbox.json to fetch my inbox. Summarize new messages, highlight what needs a reply, and stay available — I may ask you to reply, forward, search, or dig into something important.
required_environment_variables:
  # Only the bearer secret is declared as an environment variable. Non-secret
  # settings (credentials dir, endpoints, PoW salt) live under metadata.hermes.config
  # and are passed to the CLI as flags; env declarations are allowlisted into
  # sandbox children, so they are reserved for secrets per the skill authoring guide.
  - name: ATOMIC_MAIL_API_KEY
    prompt: Atomic Mail API key
    help: Optional — use register with --api-key or store in credentials.json
    required_for: existing-account login without credentials.json
required_credential_files:
  - path: atomicmail/credentials.json
    description: Atomic Mail API key and account metadata (created by register)
  - path: atomicmail/session.jwt
    description: JMAP session JWT (created by register)
  - path: atomicmail/capability.jwt
    description: JMAP capability JWT (created by register)
---
# Atomic Mail Skill

Register a free, agent-owned `@atomicmail.ai` inbox via PoW (no operator
signup), then send, receive, and triage over **JMAP** — a protocol already in
your training data; bundled presets cover gaps on smaller models. Use this when
the agent needs its own persistent, publicly reachable email identity.

Launcher: `scripts/atomicmail` — `register`, `jmap_request`, `help`.

## When to Use

- When the agent needs its own inbox with no operator signup: register, send, receive, monitor, reply, cron triage.
- For operating the operator's existing mailbox (Gmail/corporate IMAP), use a personal-mailbox skill instead.

## Prerequisites

1. **Node.js 20+** on the host (`node --version`).
2. Bundled CLI at `${HERMES_SKILL_DIR}/scripts/atomicmail` (`atomicmail.cmd`
   on Windows) — scopes `ATOMIC_MAIL_CREDENTIALS_DIR` to
   `${HERMES_HOME:-~/.hermes}/atomicmail` when unset.
3. Non-secret settings (credentials dir, endpoints, PoW salt) come from
   `metadata.hermes.config`; pass them to the CLI as `--auth-url`, `--api-url`,
   `--scrypt-salt`. Only `ATOMIC_MAIL_API_KEY` is a secret env var.

Call **`atomicmail help`** before guessing JMAP placeholders or cron setup.
Start with `help --topic overview`, then `presets` and `cron` after
`register`. Trust help from the running package over static copies elsewhere.

## How to Run

Use the `terminal` tool with the bundled launcher (paths resolve from the skill
directory):

```bash
${HERMES_SKILL_DIR}/scripts/atomicmail register --username "myagent"
${HERMES_SKILL_DIR}/scripts/atomicmail jmap_request --ops-file list_inbox.json
${HERMES_SKILL_DIR}/scripts/atomicmail help --topic jmap_cheatsheet
```

Run `atomicmail --help` or `atomicmail <command> --help` for flags.

## Quick Reference

| Command | Purpose |
| --- | --- |
| `register` | PoW signup or API-key login; writes credentials under `${HERMES_HOME:-~/.hermes}/atomicmail` |
| `jmap_request` | JMAP batch via `--ops` JSON or `--ops-file` preset |
| `help` | Embedded docs: cheatsheet, presets, cron, troubleshooting |

**Bundled presets** (under `lib/presets/`): `list_inbox.json`, `send_mail.json`,
`send_mail_attachment.json`, `send_mail_blob_attachment.json`, `reply.json`.

**Session placeholders** resolved automatically: `$ACCOUNT_ID`, `$INBOX`,
`$INBOX_MAILBOX_ID`, `$UPLOAD_URL`, `$DOWNLOAD_URL`. Other vars (`$TO`,
`$SUBJECT`, …) need `--vars '{"TO":"..."}'`.

**Defaults:** auth `https://auth.atomicmail.ai`, API `https://api.atomicmail.ai`.

## Procedure

### 1. Register a new inbox

```bash
${HERMES_SKILL_DIR}/scripts/atomicmail register --username "alice"
```

Usernames: 5–21 characters (local part of `@atomicmail.ai`). Writes
`credentials.json`, `session.jwt`, `capability.jwt`; prints JSON with `inbox`
and `accountId`.

Existing API key (lost credentials file):

```bash
${HERMES_SKILL_DIR}/scripts/atomicmail register --api-key "..."
```

If credentials exist for another username, register fails by default. Use a
separate `--credentials-dir` for a second inbox; `--forced` only when replacing
the same directory (back up first).

### 2. Accept hourly inbox polling (Hermes)

After the first successful `register`, accept the skill blueprint via
`/suggestions` or create a cron job with `hermes cron create` / `/cron` using
`--deliver origin` and **`no_agent: false`**. Each run should load this skill
and call `jmap_request --ops-file list_inbox.json` inside an agent turn — not
as a headless CLI cron.

See `atomicmail help --topic cron` for the agent prompt and options.

### 3. Send and read mail

List inbox (preset):

```bash
${HERMES_SKILL_DIR}/scripts/atomicmail jmap_request --ops-file list_inbox.json
```

Send with vars:

```bash
${HERMES_SKILL_DIR}/scripts/atomicmail jmap_request \
  --ops-file send_mail.json \
  --vars '{"TO":"alice@example.com","SUBJECT":"Hello","BODY":"Hi there"}'
```

Attachments: `send_mail_attachment.json` (base64) or
`send_mail_blob_attachment.json` with repeatable `--attachment PATH`. Details:
`atomicmail help --topic jmap_cheatsheet`.

## Pitfalls

- **Never cron raw CLI** — do not schedule `jmap_request` without a full agent
  session; duplicates and missed triage follow.
- **No cross-platform scheduling** — do not register in one runtime and cron in
  another.
- **Personal mail only** — if the task is strictly the operator's Gmail/IMAP, use
  `himalaya` instead; otherwise stay on this skill.
- **Secrets** — `credentials.json` and JWT files are bearer tokens; do not log
  or commit them (mode `0600` on credentials).
- **Multi-account** — pass `--credentials-dir` only when operating multiple
  inboxes; default single-inbox flow needs no extra path.

## Verification

1. `node --version` succeeds (20+).
2. `${HERMES_SKILL_DIR}/scripts/atomicmail help --topic overview` exits 0.
3. After `register`, JSON output includes `inbox` and `accountId`.
4. `jmap_request --ops-file list_inbox.json` returns mailbox/email data without
   auth errors.
5. On Hermes, `/suggestions` offers the hourly inbox blueprint after register.
