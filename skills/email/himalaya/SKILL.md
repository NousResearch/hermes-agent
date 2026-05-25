---
name: himalaya
description: "Himalaya CLI: IMAP/SMTP email from terminal."
version: 1.1.0
author: Eugene Ohu (gingerol)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Email, IMAP, SMTP, CLI, Communication]
    homepage: https://github.com/pimalaya/himalaya
prerequisites:
  commands: [himalaya]
---

# Himalaya Email CLI

Himalaya is a CLI email client that lets you manage emails from the terminal using IMAP, SMTP, Notmuch, or Sendmail backends.

## When to Use

Load this skill when the user asks to check, read, search, send, or manage email from the terminal, or when email-related tools (himalaya, gog) are detected in the environment.

## References

- `references/configuration.md` (config file setup + IMAP/SMTP authentication)
- `references/message-composition.md` (MML syntax for composing emails)
- `references/gog-gmail.md` (gog OAuth fallback, keychain patterns, scope constraints)
- `references/oauth-imap-flow.md` (browser OAuth flow to get IMAP-scoped tokens for himalaya)

## Prerequisites

1. Himalaya CLI installed (`himalaya --version` to verify)
2. A configuration file at `~/.config/himalaya/config.toml`
3. IMAP/SMTP credentials configured (password stored securely)

### Installation

```bash
# Pre-built binary (Linux/macOS — recommended)
curl -sSL https://raw.githubusercontent.com/pimalaya/himalaya/master/install.sh | PREFIX=~/.local sh

# macOS via Homebrew
brew install himalaya

# Or via cargo (any platform with Rust)
cargo install himalaya --locked
```

## Configuration Setup

Run the interactive wizard to set up an account:

```bash
himalaya account configure
```

Or create `~/.config/himalaya/config.toml` manually:

```toml
[accounts.personal]
email = "you@example.com"
display-name = "Your Name"
default = true

backend.type = "imap"
backend.host = "imap.example.com"
backend.port = 993
backend.encryption.type = "tls"
backend.login = "you@example.com"
backend.auth.type = "password"
backend.auth.cmd = "pass show email/imap"  # or use keyring

message.send.backend.type = "smtp"
message.send.backend.host = "smtp.example.com"
message.send.backend.port = 587
message.send.backend.encryption.type = "start-tls"
message.send.backend.login = "you@example.com"
message.send.backend.auth.type = "password"
message.send.backend.auth.cmd = "pass show email/smtp"

# Folder aliases (himalaya v1.2.0+ syntax). Required whenever the
# server's folder names don't match himalaya's canonical names
# (inbox/sent/drafts/trash). Gmail is the common case — see
# `references/configuration.md` for the `[Gmail]/Sent Mail` mapping.
folder.aliases.inbox = "INBOX"
folder.aliases.sent = "Sent"
folder.aliases.drafts = "Drafts"
folder.aliases.trash = "Trash"
```

> **Heads up on the alias syntax.** Pre-v1.2.0 docs used a
> `[accounts.NAME.folder.alias]` sub-section (singular `alias`).
> v1.2.0 silently ignores that form — TOML parses fine, but the
> alias resolver never reads it, so every lookup falls through to
> the canonical name. On Gmail this means save-to-Sent fails *after*
> SMTP delivery succeeds, and `himalaya message send` exits non-zero.
> Any caller (agent, script, user) that retries on that exit code
> will re-run the entire send — including SMTP — producing duplicate
> emails to recipients. Always use `folder.aliases.X` (plural, dotted
> keys, directly under `[accounts.NAME]`).

## Hermes Integration Notes

- **Reading, listing, searching, moving, deleting** all work directly through the terminal tool
- **Composing/replying/forwarding** — piped input (`cat << EOF | himalaya template send`) is recommended for reliability. Interactive `$EDITOR` mode works with `pty=true` + background + process tool, but requires knowing the editor and its commands
- Use `--output json` for structured output that's easier to parse programmatically
- The `himalaya account configure` wizard requires interactive input — use PTY mode: `terminal(command="himalaya account configure", pty=true)`

## Common Operations

### List Folders

```bash
himalaya folder list
```

### List Emails

List emails in INBOX (default):

```bash
himalaya envelope list
```

List emails in a specific folder:

```bash
himalaya envelope list --folder "Sent"
```

List with pagination:

```bash
himalaya envelope list --page 1 --page-size 20
```

### Search Emails

```bash
himalaya envelope list from john@example.com subject meeting
```

### Read an Email

Read email by ID (shows plain text):

```bash
himalaya message read 42
```

Export raw MIME:

```bash
himalaya message export 42 --full
```

### Reply to an Email

To reply non-interactively from Hermes, read the original message, compose a reply, and pipe it:

```bash
# Get the reply template, edit it, and send
himalaya template reply 42 | sed 's/^$/\nYour reply text here\n/' | himalaya template send
```

Or build the reply manually:

```bash
cat << 'EOF' | himalaya template send
From: you@example.com
To: sender@example.com
Subject: Re: Original Subject
In-Reply-To: <original-message-id>

Your reply here.
EOF
```

Reply-all (interactive — needs $EDITOR, use template approach above instead):

```bash
himalaya message reply 42 --all
```

### Forward an Email

```bash
# Get forward template and pipe with modifications
himalaya template forward 42 | sed 's/^To:.*/To: newrecipient@example.com/' | himalaya template send
```

### Write a New Email

**Non-interactive (use this from Hermes)** — pipe the message via stdin:

```bash
cat << 'EOF' | himalaya template send
From: you@example.com
To: recipient@example.com
Subject: Test Message

Hello from Himalaya!
EOF
```

Or with headers flag:

```bash
himalaya message write -H "To:recipient@example.com" -H "Subject:Test" "Message body here"
```

Note: `himalaya message write` without piped input opens `$EDITOR`. This works with `pty=true` + background mode, but piping is simpler and more reliable.

### Move/Copy Emails

Move to folder:

```bash
himalaya message move 42 "Archive"
```

Copy to folder:

```bash
himalaya message copy 42 "Important"
```

### Delete an Email

```bash
himalaya message delete 42
```

### Manage Flags

Add flag:

```bash
himalaya flag add 42 --flag seen
```

Remove flag:

```bash
himalaya flag remove 42 --flag seen
```

## Multiple Accounts

List accounts:

```bash
himalaya account list
```

Use a specific account (v1.2.0: `-a` is a subcommand flag, not top-level):

```bash
himalaya envelope list -a work --page-size 20
himalaya envelope list -a personal subject meeting
```

## Attachments

Save attachments from a message:

```bash
himalaya attachment download 42
```

Save to specific directory:

```bash
himalaya attachment download 42 --dir ~/Downloads
```

## Output Formats

Most commands support `--output` for structured output:

```bash
himalaya envelope list --output json
himalaya envelope list --output plain
```

## Debugging

Enable debug logging:

```bash
RUST_LOG=debug himalaya envelope list
```

Full trace with backtrace:

```bash
RUST_LOG=trace RUST_BACKTRACE=1 himalaya envelope list
```

## Gmail OAuth (gog fallback)

When himalaya fails on a Gmail/Google Workspace account with the error:

```
Application-specific password required: https://support.google.com/accounts/answer/185833
```

**DO NOT tell the user to generate an app-specific password or assume credentials are missing.** Many setups have `gog` already installed with working OAuth tokens. Check first:

```bash
which gog && gog version
```

If `gog` v0.11.0+ is present, use it to read and search Gmail via OAuth (it handles the token lifecycle):

```bash
# List recent inbox
gog -a user@domain.com gmail search "newer_than:7d" --max 20 --json

# Search inbox
gog -a user@domain.com gmail search "subject:meeting" --max 10 --json

# Read a specific thread (by thread ID from search results)
gog -a user@domain.com gmail get <thread-id>
```

`gog` credentials live at `~/Library/Application Support/gogcli/` and tokens are stored in the macOS keychain under service `gogcli`, account `token:default:user@domain.com`. OAuth is self-maintaining — no password scripts needed. See `references/gog-gmail.md` for keychain token retrieval patterns.

### Building himalaya with OAuth2 support

The Homebrew build of himalaya 1.2.0 does **not** include the `oauth2` Cargo feature. If you need native OAuth2 in himalaya, build from source:

```bash
# Check current features — if +oauth2 is missing, rebuild
himalaya --version

# Build with oauth2
cargo install himalaya --version 1.2.0 --locked --features "oauth2,smtp,imap"
```

### himalaya OAuth2 config (TOML format)

himalaya v1.2.0 OAuth2 config requires these fields (all mandatory except `pkce` which defaults false):

```toml
backend.auth.type = "oauth2"
backend.auth.method = "xoauth2"         # or "oauth-bearer"
backend.auth.client-id = "..."           # string
backend.auth.client-secret = { cmd = "/path/to/script" }  # Secret: { cmd = "..." } or { raw = "..." } or { keyring = "..." }
backend.auth.refresh-token = { cmd = "/path/to/script" }
backend.auth.access-token = { cmd = "/path/to/script" }    # can be empty if refresh-token is set
backend.auth.auth-url = "https://accounts.google.com/o/oauth2/auth"
backend.auth.token-url = "https://oauth2.googleapis.com/token"
backend.auth.pkce = false
backend.auth.scope = "https://mail.google.com/"            # single scope; use `scopes = ["...", "..."]` for multiple
```

### Critical pitfall: scope mismatch

Google's Gmail REST API scope (`https://www.googleapis.com/auth/gmail.modify`) is **not** the same as Google's IMAP/SMTP scope (`https://mail.google.com/`). If a gog refresh token was authorized for the REST API only, it will fail with XOAUTH2 on IMAP. Google responds with `{"status":"400","schemes":"Bearer","scope":"https://mail.google.com/"}` — this means the token doesn't have the right scope, not that credentials are broken.

When this happens, do NOT attempt to re-authorize via browser flow immediately — just use `gog` for reading/searching (it works fine with REST API scope). The OAuth re-auth to add the IMAP scope is a separate user-browser action.

### Account flag location (v1.2.0)

v1.2.0 moved the account flag to subcommand level, NOT top level:

```bash
# CORRECT (v1.2.0+)
himalaya envelope list -a lbs --page-size 5

# WRONG — silently ignored in v1.2.0
himalaya --account lbs envelope list
```

Verify with `himalaya account list` first to see configured accounts.

## Tips

- Use `himalaya --help` or `himalaya <command> --help` for detailed usage.
- Message IDs are relative to the current folder; re-list after folder changes.
- For composing rich emails with attachments, use MML syntax (see `references/message-composition.md`).
- Store passwords securely using `pass`, system keyring, or a command that outputs the password.
- For Gmail/Google Workspace accounts where password auth is blocked, use `gog` (OAuth) for reading/searching. See `references/gog-gmail.md` for token details and scope constraints.
- Never assume credentials are broken when you see "Application-specific password required" — check for `gog` first. Users often have OAuth already set up from prior tools.
