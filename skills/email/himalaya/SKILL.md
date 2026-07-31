---
name: himalaya
description: "Himalaya v2 mail CLI. Use when reading or sending email."
version: 2.0.0
author: community
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Email, IMAP, SMTP, CLI, Communication]
    homepage: https://github.com/pimalaya/himalaya
prerequisites:
  commands: [himalaya]
---

# Himalaya CLI v2.x

Himalaya is a Rust CLI for managing email from the terminal. **This skill targets himalaya v2.x** (v2.0.0+, post the v1→v2 breaking changes). The skill previously documented the v1.x schema; commands like `himalaya folder list` and `himalaya envelope list from x` no longer work in v2 and will trip agents loading the skill on first attempt.

This skill is separate from the Hermes Email gateway adapter. The gateway adapter lets people email the agent and uses Hermes' built-in IMAP/SMTP adapter; this skill lets the agent operate a mailbox from terminal tools and requires the external `himalaya` CLI.

## Key v2 differences from v1

| v1.x (don't use) | v2.x (current) |
|---|---|
| `folder list` | `mailbox list` |
| `folder.aliases.sent` (singular, TOML sub-table) | `mailbox.alias.sent` (plural, dotted key under account) |
| `[accounts.X] backend = { type=imap, host=..., port=..., auth={...} }` | Flat keys: `imap.server`, `imap.sasl.plain.username`, `imap.sasl.plain.password.command` |
| `envelope list from x` (positional filters) | `envelope list`; filters moved to separate `envelope search "from x"` subcommand with its own DSL |
| `message write` (no piped input) / `message reply` | `message compose --to ... --subject ... --body ...`; rich MIME via piped `mml`; legacy `template send` still exists |
| `--output json` / `plain` | `--json` (global flag) |
| Native keyring support | Removed; use `pass`, `gopass`, `secret-tool` via `password.command` |
| OAuth flows built in | None; use [ortie](https://github.com/pimalaya/ortie) as token broker |

Always verify `himalaya --version` reports **v2.0.0 or later** before using this skill.

## References

- `references/configuration.md` (v2 TOML schema, Gmail/Outlook/Fastmail/iCloud examples, mailbox aliases)
- `references/message-composition.md` (compose/reply/forward, mml integration)

## Prerequisites

1. **Himalaya v2.0.0+** installed (`himalaya --version` to verify)
2. A configuration file at one of these paths (first match wins):
   - `$XDG_CONFIG_HOME/himalaya/config.toml`
   - `$HOME/.config/himalaya/config.toml`
   - `$HOME/.himalayarc`
3. Credentials supplied either inline (`password.raw = "..."`) or via a `password.command` that prints the secret on stdout (recommended — use `pass`, `gopass`, or a custom script).

### Installation

```bash
# Pre-built binary (Linux/macOS — recommended, no sudo needed)
curl -sSL https://raw.githubusercontent.com/pimalaya/himalaya/master/install.sh | PREFIX=~/.local sh

# macOS via Homebrew
brew install himalaya

# From source (any platform with Rust)
cargo install --locked --git https://github.com/pimalaya/himalaya.git
```

The installer places the binary at `~/.local/bin/himalaya` (or `/usr/local/bin/himalaya` when run as root).

## Configuration

### The wizard

Run `himalaya` with **no subcommand** to launch the account setup wizard. It probes PACC, Thunderbird Autoconfiguration, RFC 6186 SRV records, and JMAP session discovery in parallel, then prints a ready-to-save TOML document on stdout. Redirect it into your config:

```bash
himalaya > /tmp/new-account.toml
# review, then append to your config
cat /tmp/new-account.toml >> ~/.config/himalaya/config.toml
```

### Gmail (app password)

Generate an app password at https://myaccount.google.com/apppasswords (requires 2-Step Verification), then:

```toml
[accounts.gmail]
email = "you@gmail.com"
display-name = "Your Name"
default = true

imap.server = "imaps://imap.gmail.com:993"
imap.sasl.plain.username = "you@gmail.com"
imap.sasl.plain.password.command = "pass show email/gmail-app"

smtp.server = "smtp://smtp.gmail.com:587"
smtp.starttls = true
smtp.sasl.plain.username = "you@gmail.com"
smtp.sasl.plain.password.command = "pass show email/gmail-app"

mailbox.alias.inbox = "INBOX"
mailbox.alias.sent = "[Gmail]/Sent Mail"
mailbox.alias.drafts = "[Gmail]/Drafts"
mailbox.alias.trash = "[Gmail]/Trash"
mailbox.alias.archive = "[Gmail]/All Mail"
mailbox.alias.junk = "[Gmail]/Spam"
```

The `[Gmail]/` labels must be **quoted in the shell** when used as `-m` args: `himalaya -m "[Gmail]/Drafts"`. Aliases are case-insensitive lookups; raw backend ids work verbatim too.

### Outlook / Microsoft 365 (OAuth)

Microsoft retired basic auth. Use OAuth 2.0 via `xoauth2` and a token broker like [ortie](https://github.com/pimalaya/ortie):

```toml
[accounts.outlook]

imap.server = "imaps://outlook.office365.com:993"
imap.sasl.xoauth2.username = "you@outlook.com"
imap.sasl.xoauth2.token.command = ["ortie", "token", "show", "-a", "outlook"]

smtp.server = "smtp://smtp-mail.outlook.com:587"
smtp.starttls = true
smtp.sasl.xoauth2.username = "you@outlook.com"
smtp.sasl.xoauth2.token.command = ["ortie", "token", "show", "-a", "outlook"]
```

Or skip SMTP entirely and use the Microsoft Graph REST API backend (`msgraph.auth.token.command = [...]`). Graph uses opaque folder ids — address them by name (`-m Archive`) or by well-known role (`-m inbox`).

### Fastmail

App password works for IMAP/SMTP. For the native JMAP API, replace both blocks with a single `jmap` one using an API token:

```toml
[accounts.fastmail]
imap.server = "imaps://imap.fastmail.com"
imap.sasl.plain.username = "you@fastmail.com"
imap.sasl.plain.password.command = "pass show fastmail"

smtp.server = "smtps://smtp.fastmail.com"
smtp.sasl.plain.username = "you@fastmail.com"
smtp.sasl.plain.password.command = "pass show fastmail"
```

### iCloud

Note: IMAP login uses the local-part (`johnappleseed`), SMTP login uses the full address (`johnappleseed@icloud.com`). Requires an app-specific password.

```toml
[accounts.icloud]
imap.server = "imaps://imap.mail.me.com:993"
imap.sasl.plain.username = "johnappleseed"
imap.sasl.plain.password.command = "pass show icloud"

smtp.server = "smtp://smtp.mail.me.com:587"
smtp.starttls = true
smtp.sasl.plain.username = "johnappleseed@icloud.com"
smtp.sasl.plain.password.command = "pass show icloud"

mailbox.alias.sent = "Sent Messages"
```

## Hermes integration notes

- **Reading, listing, searching, flagging, copying** — all work directly through the terminal tool.
- **Composing / replying / forwarding** — use the `message compose` subcommand (flags-only) or chain `mml` for rich MIME / attachments / PGP. The legacy `template send` (pipe a complete RFC 822 message on stdin) still works for scripted sends.
- For programmatic output (parsing in scripts), pass `--json` for structured envelopes / messages.
- **Pitfall — clap parser order matters:** search query DSL tokens (`from`, `subject`, `after`, `order by`, etc.) start consuming characters from the next flag too. Always pass `--page-size=20` (with `=`) instead of `--page-size 20`, and put query positional args **last** on the command line. Spaces between a flag like `--page-size 20` get eaten as part of the search query.
- **Pitfall — Gmail mailbox names:** `[Gmail]/Sent Mail` etc. contain `[`, `]`, and a space. Always quote in the shell.
- **Pitfall — multiple accounts:** pass `-a <name>` or `--account <name>`. The account flagged `default = true` is used when omitted.

## Common operations

### List accounts

```bash
himalaya account list
himalaya account check  # validates config
```

### List mailboxes (folders / labels)

```bash
himalaya mailbox list
himalaya mailbox list --account gmail
```

### List envelopes (default mailbox)

```bash
himalaya envelope list
himalaya envelope list --page 2 --page-size 50
himalaya envelope list --recipient          # show To: instead of From: (sent folder)
himalaya envelope list --has-attachment    # populate ATT column
```

### Search envelopes

`envelope search` takes a positional query using himalaya's cross-backend DSL. Conditions: `date <yyyy-mm-dd>`, `after <yyyy-mm-dd>`, `before <yyyy-mm-dd>`, `from <pattern>`, `to <pattern>`, `subject <pattern>`, `body <pattern>`, `flag <seen|answered|flagged|draft>`. Combine with `and`, `or`, `not`, group with parens. Sort with `order by <date|from|to|subject> [asc|desc]`.

```bash
himalaya envelope search "from alice"
himalaya envelope search "from alice and after 2026-01-01 order by date desc"
himalaya envelope search "subject meeting or body invoice"
himalaya envelope search --page-size=20 "from usvisascheduling"   # NOTE: --page-size=20 (equals form!)
```

⚠️ **Quote-protect the query** — patterns with `$` (e.g. `"body $500"`) need single quotes so the shell doesn't expand `$5`.

### Read a message

```bash
himalaya message read 42                       # rendered headers + text bodies
himalaya message read 42 --raw                 # dump raw RFC 5322 bytes (pipe to mml/w3m)
himalaya message read 42 --json                # parsed message as JSON
```

### Send a new message

The modern path uses `message compose` with flags:

```bash
himalaya message compose \
  --to you@example.com \
  --subject "Test" \
  --body "Hello from Himalaya v2" \
  --send
```

The legacy path (piped RFC 822 over stdin) still works:

```bash
cat << 'EOF' | himalaya template send
From: you@example.com
To: recipient@example.com
Subject: Test Message

Hello from Himalaya v2!
EOF
```

### Reply to a message

```bash
# Quick reply with --body
himalaya message reply 42 --body "Thanks, will do." --send

# Reply-all
himalaya message reply 42 --all --body "Looping everyone in." --send

# Quote original
himalaya message reply 42 --quote --body "See below." --send
```

### Forward

```bash
himalaya message forward 42 --to other@example.com --send
```

### Move / copy / delete

```bash
himalaya message copy --from INBOX --to Archives 42
himalaya message move --from INBOX --to Archives 42    # if backend supports move
himalaya message delete 42                              # sets \Deleted flag
himalaya message expunge                                 # permanently remove flagged
```

### Manage flags

```bash
himalaya flag add --flag seen 1:3,5
himalaya flag remove --flag seen 42
```

The `--flag` argument accepts `seen`, `answered`, `flagged`, `draft`, `recent`. Multiple message IDs can be comma-separated ranges (`1:3,5` = 1,2,3,5).

### Download attachments

```bash
himalaya attachment download 42                         # default ~/Downloads
himalaya attachment download 42 --output-dir /tmp/attach
```

## Multiple accounts

```bash
himalaya --account work envelope list
himalaya --account gmail mailbox list
```

For accounts configured with multiple backends (e.g. IMAP+JMAP), force one with `-b/--backend imap|jmap|gmail|msgraph|maildir|smtp`. Default is `auto` (first configured).

## Output formats

`--json` is a global flag that emits parsed JSON for any command:

```bash
himalaya --json envelope list --page-size=5
himalaya --json message read 42 | jq '.subject'
himalaya --json mailbox list
```

## Debugging

```bash
himalaya --log trace mailbox list
RUST_LOG=trace himalaya envelope list 2>/tmp/himalaya.log
RUST_BACKTRACE=1 himalaya envelope list
NO_COLOR=1 himalaya envelope list
```

Logs go to stderr; `--log-file <path>` writes to file directly.

## Re-using sessions (advanced)

Every invocation opens a fresh TCP+TLS+SASL handshake. To amortize that, point `imap.server` / `smtp.server` at a [sirup](https://github.com/pimalaya/sirup) Unix socket holding a pre-authenticated session:

```toml
imap.server = "unix:///run/sirup/imap.sock"
smtp.server = "unix:///run/sirup/smtp.sock"
```

## Tips

- `himalaya <subcommand> --help` is the source of truth — the help text is regenerated from the CLI definitions on every build.
- Message IDs are relative to the current mailbox; re-list after folder changes.
- For rich MIME with attachments, signatures, or encryption, chain [mml](https://github.com/pimalaya/mml) into `himalaya message send`.
- Store passwords via `pass`, `gopass`, `secret-tool`, or any shell command that prints the secret on stdout.
- The `imap.server` / `smtp.server` fields accept full URLs (`imaps://host:port`, `smtp://host:port`, `smtps://host:port`) for explicit TLS control.