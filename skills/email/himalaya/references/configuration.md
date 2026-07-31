# Himalaya v2 Configuration Reference

Authoritative source: https://github.com/pimalaya/himalaya/blob/master/config.sample.toml

This is a condensed v2 schema reference — covers the fields you'll actually
touch on a daily basis. For the full annotated sample, follow the URL above.

## Config file locations (first match wins)

- `$XDG_CONFIG_HOME/himalaya/config.toml`
- `$HOME/.config/himalaya/config.toml`
- `$HOME/.himalayarc`

Override with `himalaya -c <path>`. Multiple paths merge: `himalaya -c base:overlay`.

## Global keys (top of file)

```toml
# Default download directory for attachments. Falls back to $TMPDIR.
downloads-dir = "~/downloads"

# Table rendering
table.preset = "││──╞═╪╡┆    ┬┴┌┐└┘"          # default
table.arrangement = "dynamic"                 # or "dynamic-full-width", "disabled"

# Envelope list formatting
envelope.list.datetime-fmt = "%F %R%:z"       # ISO-ish; default
envelope.list.datetime-local-tz = false       # convert to local TZ before render
envelope.list.page-size = 50                  # default page size

# Mailbox alias map (account-level entries override)
mailbox.alias.inbox  = "INBOX"
mailbox.alias.sent   = "[Gmail]/Sent Mail"
mailbox.alias.drafts = "[Gmail]/Drafts"
mailbox.alias.trash  = "[Gmail]/Trash"

# Per-column colors (envelope list table)
envelope.list.table.id-color = "red"
envelope.list.table.flags-color = "reset"
envelope.list.table.subject-color = "green"
envelope.list.table.from-color = "blue"
envelope.list.table.to-color = "blue"
envelope.list.table.date-color = "dark_yellow"
envelope.list.table.size-color = "reset"
```

## Account block

```toml
[accounts.NAME]
default = true                        # only one account can be true
email = "you@example.com"
display-name = "Your Name"
downloads-dir = "~/downloads/NAME"    # per-account override

# Per-backend config (at least one of imap / smtp / jmap / gmail / msgraph / maildir)
imap.server = "imaps://imap.example.com:993"
imap.tls.provider = "rustls"          # or "native-tls"
imap.tls.rustls.crypto = "ring"       # or "aws"
imap.tls.cert = "/path/to/extra-ca.pem"
imap.starttls = false                 # only valid with imap://
imap.alpn = ["imap"]

# Pick ONE SASL mechanism
imap.sasl.anonymous.message = "himalaya"
imap.sasl.plain.username   = "you@example.com"
imap.sasl.plain.password.raw     = "raw-secret"           # dev only
imap.sasl.plain.password.command = "pass show example"    # recommended
imap.sasl.login.username   = "you@example.com"
imap.sasl.login.password.raw     = "***"
imap.sasl.login.password.command = ["pass", "show", "example"]
imap.sasl.oauthbearer.username = "you@example.com"
imap.sasl.oauthbearer.token.raw     = "***"
imap.sasl.oauthbearer.token.command = ["ortie", "token", "show", "-a", "example"]
imap.sasl.xoauth2.username = "you@example.com"
imap.sasl.xoauth2.token.raw     = "***"
imap.sasl.xoauth2.token.command = ["ortie", "token", "show", "-a", "example"]
imap.sasl.scram-sha-256.username = "you@example.com"
imap.sasl.scram-sha-256.password.raw     = "***"
imap.sasl.scram-sha-256.password.command = "pass show example"

# RFC 2971 ID extension (some servers require it after auth)
imap.id.auto = false
imap.id.fields = { name = true, version = true, vendor = true, support-url = true }

# RFC 5256 SORT fallback
imap.sort.fallback = false            # true = always client-side

# SMTP — same flat schema as imap.*
smtp.server = "smtps://smtp.example.com:465"
smtp.starttls = false
smtp.sasl.plain.username = "you@example.com"
smtp.sasl.plain.password.command = "pass show example"

# Gmail REST API backend (alternative to imap/smtp)
gmail.user-id = "me"                  # default
gmail.auth.token.raw     = "***"
gmail.auth.token.command = ["ortie", "token", "show", "-a", "gmail"]

# Microsoft Graph backend
msgraph.user-id = "me"
msgraph.auth.token.command = ["ortie", "token", "show", "-a", "msgraph"]

# JMAP backend (Fastmail, Stalwart, etc.)
jmap.server = "https://api.fastmail.com/jmap/session"
jmap.auth.bearer.token.command = "pass show fastmail"
jmap.identity-id = "I0123abc"          # optional: pin sender identity
jmap.drafts-mailbox-id = "M0123abc"    # optional: pin drafts folder

# Maildir backend
maildir.root = "~/Mail/example"

# Per-account mailbox aliases override global
mailbox.alias.inbox  = "INBOX"
mailbox.alias.sent   = "Sent"
mailbox.alias.drafts = "Drafts"
mailbox.alias.trash  = "Trash"
mailbox.alias.junk   = "Junk"
mailbox.alias.archive = "Archives"
```

## Server URL schemes

| Scheme | Meaning |
|---|---|
| `imaps://host:port` | Implicit TLS (typically port 993) |
| `imap://host:port` | Cleartext, optionally upgraded via STARTTLS |
| `smtps://host:port` | Implicit TLS (typically port 465) |
| `smtp://host:port` | Cleartext, optionally upgraded via STARTTLS |
| `unix:///path/to/sock` | Unix socket (for [sirup](https://github.com/pimalaya/sirup) reuse) |
| Bare `host[:port]` | Defaults to `<scheme>s://` |

## Secret storage

Three forms, per-field:

```toml
field.raw     = "literal-secret"            # dev / testing only
field.command = "pass show path/to/secret"  # shell string form
field.command = ["pass", "show", "secret"]  # argv array form (no shell interpolation)
```

The `command` form is recommended. Native keyring was removed in v2; use a
third-party tool (`pass`, `gopass`, `secret-tool`, `1password-cli`, `bitwarden`).

## Provider examples

### Gmail (app password)

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

mailbox.alias.inbox  = "INBOX"
mailbox.alias.sent   = "[Gmail]/Sent Mail"
mailbox.alias.drafts = "[Gmail]/Drafts"
mailbox.alias.trash  = "[Gmail]/Trash"
mailbox.alias.archive = "[Gmail]/All Mail"
mailbox.alias.junk   = "[Gmail]/Spam"
```

Generate the app password at https://myaccount.google.com/apppasswords (requires 2-Step Verification).

### Outlook / Microsoft 365 (OAuth)

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

Or use the Microsoft Graph REST API backend (`msgraph.auth.token.command = [...]`) — Graph has no separate SMTP block since sending also goes through Graph.

### iCloud

```toml
[accounts.icloud]
imap.server = "imaps://imap.mail.me.com:993"
imap.sasl.plain.username = "johnappleseed"             # local-part only
imap.sasl.plain.password.command = "pass show icloud"

smtp.server = "smtp://smtp.mail.me.com:587"
smtp.starttls = true
smtp.sasl.plain.username = "johnappleseed@icloud.com"  # full address
smtp.sasl.plain.password.command = "pass show icloud"

mailbox.alias.sent = "Sent Messages"
```

Generate the app-specific password at https://appleid.apple.com.

## Common mistakes

1. **`folder.alias.X`** (singular, sub-table) — that's v1.x. v2 silently
   ignores it. Always use **`mailbox.alias.X`** (plural, dotted key under
   the account block).
2. **Passwords with trailing newline** — `pass show` may include one.
   Add `| tr -d '\n'` if SASL auth fails mysteriously.
3. **Gmail folder names with `[Gmail]/`** — must be quoted in shell:
   `himalaya -m "[Gmail]/Sent Mail"`. Better: define `mailbox.alias.sent`.
4. **`backend = { type = "imap", host = ..., ... }`** — that's v1.x TOML.
   v2 uses flat keys `imap.server`, `imap.sasl.plain.username`, etc.
5. **`envelope list from alice`** — that's v1 positional. v2 uses
   `envelope search "from alice"`.
6. **Plain `password = "secret"`** — there's no such field in v2. Use
   `.raw` or `.command`.
7. **`--json` placement** — v2 made it a **global** flag. It must come
   *before* the subcommand: `himalaya --json envelope list`. Putting it
   after the subcommand silently parses as the wrong thing.