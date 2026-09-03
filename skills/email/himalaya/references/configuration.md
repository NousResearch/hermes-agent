# Himalaya Configuration Reference

Configuration file location: `~/.config/himalaya/config.toml`

## Minimal IMAP + SMTP Setup

```toml
[accounts.default]
email = "user@example.com"
display-name = "Your Name"
default = true

# IMAP backend for reading emails
imap.server = "imap.example.com:993"
imap.sasl.plain.username = "user@example.com"
imap.sasl.plain.password.raw = "your-password"

# SMTP backend for sending emails
smtp.server = "smtp.example.com:587"
smtp.starttls = true
smtp.sasl.plain.username = "user@example.com"
smtp.sasl.plain.password.raw = "your-password"

# Mailbox aliases — required whenever server folder names differ
# from himalaya's canonical names. See "Mailbox Aliases" below.
mailbox.alias.inbox = "INBOX"
mailbox.alias.sent = "Sent"
mailbox.alias.drafts = "Drafts"
mailbox.alias.trash = "Trash"
```

## Password Options

### Raw password (testing only, not recommended)

```toml
imap.sasl.plain.password.raw = "your-password"
# smtp.sasl.plain.password.raw = "your-password"
```

### Password from command (recommended)

```toml
imap.sasl.plain.password.cmd = "pass show email/imap"
# imap.sasl.plain.password.cmd = "security find-generic-password -a user@example.com -s imap -w"
```

Then run `himalaya` to set up an account (the wizard prints a ready-to-save TOML config).

## Gmail Configuration

```toml
[accounts.gmail]
email = "you@gmail.com"
display-name = "Your Name"
default = true

imap.server = "imap.gmail.com:993"
imap.sasl.plain.username = "you@gmail.com"
imap.sasl.plain.password.raw = "app-password"

smtp.server = "smtp.gmail.com:587"
smtp.starttls = true
smtp.sasl.plain.username = "you@gmail.com"
smtp.sasl.plain.password.raw = "app-password"

# Gmail folder mapping. Without these, save-to-Sent fails after
# SMTP delivery succeeds (Gmail's Sent folder is `[Gmail]/Sent Mail`,
# not `Sent`), and `himalaya message send` exits non-zero. Any
# caller that retries on that error will re-run SMTP — duplicate
# emails to recipients. Always include this block for Gmail.
mailbox.alias.inbox = "INBOX"
mailbox.alias.sent = "[Gmail]/Sent Mail"
mailbox.alias.drafts = "[Gmail]/Drafts"
mailbox.alias.trash = "[Gmail]/Trash"
```

**Note:** Gmail requires an App Password if 2FA is enabled.

## iCloud Configuration

```toml
[accounts.icloud]
email = "you@icloud.com"
display-name = "Your Name"

imap.server = "imap.mail.me.com:993"
imap.sasl.plain.username = "you@icloud.com"
imap.sasl.plain.password.raw = "app-password"

smtp.server = "smtp.mail.me.com:587"
smtp.starttls = true
smtp.sasl.plain.username = "you@icloud.com"
smtp.sasl.plain.password.raw = "app-password"
```

**Note:** Generate an app-specific password at appleid.apple.com

## Mailbox Aliases

Map himalaya's canonical mailbox names (`inbox`, `sent`, `drafts`,
`trash`) to whatever the server actually calls them:

```toml
[accounts.default]
# ... other account config ...

mailbox.alias.inbox = "INBOX"
mailbox.alias.sent = "Sent"
mailbox.alias.drafts = "Drafts"
mailbox.alias.trash = "Trash"
```

The equivalent TOML sub-section form also works:

```toml
[accounts.default.mailbox.aliases]
inbox = "INBOX"
sent = "Sent"
drafts = "Drafts"
trash = "Trash"
```

> **Note on v2.0.0 change.** Himalaya v2.0.0 renamed `folder.aliases.*`
> to `mailbox.alias.*`. If you are upgrading from v1.x, update your
> config accordingly — the old `folder.aliases.*` keys are ignored by
> v2.0.0.

## Multiple Accounts

```toml
[accounts.personal]
email = "personal@example.com"
default = true
# ... backend config ...

[accounts.work]
email = "work@company.com"
# ... backend config ...
```

Switch accounts with `--account`:

```bash
himalaya --account work envelope list
```

## Notmuch Backend (local mail)

```toml
[accounts.local]
email = "user@example.com"
# Config structure for notmuch differs — see himalaya docs.
```

## OAuth2 Authentication (for providers that support it)

```toml
# IMAP SASL OAuth2
imap.sasl.oauth2.client-id = "your-client-id"
imap.sasl.oauth2.client-secret.cmd = "pass show oauth/client-secret"
imap.sasl.oauth2.access-token.cmd = "pass show oauth/access-token"
imap.sasl.oauth2.refresh-token.cmd = "pass show oauth/refresh-token"
imap.sasl.oauth2.auth-url = "https://provider.com/oauth/authorize"
imap.sasl.oauth2.token-url = "https://provider.com/oauth/token"
```

## Additional Options

### Signature

```toml
[accounts.default]
signature = "Best regards,\nYour Name"
signature-delim = "-- \n"
```

### Downloads directory

```toml
[accounts.default]
downloads-dir = "~/Downloads/himalaya"
```

### Editor for composing

Set via environment variable:

```bash
export EDITOR="vim"
```
