# Configuration and troubleshooting

Applies to v2.1.0 unless stated otherwise. Contents: installation; accounts and secrets; backend choices; migration and diagnosis.

## Installation and discovery

Check `himalaya --version` and `himalaya --help` first. For installation, use the user's package manager or the [official installation instructions](https://github.com/pimalaya/himalaya/blob/v2.1.0/README.md); distribution packages may lag. Official release binaries use the default Cargo features. A custom build may omit backends or use a different TLS provider. Installation does not configure mail access.

Resolve config through `--config PATH`/`-c PATH` or `HIMALAYA_CONFIG`; otherwise the documented search order is `$XDG_CONFIG_HOME/himalaya/config.toml`, `$HOME/.config/himalaya/config.toml`, then `$HOME/.himalayarc`. Multiple explicit paths are merged; v2.1 uses a colon delimiter. Check path handling on Windows rather than assuming POSIX examples translate unchanged.

`himalaya configure` (alias `wizard`) runs interactive account discovery and writes configuration. Use a terminal/PTY only for requested setup. The v2.1 wizard discovers from an email address, domain, supported URL or local path; unsupported discovery requires manual configuration. First-run offers are suppressed for nonterminal stdin and JSON mode. Do not expect the wizard to work as an unattended configuration API.

## Minimal IMAP plus SMTP account

Use a side-by-side config when migrating. Keep credentials in a password-manager command; never substitute a real secret into an example, transcript or committed file.

```toml
[accounts.work]
default = true
email = "you@example.org"
display-name = "Your Name"

imap.server = "imaps://imap.example.org:993"
imap.sasl.plain.username = "you@example.org"
imap.sasl.plain.password.command = ["pass", "show", "email/work/imap"]

smtp.server = "smtp://smtp.example.org:587"
smtp.starttls = true
smtp.sasl.plain.username = "you@example.org"
smtp.sasl.plain.password.command = ["pass", "show", "email/work/smtp"]

mailbox.alias.inbox = "INBOX"
mailbox.alias.sent = "Sent"
mailbox.alias.drafts = "Drafts"
mailbox.alias.trash = "Trash"

signature = "Best regards,\nYour Name"
signature-delim = "-- \n"
```

Replace hosts, account identity and mailbox values with actual settings. SASL mechanisms must match the server. A bare server authority defaults to implicit TLS; for STARTTLS use an explicit `imap://` or `smtp://` URL and enable the corresponding `starttls` field.

`mailbox.alias` is **singular in v2**. Alias keys are case-insensitive, backend IDs are preserved, and account aliases override global entries. The `inbox` alias supplies the default for shared mailbox arguments; otherwise pass `--mailbox`. Resolve actual mailbox IDs/names from listings. Gmail via IMAP may use localized names; Gmail REST uses label identifiers. Do not copy IMAP aliases into a REST account unexamined.

Check the requested account:

```bash
himalaya --config config.v2.toml --json account list
himalaya --config config.v2.toml --account work --json account check
himalaya --config config.v2.toml --account work --json mailbox list
```

`account check` tests client setup/handshake/authentication for the selected account and allowed backends. It does not prove all permissions, queries, mailbox aliases, or message delivery work.

## Secrets and authentication

V2 secret values use `.raw` or `.command` (string or argument array), not v1 `.cmd`. Prefer command arrays. A secret command should return only the password/token, without explanatory output; a password-manager entry containing extra lines may need an appropriate secret-only command.

IMAP/SMTP offer anonymous, login, plain, oauthbearer, xoauth2 and scram-sha-256. Configure exactly one SASL mechanism per protocol. Omitting SASL skips authentication; that is for servers/proxies intentionally providing such access.

OAuth login/refresh and native keyring integration are no longer Himalaya's responsibility in v2. Use an existing external token broker (for example Ortie) or a password-manager CLI. Confirm the broker's own syntax and scopes; do not assume Himalaya refreshes a static token.

```toml
# Alternative to PLAIN, not an extra mechanism alongside it:
imap.sasl.xoauth2.username = "you@example.org"
imap.sasl.xoauth2.token.command = ["your-token-broker", "access-token"]
```

Provider policy determines whether an app password or OAuth is available. Avoid blanket claims that Gmail always requires an app password.

## Other backends

These are independent account settings, not one configuration to paste wholesale.

| Backend | Configuration starting point | Operational distinction |
| --- | --- | --- |
| JMAP | `jmap.server = "https://api.example.org/jmap/session"`; `jmap.auth.bearer.token.command = ["your-token-broker", "access-token"]` | Also supports HTTP basic or a complete authorization header. Sending can discover an identity and drafts role; pin `jmap.identity-id` / `jmap.drafts-mailbox-id` when needed. |
| Gmail REST | `gmail.auth.token.command = ["your-token-broker", "access-token"]` | OAuth bearer; optional `gmail.user-id`, default `me`. Native labels, drafts, settings and sending. |
| Microsoft Graph | `msgraph.auth.token.command = ["your-token-broker", "access-token"]` | OAuth bearer; optional `msgraph.user-id`, default `me`. Native mail folders, messages and sending. |
| Maildir | `maildir.root = "~/Mail/work"` | Local Maildir++ storage; shared copy/move and native mailbox/flag operations. |
| m2dir | `m2dir.root = "~/Mail/work-m2dir"` | Content-addressed local store with separate flag metadata; native surface differs from Maildir. |
| pimdir | `pimdir.root = "~/.local/state/neverest/work"` | Reads the store populated by Neverest; edits are staged replica mutations. A body can be absent locally. No sync loop is implied. |

For pimdir, `pimdir.source` disambiguates the replica when automatic single-source selection is insufficient. V2.1 exposes it through shared commands and `--backend pimdir`, not a top-level pimdir command.

With several backend blocks, `auto` uses the first suitable configured backend. Choose `--backend` explicitly when intent would otherwise be ambiguous; native commands always use their own backend. Sending uses the selected JMAP/Gmail/Graph storage client or available SMTP. Shared copy/move is within one account/backend.

For trusted session proxies such as Sirup, IMAP/SMTP accept `unix:///path/to/socket`; these sessions are already authenticated. Advanced TLS choices, custom CA files, IMAP ID exchange, SASL-IR overrides, sorting fallback and table/page formatting are documented in the [versioned sample](https://github.com/pimalaya/himalaya/blob/v2.1.0/config.sample.toml). Change them to resolve an observed need, not as generic setup requirements.

## Diagnose failures

For Hotmail/Outlook with the Graph backend, read [msgraph-workflows.md](msgraph-workflows.md), especially native folder IDs, Windows/MSYS path handling, ID argument preservation and OData filtering. Do not attribute malformed-ID errors to quoting without reproducing the actual argument path.

- Unknown command/option: compare version and exact `--help` first.
- Missing backend: distinguish an omitted build feature from a missing account block or unsupported operation.
- Authentication: check the selected account, secret command, token validity/scopes, advertised mechanism, host and TLS mode. Keep command output containing credentials private.
- Mailbox failure: list actual IDs, verify aliases and role semantics. A syntactically valid TOML file can still have irrelevant keys.
- Search failure: distinguish local parse errors from unsupported backend operations. CLI v2.1.0 Gmail REST and Graph reject shared search entirely; use native commands. Native queries have their own grammar. Do not diagnose authentication solely from a parse/option error. Follow the bounded recovery procedure in [execution and validation](execution-and-validation.md).
- Partial mutation: inspect returned action/count/IDs and mailbox state before retrying. See composition for send ambiguity.
- Need diagnostic logs: use `--log-level debug` and optionally `--log-file PATH`; `RUST_LOG` works when the flag is absent. Use trace only for a concrete unresolved issue and redact tokens and message content before sharing logs.

Notmuch and Sendmail are v1 backends, removed from v2. Do not configure them in a v2 account.

