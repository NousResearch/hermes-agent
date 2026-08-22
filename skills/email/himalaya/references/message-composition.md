# Message Composition with MML (MIME Meta Language)

Himalaya uses MML for composing rich (multipart, attachments, inline images, PGP) emails. MML is a simple XML-based syntax that compiles to MIME messages.

> **v2 CLI note.** The MML syntax itself is unchanged from v1.x. What changed is how you drive the composer from the CLI: v2 is flag-first (`message compose --to ... --subject ... --body ... --send` / `message reply N --body ... --send` / `message forward N --to ... --send`). Pre-v1.x editor-driven flows (`himalaya message write` opening `$EDITOR`) are gone — `message write` is now a `visible_alias` of `message compose` and behaves identically. To send a pre-written RFC 822 message, use `himalaya message send < message.eml` — `message compose` consumes stdin as the **body** and rebuilds the headers, so piping a complete RFC 5322 message into it would discard `From`/`To`/`Subject`.

## Basic Message Structure

An email message is a list of **headers** followed by a **body**, separated by a blank line:

```
From: sender@example.com
To: recipient@example.com
Subject: Hello World

This is the message body.
```

## Headers

Common headers:

- `From`: Sender address
- `To`: Primary recipient(s)
- `Cc`: Carbon copy recipients
- `Bcc`: Blind carbon copy recipients
- `Subject`: Message subject
- `Reply-To`: Address for replies (if different from From)
- `In-Reply-To`: Message ID being replied to

### Address Formats

```
To: user@example.com
To: John Doe <john@example.com>
To: "John Doe" <john@example.com>
To: user1@example.com, user2@example.com, "Jane" <jane@example.com>
```

## Plain Text Body

Simple plain text email:

```
From: alice@localhost
To: bob@localhost
Subject: Plain Text Example

Hello, this is a plain text email.
No special formatting needed.

Best,
Alice
```

## MML for Rich Emails

### Multipart Messages

Alternative text/html parts:

```
From: alice@localhost
To: bob@localhost
Subject: Multipart Example

<#multipart type=alternative>
This is the plain text version.
<#part type=text/html>
<html><body><h1>This is the HTML version</h1></body></html>
<#/multipart>
```

### Attachments

Attach a file:

```
From: alice@localhost
To: bob@localhost
Subject: With Attachment

Here is the document you requested.

<#part filename=/path/to/document.pdf><#/part>
```

Attachment with custom name:

```
<#part filename=/path/to/file.pdf name=report.pdf><#/part>
```

Multiple attachments:

```
<#part filename=/path/to/doc1.pdf><#/part>
<#part filename=/path/to/doc2.pdf><#/part>
```

### Inline Images

Embed an image inline:

```
From: alice@localhost
To: bob@localhost
Subject: Inline Image

<#multipart type=related>
<#part type=text/html>
<html><body>
<p>Check out this image:</p>
<img src="cid:image1">
</body></html>
<#part disposition=inline id=image1 filename=/path/to/image.png><#/part>
<#/multipart>
```

### Mixed Content (Text + Attachments)

```
From: alice@localhost
To: bob@localhost
Subject: Mixed Content

<#multipart type=mixed>
<#part type=text/plain>
Please find the attached files.

Best,
Alice
<#part filename=/path/to/file1.pdf><#/part>
<#part filename=/path/to/file2.zip><#/part>
<#/multipart>
```

## MML Tag Reference

### `<#multipart>`

Groups multiple parts together.

- `type=alternative`: Different representations of same content
- `type=mixed`: Independent parts (text + attachments)
- `type=related`: Parts that reference each other (HTML + images)

### `<#part>`

Defines a message part.

- `type=<mime-type>`: Content type (e.g., `text/html`, `application/pdf`)
- `filename=<path>`: File to attach
- `name=<name>`: Display name for attachment
- `disposition=inline`: Display inline instead of as attachment
- `id=<cid>`: Content ID for referencing in HTML

## Composing from CLI

### Quick send (v2 flag-based API)

```bash
himalaya message compose \
  --from you@example.com \
  --to recipient@example.com \
  --subject "Quick note" \
  --body "Hello from himalaya v2." \
  --send

# Multiple recipients, cc, bcc, attachment
himalaya message compose \
  --from you@example.com \
  --to alice@example.com --to bob@example.com \
  --cc manager@example.com \
  --attach ~/Documents/report.pdf \
  --signature "Best,\nAlice" \
  --subject "Group note" \
  --body "Hi all." \
  --send

# Append a copy to a mailbox (e.g. drafts while iterating)
himalaya message compose --to x@y.com --subject "Draft" --body "WIP" --save drafts
```

The compose command also accepts `--from`, `--body-file <PATH>`, and reads the body from stdin when neither `--body` nor `--body-file` is given.

### Quick reply / forward (v2 flag-based API)

```bash
# Reply with new body (quotes original by default; --posting-style controls layout)
himalaya message reply 42 --from you@example.com --body "Got it, thanks." --send

# Strict reply (just the original sender): pass --to with the original From address
himalaya message reply 42 --from you@example.com --to sender@example.com --body "Thanks." --send

# Reply-all: include original To/Cc recipients via --cc / --to
himalaya message reply 42 \
  --from you@example.com \
  --to sender@example.com \
  --cc teammate@example.com \
  --body "Looping everyone in." --send

# Custom quote headline and posting style
himalaya message reply 42 --from you@example.com --quote-headline "Replying inline:" --posting-style bottom --body "..." --send

# Forward
himalaya message forward 42 --from you@example.com --to other@example.com --body "FYI" --send
```

> **v2 note.** There are no `--all` / `--quote` boolean flags on `message reply`. Reply-all is "include the original recipients via `--cc` / `--to`"; quoting is the default behavior controlled by `--posting-style` (`top` / `bottom` only — `inline` is rejected) and `--quote-headline` (default: empty; no placeholder substitution).

Run `himalaya message compose --help`, `himalaya message reply --help`, and `himalaya message forward --help` for the full flag list.

### `message write` is an alias of `message compose`

```bash
himalaya message write --from you@example.com --to x@y.com --subject "..." --body "..." --send
```

> **v2 note.** In v2, `himalaya message write` is a `visible_alias` of `message compose` (alongside `new`). It does **not** open an editor — that pre-v1.x behavior is gone. For interactive composition, use an external composer like `mml compose` and pipe into `message send`.

### Reply / forward via stdin

```bash
# Pipe a body on stdin — `read_body` only consumes stdin when stdin is NOT a TTY,
# so this works under `himalaya message reply 42 < body.txt` but NOT as an
# interactive terminal command.
himalaya message reply 42 --from you@example.com < body.txt
himalaya message forward 42 --from you@example.com --to other@example.com < fwd.txt
```

For interactive (TTY-attached) composition, always supply `--body` or `--body-file` explicitly; do **not** rely on bare `himalaya message reply 42` waiting for terminal stdin — it returns an empty body when stdin is a TTY.

### Send a prepared RFC 822 message

```bash
# File path as positional arg (v2 MessageArg resolves path-or-stdin-or-inline)
himalaya message send < message.eml

# The pre-written RFC 822 message stays on the `message send` path —
# `message compose` treats stdin as the BODY and rebuilds headers, so
# piping a complete message into it would discard From/To/Subject.
# If you want to compose fresh, use the flag-based form above; if you
# want to send a prepared RFC 822 message, use `message send`.
```

`message send` routes through the account's SMTP (or JMAP submission) backend; envelope sender comes from the `From:` header and recipients from `To:`/`Cc:`/`Bcc:`. Add `--save <mailbox>` to also append a copy to a mailbox (the name is resolved through the account's `[mailbox.alias]` map).

### Prefill headers from CLI

```bash
himalaya message compose \
  --to recipient@example.com \
  --subject "Quick Message" \
  --body "Message body here"
```

### Save a draft without sending

```bash
# Compose and save to drafts (no --send)
himalaya message compose --to x@y.com --subject "Draft" --body "WIP" --save drafts

# Save a pre-written RFC 822 message to drafts WITHOUT sending it.
# For a true "save without sending", use the `message add` subcommand
# which stages the message into a mailbox with a given flag without
# routing through SMTP (see the warning callout below for why
# `message send` with `--save drafts` does NOT do what you'd expect).
himalaya message add --mailbox drafts --flag draft < message.eml
```

> ⚠️ **Why not `himalaya message send --save drafts < message.eml`?**
> v2.0.0's `MessageSendCommand::execute` calls `handler::::route(..., true)`
> unconditionally — `--save drafts` means "send, then append a copy,"
> not "save without sending." Under a "Save a draft" heading, that
> recipe would silently deliver the unfinished message. Use `message
> add --mailbox drafts --flag draft` (shown above) for true draft
> staging.

### Rich MIME via external composer (mml)

```bash
# Install mml. Upstream's Cargo package is `mime-meta-language`; the binary
# is named `mml`. Pin one contract end-to-end — the CLI shape differs between
# released tags and current master, so the install command and invocation must
# agree on the same source.
#
#   Released `v1.1.1` (stable, recommended):
#     cargo install mime-meta-language --version 1.1.1 --locked --features cli
#     # v1.1.1: `output` is a POSITIONAL argument
#     mml compose --from me@example.org /tmp/draft.eml
#     himalaya message send < /tmp/draft.eml
#
#   Current `master` (bleeding edge, output via global `-o/--output`):
#     cargo install --locked --git https://github.com/pimalaya/mml.git --rev ad50fd97786be9c94a9d758fc1f7792a03d6d378
#     # master @ ad50fd97786be9c94a9d758fc1f7792a03d6d378: output is the global `-o` / `--output` flag
#     mml compose --from me@example.org --output /tmp/draft.eml
#     himalaya message send < /tmp/draft.eml
#
# Notes:
# - Both contracts require a real path (stdout redirection breaks
#   editor-driven composition because the spawned editor needs a TTY).
# - The artifact consumed by `himalaya message send` is RFC 5322/MIME, so
#   use a `.eml` extension, not `.mml` (which is the source template format).
# - If you mix the install source with the wrong invocation, the binary will
#   fail at parse time (positional `/tmp/draft.eml` is "unexpected argument"
#   on master; `--output` is "unexpected argument" on v1.1.1).
# - The master install is pinned to a specific git rev
#   (`ad50fd97786be9c94a9d758fc1f7792a03d6d378`) so that the install and the
#   documented `--output` CLI contract stay aligned. Without `--rev`, the
#   install would resolve whatever `master` points to AT INSTALL TIME, which
#   could re-introduce the source/CLI drift this section exists to prevent.
```

This is the cleanest path for attachments, PGP signing, and inline images.

## Tips

- v2 reads `--body` from the inline string, `--body-file` from a path, or stdin when neither is given. Pick whichever fits the script.
- For Hermes integration, prefer `message compose --send` over editor-driven flows — they're deterministic and don't need `$EDITOR`.
- The `message add` subcommand (`himalaya message add --mailbox drafts --flag draft < message.eml`) still works for scripting: it stages a pre-written message into a mailbox with a given flag without routing through SMTP.
- Use `himalaya message read 42 --raw` to inspect the raw RFC 5322 bytes of a received email (there is no `message export` subcommand in v2; `message read --raw` is the equivalent).
