# Message Composition with MML (MIME Meta Language)

Himalaya uses MML for composing rich (multipart, attachments, inline images, PGP) emails. MML is a simple XML-based syntax that compiles to MIME messages.

> **v2 CLI note.** The MML syntax itself is unchanged from v1.x. What changed is how you drive the composer from the CLI: v2 is flag-first (`message compose --to ... --subject ... --body ... --send` / `message reply N --body ... --send` / `message forward N --to ... --send`). Pre-v1.x editor-driven flows (`himalaya message write` opening `$EDITOR`) are gone — `message write` is now a `visible_alias` of `message compose` and behaves identically. To send a pre-written RFC 822 message, use `himalaya message send < message.eml` or pipe to `himalaya message compose` (stdin is the fallback when no `--body`/`--body-file` is set).

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
  --to recipient@example.com \
  --subject "Quick note" \
  --body "Hello from himalaya v2." \
  --send

# Multiple recipients, cc, bcc, attachment
himalaya message compose \
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
himalaya message reply 42 --body "Got it, thanks." --send

# Strict reply (just the original sender): pass --to with the original From address
himalaya message reply 42 --to sender@example.com --body "Thanks." --send

# Reply-all: include original To/Cc recipients via --cc / --to
himalaya message reply 42 \
  --to sender@example.com \
  --cc teammate@example.com \
  --body "Looping everyone in." --send

# Custom quote headline and posting style
himalaya message reply 42 --quote-headline "Replying inline:" --posting-style bottom --body "..." --send

# Forward
himalaya message forward 42 --to other@example.com --body "FYI" --send
```

> **v2 note.** There are no `--all` / `--quote` boolean flags on `message reply`. Reply-all is "include the original recipients via `--cc` / `--to`"; quoting is the default behavior controlled by `--posting-style` (`top` / `bottom` / `inline`) and `--quote-headline`.

Run `himalaya message compose --help`, `himalaya message reply --help`, and `himalaya message forward --help` for the full flag list.

### `message write` is an alias of `message compose`

```bash
himalaya message write --to x@y.com --subject "..." --body "..." --send
```

> **v2 note.** In v2, `himalaya message write` is a `visible_alias` of `message compose` (alongside `new`). It does **not** open an editor — that pre-v1.x behavior is gone. For interactive composition, use an external composer like `mml compose` and pipe into `message send`.

### Reply / forward interactive (legacy snippets — also flag-based now)

```bash
himalaya message reply 42
himalaya message forward 42
```

These are equivalent to the flag-based variants above but with no `--body` set, so the editor-friendly path is to omit the body and let stdin fill it.

### Send a prepared RFC 822 message

```bash
# File path as positional arg (v2 MessageArg resolves path-or-stdin-or-inline)
himalaya message send < message.eml

# Or pipe stdin to message compose (when no --body / --body-file given, stdin is used)
cat message.eml | himalaya message compose --from you@example.com --send
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

# Or save a pre-written RFC 822 message to drafts
himalaya message send --save drafts < message.eml
```

### Rich MIME via external composer (mml)

```bash
# Install mml: cargo install mml
mml compose > message.mml   # interactive (or scripted)
himalaya message send < message.mml
```

This is the cleanest path for attachments, PGP signing, and inline images.

## Tips

- v2 reads `--body` from the inline string, `--body-file` from a path, or stdin when neither is given. Pick whichever fits the script.
- For Hermes integration, prefer `message compose --send` over editor-driven flows — they're deterministic and don't need `$EDITOR`.
- The `message add` subcommand (`himalaya message add --mailbox drafts --flag draft < message.eml`) still works for scripting: it stages a pre-written message into a mailbox with a given flag without routing through SMTP.
- Use `himalaya message export --full` to inspect the raw MIME structure of received emails.