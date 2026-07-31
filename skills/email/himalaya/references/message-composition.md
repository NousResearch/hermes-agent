# Message Composition with MML (MIME Meta Language)

Himalaya uses MML for composing rich (multipart, attachments, inline images, PGP) emails. MML is a simple XML-based syntax that compiles to MIME messages.

> **v2 CLI note.** The MML syntax itself is unchanged from v1.x. What changed is how you drive the composer from the CLI: v2 adds a flag-based `message compose --to ... --subject ... --body ... --send` for scripted sends and `message reply N --body ... --send` / `message forward N --to ... --send` for replies. The legacy `himalaya message write` (editor-driven) and `himalaya template send` (piped RFC 822) still work in v2.

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

# Multiple recipients, cc, bcc
himalaya message compose \
  --to alice@example.com --to bob@example.com \
  --cc manager@example.com \
  --subject "Group note" \
  --body "Hi all." \
  --send
```

### Quick reply / forward (v2 flag-based API)

```bash
# Reply with new body
himalaya message reply 42 --body "Got it, thanks." --send

# Reply-all
himalaya message reply 42 --all --body "Looping everyone in." --send

# Quote the original
himalaya message reply 42 --quote --body "See below." --send

# Forward
himalaya message forward 42 --to other@example.com --body "FYI" --send
```

Run `himalaya message compose --help`, `himalaya message reply --help`, and `himalaya message forward --help` for the full flag list.

### Interactive compose (editor-driven)

Opens your `$EDITOR`:

```bash
himalaya message write
```

> **v2 behavior.** `message write` still opens `$EDITOR` like in v1.x. From Hermes, prefer the flag-based `message compose --send` or piped `template send` paths — the editor flow requires PTY mode and a configured `$EDITOR`, which is fiddlier from an agent runtime.

### Reply (opens editor with quoted message)

```bash
himalaya message reply 42
himalaya message reply 42 --all  # reply-all
```

### Forward

```bash
himalaya message forward 42
```

### Send a prepared message from stdin (legacy v1 path, still works)

```bash
cat message.txt | himalaya template send
```

### Prefill headers from CLI

```bash
himalaya message write \
  -H "To:recipient@example.com" \
  -H "Subject:Quick Message" \
  "Message body here"
```

### Save a draft without sending

```bash
# Modern path: pipe to message add with --flag draft
himalaya message compose --to x@y.com --subject "Draft" --body "WIP" | \
  himalaya message add --mailbox drafts --flag draft

# Or use the legacy template form
cat <<'EOF' | himalaya template send --draft
From: you@example.com
To: someone@example.com
Subject: Draft

WIP content here.
EOF
```

### Rich MIME via external composer (mml)

```bash
# Install mml: cargo install mml
mml compose > message.mml   # interactive (or scripted)
himalaya message send < message.mml
```

This is the cleanest path for attachments, PGP signing, and inline images.

## Tips

- The editor opens with a template; fill in headers and body.
- Save and exit the editor to send; exit without saving to cancel.
- MML parts are compiled to proper MIME when sending.
- Use `himalaya message export --full` to inspect the raw MIME structure of received emails.
- For Hermes integration, prefer `message compose --send` over editor-driven flows — they're deterministic and don't need `$EDITOR`.