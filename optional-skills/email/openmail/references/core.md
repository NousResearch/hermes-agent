# OpenMail Core

Common CLI path: create inboxes, send mail, read incoming mail, reply in
threads, and fetch attachments. Need a key first? Use [setup.md](setup.md).

## Send

```bash
openmail send \
  --to "recipient@example.com" \
  --subject "Subject line" \
  --body "Plain text body."
```

Reply in a thread with `--thread-id thr_...`. Add HTML with
`--body-html "<p>...</p>"`. Attach files with `--attach <path>` (repeatable).
The response includes `messageId` and `threadId` — store `threadId` to continue
the conversation later.

Always reply in the existing thread. When the user asks to reply to an email,
look up the thread with `openmail threads list` first, then use `--thread-id`.
Never create a new thread unless the user explicitly asks.

## Checking for new mail

Always use `threads list --is-read false` to check for new mail. This returns
only unread threads.

```bash
openmail threads list --is-read false
```

After processing an email, mark it as read:

```bash
openmail threads read --thread-id "thr_..."
```

Do not use `messages list` to check for new mail.

## Threads

```bash
openmail threads list --is-read false
openmail threads get --thread-id "thr_..."
openmail threads read --thread-id "thr_..."
openmail threads unread --thread-id "thr_..."
```

`threads get` returns messages sorted oldest-first. Read the full thread before
replying.

## Messages

```bash
openmail messages list --direction inbound --limit 20
openmail messages list --direction outbound
```

Use `messages list` when searching across all messages by direction. For
checking new mail, use `threads list --is-read false`.

Each message includes: `id`, `threadId`, `fromAddr`, `subject`, `bodyText`,
`attachments` (with `filename`, `url`, `sizeBytes`), and `createdAt`.

## Inboxes

```bash
openmail inbox list --json
openmail inbox create --mailbox-name "support" --display-name "Support"
openmail inbox get --inbox-id "inb_..."
openmail inbox delete --inbox-id "inb_..."
```

New inboxes are live immediately. Target one with `--inbox-id` on `send`,
`threads list`, and `messages list`.

## Subagents: one inbox and one key each

When spawning a subagent that needs email, give it its own inbox and a key
scoped to that inbox. Never hand a subagent your own key.

```bash
openmail inbox create --mailbox-name "research-3" --display-name "Research 3" --json
openmail inbox keys create --inbox-id <id> --name "research-3" --json
```

Pass the subagent `OPENMAIL_API_KEY=<token>` and `OPENMAIL_INBOX_ID=<id>` in
its environment. The token is shown once; if the subagent loses it, revoke and
mint again:

```bash
openmail inbox keys revoke --inbox-id <id> --key-id <key_id>
```

When the subagent is done, `openmail inbox delete --inbox-id <id>` removes the
inbox and its mail.

## Common workflows

### Wait for a reply

1. Send a message, store the returned `threadId`.
2. Every 60 seconds: `openmail threads list --is-read false`.
3. When the expected `threadId` appears, read it:
   `openmail threads get --thread-id "thr_..."`.
4. Process the reply, then mark as read:
   `openmail threads read --thread-id "thr_..."`.

### Sign up for a service and confirm

1. Use your inbox address (`address` from `openmail inbox list --json`) as the
   registration email.
2. Submit the form or API call.
3. Poll every 60 seconds: `openmail threads list --is-read false`.
4. Look for a thread where `subject` contains "confirm" or "verify".
5. Read the thread, extract the confirmation link from `bodyText`, open it.
6. Mark as read: `openmail threads read --thread-id "thr_..."`.
