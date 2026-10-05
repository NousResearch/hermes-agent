# OpenMail Setup

## Install

```bash
npm install -g @openmail/cli   # Node.js 20+; or run: npx @openmail/cli <command>
```

## Authenticate and create the default inbox

Get an API key at https://console.openmail.sh/api-keys (free, no card). Pass it
once:

```bash
openmail init --api-key om_... --mailbox-name "agent" --display-name "Agent"
```

This creates the inbox and saves both the key and the inbox (id, address) as
defaults in `~/.openmail-cli/state.json`. After this, every command works
without `--api-key`, and `send`, `threads list`, and `messages list` need no
`--inbox-id`.

Override the saved key any time: `--api-key`, then `OPENMAIL_API_KEY`, then
`OPENMAIL_API_KEY=...` in `./.env` take precedence over the state file.

Verify:

```bash
openmail inbox list --json
```

## More inboxes

```bash
openmail inbox create --mailbox-name "support" --display-name "Support"
```

Target one inbox with `--inbox-id inb_...` on `send`, `threads list`, and
`messages list`, or set `OPENMAIL_INBOX_ID`.

## Reset

Delete `~/.openmail-cli/state.json` to forget both the saved key and the
default inbox. Also unset `OPENMAIL_API_KEY` if you set it.
