# Telegram user-client MCP connector — server

Local MCP server wrapping **Telethon** (https://docs.telethon.dev/), a real
MTProto Python client for a Telegram *user account* — not the Bot API. It
logs in exactly like the official Telegram app (phone number → code →
optional 2FA password) and can do anything that account could do manually:
read any chat it belongs to, send/edit/delete/pin messages, list group
members, send/download media.

Catalog metadata is in `../manifest.yaml`.

## Which account

This is meant to run against a **separate Telegram number dedicated to the
agent** — not the owner's personal account — since it gets full read/write
access. Treat the session file (see Auth below) like a password: anyone
with it can act as that account without logging in again.

## Setup

Venv lives **inside this directory** (`./venv`, gitignored), so the whole
connector — code and venv together — travels with the fork: any
profile/client using this same hermes-agent checkout gets it, instead of
depending on some external directory that only exists on whichever
machine happened to run the install first.

```bash
cd optional-mcps/telegram-client/server
python3 -m venv venv
PYTHONPATH="" venv/bin/pip install -r requirements.txt
```

(`PYTHONPATH=""` matters on this install — it otherwise points at Hermes's
own venv and breaks a plain `pip install`.)

The catalog manifest (`../manifest.yaml`) points `transport.command` at
`optional-mcps/telegram-client/server/venv/bin/python3` and
`transport.args` at `telegram_mcp_server.py` in this directory, both as
absolute paths under this checkout's own location. If this fork is ever
cloned somewhere other than `/home/max/.hermes/hermes-agent`, update both
paths in the manifest.

## Auth

Needs an **app registered at https://my.telegram.org/apps** (`TELEGRAM_API_ID`
/ `TELEGRAM_API_HASH`) — this is not a bot token, every Telegram client
(including the official apps) authenticates with API credentials like this.

Logging a human in cannot be a single API call — Telegram's own login flow
is phone → code → (sometimes) 2FA password — so this connector exposes it
as three tools meant to be driven through a chat dialog with the user:

1. `telegram_login_start(phone)` — sends a login code to that phone's
   Telegram app; returns `{"status": "code_sent"}`.
2. `telegram_login_code(code)` — submits the code. Returns
   `{"status": "logged_in", ...}`, or `{"status": "password_needed"}` if
   the account has Two-Step Verification.
3. `telegram_login_password(password)` — only needed after step 2 asks
   for it.

`telegram_auth_status()` / `telegram_logout()` round out the auth surface.

Once logged in, Telethon persists the session (the long-lived auth key) in
its own SQLite file at `~/.hermes/mcp-tokens/telegram-client.session` — no
need to log in again on restart unless that file is deleted or the session
is revoked from a Telegram device ("active sessions").

## Tools

19 tools, matching Telethon's real, documented high-level client methods
(signatures confirmed via `inspect.signature` on the installed library, not
guessed): dialogs (`telegram_get_dialogs`), chat/user lookup
(`telegram_get_entity_info`), message history + search
(`telegram_get_messages`, `telegram_search_global`), sending/editing/
deleting/pinning (`telegram_send_message`, `telegram_edit_message`,
`telegram_delete_messages`, `telegram_pin_message`/`telegram_unpin_message`),
read receipts (`telegram_mark_read`), group membership
(`telegram_get_participants`), media (`telegram_send_file`,
`telegram_download_media`), and `telegram_get_me`.

## Write tools

`telegram_send_message`, `telegram_edit_message`, `telegram_delete_messages`,
`telegram_pin_message`/`telegram_unpin_message`, and `telegram_send_file` act
on a live Telegram account — a message sent or deleted this way is real and
visible to other people in that chat. There is no confirmation step built
into the connector itself; that's the calling agent's responsibility.

## Scope not covered

This wraps Telethon's friendly high-level client methods, not its full raw
MTProto surface (hundreds of lower-level methods exist for things like
polls, reactions, stories, business features, etc.). Extend
`telegram_mcp_server.py` only against a method that actually exists on the
installed `telethon.TelegramClient` (`python -c "from telethon import
TelegramClient; help(TelegramClient.<name>)"`) — never guess a method name
or its parameters.
