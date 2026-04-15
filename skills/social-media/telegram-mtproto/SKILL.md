---
name: telegram-mtproto
description: Telegram MTProto client setup using Telethon for full Telegram account/bot management — managing bots via BotFather API equivalents, reading messages, changing avatars, descriptions, and all Telegram-native operations beyond what the bot token HTTP API allows.
version: 1.0.0
---

# Telegram MTProto (Telethon)

Full Telegram account access via MTProto protocol using Telethon. Use when you need to manage bots, change bot settings, read messages, interact with BotFather programmatically, or do anything the Bot API doesn't expose.

**Bot API vs MTProto**: The Bot HTTP API (bot token) can set commands, description, avatar. MTProto is a full Telegram client — can do everything including managing multiple bots, reading chats, forwarding messages, etc.

## Setup

### Prerequisites

```bash
pip3 install telethon
```

### Credentials

Obtain from https://my.telegram.org:
- `api_id` (integer)
- `api_hash` (string)
- Phone number for the target account

Public keys are hardcoded in Telethon — no need to supply them.

### Session File

Session is stored at `~/.hermes/indigo_telegram.session` (or custom path). Persists across runs. Once authenticated, no re-auth needed unless revoked.

## Authentication Flow (Headless)

MTProto auth on a headless server requires a 3-step flow because `client.start()` calls `input()` which fails without a TTY.

### Step 1: Request Code

```python
import asyncio
from telethon import TelegramClient

client = TelegramClient('/path/to/session', api_id, api_hash)

async def request_code():
    await client.connect()
    sent = await client.send_code_request(phone)
    phone_code_hash = sent.phone_code_hash
    print(f"Code sent via {sent.type}")
    # Save hash for Step 2
    import json
    with open('/tmp/telegram_auth_state.json', 'w') as f:
        json.dump({'phone_code_hash': phone_code_hash}, f)
    await client.disconnect()

asyncio.run(request_code())
```

The code is delivered via Telegram app (SentCodeTypeApp) or SMS (SentCodeTypeSms).

### Step 2: Complete Sign-In

User provides the code. Read the saved `phone_code_hash` and sign in:

```python
import asyncio, json
from telethon import TelegramClient

with open('/tmp/telegram_auth_state.json') as f:
    state = json.load(f)

client = TelegramClient('/path/to/session', api_id, api_hash)

async def sign_in():
    await client.connect()
    await client.sign_in(phone=phone, code=code, phone_code_hash=state['phone_code_hash'])
    me = await client.get_me()
    print(f"Logged in as: {me.first_name} (@{me.username or 'none'})")
    await client.disconnect()

asyncio.run(sign_in())
```

### Step 3: Verify Session

```python
asyncio.run(TelegramClient('/path/to/session', api_id, api_hash).start())
```

After first auth, the session file handles reconnection automatically.

## Common Operations

### BotFather Operations (MTProto)

```python
from telethon import functions, types

# Create a new bot via BotFather chat
# Find BotFather's entity
botfather = await client.get_entity('BotFather')

# Send command to create bot
await client.send_message(botfather, '/newbot')

# Read BotFather's reply (wait briefly)
import time; time.sleep(2)
messages = await client.get_messages(botfather, limit=3)
for m in messages:
    print(m.text)

# Send bot name, then username
await client.send_message(botfather, 'My Bot Name')
time.sleep(2)
await client.send_message(botfather, 'my_bot')
```

### Read Messages

```python
# Read from a specific chat
chat = await client.get_entity(chat_id_or_username)
messages = await client.get_messages(chat, limit=20)

# Search messages
results = await client(functions.messages.SearchRequest(
    peer=chat,
    q='search term',
    filter=types.InputMessagesFilterEmpty(),
    min_date=None,
    max_date=None,
    offset_id=0,
    add_offset=0,
    max_id=0,
    min_id=0,
    hash=0,
    limit=20
))
```

### Change Bot Settings (via Bot API is simpler)

For bot-specific settings (description, commands, avatar), prefer the direct Bot HTTP API:

```bash
curl -X POST "https://api.telegram.org/bot<TOKEN>/setMyDescription" \
  -d '{"description": "My bot description"}'

curl -X POST "https://api.telegram.org/bot<TOKEN>/setMyCommands" \
  -d '{"commands": [{"command": "start", "description": "Start the bot"}]}'
```

### Get Dialog List (All Chats)

```python
dialogs = await client.get_dialogs(limit=50)
for d in dialogs:
    print(f"{d.name} (@{d.entity.username or 'private'}) [id={d.entity.id}]")
```

## Pitfalls


- **`client.start()` uses `input()`** — fails on headless servers. Use `connect()` + `send_code_request()` + `sign_in()` instead.
- **Installation Pitfall** — If receiving `externally-managed-environment` errors (PEP 668), use `python3 -m pip install telethon --break-system-packages` to force installation in restricted container environments.
- **`phone_code_hash` is required** — must capture it from `send_code_request()` return value and pass to `sign_in()`. If the connection drops, re-request the code to get a fresh hash.
- **Session expiration** — sessions persist until manually revoked in Telegram settings. If you get `auth key unregistered`, re-run the flow.
- **Rate limits on BotFather interaction** — BotFather has anti-spam. Add `time.sleep(2-3)` between messages.
- **2FA / Cloud Password** — if the account has 2FA enabled, `sign_in()` raises `SessionPasswordNeededError`. Handle with `await client.sign_in(password='your_2fa_password')`.
- **Code delivery method** — `SentCodeTypeApp` sends to Telegram app (faster), `SentCodeTypeSms` sends via SMS (slower, fallback).
