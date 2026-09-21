# Discord REST API — Channel Management via Kensei Bot

Reference for programmatic channel operations using Kensei's bot token and Discord's REST API v10.

## Prerequisites

- Bot token with `MANAGE_CHANNELS` (16) + `MANAGE_ROLES` (268435456) permissions
- Guild ID known (`https://discord.com/channels/<GUILD>/<CHANNEL>`)
- Guild ID should be set in root `config.yaml`: `guild_id: '<ID>'`

## Auth pattern

```python
import json, urllib.request, urllib.error

TOKEN = '<discord_bot_token>'
HEADERS = {
    'Authorization': f'Bot {TOKEN}',
    'Content-Type': 'application/json',
    'User-Agent': 'DiscordBot (https://hermes-agent.nousresearch.com, 1.0)'
}
GUILD = '<guild_id>'
```

## List channels

```python
req = urllib.request.Request(f'https://discord.com/api/v10/guilds/{GUILD}/channels', headers=HEADERS)
channels = json.loads(urllib.request.urlopen(req).read())
```

Returns list of channel objects with: `id`, `name`, `type`, `parent_id`, `position`, `topic`, `permission_overwrites`.

## Channel types

| Name | type | description |
|---|---|---|
| GUILD_TEXT | 0 | Text channel |
| GUILD_VOICE | 2 | Voice channel |
| GUILD_CATEGORY | 4 | Category |
| GUILD_FORUM | 15 | Forum channel |

## Create channel

```python
data = {'name': name, 'type': channel_type}
# Optional for text/forum: 'topic', 'parent_id' (category id)
# Voice channels do NOT support 'topic' — omit it or Discord returns 50035 Invalid Form Body

req = urllib.request.Request(
    f'https://discord.com/api/v10/guilds/{GUILD}/channels',
    data=json.dumps(data).encode(),
    headers=HEADERS,
    method='POST'
)
result = json.loads(urllib.request.urlopen(req).read())
```

**Pitfall:** Voice channels reject the `topic` field entirely. Only set `name` and `type` for type=2.

## Move channel to category

```python
data = json.dumps({'parent_id': target_category_id}).encode()
req = urllib.request.Request(
    f'https://discord.com/api/v10/channels/{channel_id}',
    data=data, headers=HEADERS, method='PATCH'
)
```

## Set permission overwrites

Requires bot to have `MANAGE_ROLES` permission (bit 28, value 268435456).
`MANAGE_CHANNELS` (bit 4, value 16) alone is insufficient — Discord's PATCH endpoint checks Manage Roles for overwrite operations.

```python
overwrites = [
    # Deny @everyone view by default (for restricted channels)
    {
        'id': '@everyone_role_id',
        'type': 0,  # 0=role, 1=user
        'deny': '1024',  # VIEW_CHANNEL
        'allow': '0'
    },
    # Allow specific bot roles
    {
        'id': bot_role_id,
        'type': 0,
        'allow': '117760',  # VIEW + SEND + EMBED + ATTACH + READ_HISTORY
        'deny': '0'
    }
]
data = json.dumps({'permission_overwrites': overwrites}).encode()
req = urllib.request.Request(
    f'https://discord.com/api/v10/channels/{channel_id}',
    data=data, headers=HEADERS, method='PATCH'
)
```

### Permission constants

| Permission | Bit | Value | Description |
|---|---|---|---|
| CREATE_INSTANT_INVITE | 0 | 1 | |
| MANAGE_CHANNELS | 4 | 16 | |
| MANAGE_ROLES | 28 | 268435456 | Required for permission overwrites |
| VIEW_CHANNEL | 10 | 1024 | |
| SEND_MESSAGES | 11 | 2048 | |
| MANAGE_MESSAGES | 14 | 16384 | |
| EMBED_LINKS | 14 | 16384 | |
| ATTACH_FILES | 15 | 32768 | |
| READ_MESSAGE_HISTORY | 16 | 65536 | |
| CONNECT (voice) | 20 | 1048576 | |
| SPEAK (voice) | 21 | 2097152 | |
| USE_VAD (voice) | 22 | 4194304 | |

### Common permission integers

| Scenario | Value | Bits |
|---|---|---|
| Text access | `117760` | VIEW + SEND + EMBED + ATTACH + HISTORY |
| Text + Manage Channels | `117776` | Above + MANAGE_CHANNELS |
| Text + Manage Channels + Manage Roles | `268553232` | Above + MANAGE_ROLES |
| Voice access | `5243936` | VIEW + CONNECT + SPEAK + VAD |

## Error codes

| Code | Message | Meaning |
|---|---|---|
| 50013 | Missing Permissions | Bot lacks Manage Roles or Manage Channels |
| 50033 | Channel already exists | Name taken in guild |
| 50035 | Invalid Form Body | Bad request data (e.g. topic on voice channel) |

## Bot role identification

Each bot gets a managed role created by Discord. The role name matches the bot name and its `tags.bot_id` field equals the bot's user ID.

To find bot role IDs:

```python
req = urllib.request.Request(f'https://discord.com/api/v10/guilds/{GUILD}/roles', headers=HEADERS)
roles = json.loads(urllib.request.urlopen(req).read())
for r in roles:
    if r.get('tags', {}).get('bot_id'):
        print(f"{r['name']}: role_id={r['id']} bot_id={r['tags']['bot_id']}")
```

To extract bot user ID from token:

```python
import base64
bot_user_id = base64.b64decode(token.split('.')[0] + '==').decode()
```

## Verifying bot permissions

```python
# Check bot's guild member record for assigned roles
req = urllib.request.Request(f'https://discord.com/api/v10/guilds/{GUILD}/members/{bot_user_id}', headers=HEADERS)
member = json.loads(urllib.request.urlopen(req).read())

# Cross-reference role permissions
for r in roles:
    if r['id'] in member['roles']:
        perms_int = int(r.get('permissions', '0'))
        print(f"{r['name']}: ADMIN={bool(perms_int & 8)}")
        print(f"  MANAGE_CHANNELS={bool(perms_int & 16)}")
        print(f"  MANAGE_ROLES={bool(perms_int & 268435456)}")
