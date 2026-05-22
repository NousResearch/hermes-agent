# Rocket.Chat Plugin für Hermes Agent

Dieses Plugin verbindet Hermes Agent mit einem selbst-gehosteten Rocket.Chat-Server.
Es nutzt die REST API v1 für ausgehende Nachrichten und das DDP WebSocket für eingehende.

> **Plugin-Pfad:** `plugins/platforms/rocketchat/`
> **Basiert auf:** PR #14869 von @cyb0rgk1tty, umgestellt auf das moderne Plugin-Format (IRC-Referenz)

---

## Quick Start

### 1. Bot auf Rocket.Chat erstellen

1. Als Admin in Rocket.Chat einloggen
2. **Admin** → **Users** → **New**
3. Username: `hermes-bot`, Rolle: `bot`
4. Speichern

### 2. Personal Access Token generieren

1. Als Bot-User einloggen
2. **Account** → **Personal Access Tokens**
3. Name eingeben (z.B. `hermes-gateway`)
4. **☑ Ignore Two Factor Authentication** anhaken (wichtig!)
5. **Token** und **User ID** sofort kopieren

### 3. Konfigurieren

Entweder per Wizard:
```bash
hermes gateway setup
```
→ Rocket.Chat auswählen → URL, Token, User ID eingeben

Oder manuell in `~/.hermes/.env`:
```bash
ROCKETCHAT_URL=https://rc.example.com
ROCKETCHAT_TOKEN=dein_pat_token
ROCKETCHAT_USER_ID=deine_bot_user_id
ROCKETCHAT_ALLOWED_USERS=deine_user_id
```

### 4. Gateway neustarten

```bash
systemctl restart hermes-gateway
# oder per Telegram: /restart
```

---

## Environment Variables

| Variable | Pflicht | Default | Beschreibung |
|----------|---------|---------|--------------|
| `ROCKETCHAT_URL` | ✅ | — | Server-URL (z.B. https://rc.example.com) |
| `ROCKETCHAT_TOKEN` | ✅ | — | Personal Access Token (PAT) |
| `ROCKETCHAT_USER_ID` | ✅ | — | Bot-User-ID (`_id`) |
| `ROCKETCHAT_ALLOWED_USERS` | — | `""` | Erlaubte User-IDs (komma-getrennt) |
| `ROCKETCHAT_ALLOW_ALL_USERS` | — | `false` | Alle User erlauben (dev only) |
| `ROCKETCHAT_HOME_CHANNEL` | — | — | Room-ID für Cron-Benachrichtigungen |
| `ROCKETCHAT_REQUIRE_MENTION` | — | `true` | @mention-Pflicht in Channels |
| `ROCKETCHAT_FREE_RESPONSE_CHANNELS` | — | — | Rooms ohne @mention-Pflicht |
| `ROCKETCHAT_REPLY_MODE` | — | `off` | `thread` für verschachtelte Replies |

---

## Features

| Feature | Status |
|---------|--------|
| DDP WebSocket (Inbound) | ✅ `__my_messages__` Subscription |
| REST API (Outbound) | ✅ `chat.postMessage` |
| File Upload | ✅ Zwei-Step `rooms.media` |
| Attachment Download | ✅ Inkl. Image/Audio/Document-Cache |
| Thread Support | ✅ Via `tmid` |
| Mention Gating | ✅ Konfigurierbar pro Room |
| Typing Indicator | ✅ Rocket.Chat 8.x-kompatibel |
| Reconnect | ✅ Exponential Backoff (2s–60s) |
| Cron Delivery | ✅ REST-only One-Shot Sender |
| Setup Wizard | ✅ `hermes gateway setup` |
| Plugin Discovery | ✅ Auto-discover als `kind: platform` |
| Emoji Reactions | ❌ (PR #14869 hatte keine) |

---

## Troubleshooting

| Problem | Lösung |
|---------|--------|
| `totp-required` | PAT ohne "Ignore Two Factor" erstellt → neu generieren |
| "Failed to authenticate" | `curl -H "X-Auth-Token: TOKEN" -H "X-User-Id: ID" https://rc/api/v1/me` prüfen |
| Bot antwortet nicht | Bot in den Channel einladen + `ROCKETCHAT_ALLOWED_USERS` prüfen |
| WS disconnects | nginx `proxy_read_timeout 600s` setzen, Mongo Replica Set prüfen |
| Rate-limited (429) | Rocket.Chat Rate Limiter für Bot-IP entschärfen |

---

## Verifikation

Nach Konfiguration sollte `hermes status` zeigen:
```
Rocket.Chat 🚀 ✓ configured (plugin)
```

Test per DM an den Bot in Rocket.Chat.

---

## Architektur

```
Rocket.Chat ←── REST /api/v1/chat.postMessage ──→ Hermes Agent
           ←── DDP WebSocket stream-room-messages ──→ (Inbound)
```

- **Auth:** Personal Access Token (funktioniert für REST + DDP)
- **Room-Detection:** `rooms.info` + Lazy Cache
- **System Messages:** Gefiltert via `t`-Feld (join/leave/role etc.)
