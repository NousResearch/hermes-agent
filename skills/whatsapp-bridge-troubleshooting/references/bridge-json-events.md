# Bridge JSON events

The WhatsApp bridge (`bridge.js`) emits JSON lines to stdout for every message decision. These are the event types you may see when reading `/proc/<pid>/fd/1`.

## Intake / filtering events (from `messages.upsert`)

### `ignored`

Message was not forwarded to the gateway. Always paired with a `reason` field.

| reason | when | fix |
|---|---|---|
| `allowlist_mismatch` | non-self message from a sender not on `WHATSAPP_ALLOWED_USERS` | add sender to allowlist |
| `allowlist_mismatch_owner_chat` | bot-mode owner message, chatId is bot's own LID, allowlist check failed (no forward LID mapping) | see class skill Step 5 |
| `self_chat_mismatch` | self-chat mode but chatId is not the bot's own number or LID | not actionable — expected for non-self chats |
| `self_chat_mode_rejects_non_self` | self-chat mode, message is not fromMe (from another person) | not actionable — self-chat only responds to own messages |
| `drop_disabled` | bot mode, `fromMe: true`, `WHATSAPP_FORWARD_OWNER_MESSAGES` is false | set `WHATSAPP_FORWARD_OWNER_MESSAGES=true` |
| `drop_echo` | `fromMe: true` and the messageId matches a recently sent outbound ID | not actionable — prevents echo loops |
| `from_me_group` | `fromMe: true` in a group chat | not actionable — group echoes ignored |
| `from_me_status` | `fromMe: true` for a status update | not actionable |
- `group_policy_rejected` | group message, group policy is `disabled` or `pairing` or allowlist doesn't match | check `WHATSAPP_GROUP_POLICY` in `.env`; ensure at least one other participant is in the group (see skill Step 7) |

### `connected`

WhatsApp session established. Useful to confirm the bridge successfully authenticated.

### `qr`

Pairing QR code generated. Used during initial setup.

### `debug`

Diagnostic event from `emitDebugEvent`. Fields vary by `stage`.

## Relevant source locations

- Message intake loop: `bridge.js` ~line 506 (`sock.ev.on('messages.upsert', ...)`)
- Owner message gate: `owner_message_gate.js` — `classifyOwnerMessageGate()`
- Allowlist matching: `allowlist.js` — `matchesAllowedUser()`, `expandWhatsAppIdentifiers()`
- Self-chat check: `bridge.js` ~line 602 (self-chat mode branch)
