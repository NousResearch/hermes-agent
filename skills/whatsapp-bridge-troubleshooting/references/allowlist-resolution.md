# Allowlist resolution for WhatsApp bridge

How the bridge decides whether an incoming message's sender is on the `WHATSAPP_ALLOWED_USERS` list.

## Entry points

- `matchesAllowedSender(senderId, senderAltId, allowedUsers, sessionDir)` — used for non-self messages. Checks both `senderId` and `senderAltId` (the alternate JID form Baileys supplies for multi-device).
- `matchesAllowedUser(senderId, allowedUsers, sessionDir)` — core check. Resolves identifiers to a set of aliases, then tests set membership.

## Resolution chain: `expandWhatsAppIdentifiers`

Given a raw identifier (phone number or LID), the function builds a set of all known aliases:

1. Normalize the input: strip `+`, strip everything from `:` onward, strip everything from `@` onward → plain digits.
2. Walk both `lid-mapping-<id>.json` (LID→phone) and `lid-mapping-<id>_reverse.json` (phone→LID) files in the session dir.
3. For each newly discovered alias, repeat from step 2 (BFS until no new aliases).

## The LID mapping file gap

Baileys creates **reverse** mapping files (`lid-mapping-<LID>_reverse.json` containing the phone number) but not **forward** mapping files (`lid-mapping-<phone>.json` containing the LID).

This means:
- Looking up a phone number → finds the LID via the reverse file → works.
- Looking up a LID → finds nothing via the forward file (missing) → only resolves to itself → **fails to match a phone-number allowlist**.

This is why bot-mode self-chat (where `chatId` is the bot's own LID) fails the allowlist check even when the phone number is allowed.

## The `botIds` array

In the `messages.upsert` handler, `botIds` is pre-computed as:

```js
const botIds = Array.from(new Set([
  normalizeWhatsAppId(sock.user?.id),   // e.g. "33687726682:6@s.whatsapp.net" → "33687726682"
  normalizeWhatsAppId(sock.user?.lid),  // e.g. "19211439063084@lid" → "19211439063084"
]).filter(Boolean));
```

These are the bridge's own identities. A self-chat check is `botIds.includes(normalizeWhatsAppId(chatId))`.

## Interaction with bot mode vs self-chat mode

- **Self-chat mode** (`WHATSAPP_MODE=self-chat`): uses a separate self-chat check (lines ~602-633) that compares `chatId` against the bot's own number/LID directly. Does NOT go through the allowlist.
- **Bot mode** (`WHATSAPP_MODE=bot`): for `fromMe: true` messages, goes through `classifyOwnerMessageGate` which calls `matchesAllowedUser(chatId, ...)`. This is where the LID→phone gap bites.
