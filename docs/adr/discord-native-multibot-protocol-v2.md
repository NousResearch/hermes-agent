# ADR: Discord Native Multi-Bot Protocol v2

- Status: accepted for Slice 0A contract
- Datum: 2026-06-14
- Rozsah: dokumentace a default-off config schema; bez runtime wiring

## Kontext

Hermes dnes podporuje Discord jako legacy single-bot adaptér. Pro pracovní Discord
server s více participant agenty potřebujeme protokol, kde je transportní identita
viditelná a routovatelná stejně přirozeně jako lidský Discord účet: vlastní bot
user, avatar, jméno a mention pro každého agenta.

Slice 0A pouze fixuje rozhodnutí a config kontrakt. Nesmí spouštět nové Discord
klienty, durable ingest, outbox sendery, gateway routing ani multi-token runtime.

## Rozhodnutí

1. **Native multi-bot MVP je závazná cesta.** Každý participant agent používá
   vlastní Discord Application, Bot User, token, avatar, jméno a mention.
   Hermes profile zůstává runtime/intelligence identita; Discord Bot User je
   transportní identita.
2. **Mention je first-class routing primitive.** Explicitní mention agenta je
   primární operátorův způsob, jak směrovat práci na konkrétního agenta.
3. **Webhook fallback není default MVP path.** Webhook smí být pouze explicitní
   diagnostický/projekční fallback pro operátora, ne běžná cesta pro aktivní v2.
4. **Agent-agent Discord komunikace je projekce, ne zdroj pravdy.** Autoritativní
   handoff/consult/review práce vzniká z interních eventů, ne z toho, že bot na
   Discordu zmíní jiného bota.
5. **Restart safety je požadavek protokolu.** Topics, session mapping,
   approvals, handoffs, inbound deliveries, outbox deliveries, `message_map` a
   `route_decisions` musí být v dalších slicích durable a obnovitelné po restartu.
6. **Route priority:** human explicit mention > human reply-to-agent > human
   default intake Bohumil > fail/clarify.
7. **Bot-loop prevention:** zprávy od registrovaného Hermes bota, externího bota,
   webhooku nebo systémové zprávy nikdy nevytvářejí Discord-originated inbound
   deliveries, ani když mentionují registrovaného agent bota.
8. **Durable schema source-of-truth požadavky:** `message_map` a
   `inbound_deliveries` pole popsaná v protocol reference jsou závazné vstupy pro
   pozdější durable store slice.

## Důsledky

- Legacy single-token Discord zůstává kompatibilní a může dál používat
  `platforms.discord.token` nebo `DISCORD_BOT_TOKEN`.
- V2 se aktivuje pouze přes `discord_native_multibot.enabled: true` a nebere si
  tokeny z legacy Discord configu/env.
- Config ukládá pouze `token_secret_ref`; plaintext Discord tokeny, `token`,
  `bot_token`, `discord_token` a `DISCORD_BOT_TOKEN` klíče jsou ve v2 identitách
  odmítnuté.
- Žádná část Slice 0A nesmí řešit runtime connect/send/ingest.
