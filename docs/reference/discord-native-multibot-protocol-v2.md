# Discord Native Multi-Bot Protocol v2 Reference

Tento dokument je Slice 0A kontrakt pro budoucí Discord v2 implementaci. Popisuje
semantiku, config a durable požadavky, ale nezavádí runtime wiring.

## MVP model identit

Native multi-bot MVP používá jednu Discord Application/Bot User identitu na
participant agenta. Základní mapování identity:

- `agent_id`: stabilní Hermes agent slug, např. `bohumil`.
- `hermes_profile`: Hermes runtime profile pro inteligenci a lokální state.
- `discord_application_id`: Discord application ID daného agenta.
- `discord_bot_user_id`: Discord bot user ID, které se používá pro mention a
  route decisions.
- `token_secret_ref`: odkaz na secret, nikdy plaintext token.
- `capabilities`: deklarované schopnosti identity, např. `intake`, `reply`,
  `approve_projection`.
- `allowed_scopes.guild_ids`: allowlist guildů pro danou identitu.

## Config schema

Default je inertní/off:

```yaml
discord_native_multibot:
  enabled: false
  mode: off
  guild_allowlist: []
  default_intake_agent_id: bohumil
  identities: []
```

Příklad aktivní identity:

```yaml
discord_native_multibot:
  enabled: true
  mode: active
  guild_allowlist: ["333333333333333333"]
  default_intake_agent_id: bohumil
  identities:
    - agent_id: bohumil
      hermes_profile: default
      discord_application_id: "111111111111111111"
      discord_bot_user_id: "222222222222222222"
      token_secret_ref: "secret://hermes/discord/bohumil-token"
      capabilities: [intake, reply, approve_projection]
      allowed_scopes:
        guild_ids: ["333333333333333333"]
```

### Rollout modes

- `off`: v2 je inertní. Nevznikají v2 Discord klienti, durable ingest, workers,
  outbox rows ani Discord sends. Legacy single-token Discord může dál běžet.
- `shadow`: config/diagnostický režim pro budoucí route diagnostics; Slice 0A ho
  pouze validuje jako povolenou hodnotu.
- `listen_only`: budoucí durable ingest bez Hermes invocation a bez outbox sendů;
  v Slice 0A pouze vyžaduje `enabled=true` a neprázdný `guild_allowlist`.
- `active`: budoucí plná v2 cesta; v Slice 0A vyžaduje `enabled=true`, neprázdný
  `guild_allowlist`, `default_intake_agent_id` a enabled identitu se stejným
  `agent_id`.

Valid modes jsou přesně: `off`, `shadow`, `listen_only`, `active`.

### Secret ref kontrakt

- V2 identity ukládají pouze `token_secret_ref`.
- Default povolené schéma v Slice 0A je `secret://...`.
- Legacy `DISCORD_BOT_TOKEN`, `platforms.discord.token` a `PlatformConfig.token`
  nesmí nikdy splnit v2 token požadavky.
- Ve v2 identity configu jsou odmítnuté klíče `token`, `bot_token`,
  `discord_token`, `DISCORD_BOT_TOKEN` a varianty stejného významu.
- Safe dumps/redacted snapshots nesmí obsahovat resolver output ani plaintext
  token-like hodnoty.

## Routing kontrakt

Routing priorita pro human-authored Discord zprávy:

1. human explicit mention,
2. human reply-to-agent,
3. human default intake Bohumil,
4. fail/clarify.

Mention je first-class routing primitive. Multi-mention v pozdější implementaci
znamená delivery pro každého explicitně mentionovaného agenta, ne fanout na
nezmíněné agenty.

## Bot-loop prevention

Discord-originated inbound deliveries smí vznikat pouze pro `author_kind=human`.
Následující author kinds nikdy nevytvářejí inbound deliveries, i když mentionují
registrovaného agent bota:

- `registered_bot`,
- `external_bot`,
- `webhook`,
- `system`.

Takové zprávy mohou být v dalších slicích uložené/projektované do `message_map` a
`route_decisions` jako diagnostika, ale nesmí spustit agent práci.

## Agent-agent komunikace

Agent-agent handoff/consult/review je autoritativně řízené interními eventy:

- `handoff.requested`,
- `consult.requested`,
- `review.requested`.

Discord zpráva reprezentující takový event je projekce. Re-ingest projekce může v
budoucnu aktualizovat `message_map` nebo diagnostické `route_decisions`, ale
nesmí vytvořit druhou inbound delivery pro target agenta.

## Restart-safety durable požadavky pro další slices

Pozdější durable store musí zachovat po restartu minimálně:

- topics,
- topic × agent sessions,
- approvals,
- handoffs,
- inbound deliveries,
- outbox deliveries a outbox parts,
- `message_map`,
- `route_decisions`.

### `message_map` minimální source-of-truth pole

- `discord_message_id primary key`, `guild_id`, `channel_id`, nullable
  `thread_id`, nullable `parent_channel_id`;
- `direction` (`inbound|outbound|projection`), nullable `agent_id`, nullable
  `delivery_key`, nullable `outbox_delivery_id`, nullable `agent_event_id`;
- `author_id`, `author_kind` (`human|registered_bot|external_bot|webhook|system`),
  nullable `author_bot_user_id`, nullable `source_client_agent_id`,
  `mentions_json`;
- `created_at`, `updated_at`, `payload_json`.

### `inbound_deliveries` minimální source-of-truth pole

- `delivery_key primary key`;
- `source_type` enum `discord_message|internal_event`;
- `source_id` stable source identifier (`discord_message_id` nebo
  `agent_event_id`);
- nullable `discord_message_id`;
- nullable `agent_event_id`;
- `target_agent_id`, `topic_id`, `route_reason`, `author_kind`, `payload_json`;
- `status`, nullable `lease_owner`, nullable `lease_until`, `attempts`,
  `created_at`, `updated_at`, `state_version`;
- unique idempotency constraint na `(source_type, source_id, target_agent_id)`.

## Webhook fallback

Webhook fallback je explicitní diagnostický/projekční fallback. Není rollout mode
pro aktivaci v2 a není default MVP cesta.
