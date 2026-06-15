# Discord Native Multi-Bot Protocol v2 Implementation Plan

> Stav po průzkumu codebase: Discord transport je dnes implementovaný jako bundled platform plugin v `plugins/platforms/discord/adapter.py`, ne jako `gateway/platforms/discord.py`. Gateway core používá `gateway/run.py`, `gateway/config.py`, `gateway/session.py` a plugin registry v `gateway/platform_registry.py`. Existuje velká sada Discord testů, z nichž část už míří na primary UI/structured approvals a část importuje starší alias `gateway.platforms.discord`; to je migrační riziko a musí být řešeno jako kompatibilitní shim nebo test update.

## Goal

Dodat **Hermes Discord communication protocol v2** jako default-off MVP pro **native multi-bot Discord workspace**:

- každý participant agent má vlastní Discord Application + Bot User + token + avatar/name + mention;
- mention je first-class routing primitive;
- Hermes profile = intelligence/runtime identity, Discord bot identity = transport identity;
- Discord topic/thread = téma/práce, ne agent;
- session mapping je `topic × agent`;
- Bohumil je participant agent a default intake pro topic bez mentionu;
- human-authored mention routuje, bot-authored mention jiného agenta nespouští;
- agent-agent handoff/consult/review vzniká pouze z interních eventů a Discord je jen projekce;
- restart gateway nesmí ztratit topic state, approvals, handoffs, inbox/outbox, message map ani session mapping a nesmí způsobit duplicitní odpovědi.

## Architecture

### Runtime components

1. **Single durable multi-token Discord gateway process**
   - jeden proces gateway, N Discord clients/tokens;
   - izolace per `agent_id`, ale sdílený durable protocol store;
   - listen-only/shadow režim před aktivním outboxem.

2. **Identity registry**
   - mapuje `agent_id ↔ hermes_profile ↔ discord_application_id ↔ discord_bot_user_id ↔ token_secret_ref ↔ capabilities/allowed_scopes`;
   - žádné plaintext tokeny v configu, DB ani logu.

3. **Durable protocol store**
   - tabulky: `identity_registry`, `topics`, `topic_agent_sessions`, `inbound_deliveries`, `outbox_deliveries`, `outbox_parts`, `message_map`, `route_decisions`, `approvals`, `handoffs`, `agent_events`, `leases`, `reconciliation_runs`;
   - idempotency keys:
     - inbound: `source_type + source_id + target_agent_id`, kde `source_type ∈ {discord_message, internal_event}`;
     - outbound: stable `idempotency_key` unique per logical response/projection;
     - outbound part send: `outbox_delivery_id + part_index`;
   - leases + state version pro concurrency a replay.

4. **Routing**
   - priorita: explicit mention > reply-to-agent > default intake Bohumil > policy fail/clarify;
   - multi-mention = delivery per mentioned agent, žádný fanout na non-mentioned agents;
   - registered Hermes bot / external bot / webhook / system authored mentions se ukládají/projektují jen do `message_map`/`route_decisions`, ale vytvářejí zero inbound deliveries.

5. **Internal event source of truth**
   - agent-agent handoff/consult/review pouze přes interní eventy: `handoff.requested`, `consult.requested`, `review.requested`;
   - Discord zprávy jsou projekce těchto eventů, ne autoritativní trigger;
   - durable `agent_events` + `inbound_deliveries(source_type='internal_event')` jsou jediný autoritativní zdroj target-agent práce pro agent-agent komunikaci;
   - re-ingest Discord projekce interního eventu smí vytvořit/update `message_map` a diagnostický `route_decisions` záznam, ale nesmí vytvořit další inbound delivery.

6. **Onboarding CLI**
   - `sync`, `plan`, `handoff-missing-secrets`, `verify`, `invite-links`, `reconcile-guild`;
   - secret handoff přes MCP/standalone flow, config obsahuje pouze `token_secret_ref`.

## Tech Stack

- Python 3.11, pytest.
- `discord.py` přes existing lazy dependency `tools.lazy_deps.ensure("platform.discord")`.
- Existing gateway: `gateway/run.py`, `gateway/platforms/base.py`, `gateway/session.py`, `gateway/config.py`.
- Existing Discord adapter/plugin: `plugins/platforms/discord/adapter.py`, `plugins/platforms/discord/plugin.yaml`.
- Existing stores/patterns: `gateway/session.py` JSON+SQLite bridge, `hermes_state.py` `SessionDB`, JSON stores v `gateway/discord_approvals.py` a `gateway/discord_dashboard.py` pro krátkodobé MVP vzory.
- New durable protocol store: preferovat SQLite pod profile-aware Hermes home (`get_hermes_home()/gateway/discord_protocol_v2.sqlite3`) s atomickými migracemi; JSON jen pro read-only export/diagnostiku.
- Secret refs: navázat na Hermes config/env secret-ref mechanismy v `hermes_cli/config.py`; tokeny načítat až při runtime connectu.

## Current codebase findings

### Existing relevant files

- `plugins/platforms/discord/adapter.py`
  - single-token `DiscordAdapter` s `commands.Bot`, `connect()`, `disconnect()`, `send()`, `edit_message()`, `_handle_message()`, slash commands, approvals buttons, auto-threading, voice, allowed mentions;
  - už má bot filtering (`DISCORD_ALLOW_BOTS`) a multi-agent mention filtering, ale jen pro single bot klienta;
  - dedupe je in-memory `MessageDeduplicator`, ne durable per target agent;
  - token bere z `PlatformConfig.token` / `DISCORD_BOT_TOKEN`.
- `plugins/platforms/discord/plugin.yaml`
  - deklaruje `DISCORD_BOT_TOKEN`; pro v2 musí přejít na registry/secret refs.
- `gateway/run.py`
  - lifecycle gateway, adapter creation přes `platform_registry`, message handling, session creation, approvals, restart/drain, delivery router;
  - nyní instancuje jeden adapter per `Platform.DISCORD`, ne multi-client workspace.
- `gateway/config.py`
  - `Platform.DISCORD`, `PlatformConfig.token`, env override pro `DISCORD_BOT_TOKEN`, `DISCORD_HOME_CHANNEL`, `DISCORD_REPLY_TO_MODE`;
  - pro v2 přidat default-off schema bez plaintext tokenů.
- `gateway/session.py`
  - `SessionSource` už má Discord-relevantní fields: `guild_id`, `parent_chat_id`, `thread_id`, `message_id`, `is_bot`;
  - `build_session_key()` je dnes platform/chat/thread/user centric, ne `topic × agent`.
- `gateway/platforms/base.py`
  - `BasePlatformAdapter`, `MessageEvent`, `SendResult`, handler/typing/send contract.
- `gateway/discord_approvals.py`, `gateway/discord_dashboard.py`, `gateway/discord_kanban.py`, `gateway/discord_event_renderer.py`
  - užitečné MVP patterny pro structured state/rendering, ale JSON-backed stores nejsou dostatečné pro v2 restart/idempotency.
- `cron/discord_delivery.py`, `tools/discord_tool.py`
  - Discord output sanitization a REST introspection dnes spoléhají na `DISCORD_BOT_TOKEN`; v2 musí být agent-scoped token resolution.
- `hermes_cli/gateway.py`, `hermes_cli/main.py`, `hermes_cli/config.py`
  - místo pro onboarding CLI subcommands a config validation.

### Existing tests to preserve/extend

- `tests/gateway/test_discord_*`: 48 Discord-related test files.
- Key files for v2 gates:
  - `tests/gateway/test_discord_bot_filter.py`
  - `tests/gateway/test_discord_bot_auth_bypass.py`
  - `tests/gateway/test_discord_allowed_mentions.py`
  - `tests/gateway/test_discord_reply_mode.py`
  - `tests/gateway/test_discord_thread_persistence.py`
  - `tests/gateway/test_discord_primary_session_mapping.py`
  - `tests/gateway/test_discord_primary_security.py`
  - `tests/gateway/test_discord_primary_rollout.py`
  - `tests/gateway/test_discord_approvals.py`
  - `tests/gateway/test_discord_approval_auth.py`
  - `tests/gateway/test_discord_fakes.py`, `tests/gateway/discord_fakes.py`
- Existing drift/blocker: several new primary tests import `DiscordPrimaryUIConfig` from `gateway.config`, but current `gateway/config.py` scan did not find that class. Plan includes an early schema-contract task to reconcile this before v2 implementation.

## Assumptions

- `agent_id` is a stable slug, e.g. `bohumil`, `reviewer`, `planner`.
- Bohumil maps to a Hermes profile (default can be `default` initially) and a native Discord bot identity.
- Secret refs can be implemented independently of a specific external vault for MVP via a resolver interface; tests use fake resolver.
- Existing single-token Discord plugin remains supported behind legacy config until v2 is explicitly enabled.

## Feature flag / rollout

- New config flag: `discord_native_multibot.enabled: false` by default.
- Rollout modes and exact semantics:
  - `off`: v2 is inert. Do not instantiate v2 clients, store, workers, or outbox. Legacy single-token Discord behavior may continue unchanged through existing config/env paths.
  - `shadow`: validate config, resolve safe identity metadata, optionally connect fake/test clients in unit tests, persist only route diagnostics if explicitly wired for a shadow harness; no Hermes invocation, no outbox rows, no Discord sends. Production startup must remain safe if token refs are unresolved.
  - `listen_only`: connect configured v2 Discord clients, persist durable topics, `message_map`, `route_decisions`, and eligible human-originated inbound deliveries for diagnostics/replay; do not invoke Hermes and do not create/send response outbox deliveries.
  - `active`: full v2 path: durable ingest, Hermes invocation, internal events, native outbox, reconciliation. Must still keep webhook fallback disabled unless explicitly configured as diagnostic-only.
- Validation rules:
  - `mode=active` requires `enabled=true`, a non-empty `guild_allowlist`, and `default_intake_agent_id` resolving to an enabled identity with `intake` capability.
  - `mode=listen_only` requires `enabled=true` and a non-empty `guild_allowlist`.
  - `mode=shadow` may be used with incomplete token refs only when it does not connect real Discord clients.
  - Legacy `DISCORD_BOT_TOKEN` / `platforms.discord.token` must never satisfy v2 identity token requirements.
  - `off` is the only valid mode when `enabled=false` unless the config loader intentionally normalizes disabled non-off values back to an inert diagnostics-only state and warns.
- `diagnostic_webhook_fallback` is not a rollout mode for v2 activation; it is an explicit operator-only fallback/projection path and never default.
- MVP rollout order: default-off contract → single guild allowlist → `shadow` route diagnostics → `listen_only` durable ingest → `active` for Bohumil only → add one secondary participant → multi-mention → approvals/handoffs.

## Secret ref contract

- Persist only `token_secret_ref` for v2 identities.
- Allowed schemes:
  - required production/default scheme: `secret://...`;
  - optional development/test scheme: `env://ENV_VAR_NAME`, accepted only when explicitly enabled by config/test resolver policy.
- Reject in v2 identity config:
  - raw Discord token-looking strings in any token ref field;
  - keys named `token`, `bot_token`, `discord_token`, `DISCORD_BOT_TOKEN`, or case/underscore variants;
  - attempts to reuse legacy `DISCORD_BOT_TOKEN` as the v2 credential source;
  - resolver output/token values appearing in serializable config, DB rows, diagnostics, exceptions, or logs.
- Resolver contract:
  - `SecretResolver.resolve(ref) -> SensitiveToken` (or equivalent) returns a token only at runtime connect/send boundaries;
  - resolver output is never persisted, repr'd, logged, included in config dumps, or returned by diagnostics/CLI;
  - failures must raise redacted exceptions containing the ref scheme/path only as allowed by policy, never the token value.
- Required secret tests:
  - no plaintext token in config dump/snapshot;
  - no plaintext token in DB rows, including identity registry, route decisions, message map, inbound/outbox payloads;
  - no plaintext token in exceptions/logs/caplog/CLI stdout/stderr;
  - raw token or legacy token key in v2 identity config fails validation.

## Non-goals

- Webhooks/single-bot identity as default MVP path.
- Discord-as-kanban as source of truth.
- Bot-to-bot mention triggers.
- Per-thread = per-agent mapping.
- Storing plaintext Discord bot tokens in `config.yaml`, DB, logs, snapshots, or test fixtures.
- Agent-agent coordination inferred from Discord messages; it must come from internal events.

---

# Implementation tasks

## Slice 0A — ADR/protocol contract + default-off config only

This is the first safe implementation slice. It may create Markdown docs and a default-off config schema/validation contract, but it must not implement Discord runtime clients, durable ingest, workers, outbox sends, or gateway wiring.

### Task 0A.1 — Write ADR and protocol reference

**Objective**
Freeze the v2 semantics before any runtime implementation.

**Files**
- Create directory if missing: `docs/adr/`
- Create directory if missing: `docs/reference/`
- Create: `docs/adr/discord-native-multibot-protocol-v2.md`
- Create: `docs/reference/discord-native-multibot-protocol-v2.md`
- Do not update runtime code in this task.

**Required decisions to document**
1. Native multi-bot MVP is the committed path: each participant agent uses its own Discord Application/Bot User/token/avatar/name/mention.
2. Mention is a key feature and first-class routing primitive.
3. Webhook fallback is explicit diagnostic/projection fallback only, not default MVP path.
4. Agent-agent Discord communication is projection, not source of truth; authoritative handoff/consult/review work comes from internal events.
5. Restart safety is required for topics, sessions, approvals, handoffs, inbound deliveries, outbox deliveries, message map, and route decisions.
6. Route priority: human explicit mention > human reply-to-agent > human default intake Bohumil > fail/clarify.
7. Bot-loop prevention: registered Hermes bot, external bot, webhook, and system authored messages never create Discord-originated inbound deliveries, even when they mention a registered agent bot.
8. `message_map` and `inbound_deliveries` durable schema fields listed in this plan are source-of-truth requirements for later slices.

**Acceptance tests / verification**
- `python -m pytest tests/test_packaging_metadata.py -q`
- Manual review gate: `git diff --name-only` for this task shows only Markdown/docs paths under `docs/`.

### Task 0A.2 — Add default-off config schema contract

**Objective**
Add a v2 config contract with secret refs only, exact rollout-mode validation, and no runtime Discord activation.

**Files**
- Modify: `gateway/config.py`
- Modify if needed for config load/dump/redaction only: `hermes_cli/config.py`
- Test: `tests/gateway/test_discord_native_multibot_config.py` (new)
- Existing drift/blocker to handle before depending on primary UI config tests: `DiscordPrimaryUIConfig` expected by primary tests may not exist in `gateway/config.py`; either reconcile the class/alias or mark the exact existing tests as blocked with a focused follow-up before any v2 runtime slice.

**Config shape**
```yaml
discord_native_multibot:
  enabled: false
  mode: off
  guild_allowlist: []
  default_intake_agent_id: bohumil
  identities:
    - agent_id: bohumil
      hermes_profile: default
      discord_application_id: "..."
      discord_bot_user_id: "..."
      token_secret_ref: "secret://hermes/discord/bohumil-token"
      capabilities: [intake, reply, approve_projection]
      allowed_scopes:
        guild_ids: ["..."]
```

**Validation contract**
1. Defaults: absent config parses to `enabled=false`, `mode=off`, empty `guild_allowlist`, and no identities.
2. Valid modes: `off`, `shadow`, `listen_only`, `active` only.
3. `enabled=false` keeps v2 inert; legacy `platforms.discord.token` / `DISCORD_BOT_TOKEN` behavior remains intact for legacy Discord only.
4. `mode=active` requires `enabled=true`, non-empty `guild_allowlist`, `default_intake_agent_id`, and an enabled identity matching the default intake agent.
5. `mode=listen_only` requires `enabled=true` and non-empty `guild_allowlist`.
6. Legacy `DISCORD_BOT_TOKEN` must not satisfy v2 token requirements.
7. Identity credentials must use `token_secret_ref` with allowed schemes from the Secret ref contract; raw tokens and legacy token keys are rejected.
8. Safe config dumps/redacted snapshots must not include resolver output or token-like plaintext.

**Acceptance tests / verification**
- `python -m pytest tests/gateway/test_discord_native_multibot_config.py -q`
- The new test file must include cases for:
  - default absent config is off/inert;
  - exact `off`, `shadow`, `listen_only`, `active` semantics/validation;
  - active requires `enabled`, `guild_allowlist`, and default intake identity;
  - `DISCORD_BOT_TOKEN` does not satisfy v2 identity token requirements;
  - `secret://...` accepted and raw token/`token`/`bot_token`/`discord_token`/`DISCORD_BOT_TOKEN` keys rejected;
  - config serialization/redacted dumps contain no plaintext token or resolver output.
- Manual review gate: no v2 runtime clients, gateway worker, durable ingest, or outbox sender are implemented in Slice 0A.

---

## Slice 0B — Durable schema migrations and store contract

### Task 0B.1 — Durable schema migrations

**Objective**
Create a durable store contract for inbox/outbox/topic/session/message_map/approvals/handoffs/internal events, with internal events as source of truth for agent-agent work.

**Files**
- Create: `gateway/discord_protocol_v2_store.py`
- Create: `gateway/discord_protocol_v2_schema.sql` or inline migration constants in the store module
- Test: `tests/gateway/test_discord_protocol_v2_store.py` (new)

**Schema contract**
1. `identity_registry(agent_id primary key, hermes_profile, discord_application_id, discord_bot_user_id, token_secret_ref, capabilities_json, scopes_json, enabled, version)`.
2. `topics(topic_id primary key, guild_id, channel_id, thread_id nullable, parent_channel_id nullable, title, state_json, version)`.
3. `topic_agent_sessions(topic_id, agent_id, hermes_session_id, session_key, state, version, unique(topic_id, agent_id))`.
4. `message_map` must include at minimum:
   - `discord_message_id primary key`, `guild_id`, `channel_id`, nullable `thread_id`, nullable `parent_channel_id`;
   - `direction` (`inbound|outbound|projection`), nullable `agent_id`, nullable `delivery_key`, nullable `outbox_delivery_id`, nullable `agent_event_id`;
   - `author_id`, `author_kind` (`human|registered_bot|external_bot|webhook|system`), nullable `author_bot_user_id`, nullable `source_client_agent_id`, `mentions_json`;
   - `created_at`, `updated_at`, `payload_json`.
5. `agent_events` must persist internal source-of-truth events:
   - `agent_event_id primary key`, `event_type`, nullable `source_agent_id`, `target_agent_id`, `topic_id`, `payload_json`, `status`, `created_at`, `version`.
6. `inbound_deliveries` must support Discord and internal-event sources:
   - `delivery_key primary key`;
   - `source_type` enum `discord_message|internal_event`;
   - `source_id` stable source identifier (`discord_message_id` for Discord, `agent_event_id` for internal event);
   - nullable `discord_message_id`;
   - nullable `agent_event_id`;
   - `target_agent_id`, `topic_id`, `route_reason`, `author_kind`, `payload_json`;
   - `status`, nullable `lease_owner`, nullable `lease_until`, `attempts`, `created_at`, `updated_at`, `state_version`;
   - unique idempotency constraint on `(source_type, source_id, target_agent_id)`.
7. `outbox_deliveries` must include stable source correlation and idempotency:
   - `outbox_delivery_id primary key`, `idempotency_key unique`, `target_agent_id`, `topic_id`, `channel_id`, nullable `thread_id`;
   - nullable `source_inbound_delivery_key`, nullable `source_agent_event_id`;
   - `delivery_kind` (`response|projection|diagnostic`), `payload_json`, `status`, nullable `lease_owner`, nullable `lease_until`, `attempts`, `created_at`, `updated_at`, `state_version`;
   - normal response key format: `response:{inbound_delivery_key}:{target_agent_id}`;
   - projection key format: `projection:{agent_event_id}:{target_agent_id}`.
8. `outbox_parts(outbox_delivery_id, part_index, status, discord_message_id nullable, unique(outbox_delivery_id, part_index))`.
9. `route_decisions(decision_id primary key, source_type, source_id, topic_id, author_kind, decision, target_agent_ids_json, reason, created_at, payload_json)` for diagnostics, including zero-delivery decisions.
10. `approvals`, `handoffs`, and `reconciliation_runs` must be restart-safe and refer to stable IDs, not Discord message text.

**Behavior contract**
1. Only `author_kind=human` may create Discord-originated inbound deliveries (`source_type='discord_message'`).
2. `registered_bot`, `external_bot`, `webhook`, and `system` authored mentions create zero inbound deliveries and must produce a diagnostic `route_decisions` row.
3. Internal handoff/consult/review creates `agent_events` and then at most one `inbound_deliveries(source_type='internal_event')` per target agent.
4. Discord projection of an internal event may write `message_map` but must not create a second delivery for the target agent.
5. Add idempotent insert APIs and lease APIs.
6. Add crash-safe state transitions: inbound `pending → leased → completed/failed/retryable`; outbox `pending → leased/sending → sent/acked/uncertain → reconciled`.

**Tests/verification**
- `python -m pytest tests/gateway/test_discord_protocol_v2_store.py -q`
- Restart gate unit test: create store, insert pending deliveries/events/outbox/message_map, close/reopen, assert all state survives.
- Idempotency gates:
  - replay of the same Discord message for the same target returns one delivery via unique `(source_type, source_id, target_agent_id)`;
  - replay of the same internal handoff event for the same target returns one delivery;
  - projection/re-ingest of an internal event Discord message creates/updates `message_map` and `route_decisions` but creates zero additional inbound deliveries.

---

## Slice 1 — Identity registry + onboarding

### Task 1.1 — Identity registry service

**Objective**
Load v2 identities, resolve secret refs only at runtime, expose safe identity metadata to gateway.

**Files**
- Create: `gateway/discord_identity_registry.py`
- Modify: `gateway/config.py`
- Test: `tests/gateway/test_discord_identity_registry.py` (new)

**Steps**
1. Add `DiscordIdentityRegistry.load(config, store, secret_resolver)`.
2. Validate uniqueness:
   - unique `agent_id`;
   - unique `discord_application_id`;
   - unique `discord_bot_user_id`;
   - no duplicate token secret refs unless explicitly allowed for diagnostics.
3. Add `resolve_token(agent_id)` that returns token only in memory and never logs it.
4. Add safe `redacted_snapshot()` for diagnostics.

**Tests/verification**
- Duplicate ID tests fail closed.
- Plaintext token test: passing a token-like value where `token_secret_ref` is expected fails validation.
- Log redaction test with `caplog`: token value never appears.

### Task 1.2 — Onboarding CLI skeleton

**Objective**
Add default-off operator CLI for plan/sync/verify/invite/reconcile without connecting the active gateway.

**Files**
- Create: `hermes_cli/discord_native.py`
- Modify: `hermes_cli/main.py`
- Modify: `hermes_cli/gateway.py` only if gateway subcommand nesting is preferred
- Test: `tests/hermes_cli/test_discord_native_cli.py` (new)

**Steps**
1. Add commands:
   - `hermes discord-native plan`
   - `hermes discord-native sync`
   - `hermes discord-native handoff-missing-secrets`
   - `hermes discord-native verify`
   - `hermes discord-native invite-links`
   - `hermes discord-native reconcile-guild`
2. `plan` prints desired identities/scopes and missing secrets with redacted refs.
3. `handoff-missing-secrets` invokes a secret-handoff provider abstraction, not chat plaintext.
4. `verify` checks bot user IDs, guild membership, intents, scopes, and no duplicate bot identities.
5. `invite-links` builds OAuth invite URLs from application IDs/scopes.
6. `reconcile-guild` compares registry to guild bots/channels/threads and outputs actions.

**Tests/verification**
- CLI snapshot tests with fake config and fake Discord REST.
- Secret gate: `handoff-missing-secrets` output contains URL/ref only, never token.
- No network in unit tests; Discord REST calls behind injectable client.

### Task 1.3 — Secret resolver and handoff integration

**Objective**
Formalize secret handling for multi-token gateway.

**Files**
- Create: `gateway/secret_refs.py` or reuse existing secret-ref helpers in `hermes_cli/config.py` if present
- Modify: `hermes_cli/discord_native.py`
- Test: `tests/gateway/test_discord_secret_refs.py` (new)

**Steps**
1. Define `SecretResolver` interface: `resolve(ref) -> SensitiveToken` (or equivalent runtime-only wrapper).
2. Enforce allowed ref schemes from the Secret ref contract: `secret://...` by default, optional `env://ENV_VAR_NAME` only under explicit dev/test resolver policy.
3. Reject raw Discord token-looking values and keys named `token`, `bot_token`, `discord_token`, or `DISCORD_BOT_TOKEN` in v2 identity config.
4. Implement fake/test resolver and env/dev resolver if needed; resolver output is runtime-only and must not be serializable.
5. Add redaction helper for refs and token-like values.
6. Add log/exception filter coverage for gateway logs.

**Tests/verification**
- `python -m pytest tests/gateway/test_discord_secret_refs.py -q`
- Secret-handling acceptance gate:
  - no plaintext token in DB rows;
  - no plaintext token in config serialization;
  - no plaintext token in exceptions/logs/caplog;
  - no plaintext token in CLI stdout/stderr.

---

## Slice 2 — Multi-token gateway listen-only / durable ingest

### Task 2.1 — Extract single-bot client wrapper from DiscordAdapter

**Objective**
Make existing Discord client logic reusable per identity without changing legacy behavior.

**Files**
- Modify: `plugins/platforms/discord/adapter.py`
- Create: `plugins/platforms/discord/client_runtime.py` (optional extraction)
- Test: existing Discord adapter tests + new `tests/gateway/test_discord_multiclient_runtime.py`

**Steps**
1. Preserve legacy `DiscordAdapter` API for `Platform.DISCORD`.
2. Extract reusable client creation/event registration into `DiscordClientRuntime` with params:
   - `agent_id`, `bot_user_id`, `token_resolver`, `allowed_mentions`, callbacks.
3. Keep voice/slash features legacy-only until v2 active mode explicitly opts in.
4. Do not introduce multi-token behavior in this task; only extraction with tests green.

**Tests/verification**
- `python -m pytest tests/gateway/test_discord_connect.py tests/gateway/test_discord_imports.py tests/gateway/test_discord_bot_filter.py -q`
- Legacy behavior unchanged when `discord_native_multibot.enabled=false`.

### Task 2.2 — New multi-token platform adapter/process wrapper

**Objective**
Start N Discord clients in one gateway process in `shadow/listen_only` mode.

**Files**
- Create: `plugins/platforms/discord/native_multibot.py`
- Modify: `plugins/platforms/discord/adapter.py` or `plugins/platforms/discord/__init__.py` for registration hook
- Modify: `plugins/platforms/discord/plugin.yaml`
- Modify: `gateway/run.py` only where v2 adapter needs gateway runner injection
- Test: `tests/gateway/test_discord_native_multibot_gateway.py` (new)

**Steps**
1. Add `DiscordNativeMultibotAdapter(BasePlatformAdapter)` or a companion runtime owned by the Discord plugin.
2. When v2 disabled, plugin registration returns existing `DiscordAdapter`.
3. When v2 enabled, create one runtime per identity, all sharing durable store and routing engine.
4. Use per-token locks keyed by `token_secret_ref`/hash, not raw token.
5. Listen-only mode stores inbound messages and route decisions but does not call Hermes or send replies.

**Tests/verification**
- Fake `discord.py` clients: two identities connect, both receive events, no token printed.
- Startup failure isolation: one bad token marks that identity unhealthy, other identities remain connected.
- Gateway default-off gate: legacy tests still pass with v2 absent/disabled.

### Task 2.3 — Durable ingest and topic normalization

**Objective**
Normalize Discord channel/thread messages into durable topics and inbound candidates.

**Files**
- Create: `gateway/discord_protocol_v2_ingest.py`
- Modify: `plugins/platforms/discord/native_multibot.py`
- Test: `tests/gateway/test_discord_protocol_v2_ingest.py` (new)

**Steps**
1. Define `topic_id = guild_id/channel_id/thread_id-or-root` deterministic key.
2. Store `topics` row for every observed allowed message.
3. Store `message_map` for every allowed observed Discord message and projection with the required fields: `author_id`, `author_kind`, `author_bot_user_id`, `source_client_agent_id`, `mentions_json`, source correlation fields, and payload metadata.
4. Classify `author_kind = human|registered_bot|external_bot|webhook|system` using the identity registry plus Discord webhook/system/bot flags.
5. Only `author_kind=human` may become a Discord-originated inbound delivery candidate; all other author kinds are diagnostics/projection only.
6. Ignore or record Discord system messages consistently with existing adapter behavior, but they must create zero inbound deliveries.

**Tests/verification**
- Same thread after restart maps to same topic.
- Parent channel and thread ID are preserved.
- Ingest is idempotent on repeated Discord event/replay.
- Registered Hermes bot, external bot, webhook, and system messages persist message/route diagnostics but produce zero inbound delivery candidates.

---

## Slice 3 — Mention/reply/default routing

### Task 3.1 — Routing engine

**Objective**
Implement deterministic routing independent from Discord adapter side effects.

**Files**
- Create: `gateway/discord_protocol_v2_routing.py`
- Modify: `gateway/discord_protocol_v2_store.py`
- Test: `tests/gateway/test_discord_protocol_v2_routing.py` (new)

**Steps**
1. Build mention parser from Discord user mentions and registry `discord_bot_user_id`.
2. Route priority applies only to `author_kind=human` Discord messages:
   1. explicit human-authored mention;
   2. reply-to-agent from `message_map`;
   3. default intake agent `bohumil` for unmentioned human topic message;
   4. fail/clarify when policy disallows default intake.
3. Multi-mention creates one inbound delivery per mentioned agent.
4. No fanout to non-mentioned agents.
5. Non-human author kinds (`registered_bot`, `external_bot`, `webhook`, `system`) always create zero Discord-originated inbound deliveries, even when they mention registered agent bots; record a diagnostic `route_decisions` row with the suppression reason.
6. Bot-authored/projection messages may update `message_map` and projected state only.

**Tests/verification**
- Human `@agentA @agentB` creates exactly two deliveries.
- Human mentions unknown bot → policy fail/clarify, no Hermes invocation.
- Human reply to message authored by agentB routes to agentB without mention.
- No mention routes to Bohumil only.
- Registered Hermes bot `@agentB` creates zero inbound deliveries and one diagnostic route decision.
- External bot `@agentB` creates zero inbound deliveries and one diagnostic route decision.
- Webhook `@agentB` creates zero inbound deliveries and one diagnostic route decision.
- System message/mention creates zero inbound deliveries and one diagnostic route decision when recorded.

### Task 3.2 — Bot-loop prevention acceptance gate

**Objective**
Prevent infinite bot-to-bot Discord loops by construction.

**Files**
- Test: `tests/gateway/test_discord_protocol_v2_bot_loop_prevention.py` (new)
- Modify: routing/ingest only if tests fail.

**Steps**
1. Simulate agentA output mentioning agentB in Discord.
2. Store it as projection in `message_map` with `author_kind=registered_bot`, `source_client_agent_id=agentA`, `author_bot_user_id` set, and `mentions_json` containing agentB.
3. Re-ingest as Discord bot-authored event after reconnect/replay.
4. Assert no inbound delivery for agentB and assert a diagnostic `route_decisions` row explains suppression.
5. Simulate external bot, webhook, and system messages mentioning agentB; each must create zero inbound deliveries and a diagnostic route decision.
6. Simulate internal `handoff.requested` from agentA to agentB and assert delivery is created from `source_type=internal_event`, not from Discord mention.
7. Re-ingest the Discord projection of that internal event and assert it does not create another delivery.

**Tests/verification**
- Required acceptance command:
  - `python -m pytest tests/gateway/test_discord_protocol_v2_bot_loop_prevention.py -q`
- Must pass before active outbox can be enabled.
- Covered cases: registered Hermes bot mention, external bot mention, webhook mention, system mention, internal handoff replay, and Discord projection replay.

### Task 3.3 — Session mapping per topic + agent

**Objective**
Replace/extend session mapping for v2 without breaking legacy `build_session_key()`.

**Files**
- Modify: `gateway/session.py` only to add v2 helper or bridge, not to break legacy keys
- Create/modify: `gateway/discord_protocol_v2_sessions.py`
- Test: `tests/gateway/test_discord_protocol_v2_session_mapping.py` (new)
- Existing related: `tests/gateway/test_discord_primary_session_mapping.py`

**Steps**
1. Add v2 helper:
   - `build_discord_v2_session_key(topic_id, agent_id) -> "discord:v2:topic:{topic_id}:agent:{agent_id}"`.
2. Store mapping in `topic_agent_sessions`, not only `sessions.json`.
3. Reuse `SessionStore`/`SessionDB` for Hermes transcript `session_id`, but durable v2 mapping owns `topic × agent`.
4. Keep thread as topic/work unit, not agent identity.

**Tests/verification**
- Same topic + same agent reuses session after restart.
- Same topic + different agents get different sessions.
- Different topics + same agent get different sessions.
- Legacy Discord session tests unchanged when v2 off.

---

## Slice 4 — Hermes runtime invocation + native outbox

### Task 4.1 — Agent invocation worker

**Objective**
Turn durable inbound deliveries into Hermes profile-specific agent runs.

**Files**
- Create: `gateway/discord_protocol_v2_worker.py`
- Modify: `gateway/run.py` to start/stop worker when v2 active/listen mode requires it
- Test: `tests/gateway/test_discord_protocol_v2_worker.py` (new)

**Steps**
1. Lease pending inbound deliveries by `delivery_key`.
2. Resolve `agent_id → hermes_profile` from identity registry.
3. Build `SessionSource` with platform Discord, topic fields, target agent fields in metadata/channel prompt if needed.
4. Invoke existing Hermes runtime path in `GatewayRunner` without changing legacy adapter flow.
5. Persist generated response to `outbox_deliveries`; do not send inline. Use `idempotency_key=response:{inbound_delivery_key}:{target_agent_id}` and populate `source_inbound_delivery_key`.

**Tests/verification**
- Fake agent returns response; outbox row is created once.
- Worker crash before completion leaves delivery retryable after lease expiry.
- Worker replay after completion does not create duplicate outbox rows.

### Task 4.2 — Native outbox sender

**Objective**
Send agent responses through the correct native Discord bot identity with durable idempotency.

**Files**
- Create: `gateway/discord_protocol_v2_outbox.py`
- Modify: `plugins/platforms/discord/native_multibot.py`
- Test: `tests/gateway/test_discord_protocol_v2_outbox.py` (new)

**Steps**
1. Split messages according to Discord 2000 char limit; use existing chunking behavior from `DiscordAdapter.send()` where possible.
2. `outbox_deliveries.idempotency_key` is unique and based on stable source correlation, not transient send attempts:
   - normal response: `response:{inbound_delivery_key}:{target_agent_id}` with `source_inbound_delivery_key` populated;
   - projection of an internal event: `projection:{agent_event_id}:{target_agent_id}` with `source_agent_event_id` populated.
3. Part-level send idempotency remains `outbox_delivery_id + part_index` in `outbox_parts`.
4. Send with the target agent’s Discord client/token.
5. Persist Discord message IDs into `outbox_parts` and `message_map` with `direction=outbound|projection`, `source_client_agent_id`, `author_kind=registered_bot`, `author_bot_user_id`, and `mentions_json`.
6. On send exception after uncertain network result, mark `uncertain`, not `failed`; reconciliation handles it.

**Tests/verification**
- Same logical response (`response:{inbound_delivery_key}:{target_agent_id}`) processed twice creates/sends one outbox delivery.
- Same projection (`projection:{agent_event_id}:{target_agent_id}`) processed twice creates/sends one outbox delivery.
- Multi-part response persists every part with stable `part_index`.
- Wrong/missing client for agent fails closed and remains retryable.

### Task 4.3 — Reconciliation after crash

**Objective**
Recover `sending/uncertain` outbox without duplicates after gateway restart.

**Files**
- Modify: `gateway/discord_protocol_v2_outbox.py`
- Create: `gateway/discord_protocol_v2_reconcile.py`
- Test: `tests/gateway/test_discord_protocol_v2_restart_resilience.py` (new)

**Steps**
1. On startup, scan `outbox_deliveries` in `leased/sending/uncertain` with expired lease.
2. Use `message_map` and recent Discord history fetch (fake in tests) to decide if message already sent.
3. If found, mark `acked` and map Discord ID.
4. If not found and retry budget remains, enqueue send.
5. Record reconciliation run for audit.

**Tests/verification**
- Acceptance restart gate:
  - inbound delivery leased before crash is retried once;
  - outbox marked `sending` before crash is not duplicated if Discord message exists;
  - pending approvals/handoffs/topic/session mappings survive close/reopen;
  - no duplicate response rows or Discord sends.
- Required command:
  - `python -m pytest tests/gateway/test_discord_protocol_v2_restart_resilience.py -q`

---

## Slice 5 — Durable approvals/handoffs

### Task 5.1 — Durable approvals v2

**Objective**
Move approvals needed by v2 into durable store while preserving existing structured approval behavior.

**Files**
- Modify: `gateway/discord_approvals.py` or create `gateway/discord_protocol_v2_approvals.py`
- Modify: `gateway/run.py` where approval callbacks are created/handled
- Test: `tests/gateway/test_discord_protocol_v2_approvals.py` (new)
- Existing: `tests/gateway/test_discord_approvals.py`, `tests/gateway/test_discord_approval_auth.py`

**Steps**
1. Store approval state in SQLite with stable `approval_id`, `topic_id`, `agent_id`, `requesting_event_id`, `status`.
2. Button/component custom IDs contain opaque IDs only, no secrets.
3. Approval policy uses registry capabilities/scopes plus existing owner/admin controls.
4. Restart with pending approvals should keep buttons resolvable and state pending.

**Tests/verification**
- Existing approval tests remain green.
- Pending approval survives store reopen.
- Duplicate approve/deny is idempotent and audited.

### Task 5.2 — Internal handoff/consult/review events

**Objective**
Support agent-agent collaboration via internal events only.

**Files**
- Create: `gateway/discord_protocol_v2_events.py`
- Create: `gateway/discord_protocol_v2_handoffs.py`
- Modify: `gateway/discord_protocol_v2_worker.py`
- Test: `tests/gateway/test_discord_protocol_v2_handoffs.py` (new)

**Steps**
1. Define event envelope:
   ```json
   {"agent_event_id":"evt_...","event_type":"handoff.requested","source_agent_id":"agentA","target_agent_id":"agentB","topic_id":"...","payload":{...}}
   ```
2. Persist the envelope in `agent_events`; this row is the source of truth.
3. Only internal event API can create target-agent delivery for handoff/consult/review, using `inbound_deliveries(source_type='internal_event', source_id=agent_event_id, agent_event_id=agent_event_id, target_agent_id=...)`.
4. Projection outbox rows for Discord use `idempotency_key=projection:{agent_event_id}:{target_agent_id}` and populate `source_agent_event_id`.
5. Discord projection message may mention target bot, but routing engine must ignore that bot-authored/projection mention and emit only diagnostics.
6. Durable handoff states: `requested`, `accepted`, `declined`, `completed`, `cancelled`.

**Tests/verification**
- Internal handoff creates one delivery for target agent with `source_type=internal_event`.
- Replay of the same internal handoff creates no duplicate delivery.
- Discord replay of projection creates zero additional deliveries and records a diagnostic route decision.
- Handoff state survives restart and is not duplicated.

---

## Slice 6 — Ops diagnostics/fallback

### Task 6.1 — Diagnostics surfaces

**Objective**
Make v2 observable before/after activation.

**Files**
- Create: `gateway/discord_protocol_v2_diagnostics.py`
- Modify: `hermes_cli/discord_native.py`
- Modify: `gateway/status.py` if runtime status should include v2 health
- Test: `tests/gateway/test_discord_protocol_v2_diagnostics.py` (new)

**Steps**
1. Add health snapshot:
   - identities connected/unhealthy;
   - pending inbound/outbox counts;
   - expired leases;
   - uncertain sends;
   - last reconciliation status;
   - secret refs missing/unresolved (redacted).
2. CLI `verify` and `reconcile-guild` use same diagnostics core.
3. Add log redaction for token-like values and secret refs.

**Tests/verification**
- Diagnostic output contains agent IDs, bot user IDs, state counts.
- Diagnostic output never contains token values.

### Task 6.2 — Webhook fallback/diagnostic mode only

**Objective**
Keep webhook/single-bot path explicit and non-default.

**Files**
- Modify: `gateway/platforms/webhook.py` only if fallback projection needs a documented bridge
- Modify: `website/docs/user-guide/messaging/discord.md`
- Test: `tests/gateway/test_discord_protocol_v2_webhook_fallback.py` (new)

**Steps**
1. Add config validation: webhook fallback cannot be auto-enabled by v2.
2. Diagnostic fallback can send a projection of an internal event, but cannot receive authoritative agent-agent triggers.
3. Webhook-authored messages are `author_kind=webhook`; they must produce zero inbound deliveries even if they mention registered agent bots, and should emit diagnostic route decisions.
4. Document operator-only use.

**Tests/verification**
- v2 active without webhook config does not instantiate webhook fallback.
- webhook-authored mention cannot trigger target agent.

---

## Cross-cutting acceptance gates

### Gate 0A — First slice readiness

Run:
```bash
python -m pytest tests/gateway/test_discord_native_multibot_config.py -q
python -m pytest tests/test_packaging_metadata.py -q
```

Expected:
- absent v2 config is default-off/inert;
- `off`, `shadow`, `listen_only`, `active` validation matches the rollout semantics above;
- `active` requires `enabled=true`, `guild_allowlist`, and a valid default intake identity;
- legacy `DISCORD_BOT_TOKEN` does not satisfy v2 identity token requirements;
- only secret refs are accepted for v2 identities and safe dumps contain no resolver output/token value;
- implementation diff for Slice 0A contains docs/default-off config/test contract only, no runtime Discord clients/ingest/workers/outbox.

### Gate A — Default-off and legacy compatibility

Run:
```bash
python -m pytest \
  tests/gateway/test_discord_connect.py \
  tests/gateway/test_discord_send.py \
  tests/gateway/test_discord_reply_mode.py \
  tests/gateway/test_discord_allowed_mentions.py \
  tests/gateway/test_discord_bot_filter.py \
  tests/gateway/test_discord_bot_auth_bypass.py \
  tests/gateway/test_discord_slash_commands.py \
  -q
```

Expected:
- legacy Discord remains unchanged when `discord_native_multibot.enabled=false`;
- no new config required for existing users;
- v2 `off` mode does not instantiate v2 clients, store, workers, or outbox.

### Gate B — Routing correctness

Run:
```bash
python -m pytest \
  tests/gateway/test_discord_protocol_v2_routing.py \
  tests/gateway/test_discord_protocol_v2_bot_loop_prevention.py \
  tests/gateway/test_discord_protocol_v2_session_mapping.py \
  -q
```

Expected:
- explicit mention > reply-to-agent > Bohumil default intake > fail/clarify;
- multi-mention delivers only to mentioned agents;
- only `author_kind=human` Discord messages create Discord-originated inbound deliveries;
- registered Hermes bot, external bot, webhook, and system authored mentions create zero inbound deliveries and produce diagnostic route decisions;
- internal handoff creates delivery from `source_type=internal_event`; Discord projection replay creates zero additional deliveries.

### Gate C — Restart resilience

Run:
```bash
python -m pytest tests/gateway/test_discord_protocol_v2_restart_resilience.py -q
```

Expected:
- pending approvals/handoffs/topic state/outbox/message_map/session mapping survive restart;
- `sending/uncertain` outbox reconciles;
- no duplicate Hermes invocation and no duplicate Discord send.

### Gate D — Secret handling

Run:
```bash
python -m pytest \
  tests/gateway/test_discord_secret_refs.py \
  tests/gateway/test_discord_identity_registry.py \
  tests/hermes_cli/test_discord_native_cli.py \
  -q
```

Expected:
- no plaintext tokens in config dumps, DB rows, logs, CLI output, exceptions, or snapshots;
- only `token_secret_ref` is persisted;
- token values exist only in memory during connect/send.

### Gate E — Active mode smoke with fakes

Run:
```bash
python -m pytest \
  tests/gateway/test_discord_native_multibot_gateway.py \
  tests/gateway/test_discord_protocol_v2_worker.py \
  tests/gateway/test_discord_protocol_v2_outbox.py \
  -q
```

Expected:
- two fake Discord clients can ingest messages concurrently;
- correct Hermes profile is invoked per target agent;
- each agent sends via its own bot identity.

## Migration/rollout checklist

1. Ship Slice 0A: ADR/reference docs + default-off config contract with `enabled=false` and no runtime v2 activation.
2. Ship durable schema/store with internal events, message map, route decisions, and idempotent inbox/outbox.
3. Add CLI `plan/verify` and require all identities to pass before `shadow`.
4. Enable `shadow` for one guild; compare route decisions with operator expectations.
5. Enable `listen_only`; validate durable ingest, topic mapping, bot-loop diagnostics, and no response outbox sends.
6. Enable `active` for Bohumil only.
7. Add second participant agent and validate mention/reply routing.
8. Enable handoff/consult/review internal events.
9. Keep webhook fallback disabled unless operator explicitly enables diagnostic mode.

## Risks / blockers

- **Config/test drift:** existing primary tests import `DiscordPrimaryUIConfig`, but current scanned `gateway/config.py` does not define it. Resolve before building v2 on top of primary UI semantics.
- **Adapter size/coupling:** `plugins/platforms/discord/adapter.py` is large and mixes transport, slash UI, approvals, voice, threading, and routing; extraction must be incremental and test-driven.
- **Token handling migration:** legacy `DISCORD_BOT_TOKEN` and `PlatformConfig.token` are plaintext-friendly; v2 must avoid reusing those paths for multi-bot identities.
- **Discord privileged intents:** message content/member intents differ per bot application; onboarding `verify` must catch missing intents before active mode.
- **Concurrency:** multiple clients for same guild/topic can observe the same message; durable idempotency must be in store, not in-memory dedupe.
- **Crash uncertainty:** Discord send can succeed while process crashes before DB ack; reconciliation is mandatory before active rollout.
- **Bot loops:** existing single-bot `DISCORD_ALLOW_BOTS` behavior is not enough for v2; bot-authored mention suppression must be enforced in routing engine and tested.
