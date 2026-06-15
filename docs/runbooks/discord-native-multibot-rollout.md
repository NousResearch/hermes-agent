# Discord Native Multi-Bot v2 Rollout Runbook

This runbook is for operators rolling out Discord native multi-bot v2. All CLI diagnostics are offline-safe by default: they read config/state, use fake or injected clients in tests, and do not call Discord unless a future operator-gated live provider is explicitly added.

## Rollout phases

1. **off**
   - Keep `discord_native_multibot.enabled: false` and `mode: off`.
   - Create Discord applications and bot users, but do not start production traffic.
   - Run `hermes discord-native plan` and confirm desired identities, guild allowlist, and missing secret refs.
2. **shadow**
   - Configure identities for Bohumil and the second agent with token secret refs and guild scopes.
   - Keep active response delivery disabled; use stored events/projections only for observation.
   - Run `hermes discord-native sync --dry-run`, then `sync` to persist safe identity metadata only.
3. **listen_only**
   - Set `enabled: true`, `mode: listen_only`, and a narrow `guild_allowlist`.
   - Bots may observe/ingest according to the v2 lane, but must not produce autonomous user-visible responses.
   - Run `hermes discord-native verify` and `reconcile-guild`; without an injected client these return `not_implemented_without_client` and `network: not_used`.
4. **active Bohumil**
   - Make Bohumil the `default_intake_agent_id` and enable only Bohumil for active response handling.
   - Confirm Message Content Intent, Server Members Intent, invite scopes, and channel permissions.
5. **active second agent**
   - Enable the second agent identity after Bohumil has stable delivery and reconciliation.
   - Re-run reconcile and check for missing/extra bot users in each allowed guild.
6. **handoff**
   - Enable handoff/routing rules after both bots are present and healthy.
   - Use outbox/reconciliation diagnostics for uncertain sends before increasing scope.

## Gateway/config entrypoint

Native v2 is instantiated by the normal Discord platform plugin. The gateway only creates the Discord adapter when the Discord platform itself is enabled, so production rollout requires both switches:

```yaml
platforms:
  discord:
    enabled: true

discord_native_multibot:
  enabled: true
  mode: listen_only  # then active after the rollout gates pass
```

Do **not** use legacy `DISCORD_BOT_TOKEN` (or `platforms.discord.token`) for v2 identities. V2 identities must use `token_secret_ref: "secret://..."`; runtime resolution comes from the gateway secret resolver only, for example `HERMES_SECRET_REFS_JSON` or a `HERMES_SECRET_REF_<SHA256(secret://...)>` environment variable. Legacy single-bot credentials remain legacy-only.

## Discord application checklist

For each agent identity:

- Create a separate Discord application and bot user in the Discord Developer Portal.
- Record the Application ID as `discord_application_id` and the bot user ID as `discord_bot_user_id`.
- Store the bot token only through a secret ref such as `secret://hermes/discord/bohumil-token`; never paste plaintext tokens into config, logs, or CLI output.
- Enable **Message Content Intent**. Enable **Server Members Intent** when membership/role checks or guild reconciliation need member data.
- Invite with scopes `bot` and `applications.commands`.
- Grant at least View Channels, Send Messages, Read Message History, Attach Files, and Embed Links. Add Send Messages in Threads and Add Reactions when needed.
- Limit `guild_allowlist` and per-identity `allowed_scopes.guild_ids` before entering `listen_only` or `active`.

## Operator diagnostics

Offline/default commands:

```bash
hermes discord-native plan
hermes discord-native sync --dry-run
hermes discord-native sync
hermes discord-native invite-links --permissions 274878286912
hermes discord-native verify
hermes discord-native reconcile-guild --guild-id <guild-id>
```

Expected safety properties:

- `plan` shows desired identities, mode, guild allowlist, and missing or unchecked secret refs.
- `verify` without a fake/injected client does **not** connect to Discord; it returns `network: not_used` and `not_implemented_without_client`.
- `verify` with a fake/injected client reports bot user ID mismatch, missing guild membership, Message Content Intent, scopes, permissions, and represented secret presence.
- `reconcile-guild` compares desired vs. actual bot users and reports present, missing, and extra bot IDs.
- Outputs redact plaintext tokens and full `secret://...` / `env://...` paths.

Future live checks must be explicitly operator-gated. The current `--operator-network` flag is a gate marker only and returns `operator_network_not_implemented`; it does not perform network I/O.

## Rollback

Rollback in the reverse order and verify after each step:

1. **active → listen_only**: stop user-visible sends while keeping observation enabled.
2. **listen_only → shadow**: stop ingestion/active delivery and keep only offline projections/metadata.
3. **shadow → off**: set `mode: off`, then `enabled: false`.
4. If a token may be exposed, reset it in Discord, rotate the backing secret, and re-run `plan`/`verify` to confirm only redacted refs appear.

Do not use webhook fallback as a v2 control plane. If configured for diagnostics, it is projection-only and must not trigger agent-to-agent delivery.
