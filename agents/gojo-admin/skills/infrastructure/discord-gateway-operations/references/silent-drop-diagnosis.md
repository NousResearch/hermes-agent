# Silent-Drop Diagnosis: Bot Connected but Never Responds

## Scenario

A specialist Discord gateway is running (`systemctl active (running)`), reports `Connected as BotName#1234` in its gateway log, but never responds to any message — even when @-mentioned. No errors, no retry loops, no `PrivilegedIntentsRequired`.

## Root cause

The profile's `config.yaml` is **missing the `discord:` section entirely**. This means:

1. `apply_yaml_config_fn()` (adapter.py line 6232) never fires
2. `DISCORD_REQUIRE_MENTION` env var is never set → defaults to `"true"`
3. `DISCORD_FREE_RESPONSE_CHANNELS` env var is never set → empty set
4. Every message hits this guard (adapter.py line 4703):
   ```python
   if require_mention and not is_free_channel and not in_bot_thread:
       if self._client.user not in message.mentions and not mention_prefix:
           return  # <-- silent drop
   ```
5. The multi-agent filter above it (line 837) catches messages where *other* bots are `@`-mentioned but we aren't. When no mentions exist at all (e.g. bot-to-bot text like "Dezzy — please review"), `message.mentions` is empty, so neither filter catches it — the `require_mention` check handles the drop.

**Key insight:** the gateway connects and authenticates successfully. The Discord API confirms the connection. Message routing is configured *after* connection via env vars. Missing config = silent drop.

## Detection

```bash
# Check if discord: section exists
grep -c '^discord:' /home/kensei/.hermes/profiles/<profile>/config.yaml
# Output: 0 → missing

# Compare against a known-good specialist
grep -A15 '^discord:' /home/kensei/.hermes/profiles/wesker/config.yaml

# Check gateway log for message processing evidence
grep -c 'Processing message\|Sending response\|free_response' \
  /home/kensei/.hermes/profiles/<profile>/logs/gateway.log
# Output: 0 → silent drop
```

## Fix

Add the `discord:` section to the profile's `config.yaml`. Minimum viable:

```yaml
discord:
  require_mention: true
  free_response_channels: '<HOME_CHANNEL_ID>'     # from profile .env DISCORD_HOME_CHANNEL
  allowed_channels: ''
  auto_thread: true
  reactions: true
  channel_prompts: {}
  extra:
    gateway_restart_notification: false
  server_actions: ''
```

Then restart:

```bash
sudo systemctl restart hermes-gateway-<profile>
```

No token change needed. No re-auth. The `discord:` section only controls env vars for message routing.

## Batch fix (all profiles at once)

When multiple profiles lack the section, patch the config files in a loop:

```bash
for p in profile1 profile2 profile3; do
  env_file="/home/kensei/.hermes/profiles/$p/.env"
  channel_id=$(grep -oP 'DISCORD_HOME_CHANNEL=\K\d+' "$env_file")
  # ... insert discord: section into $p/config.yaml ...
done

sudo systemctl restart hermes-gateway-{profile1,profile2,profile3}
```

### Pitfalls in batch fixes

- **Do not patch after `always_skills:` if a `discord:` section already exists.** You'll create a duplicate, which causes the config parser to merge them, usually keeping the first (empty) one. Always check `grep -c '^discord:'` first. If count > 1, remove the old empty section.
- **channel_prompts entries on Kensei's root config must be migrated**, not duplicated. If Kensei's root `config.yaml` has a `channel_prompts` entry for `#ai-learning-qa`, that entry forces Kensei to speak as MrHermagi in that channel. Move it to MrHermagi's profile config, not copy it.
- **After the fix, restart ALL gateways**, not just the one you edited. The channel directory cache is per-gateway.

## Incident timeline (2026-05-24)

| Time | Event |
|---|---|
| 00:43 | Kensei posts UX review brief in `#design-review` as "Dezzy — please review..." |
| 00:43 | Dezzy silently drops it — `message.mentions` empty, `require_mention=true`, no `free_response_channels` (no `discord:` section at all) |
| 01:00 | Dezzy `discord:` section added + gateway restart |  
| 01:20 | Remaining 7 specialists checked — all 7 also missing the section. Patched with `free_response_channels` per domain |
| 01:30 | 6 profiles found with duplicate `discord:` sections (old empty + new populated). Cleaned up |
| 01:40 | Root Kensei config had `#ai-learning-qa` configured as free_response with MrHermagi prompt — migrated to MrHermagi |
| 01:45 | `allow_bots: mentions` added to all 9 profiles |
