# Profile-Based Routing for Inbound Messages

> **Audience:** Gateway operators and contributors
> **Source files:** `gateway/profile_routing.py`, `gateway/run.py` (`_profile_name_for_source`), `gateway/platforms/base.py` (`build_source`), `gateway/config.py`
> **Related:** [Session Lifecycle](session-lifecycle.md), `docs/design/profile-builder.md`

## Overview

By default a single gateway run uses one profile (memory, persona, tools). **Profile-based
routing** lets one gateway instance serve **multiple isolated profiles**, selecting which
profile handles an inbound message based on *where the message came from* — the platform,
server (`guild_id`), channel (`chat_id`), and/or thread (`thread_id`).

This is the inbound counterpart to multiplexing: instead of running N gateways, run one
gateway and route per-community / per-channel / per-thread to a dedicated profile. Each
profile keeps fully isolated state (`MEMORY.md`, `USER.md`, `SOUL.md`, sessions, tools).

Routing is **platform-generic**: it works for Discord, Telegram, Feishu, Slack, and every
adapter — not just Discord.

## Configuring routes

Routes live under `profile_routes` in `config.yaml`. Both the top-level and the nested
`gateway.profile_routes` forms are accepted (the nested form is what
`hermes config set gateway.profile_routes ...` writes).

```yaml
profile_routes:
  # Route an entire Discord server (guild) to one profile.
  - name: server-default
    platform: discord
    guild_id: "1234567890"
    profile: server-profile

  # Override a specific channel within that server with a different profile.
  - name: support-channel
    platform: discord
    guild_id: "1234567890"
    chat_id: "9876543210"
    profile: support-profile

  # Pin a Telegram group to a profile (Telegram has no guild_id — chat_id only).
  - name: tg-group
    platform: telegram
    chat_id: "-1001234567890"
    profile: tg-profile

  # Route a single Discord thread.
  - name: standup-thread
    platform: discord
    guild_id: "1234567890"
    chat_id: "9876543210"
    thread_id: "1111111111"
    profile: standup
```

### Fields

| Field | Required | Description |
|---|---|---|
| `name` | yes | Human-readable route identifier (used in logs). |
| `platform` | yes | Adapter platform: `discord`, `telegram`, `feishu`, `slack`, … |
| `profile` | yes | Target profile name (must exist under `~/.hermes/profiles/<name>`). |
| `guild_id` | no | Server/guild (Discord). |
| `chat_id` | no | Channel/group/DM id. |
| `thread_id` | no | Thread id within a channel. |
| `enabled` | no | Default `true`; set `false` to disable a route without removing it. |

## Matching rules

A route matches an inbound source when **every discriminator the route declares is satisfied**
(conjunctive / AND). A field the route leaves unset is ignored.

- **`platform`** must equal the source platform exactly.
- **`thread_id`** (if set) must equal the source thread id.
- **`chat_id`** (if set) must match the source channel **or** its parent — a thread in a
  channel matches the channel's route (hierarchical match for Discord forums/threads).
- **`guild_id`** (if set) must equal the source guild.

> A route declaring **both** `guild_id` and `chat_id` requires both to hold. A channel match
> alone does not satisfy a guild constraint — this is intentional and tested.

When multiple routes match, the **most specific** one wins. Specificity is additive:

| Discriminator | Weight |
|---|---|
| `thread_id` | 8 |
| `chat_id` | 4 |
| `guild_id` | 2 |
| (platform only) | 1 |

So a thread route (8) beats a channel route (4) beats a guild route (2) within the same server.
If no route matches, the message uses the default/active profile.

## How it works at runtime

1. An inbound message arrives at a platform adapter.
2. `BasePlatformAdapter.build_source` builds the `SessionSource` for the message. Every
   adapter carries a back-reference to the running `GatewayRunner`
   (`gateway_runner`, injected in `gateway/run.py`), so it asks the runner to resolve the
   target profile via `_profile_name_for_source`.
3. `_profile_name_for_source` runs the configured routes through `match_profile_route` and
   stamps `source.profile` with the winning route's profile (or leaves it unset).
4. Downstream, `_resolve_profile_home_for_source` chooses the profile home directory
   (`source.profile` → active profile → `default`) and the session is scoped per-profile, so
   each routed community gets isolated memory and conversation state.

Because `gateway_runner` is injected for **all** adapters (declared on `BasePlatformAdapter`),
every platform goes through this path — not just Discord.

## Migration / coexistence with multiplexing

`profile_routes` is independent of `gateway.multiplex_profiles`. Multiplexing splits the
gateway across model credentials; profile routing splits conversation state across profiles.
They compose: you may multiplex credentials while also routing channels to distinct profiles.
