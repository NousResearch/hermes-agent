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
| (platform only) | 0 |

So a thread route (8) beats a channel route (4) beats a guild route (2) within the same server.
If no route matches, the message uses the default/active profile.

## Strict private Telegram topics

An owner-private Telegram bot can opt into a stricter admission layer under
`platforms.telegram.extra.topic_routing`. Unlike ordinary `profile_routes`, strict topic
routing has no default, root-DM, lobby, or last-active fallback: the update must carry an
exact registered `(chat_id, message_thread_id)` pair. Every strict route must also have an
exact `profile_routes` entry selecting the same profile, and multiplexing must be enabled.

```yaml
gateway:
  multiplex_profiles: true
  profile_routes:
    - {name: companion-topic, platform: telegram, chat_id: "123456789", thread_id: "1", profile: companion}
    - {name: knowledge-topic, platform: telegram, chat_id: "123456789", thread_id: "77", profile: knowledge}
    - {name: tutor-topic, platform: telegram, chat_id: "123456789", thread_id: "78", profile: tutor}

platforms:
  telegram:
    extra:
      topic_routing:
        mode: strict
        routes:
          - {chat_id: "123456789", thread_id: 1, profile: companion}
          - {chat_id: "123456789", thread_id: 77, profile: knowledge}
          - {chat_id: "123456789", thread_id: 78, profile: tutor}
        # Lazy plugin factories are inert unless explicitly enabled here.
        hooks:
          - {profile: companion, plugin: companion-actions}
```

Each `hooks` entry binds a route profile to the exact plugin id that owns its lazy factory.
Hook state is resolved from that routed profile's home, even though the factory is
instantiated by the default multiplex gateway. Once Telegram is connected, the adapter
starts hook lifecycle work (including durable exact-envelope reconciliation) and awaits
its stop hook before transport teardown.
Plugin discovery only registers lazy factories; enabling a hook constructs it at adapter
startup. An unknown, wrong-owner, duplicate,
unrouted, misbound, or misconfigured hook fails startup. Omitting `hooks` leaves all plugin
credentials and state untouched while the three isolated conversation profiles continue to
route normally. Replies, media, callback actions, and receipts produced by a hook remain
bound to the exact origin chat and thread, including explicit General topic id `1`.
Status caches include that thread id, and a later edit or delete is accepted only for a
message id recorded as created in the same exact strict origin during this gateway run.

The bundled `sol-food` hook is one example: when explicitly bound to a strict route it
uses `/food <description>` or one bounded photo to create a noncanonical proposal, then
presents opaque origin-bound choices. Health remains the only canonical writer, and no
write occurs before an authenticated Confirm callback.

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

## Relationship to multiplexing

`profile_routes` requires `gateway.multiplex_profiles: true`. Multiplexing is what
activates the per-profile runtime scope (per-profile `HERMES_HOME`, secret scope, and
profile-namespaced session keys); routing is the decision layer that picks *which*
profile a given guild/channel/thread lands in. With multiplexing off, `profile_routes`
is ignored entirely — behavior is byte-identical to a single-profile gateway.
