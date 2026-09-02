# Multiplexing Gateway

One gateway process can serve every profile in the install. The mode is opt-in
(`gateway.multiplex_profiles`, default `false`), and everything it changes
reverts the moment the flag is off. This document is the design rationale
referenced from `agent/secret_scope.py` ("Workstream A"): what is isolated per
profile, the mechanism that isolates it, and what deliberately stays
process-global.

## Overview

Without multiplexing, one gateway process serves exactly one profile — its
`.env`, sessions, skills, and platform adapters — and multi-profile installs
run one process per profile. Multiplexing collapses that into a single
process: the default profile plus every served named profile get their own
adapters, secrets, sessions, and cron ticks, while sharing one event loop, one
HTTP listener, one process lock, and one status surface.

The design constraint that shapes everything below: **profile A's turns must
never observe profile B's state**. Secrets, homes, sessions, and adapter lanes
are isolated per profile; anything that cannot yet be isolated fails closed or
is documented as a known limitation at the end of this document.

## The mode flag

- Config: `gateway.multiplex_profiles: true` (also accepted at top level).
  Parsed in `gateway/config.py` with precedence env > config > default.
- Env override: `GATEWAY_MULTIPLEX_PROFILES` accepts explicit truthy/falsy
  tokens only; a blank or unrecognized value returns "no override" so an empty
  deployment secret cannot shadow a config opt-in.
- At startup, `GatewayRunner.__init__` calls
  `agent.secret_scope.set_multiplex_active(...)` once. `_MULTIPLEX_ACTIVE` is
  a plain module global, not a contextvar: it describes the deployment mode,
  not a per-task value. Its only job is to arm the fail-closed behavior in
  `get_secret()`.

## Scope composition

Every inbound event composes the same two context-local scopes before any
profile-owned code runs:

```
platform event
   │
   ▼
profile_routes match ──► served-set check ──► SessionSource.profile stamped
   │                                           (gateway/profile_routing.py)
   ▼
_profile_runtime_scope(profile_home)           (gateway/run.py)
   ├── set_hermes_home_override(home)          config / state.db / skills /
   │                                           memory / sessions resolve here
   └── set_secret_scope(profile .env + secret sources)
   │                                           provider keys, platform tokens
   ▼
agent turn (worker thread via copy_context())
   │
   ▼
scope unwound in finally
```

`_profile_runtime_scope` wraps every seam where profile-owned code executes:
secondary adapter startup, connect and reconnect, the primary platform event
handler, inbound preprocessing, `/model` and session-info resolution,
background tasks, and the agent turn itself. Config reloads run under the
default profile's scope so global gateway settings (`#64674`) resolve
consistently.

Both scopes are `contextvars`, so they propagate into executor worker threads
via `copy_context()` and unwind deterministically — nothing is written to
`os.environ`, ever.

## Workstream A: context-local secret scope

`agent/secret_scope.py` exists because the obvious implementation — union all
profile `.env` files into `os.environ` — leaks profile A's keys into profile
B's turns and into every subprocess spawned with `env=dict(os.environ)`.

- `build_profile_secret_scope(home)` merges the profile's `.env` with its
  configured secret sources, skipping globals.
- `set_secret_scope(mapping)` installs it for the current task.
- `get_secret(name)` resolves: global allowlist → active scope → fallback.
  The fallback is the load-bearing part:
  - multiplexing **off**: reads `os.environ`, so single-profile gateways and
    every non-gateway caller behave exactly as before;
  - multiplexing **on**, no scope installed: **raises `UnscopedSecretError`**
    rather than silently reading the process environment. An un-migrated call
    site fails loud at that exact line instead of leaking another profile's
    value.
- A small allowlist (`HERMES_HOME`, `HERMES_PROFILE`, proxy settings,
  `API_SERVER_*` listener settings — but deliberately not `API_SERVER_KEY`)
  stays global because those describe the process, not a profile.

Because the per-turn `.env` reload is a no-op under multiplexing, rotated
credentials are picked up through the profile scope on the next turn — never
via `os.environ`. This holds at the loader boundary, not just the gateway's
reload helper: `hermes_cli.env_loader.load_hermes_dotenv` skips the
process-global load whenever multiplexing is active *and* a profile-home
override is installed (import-time and cron callers hit it mid-turn), while
still hydrating the profile's external secret sources into its private
snapshot (`#77562`). The unscoped startup load is unchanged.

The same scope-authoritative rule covers the other `os.environ` seams a
routed turn can reach: `${VAR}` / `${env:VAR}` references in a profile's
`config.yaml` resolve through `get_secret` when a scope is installed
(`#84079`), and `.env` writes made under a scope (`save_env_value`, e.g. a
`/pair` grant mirror) update the installed scope mapping instead of the
process environment (`#88441`).

## The HERMES_HOME override

`hermes_constants.py` holds a context-local override consulted by
`get_hermes_home()` before the `HERMES_HOME` env var. Everything that resolves
paths through it — config, `state.db`, skills, memory, SOUL, sessions, kanban,
goals, plugin discovery, MCP startup — follows the active profile
automatically. `get_process_hermes_home()` exists for the few machine-level
assets that must not follow the override. `hermes_home_key()` gives
per-home registries a stable scope key. A one-shot warning (`#18594`) fires if
profile-scoped code runs without the override where one is expected.

## Inbound routing

`gateway.profile_routes` maps `(platform, guild_id, chat_id, thread_id)` to a
profile; matching is conjunctive, most-specific-first, with parent-chain chat
matching for threads. Routing only runs when multiplexing is active, and a
matched route whose target is outside the served set is rejected (the event is
dropped, not misdelivered). Full schema and matching rules:
`docs/profile-routing.md`.

## Serving selected profiles

`profiles_to_serve(multiplex, profile_allowlist)` in `hermes_cli/profiles.py`
is the single chokepoint for which profiles a multiplexer serves: default plus
every valid profile directory, optionally filtered by allowlist. A malformed
allowlist fails safe to default-only. The served set gates adapter startup,
cron ticking (`#69377`), `/p/<profile>/` HTTP admission, route eligibility,
and the runtime status surface. An excluded profile stays installed and can
still run its own standalone gateway.

## Per-profile persistence

`SessionStore` binds no database handle at construction (`#88532`). Session
DB handles are resolved at call time through the active HERMES_HOME override —
one cached handle per resolved `profiles/<name>/state.db` — so sessions land
in the owning profile's store even when the store object itself is shared.
Pairing stores are constructed per served profile.

## Per-bot session lanes

Session keys are namespaced by profile (`agent:main` for default,
`agent:<name>` for named profiles). Adapters carry `_owner_profile`
(installed at adapter configuration time, before any inbound event) because
adapter ingress runs before `SessionSource.profile` is stamped;
`_session_key_profile` resolves source stamp → owner profile → store
resolver. Text/media batching, active-session tracking, and the busy-session
guard are all keyed per lane, so two bots sharing a chat do not share a
session lane.

## Control plane

Desktop plugins reach the gateway only through the ws JSON-RPC door, so
profile enumeration and configuration live in
`tui_gateway/methods_profiles.py`: `profiles.list`, `profiles.create`,
`profiles.describe`, `profiles.configure`, `profiles.set_asset`,
`profiles.get_asset`. Reads and writes run under the target profile's
HERMES_HOME override. Asset writes are atomic, type- and size-capped.

## Failure modes

- Fatal at startup: multiplex config errors and a secondary profile enabling a
  port-binding platform (`MultiplexConfigError`,
  `SecondaryPortBindingConfigError`) — one shared HTTP listener is owned by
  the default profile.
- Skipped, not fatal: a single misconfigured secondary adapter is skipped with
  a warning rather than taking down the multiplexer.
- Fail-closed: unscoped `get_secret()` under multiplexing raises; a routed
  event targeting an unserved profile is dropped; an unscoped `/p/` request
  enters the default profile's scope (`#61276`) rather than an undefined one.
- Fallback: an external `cron.provider` does not support multiplexing and
  falls back to the built-in ticker with a warning.

## Known limitations

Process-global state that is not yet profile-scoped:

| Surface | State at time of writing |
| --- | --- |
| MCP discovery and tool registration | Process-global; the first profile to build an agent wins the discovery slot. Full per-profile MCP registries are tracked in `#67605`. |
| Terminal / sandbox env (`TERMINAL_*`) | Global by allowlist; tools read it from the process environment. |
| Built-in tool registry | Built-ins are process-global; plugin-registered tools are overlaid per profile via `hermes_home_key()`. |
| Provider/capability registries | Same hybrid overlay pattern (browser, image-gen, TTS, transcription, video-gen, web-search, secret sources). |
| HTTP listener, relay ingress, process lock | One per process, owned by the default/active profile. Per-profile `runtime_status.json` is still written. |

## Non-goals

Multiplexing isolates *profiles*; it does not authenticate or authorize *end
users*. A profile is a configuration, not a person: the gateway trusts its
transport and its routing table to decide which profile an event belongs to.
Request-level identity and per-user authorization above the profile layer are
out of scope for this document.

## Related

- `docs/profile-routing.md` — inbound routing schema and matching rules.
- `website/docs/user-guide/multi-profile-gateways.md` — user-facing guide,
  including the standalone one-gateway-per-profile alternative.
- `agent/secret_scope.py`, `hermes_constants.py`, `gateway/profile_routing.py`,
  `gateway/run.py` (`_profile_runtime_scope`), `hermes_cli/profiles.py`
  (`profiles_to_serve`), `gateway/session.py`, `tui_gateway/methods_profiles.py`.
