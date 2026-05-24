# Gateway Guide

This directory owns the messaging gateway, session orchestration, platform
adapters, gateway hooks, and platform-specific command routing.

## Runtime Shape

`gateway/run.py` owns the gateway runner and active session cache.
`gateway/session.py` owns session-level structures. `gateway/platforms/` holds
adapters for Telegram, Discord, Slack, WhatsApp, Matrix, Signal, email, SMS,
webhook, API server, and other platforms.

Gateway sessions cache `AIAgent` instances per session. Destroying that cache
breaks prompt caching and can change behavior across turns.

## Command Registry

Gateway command names and help should derive from `hermes_cli/commands.py`.
Do not maintain a separate command list unless the platform API forces a local
projection.

When adding a gateway-available slash command:

1. Add or update the `CommandDef`.
2. Add gateway dispatch in `gateway/run.py`.
3. Update platform-specific routing only when that platform has its own command
   shape, such as Slack subcommands or Telegram bot menus.

## Two Message Guards

When an agent is running, incoming messages pass through two guards:

1. Base adapter queues messages in `_pending_messages` when the session is
   active.
2. Gateway runner intercepts control commands such as `/stop`, `/new`,
   `/queue`, `/status`, `/approve`, and `/deny`.

Any command that must reach the runner while the agent is blocked must bypass
both guards and dispatch inline. Do not route approval/control commands through
background processing that races session lifecycle.

## Platform Adapters

Platform adapters should:

- use profile-safe paths,
- acquire scoped locks for unique credentials such as bot tokens,
- release locks on disconnect/stop,
- keep platform-specific payload parsing at the adapter edge,
- share command/session behavior through gateway core helpers.

See Telegram for scoped-lock patterns and `ADDING_A_PLATFORM.md` for platform
authoring guidance.

## Gateway Hooks

Gateway hooks live under `~/.hermes/hooks/<name>/` with `HOOK.yaml` and
`handler.py`. Built-in hooks can live under `gateway/builtin_hooks/`, which is
currently an extension point.

Hook events include gateway startup, session start, agent step, and command
wildcards. Hook failures should be isolated and logged without taking down the
gateway unless the hook is explicitly enforcing a block.

## Cron and Gateway

Cron usually runs from the gateway tick loop. Cron deliveries are not mirrored
into the target chat session; they are their own sessions with delivery frames.

Do not assume gateway conversation context is available in cron jobs unless the
job prompt explicitly includes it.

## Background Process Notifications

When terminal background processes complete, the gateway watcher can trigger a
new agent turn. Respect `display.background_process_notifications`.
