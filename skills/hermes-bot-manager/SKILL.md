---
name: hermes-bot-manager
description: Manage Hermes Bot Mode profiles, routines, and groups.
version: 1.1.0
author: PatchworkMD
license: MIT
tags:
  - hermes
  - bots
  - bot-mode
  - profiles
  - cron
  - desktop
related_skills:
  - hermes-skill-lifecycle
  - skill-hub-verification
---

# Hermes Bot Manager

Manage Hermes Bot Mode from the desktop or CLI without inventing a second
control plane.

The Nous documentation describes Bot Mode this way: **“Bot Mode turns your
Hermes profiles into a roster of named Bots.”** It also says: **“There is no
new primitive to learn: a Bot is a Hermes profile.”** Bot Mode is a UI over the
profile primitive. Use the profile, chat, cron, and peer commands that already
exist instead of creating replacement state.

## When to Use

Use this skill when a user needs to create, inspect, or operate a Hermes Bot
Mode profile, routine, group, Bot Chat, or cross-machine peer.

## Prerequisites

This skill uses the existing Hermes CLI, desktop Bot Mode, and the `terminal`
tool. It has no external package or API-key dependency. Peer operations require
a user-supplied gateway URL and private API-server key; never request or print
that key in chat.

## How to Run

Start with one of the routing modes below. These are skill modes, not new
Hermes CLI commands. Use the official Hermes commands shown in each mode, then
run `verify` against the actual destination.

## Official documentation

Use these as the source of truth when this skill and the installed Hermes
version differ:

- [Bot Mode: A Roster of Agents](https://hermes-agent.nousresearch.com/docs/user-guide/bot-mode)
- [Desktop App](https://hermes-agent.nousresearch.com/docs/user-guide/desktop)
- [Profile Commands](https://hermes-agent.nousresearch.com/docs/reference/profile-commands)
- [CLI Commands](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)
- [Cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)

The docs state that Bot Mode ships built into the desktop app and is on by
default. It does not require a separate Bot Mode install. This skill adds an
operator workflow and command vocabulary; it does not patch Hermes core.

## Quick Reference

These routing modes are the supported entry points for this skill. When a user
starts with one, perform that slice and verify the result before moving to
another.

```text
/bot-manager overview [bot]
/bot-manager create <name>
/bot-manager inspect <name>
/bot-manager routine <bot> <schedule> <task>
/bot-manager group <group> [bot ...]
/bot-manager message <bot> <text>
/bot-manager peer add <name> <url>
/bot-manager peer list
/bot-manager peer dm <peer>[/<agent>] <text>
/bot-manager peer remove <name>
/bot-manager verify <bot>
/bot-manager disable
```

If the user says “bot manager” without a subcommand, start with `overview`
and ask only for the missing bot name or intended operation. Do not create a
profile, routine, group, peer, or external message from an ambiguous request.

## `overview`

Explain the parity between Bot Mode and the CLI:

| Bot Mode | CLI / filesystem |
| --- | --- |
| Chat with a Bot | `hermes -p <bot> chat` |
| Bot files, skills, memory | `~/.hermes/profiles/<bot>/` |
| Routines | `hermes cron list`, filtered to `[bot:<name>] ...` |
| Create / inspect profiles | `hermes profile create`, `hermes profile list` |
| Bot-to-bot DM | `hermes -p <bot> chat --in ~ -c "Bot Chat" --create-if-missing -Q -q "Message from 🤖 <sender> (@<sender>): ..."` |
| Cross-machine Bot-to-bot DM | `hermes peer dm <peer>[/<agent>] "message"` |

A Bot has its own role, model, memory, skills, avatar, and canonical Bot Chat.
The canonical Bot Chat is persistent. In that chat, `/new` and `/reset` are
rerouted to `/compact`; regular sessions keep their normal `/new` behavior.

## `create <name>`

Use the desktop **New Agent** flow or the CLI equivalent:

```bash
hermes profile create <name>
hermes profile list
```

The documented quick path is **Name, Title, Description**. The Advanced
surface can clone an existing profile or create a fresh one, pin a model and
provider, set a custom `SOUL.md`, and enable specific skills, toolsets, and MCP
servers. With multiple registered connections, **Create on** chooses the
machine that owns the profile.

After creating a Bot, verify that its canonical Bot Chat exists:

```bash
hermes -p <name> chat
```

Do not edit the default profile when the requested Bot has a different name.
Do not copy credentials into a new profile. Hermes profile paths are private;
never put them in public copy.

## `inspect <name>`

Read the existing profile before changing it. Inspect only the requested
profile's config, identity, skills, routines, and latest Bot Chat state.

```bash
hermes profile list
hermes -p <name> chat
hermes cron list
```

The desktop profile editor is the documented place to change avatar, title,
description, model pin, skills, toolsets, MCP servers, and `SOUL.md`. Direct
file edits are a fallback for local maintenance, not a reason to create a
second profile or rewrite unrelated settings.

## `routine <bot> <schedule> <task>`

The Bot Mode docs define routines as plain Hermes cron jobs namespaced:

```text
[bot:<name>] <routine>
```

Create or edit the routine through the Routines pane or the cron system. Verify
with:

```bash
hermes cron list
```

The run lands in the Bot's own chat history. For change-driven routines, the
cron model supports `monitor_script` or `monitor_url`; for incremental work,
use `continuity`; for routing output, use `deliver`. Keep one routine per
actual recurring behavior. Do not create duplicate schedules to simulate
progress.

## `group <group> [bot ...]`

Use the desktop roster: right-click a Bot, choose **Move to group**, then open
the group header to create a shared room. The documented group-chat limits are
2–6 Bots, up to three serial rounds, and ten messages per send. Bots may be
silent when they have nothing new to add. Use `@name` mentions to scope a turn.

A group member keeps its own persistent `Group: <name>` session. Groups can
span registered machines; use the disambiguated `@name-device` handle when
needed. Do not make a second group-chat daemon or bus.

## `message <bot> <text>`

For a local Bot Chat, use the documented CLI shape:

```bash
hermes -p <bot> chat --in ~ -c "Bot Chat" --create-if-missing -Q -q "Message from 🤖 <sender> (@<sender>): <text>"
```

In a Bot Chat, `@name` hands work to another Bot. Delivery is per-invocation:
the receiving Bot picks the message up the next time it runs. Live interruption
of a Bot mid-conversation is not implied.

The backend teaches the messaging protocol to each canonical Bot Chat when
`agent.bot_mode_protocol` is enabled. The documented default is on:

```yaml
agent:
  bot_mode_protocol: true
```

Only canonical Bot Chats receive that protocol section. Regular sessions and
`SOUL.md` remain untouched.

## `peer add|list|dm|remove`

For Bot-to-bot DMs across machines, the official CLI reference defines:

```bash
hermes peer add <name> --url http://host:port --key <API_SERVER_KEY>
hermes peer list
hermes peer dm <peer>[/<agent>] "message"
hermes peer remove <name>
```

A peer is another Hermes gateway running the `api_server` platform. `peer dm`
delivers to the remote agent's canonical Bot Chat, runs one agent turn, and
prints the reply. The peer URL and name live in `config.yaml`; the key belongs
in the private environment file as `HERMES_PEER_<NAME>_KEY`. Never print or
publish the key.

Do not set up a peer without an explicit machine, URL, credential, and
network-scope request. Do not confuse a peer with a desktop Connections entry.

## `verify <bot>`

Run the smallest checks that prove the requested state:

```bash
hermes profile list
hermes -p <bot> chat
hermes cron list
```

Then check the actual destination: Bot Chat history for a local message,
`hermes peer dm` output for a peer message, or the desktop roster for a group
or identity change. A profile listing alone does not prove delivery.

Report **unproven—not absent** when the requested runtime, delivery, device, or
connection evidence is missing.

## `disable`

Bot Mode is a bundled desktop plugin. Disable it in **Settings → Plugins →
Bots**. The docs state that profiles, sessions, and cron jobs remain intact;
disabling the UI does not delete the underlying profile data.

## Common mistakes

- Editing the wrong profile directory. Confirm the profile name first.
- Treating Bot Mode as a new backend primitive. A Bot is a profile.
- Forgetting the `[bot:<name>]` routine namespace.
- Assuming Bot Chats behave like regular sessions. Canonical Bot Chats are
  persistent.
- Claiming live interrupt. Bot-to-bot delivery is per-invocation.
- Exposing `.env` keys, API server keys, home paths, or private machine names.
- Creating a replacement bot, bus, daemon, schedule, or control plane.

## References

- Official Nous/Hermes Bot Mode guide: `user-guide/bot-mode.md`
- Official Nous/Hermes desktop guide: `user-guide/desktop.md`
- Official CLI reference: `reference/cli-commands.md`
- Official cron guide: `user-guide/features/cron.md`
- Public docs: https://hermes-agent.nousresearch.com/docs
