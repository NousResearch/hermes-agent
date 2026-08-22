# Nous/Hermes Bot Mode source notes

This reference records the official documentation used by
`hermes-bot-manager`. The public URLs are the source of truth for future Hermes
releases.

## Primary sources

- [Bot Mode: A Roster of Agents](https://hermes-agent.nousresearch.com/docs/user-guide/bot-mode)
- [Desktop App](https://hermes-agent.nousresearch.com/docs/user-guide/desktop)
- [CLI Commands](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)
- [Cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)

## Wording used in the skill

The Bot Mode guide defines the model directly:

> Bot Mode turns your Hermes profiles into a roster of named Bots.

It also states:

> There is no new primitive to learn: a Bot is a Hermes profile.

The same guide documents these operational facts:

- Bot Mode ships built into the desktop app and is on by default.
- Each Bot has a canonical, persistent Bot Chat.
- Routines are plain Hermes cron jobs namespaced `[bot:<name>] <routine>`.
- Bots can message one another with `@name` and through the CLI.
- Bot-to-bot delivery is per-invocation; live interruption is not implied.
- The `agent.bot_mode_protocol` setting controls the protocol injected into
  canonical Bot Chats and defaults to on.
- The CLI parity table maps Bot Chat, profile files, routines, and profile
  creation to existing Hermes commands.

The CLI reference documents `hermes peer add`, `list`, `dm`, and `remove` for
Bot-to-bot DMs across machines. Peer keys are credentials and stay in the
private environment file.

This file is a compact attribution and maintenance note, not a replacement for
the official documentation.
