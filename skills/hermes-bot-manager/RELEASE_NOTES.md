# Hermes Bot Manager

PatchworkMD release of a Hermes-first skill for managing Bot Mode profiles,
routines, groups, Bot Chat, and cross-machine peers.

## This revision

- Adds `/bot-manager` routing modes for overview, create, inspect, routine,
  group, message, peer, verify, and disable.
- Grounds the workflow in the official Nous/Hermes Bot Mode, Desktop, CLI, and
  Cron documentation.
- Adds `references/nous-bot-mode.md` with source links and quoted terminology.
- Keeps the public package attributed to PatchworkMD.

This skill does not add new Hermes core commands. Its subcommands are routing
vocabulary for the skill; the commands it invokes are existing Hermes CLI or
desktop operations.
