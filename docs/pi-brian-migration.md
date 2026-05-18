# pi-brian -> Hermes migration map

Status: in progress.

This fork keeps migration split by concern so we do not rebuild the same fragility that accumulated in `pi-brian`.

## Stream 1 - memory

Implemented in this fork:

- `plugins/memory/pi_brian_mem0/` — self-hosted Mem0 provider for long-tail recall
- `scripts/pi_brian_migration/export_selfhosted_mem0.py`
- `scripts/pi_brian_migration/build_hot_memory.py`
- `scripts/pi_brian_migration/install_hot_memory.py`
- `docs/pi-brian-migration-memory.md`

Target state:

- hot memory in `USER.md` + `MEMORY.md`
- long-tail semantic recall in self-hosted Mem0
- manual review gate for sensitive memory before import

## Stream 2 - messaging runtime

Target Hermes features:

- Telegram gateway as primary chat surface
- Discord optional
- approvals and busy-input behavior via Hermes gateway config
- slash-command layer for operator actions

Port from pi-brian:

- `/status`
- `/task`, `/whyfailed`, `/lastfailures`, `/lastcompletions`
- `/usage`
- approval/restart/operator workflows

Likely implementation shape:

- Hermes slash commands or plugin tools
- small repo-local plugin for Brian-specific operator commands

## Stream 3 - recurring digests and reminders

Target Hermes features:

- `cronjob` for weekday morning digest
- script-only watchdog jobs for low-cost checks
- background delivery to Telegram home channel

Port from pi-brian:

- morning digest schedule and formatting preference
- war digest cadence
- reminder semantics
- quiet/silent tick behavior

Likely implementation shape:

- cron jobs with explicit self-contained prompts
- optional script-only prechecks for wake/no-wake behavior

## Stream 4 - research publishing

Target Hermes features:

- Hermes terminal/process/delegation
- OpenCode skill
- repo-local skills or plugin helpers

Port from pi-brian:

- research prompt contract
- artifact generation rules
- `research.briankeefe.dev` publishing path
- lane split: Idea Filter vs Research

Likely implementation shape:

- Hermes skill for deep research + publish contract
- keep existing Python render/index scripts from `pi-brian`
- deliver completion links back to Telegram via final response or `send_message`

## Stream 5 - homelab and Home Assistant

Target Hermes features:

- built-in Home Assistant platform/tools
- SSH terminal backend for host checks
- script-only cron watchdogs for server health

Port from pi-brian:

- poseidon-centric diagnostics
- homelab/server checks
- selected Home Assistant workflows

Likely implementation shape:

- Hermes `homeassistant` toolset
- SSH backend presets
- low-cost cron monitors

## Stream 6 - background OpenCode workflows

Target Hermes features:

- OpenCode bundled skill
- background PTY sessions via terminal/process tools
- cron for work that must outlive a chat turn

Port from pi-brian:

- spawn/resume/steer/cancel semantics
- result capture for long-running work
- repo-scoped workdir discipline

Likely implementation shape:

- thin Brian plugin or skill wrapping OpenCode session patterns
- no Trigger in critical path

## Not being ported as first-class runtime

- Trigger shadow-mode execution path
- current stale Trigger run store model
- current watchdog/deploy-state coupling as-is

Reason:

- too much state drift
- not needed for first working Hermes cutover

## Cutover order

1. memory
2. Telegram gateway + auth
3. morning digest + reminders
4. research publish path
5. homelab / Home Assistant
6. advanced operator commands

## Current local artifacts

- Hermes fork: `/home/brian/code/hermes-brian`
- draft memory export bundle: `/tmp/opencode/hermes-brian-memory/`
