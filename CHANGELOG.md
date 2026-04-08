# Changelog

All notable changes to Hermes Agent will be documented in this file.

## [0.7.1] - 2026-04-08

### Added

- `/new <message>` inline support: reset the session and immediately process the trailing text as the first message in the fresh session, saving a round-trip on every topic switch
- Works on all gateway platforms (Telegram, Discord, Slack, etc.) and the interactive CLI
- `/reset <message>` alias behaves identically

### Fixed

- Race condition in gateway session sentinel: the sentinel is now set before any async work in `_reset_with_inline`, preventing concurrent agent dispatch during the reset window
