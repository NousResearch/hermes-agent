# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **TUI clipboard copy** — Native tool probing on Linux now short-circuits immediately when `$DISPLAY` and `$WAYLAND_DISPLAY` are both unset, avoiding wasted time and silent failures in headless environments (Docker, CI). (Hermes Ink / osc.ts)
- **TUI debug visibility** — Added `HERMES_TUI_DEBUG_CLIPBOARD` environment variable. When set, the TUI logs which clipboard mechanism is used, probe results, and why OSC 52 might be suppressed. Helps users and operators diagnose copy failures.
- **Dashboard clipboard logging** — Silent failures in OSC 52 → Clipboard API bridge and direct `Ctrl+Shift+C` copy are now logged to the browser console with explanatory warnings, replacing empty catch blocks. Makes clipboard permission issues and gesture-timeout failures visible during development.
- **Documentation** — Added comprehensive "Clipboard Troubleshooting" section to README covering OSC 52 verification, tmux configuration, Docker/headless constraints, environment variables, and dashboard caveats. AGENTS.md now documents all clipboard-related environment variables and known failure modes.

### Changed

- Desktop and dashboard clipboard error handling is now consistent: all Clipboard API rejections and native tool failures produce diagnostic logs rather than being swallowed.

</content>