# Companiondex — Animated Companion Mascots

Browse, install, and select animated "companion" mascots from the public
[companiondex](https://github.com/crafter-station/companiondex) gallery. An installed companion
reacts to agent activity (idle, running a tool, reviewing, error, done) across
the Hermes CLI, TUI, and desktop app. This skill drives the `hermes companions` CLI
and the `display.companion` config — it does not generate sprites.

## When to Use

- The user wants a desktop/terminal mascot or asks about "companions" / companiondex.
- The user wants to change, preview, or disable the active companion.
- Diagnosing why a companion isn't showing (terminal graphics support, config).

## Prerequisites

- Network access to `companiondex.dev` for the gallery/manifest (read-only, no auth).
- Pillow (a core Hermes dependency) for sprite decoding — already installed.
- For full-fidelity terminal rendering: a graphics-capable terminal (kitty,
  Ghostty, WezTerm, iTerm2, or sixel). Otherwise a truecolor Unicode
  half-block fallback is used automatically.

## How to Run

Use the `terminal` tool to run `hermes companions <subcommand>`.

## Quick Reference

| Goal | Command |
| --- | --- |
| Browse the gallery | `hermes companions list` (add a substring to filter: `hermes companions list cat`) |
| List installed companions | `hermes companions list --installed` |
| Install a companion | `hermes companions install <slug>` (add `--select` to make it active) |
| Set the active companion | `hermes companions select <slug>` (omit slug for a picker) |
| Resize the companion everywhere | `hermes companions scale <factor>` (e.g. `0.5`, clamped 0.1–3.0) |
| Preview/animate in terminal | `hermes companions show [slug] [--cycle] [--state run]` |
| Disable the companion | `hermes companions off` |
| Remove a companion | `hermes companions remove <slug>` |
| Diagnose setup | `hermes companions doctor` |

## Procedure

1. Find a companion: `hermes companions list <query>` and note its `slug`.
2. Install + activate: `hermes companions install <slug> --select`.
3. Preview it: `hermes companions show` (Ctrl+C to stop).
4. Confirm setup: `hermes companions doctor` — shows the resolved companion, configured
   render mode, detected terminal graphics protocol, and effective mode.

Companions install into `<HERMES_HOME>/companions/<slug>/` (profile-aware). Selecting a companion
writes `display.companion.slug` + `display.companion.enabled` to `config.yaml`.

## Configuration

Under `display.companion` in `config.yaml`:

- `enabled` (bool) — master on/off.
- `slug` (str) — active companion; empty = first installed.
- `render_mode` — `auto` (detect) | `kitty` | `iterm` | `sixel` | `unicode` | `off`.
- `scale` (float) — on-screen size of the native 192×208 frames (default 0.33,
  clamped 0.1–3.0). One knob resizes every surface; set it with
  `hermes companions scale <factor>`, the `/companion scale` slash command, or the desktop
  Appearance slider.
- `unicode_cols` (int) — width in columns for the Unicode fallback.

## Pitfalls

- A companion only shows once one is installed AND selected (`enabled: true`).
- Inside a pipe/redirect (no TTY) terminal rendering is disabled by design.
- The companiondex npm CLI installs to `~/.codex/companions`; Hermes uses its own
  profile-scoped `<HERMES_HOME>/companions/` instead — install through `hermes companions`.

## Verification

- `hermes companions doctor` reports `✓ ready` when a companion is installed, selected,
  enabled, and Pillow is importable.
