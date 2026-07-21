---
name: hermes-themes
description: "Author a Hermes color theme that skins every surface."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [theme, skin, appearance, cli, tui, desktop, self-config]
    related_skills: []
---

# Hermes Themes Skill

Author a Hermes **skin** — one YAML file that themes the CLI, the TUI, and the
desktop GUI at once. The skin engine (`hermes_cli/skin_engine.py`) resolves the
active skin and the gateway pushes it to every surface, so a file dropped in
`~/.hermes/skins/` is the theme analogue of a plugin: no code, all surfaces. This
skill covers writing a good skin and activating it; it does not build GUI theme
editors or ship built-in presets.

## When to Use

- The user asks for a custom look ("make me a synthwave theme", "dark forest
  vibes", "match my brand colors") for Hermes itself.
- The user wants the CLI/TUI/desktop to share one coordinated palette.

## Prerequisites

- Write access to the Hermes home dir — `~/.hermes` by default, or `$HERMES_HOME`
  / the active profile's dir. Skins live in `<hermes-home>/skins/`.
- Native tools: `write_file` (create the YAML), `read_file` / `search_files`
  (inspect existing skins), `patch` (set `display.skin`).

## How to Run

1. Pick a lowercase, hyphen-safe `name` (e.g. `synthwave`).
2. Copy `templates/skin.yaml` and fill in the palette (keep every key — missing
   keys inherit the `default` skin).
3. `write_file` it to `<hermes-home>/skins/<name>.yaml`.
4. Activate it (see Procedure). Confirm the change landed.

## Quick Reference

Load-bearing color keys (hex, `#rrggbb`). The desktop GUI derives its whole
palette from these; the TUI and CLI read the terminal-oriented ones directly.

| Key | Drives |
|---|---|
| `background` | Base surface — GUI + TUI status bar seed. Set it. |
| `ui_accent` / `banner_accent` | Brand accent: buttons, rings, primary. |
| `banner_title` | Headings / primary text. |
| `banner_text` / `ui_text` | Body foreground. |
| `banner_border` / `ui_border` | Borders. |
| `banner_dim` | Muted / secondary text. |
| `ui_ok` / `ui_warn` / `ui_error` | Semantic status colors. |
| `status_bar_bg` / `status_bar_text` | TUI status bar. |
| `response_border` | CLI response box. |

`branding` (`agent_name`, `welcome`, `goodbye`, `prompt_symbol`, `help_header`),
`spinner` (faces/verbs/wings), and `tool_prefix` are optional flavor. See the
full schema in `hermes_cli/skin_engine.py`.

## Procedure

1. **Design the palette.** Choose a `background` first, then an `ui_accent` that
   clears WCAG AA against it (~4.5:1) so labels stay legible — the GUI enforces
   contrast but a low-contrast accent still looks washed out. Keep
   `ui_ok`/`ui_warn`/`ui_error` recognizably green/amber/red.
2. **Write the file** to `<hermes-home>/skins/<name>.yaml`. Every top-level
   `colors` key from the template should be present.
3. **Activate.** Set `display.skin: <name>` in `<hermes-home>/config.yaml` with
   `patch` (create the `display:` block if absent). This is the source of truth
   all surfaces read.
   - **Desktop**: repaints automatically after the current turn (and the skin
     appears in Appearance / `Cmd-K` / `/skin`).
   - **CLI / TUI**: run `/skin <name>` for an immediate switch, or it loads on
     next start.
4. **Confirm** and tell the user how to switch back (`/skin default`).

## Pitfalls

- **Don't hardcode `~/.hermes`** when a profile is active — resolve the real home
  from `$HERMES_HOME` first, falling back to `~/.hermes`.
- **Keep `#rrggbb` hex.** Shorthand `#rgb`, `rgb()`, and named colors are not
  guaranteed to parse on every surface.
- **Set `background`.** Without it the GUI has to guess a base surface from text
  luminance — usable, but you lose control of the app background.
- **Name collisions**: a skin named like a desktop built-in (`mono`, `slate`,
  `cyberpunk`, `nous`, `midnight`, `ember`) won't override that built-in on the
  GUI. Pick a fresh name.
- **Don't rebuild config.yaml** — `patch` only the `display.skin` line so you
  don't clobber the user's other settings.

## Verification

- `read_file` the written `<hermes-home>/skins/<name>.yaml` and confirm valid
  YAML with the intended `name` and `colors`.
- `read_file` `<hermes-home>/config.yaml` and confirm `display.skin: <name>`.
- Ask the user to confirm the new look, or check the current surface repainted.
