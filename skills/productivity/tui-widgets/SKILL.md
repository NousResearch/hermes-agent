---
name: tui-widgets
description: Author live widget apps for the Hermes TUI dock.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [tui, widgets, sdk, ui]
    category: productivity
---

# TUI Widgets Skill

Author widget apps for the Hermes TUI (`hermes --tui`): glanceable ambient
panels docked above the status bar, or modal overlays that own the keyboard.
Widgets are plain ESM files the TUI loads at startup — no build step, no
repo changes. This skill does not cover desktop-app or web-dashboard
widgets.

## When to Use

- The user asks for a live panel in the TUI (ticker, clock, countdown,
  status card, API-backed readout).
- The user wants a custom modal tool (picker, calculator, viewer) bound to
  a slash command.

## Prerequisites

- The TUI must be in use (`hermes --tui`). Widgets do not render in the
  classic CLI or messaging platforms.
- Network-backed widgets need whatever credentials their API needs; fetch
  failures must land as an error phase, never a crash.

## How to Run

1. Use `write_file` to create `~/.hermes/tui-widgets/<name>.mjs` (see
   `templates/clock.mjs` for a complete working widget).
2. If the TUI is running it hot-loads the file within ~a second (the
   widgets directory is watched); `/widgets-reload` forces a rescan.
3. The widget's id becomes its slash command automatically (`/<id>`), with
   its `help` in the `/` completion popover. No other registration exists.

## Quick Reference

A widget file default-exports `register(sdk)`:

```js
export default function register(sdk) {
  const { Box, Text, defineWidgetApp, h } = sdk

  defineWidgetApp({
    id: 'clock',                    // slash command name
    help: 'live clock in the dock', // `/` completion metadata
    mode: 'ambient',                // 'ambient' docks; 'modal' takes input
    init: arg => ({ label: arg.trim() || 'UTC' }),   // null = print usage
    reduce: (state, { ch, key }) => (key.escape || ch === 'q' ? null : state),
    render: ({ state, t }) => h(sdk.Dialog, { width: 24 }, h(Text, { color: t.color.label }, state.label))
  })
}
```

`sdk` contents: `defineWidgetApp`, `openWidget`, `updateWidget`, `isCtrl`,
`React`, `h` (createElement — no JSX in .mjs), components `Box`, `Text`,
`Dialog`, `Overlay`, `WidgetGrid`, `GridAreas`.

Contract essentials:

- `mode: 'ambient'` — docks above the status bar, captures no input, the
  command toggles it; `render` returns a CARD (usually `Dialog`), never
  `Overlay`.
- `mode: 'modal'` (default) — owns every keypress; `reduce` returns next
  state, the same reference to swallow a key, or `null` to close; `render`
  wraps content in `Overlay` for placement.
- Async data: fire the fetch from `init`, land results with
  `sdk.updateWidget(app, fn)` — it no-ops if the widget was closed, so a
  late reply can never resurrect it.
- Animation: own a timer inside a component via `React.useState` +
  `React.useEffect` (see the template); keep intervals ≥ 250ms.
- Colors: ALWAYS theme tones (`t.color.primary/label/muted/ok/error/…`),
  never hardcoded hexes — widgets must survive `/skin` and light/dark.

## Procedure

1. Pick `id`, `mode`, and the state shape; keep state serializable.
2. Write the file from the template; wire data via `init` + `updateWidget`.
3. `/<id>` to launch (hot-loaded on write); relaunch `/<id>` to dismiss an
   ambient widget.
4. Iterate: edit the file — it hot-reloads on save (last-writer-wins, the
   fresh definition shadows the old one). Relaunch `/<id>` to remount.

## Pitfalls

- No JSX and no bare imports in `.mjs` — everything comes from the `sdk`
  parameter; `h(...)` builds elements.
- Don't ship a modal without a close path (`Esc`/`q` returning `null`).
- Ambient widgets must stay small (≤ ~6 rows) — the dock sits between the
  transcript and the status bar.
- A thrown `register()` is logged and skipped; check
  `~/.hermes/logs/tui_gateway_crash.log` if a widget never appears.

## Verification

Run `/widgets-reload` — the transcript line must list the file under
`loaded:`. Then `/<id>`: an ambient widget appears docked right, above the
status bar, while the composer keeps accepting input; `/<id>` again removes
it.
