# TUI Widgets — Live Panels for the Ink TUI Dock

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
4. Auto-open (no command needed): end `register(sdk)` with
   `sdk.openWidget(app, app.init(''))` — the widget docks itself the moment
   the file loads. Only do this when the user asked for it; note it re-docks
   on every `/widgets-reload`.

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
`Dialog`, `Overlay`, `WidgetGrid`, `GridAreas`, and loaders `Shimmer`,
`ShimmerRows`, `useShimmerPhase` — use `ShimmerRows` for loading phases
instead of a bare "loading…" line.

Expand/collapse: `sdk.Accordion` — the same primitive the session panel's
tool/skill sections use. `h(Accordion, { t, title: 'details', count: 3,
defaultOpen: false }, body)` toggles on CLICK (works in ambient widgets,
which receive no keys); modal apps may pass `open` + `onToggle` to drive it
from reducer state instead.

Stable sizing (cards must NEVER resize while ticking):

- Give `Dialog` an explicit `width`; charts already return exactly the
  `width` you ask for (short series pad-left while history warms up).
- Pad dynamic numbers: `String(v).padStart(6)` — `51 ms` → `112 ms` must
  not change the line length.
- Keep row counts constant per phase; swap content, not structure.

Charts (pure string builders — color the result with theme tones):

- `sdk.sparkline(series, width?)` → `▂▃▅▇█▆` one-row trend
- `sdk.sparkRows(series, width, rows)` → multi-row column chart (top line
  first) — the mission-control panel look; taller cells gain resolution
- `sdk.gauge(ratio, width)` → `█████░░░` fill bar for a 0..1 value
- `sdk.hbars(values, width)` → horizontal bar chart, one bar per value,
  eighth-block tips, scaled to the max

Keep a rolling series in component state (push per tick, cap ~120 samples)
and render `sparkRows` for dashboard panels, `sparkline` for one-liners.

Contract essentials:

- `mode: 'ambient'` — captures no input, the command toggles it; `render`
  returns a CARD (usually `Dialog`), never `Overlay`.   Placement via `zone` — every zone RESERVES real space (nothing ever
  paints over the transcript):
  - Docks (chrome rows): `dock-top` (under the top status bar),
    `dock-bottom` (default — above the bottom one).
  - Rails (side columns beside the transcript; text reflows around them):
    `top-left`, `top-right`, `bottom-left`, `bottom-right` — corner names
    pick the rail side and its top/bottom anchor. Set `width` on the app
    to the card's width (match your Dialog width; default 44) — the rail
    reserves exactly that many columns.
  Map the user's words to the nearest zone: "top right" → `top-right`,
  "above/next to the status bar" → a dock. Rails suit narrow cards
  (~30-46 cols); full-width or short-and-wide content belongs in a dock.
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
