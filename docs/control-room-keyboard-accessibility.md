# Control Room — Keyboard Accessibility (CR-604)

Requirement: opening, list navigation, focus trap, Escape return, readable
status labels, no mouse dependency. Verified per surface below.

## CLI

- **Open:** `Ctrl+P` anywhere outside a modal prompt (bound in `cli.py`,
  filtered by `_editor_filter` so it never fires mid-composer — test-covered).
- **Readable status:** the status bar segment renders attention counts as text
  (`needs-you: N · running: N` style) with no colour-only semantics; the
  `/control` command renders the same contract as plain text.
- **No mouse dependency:** full keyboard path.
- **Tests:** `tests/control_room/test_cli_phase3.py` (binding + segment),
  status bar suite 20/20.

## Ink TUI

- **Open/close:** `Ctrl+P` toggles; `Esc` closes from the blocked-overlay path
  (the same branch as the agents overlay — focus is trapped by `$isBlocked`).
- **List navigation:** the overlay is built on the `useMenu` primitive — arrow
  keys move the active row, Enter activates, Esc returns.
- **Focus trap:** `controlRoom` is part of the `$isBlocked` aggregate; while
  open, composer input is blocked and keys route to the overlay handler.
- **Readable labels:** rows render kind + status text; the badge renders
  `needs-you N` (and renders **nothing** until the first RPC resolves — never
  a fake zero).
- **Tests:** `ui-tui/src/__tests__/controlRoomOverlay.test.tsx` — 10/10 green
  (overlay state, Ctrl+P toggle, close, reset, badge export, RPC method name,
  global handler wiring).

## Native Desktop

- **Open/close:** `Ctrl/Cmd+P` via the `nav.controlRoom` keybind (registered
  through KEYBINDS_AREA, shown in the shortcuts sheet); the palette keeps
  `Cmd/Ctrl+K`.
- **List navigation:** command-center sections use the existing
  `useRouteEnumParam` tab navigation — arrow/tab + Enter, no mouse.
- **Focus:** the overlay is modal (full-screen command-center), Esc closes.
- **Tests:** `apps/desktop/src/lib/keybinds/control-room.test.tsx` — keybind
  contract 3/3 verified via minimal vitest config (full desktop vitest is
  env-blocked; see fixture matrix gap).

## Kensei Dashboard (web)

- **Open:** `Ctrl+P` navigates to `/control-room` (App.tsx global handler);
  `Cmd/Ctrl+K` unchanged for the command bar.
- **List navigation:** sections are tab buttons (keyboard reachable), rows are
  links (Enter activates), Esc isn't needed for a route.
- **Readable labels:** the HealthStrip renders `● N need you` in text; the
  page header shows `profile: <name> · snapshot vN`; unavailable capabilities
  render as explicit "not wired in this runtime" hints, never dead controls.
- **Tests:** `web/src/pages/ControlRoom.test.tsx` — 4/4 green (home tiles,
  needs-you count, unavailable caps, deep-links).

## Summary

| Surface | Open | Navigate | Close | No mouse | Tests |
|---------|------|----------|-------|----------|-------|
| CLI | Ctrl+P | arrow/native | Esc/native | ✅ | ✅ |
| TUI | Ctrl+P | arrows via useMenu | Esc | ✅ | ✅ 10 |
| Desktop | Ctrl/Cmd+P | tabs | Esc | ✅ | ✅ 3 (env-blocked rest) |
| Dashboard | Ctrl+P | tabs/links | route back | ✅ | ✅ 4 |

No surface requires a mouse to open or operate Control Room.
