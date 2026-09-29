# Mission Control desktop workspace prototype review

## Status

Prototype reference only. This is not a merge-ready feature proposal and does
not ask the Hermes team to maintain the external repository as-is.

Reference implementation:
https://github.com/jarvisstark2026/mission-control-center-react-ts

## Why this is being shared

Mission Control is an experimental Tauri/React desktop workspace manager built
around a widget-based, multi-workspace operating surface. It is unfinished, but
some of its UX and integration patterns may be useful when thinking about the
Hermes Desktop app.

The project is being shared for code and idea review only. The intended review
question is: are any of these patterns worth adapting, rewriting, or using as
inspiration inside Hermes Desktop?

## Prototype areas that may be relevant

- Widget-based multi-workspace layout with draggable, resizable, pinnable, and
  fill/restore widgets.
- A desktop command rail and tracker-style menu for quickly finding and
  focusing workspace surfaces.
- Hermes-style HUD concepts for a live agent surface, including quick chat,
  voice/listening state, and audio-reactive HUD visuals.
- Agent Control concepts for bridge setup, endpoint readiness, diagnostics,
  local bridge lifecycle, and per-workspace live layout toggles.
- Proposal/task history concepts that separate explicit proposal workflows from
  direct trusted UI actions.
- Local-first goals, workflows, evidence, command inbox, notifications, and
  integration registry surfaces.
- Theme, layout, and desktop quality-of-life behavior for users who keep many
  tools open at once.

## Important limitations

- Many widgets are incomplete or prototype-only.
- The app is not production-ready and should not be merged wholesale.
- The browser widget was removed after native WebView experiments proved too
  fragile for the intended widget-composited layout.
- Hermes integration needs a fresh validation pass after reinstalling the
  Hermes gateway and API server.
- Some concepts intentionally trade implementation completeness for rapid UX
  exploration.

## Suggested review scope

The useful review target is the idea and implementation pattern level, not
line-by-line readiness:

1. Whether a widget-based desktop workspace is useful for Hermes Desktop.
2. Whether a Hermes HUD / voice surface should exist as a first-class desktop
   interaction model.
3. Whether Agent Control-style diagnostics would improve bridge/API setup.
4. Whether direct trusted UI actions and proposal workflows should be separated.
5. Whether any layout, menu, theme, or workspace quality-of-life patterns are
   worth recreating in the Hermes codebase.

## Non-goals

- This does not propose replacing Hermes Desktop.
- This does not request urgent review.
- This does not ask maintainers to merge or maintain the external prototype.
- This does not introduce runtime dependencies or code changes to Hermes Agent.

