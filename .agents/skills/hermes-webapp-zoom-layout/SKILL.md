---
name: hermes-webapp-zoom-layout
description: Diagnosing and fixing CSS zoom vs. getBoundingClientRect() drift in PersistentTerminal and terminalSelectionAnchor, avoiding KeepAlivePaneHost vs. PersistentTerminal overlap, maintaining TerminalRail full-height stretch and shrink-0 spacing, and verifying parameterized zoom tests.
---

# Hermes Webapp Zoom & Layout Skill

Based on: project-specific (Hermes Desktop/Webapp)

This skill covers diagnosing and fixing layout drift caused by browser CSS zoom in Hermes Desktop and Webapp components.

## How to Diagnose and Fix Zoom and Layout Issues

1. **Unscale Fixed Overlay Coordinates (`PersistentTerminal`, `terminalSelectionAnchor`)**
   - When `document.documentElement.style.zoom` is active (`!== 1`), `getBoundingClientRect()` returns viewport-scaled coordinates.
   - When applying these to `position: fixed` elements inside a zoomed `<html>`, divide `top`, `left`, `right`, and `bottom` by `slot.currentCSSZoom || parseFloat(document.documentElement.style.zoom) || 1`.
   - Use `Math.floor(r.right / zoom) - left` for width to prevent 1px encroachment onto adjacent panel borders.

2. **Stacking Context & Rail Alignment (`KeepAlivePaneHost` vs. `PersistentTerminal`)**
   - Account for stacking contexts created by `KeepAlivePaneHost` (`z-0`) and root overlays (`PersistentTerminal` at `z-4`).
   - Keep overlay widths strictly bounded (`<= slot.right`) to prevent painting over `TerminalRail`.
   - Ensure `TerminalRail` stretches full height (`h-full items-stretch`, `self-stretch`, `shrink-0` on `<li>` items, and clean gap spacing).

3. **Parameterized Testing**
   - Verify layout calculations across parameterized zoom levels (`0.9`, `1`, `1.1`, `1.25`, `1.5`) in Vitest suites.
   - Note known accepted behaviors: 125% row quantization residual space and upstream xterm.js mouse-selection limitations under non-1 zoom.

## What this skill does NOT do
- Never patches upstream xterm.js internals.
- Never adds layout hacks for accepted 125% row quantization.
- Never touches chat tool cards.
