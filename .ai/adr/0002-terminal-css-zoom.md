# ADR-0002: Integrated Terminal Alignment Under CSS Zoom & Pane Layout

- **Status:** Accepted
- **Date:** 2026-11-23
- **Scope:** `apps/desktop/src/app/right-sidebar/terminal/` (`persistent.tsx`, `selection.ts`, `instance.tsx`, `chrome.tsx`, `rail.tsx`, `rail-new-terminal.tsx`, `use-terminal-session.ts`, `terminal-session-attempt.ts`)

## Context

In the browser-hosted Webapp, UI scaling is implemented via `document.documentElement.style.zoom` (`apps/desktop/src/lib/browser-zoom.ts`). Meanwhile, `PersistentTerminal` mounts a single `position: fixed` overlay (`z-index: 4`) at the layout root and positions it by copying `slot.getBoundingClientRect()` from `<TerminalSlot />` (which lives inside `KeepAlivePaneHost` at `z-index: 0`).

When `zoom !== 1` (e.g., `1.1`), `getBoundingClientRect()` returns viewport-scaled coordinates, and applying those values directly to `position: fixed` inside the zoomed `<html>` multiplied `top`, `left`, `width`, and `height` by `zoom` a second time. This caused two visible defects:
1. A ~46px vertical gap between the `TERMINAL` pane header and the top of xterm (`top` shifted down by `10%`).
2. The opaque terminal overlay overflowing rightward over `TerminalRail`, covering the `+` (New terminal) button while leaving only the top tab icon visible above the shifted `top` edge.

Additional layout and session defects also affected the integrated terminal:
- `TerminalRail` did not stretch to full pane height when its flex container omitted `h-full items-stretch` / `self-stretch`, cutting its left vertical border short.
- Rail `<li>` items lacked `shrink-0` and had tight `gap-0.5` spacing.
- Fresh persistent browser shells sent an empty initial frame with `replay: true`, which bypassed `stripBootGap`, and expired sessions did not reset cursor position to row 0 before printing the expired notice or starting a new shell.

## Decision

1. **Unscale fixed overlay coordinates by CSS zoom (`persistent.tsx`):** Divide `r.top`, `r.left`, `r.right`, and `r.bottom` by `slot.currentCSSZoom || parseFloat(document.documentElement.style.zoom) || 1` (guarded for finite positive values). Use `Math.floor(r.right / zoom) - left` for `width` so fractional zoom rounding never encroaches 1px onto `TerminalRail`'s `border-l`.
2. **Unscale selection popover anchor (`selection.ts`):** Apply the same CSS zoom divisor in `terminalSelectionAnchor` so the floating "Add to chat" button aligns with xterm selections under zoom.
3. **Full-height rail and non-overlapping icon spacing (`chrome.tsx`, `rail.tsx`, `rail-new-terminal.tsx`):** Add `h-full items-stretch` on the terminal chrome row, `self-stretch` on `TerminalRail`, `shrink-0` on rail `<li>` items, and `gap-1.5 py-1.5` spacing.
4. **Top-anchored output and full-height fit (`instance.tsx`, `use-terminal-session.ts`, `terminal-session-attempt.ts`):** Add `flex-1` to `HOST_CLASS`, run `fitRef.current?.()` on activation even when `status !== 'open'`, reset via RIS (`\u001bc`) on `expired` exit/retry (preserving scrollback on transport-loss `disconnected` retries), and only mark `hadReplay = true` when a replay frame has non-empty content so `stripBootGap` runs on fresh shells.

## Consequences

- At 100% and 110% UI scale, the terminal prompt starts flush below the `TERMINAL` header, the `+` button is clearly visible and spaced, and the right vertical rail border runs uninterrupted to the bottom.
- Parameterized Vitest coverage (`0.9`, `1`, `1.1`, `1.25`, `1.5`) protects both `PersistentTerminal` and `terminalSelectionAnchor`.
- **Known upstream limitation (unchanged):** Mouse click/selection hit-testing inside xterm.js's canvas under CSS `zoom !== 1` is an upstream xterm.js limitation and is deliberately left untouched.
