[Leia em português](readme-ptbr.md)

# One Dark Pro Theme & Integrated Terminal Polish (Hermes Desktop / Webapp)

This document summarizes the Atom One Dark Pro theme preset and integrated terminal refinements maintained on the local `theme/one-dark-sync` branch of Hermes Agent (`apps/desktop/`).

## Overview

The built-in `one-dark` backend skin is mapped to an exact Atom One Dark Pro renderer preset (`apps/desktop/src/themes/one-dark.ts` + `apps/desktop/src/themes/one-dark.css`) guarded by runtime provenance (`Symbol('one-dark-pro')` and `data-hermes-palette='one-dark'`), keeping user-installed themes unaffected and ensuring clean teardown when switching themes.

## Key Changes

### Theme & Typography (`apps/desktop/src/themes/`)
- **Core surfaces:** Chrome/sidebar `#282c34`, editor/card/popover/terminal `#21252b`, inputs/composer `#1d1f23`, muted/user bubble `#2c313a`, continuous borders `#181a1f`.
- **Accents & markdown:** Headings and primary accent `#e06c75`, bold `#d19a66`, links `#61afef`, list markers (`li::marker`, `--tw-prose-bullets`, `--tw-prose-counters`) `#abb2bf`, checkboxes `#c678dd`.
- **Text selection (`::selection`):** `background: rgba(97, 175, 239, 0.4)` and `color: #e6e6e6`.
- **UI chrome:** Continuous sidebar/content seam (`--sidebar-edge-border: #181a1f`), 2px active pane-tab underline (`pane-tab.tsx`), sidebar row hairlines (`sidebar/index.tsx`, `session-row.tsx`), and live "Thinking" wave animation (`od-sun` in `message-parts.tsx` with `prefers-reduced-motion` fallback).

### Integrated Terminal (`apps/desktop/src/app/right-sidebar/terminal/`)
- **High-contrast xterm palette:** Explicit `terminal` and `darkTerminal` palettes in `one-dark.ts` against `--ui-terminal-surface-background: #21252b` (primary foreground `#d7dae0` at **10.99:1** WCAG contrast; all readable ANSI colors >= **4.82:1**).
- **CSS zoom alignment:** `PersistentTerminal` (`persistent.tsx`) and `terminalSelectionAnchor` (`selection.ts`) unscale `getBoundingClientRect()` by `currentCSSZoom` (fallback `documentElement.style.zoom || 1`), with `Math.floor` on the right edge so the fixed overlay aligns flush under the `TERMINAL` header and never covers `TerminalRail` or its `+` button.
- **Rail & boot hygiene:** Full-height `TerminalRail` border (`chrome.tsx`, `rail.tsx`), non-overlapping icon spacing (`shrink-0`, `gap-1.5`), full-height xterm fit (`instance.tsx`, `use-terminal-session.ts`), and top-anchored fresh/expired shell output (`terminal-session-attempt.ts`).

## Build & Test

Run commands from the repository root (`hermes-agent`), using the pinned Node/pnpm runtime matching `pm/lock.json` (`--no-install` reuses `.build/webapp-workspace`):

```bash
export PATH="$HOME/.hermes/tools/pnpm-12.15.0-linux-x64:$HOME/.hermes/tools/node-26.12.1-linux-x64/bin:$PATH"

# Production webapp build (no dependency install)
node apps/desktop/scripts/build-webapp.mjs --no-install

# Theme + integrated terminal unit/CSSOM tests
cd .build/webapp-workspace/apps/desktop
./node_modules/.bin/vitest run \
  src/themes \
  src/app/right-sidebar/terminal/persistent.test.tsx \
  src/app/right-sidebar/terminal/selection.test.ts \
  src/app/right-sidebar/terminal/use-terminal-session.test.tsx \
  src/app/right-sidebar/terminal/instance.test.tsx \
  src/app/right-sidebar/terminal/rail.test.tsx
```

## Project Documentation
- Architecture decisions: [`.ai/adr/`](../../.ai/adr/)
- Known pitfalls & behaviors: [`.ai/pitfalls.md`](../../.ai/pitfalls.md)
- Technical rules & boundaries: [`.ai/regras.md`](../../.ai/regras.md), [`.ai/teto.md`](../../.ai/teto.md)
- Available agent skills: [`.agents/skills/README.md`](../../.agents/skills/README.md)
