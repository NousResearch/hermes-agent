# ADR-0001: One Dark Pro Theme Preset & Renderer Surface Policy

- **Status:** Accepted
- **Date:** 2026-11-23
- **Scope:** `apps/desktop/src/themes/one-dark.ts`, `apps/desktop/src/themes/one-dark.css`, and desktop theme wiring

## Context

The Hermes Desktop/Webapp renderer converts backend YAML skins (`~/.hermes/skins/<name>.yaml`) into desktop theme tokens via `skinToDesktopTheme`, and applies additional CSS mixes and glass defaults in `styles.css`. For Atom One Dark Pro, generic skin derivation distorted surface contrast, muted text readability, input chrome opacity, and integrated terminal legibility.

## Decision

1. **Runtime provenance guard:** Identify the built-in One Dark Pro backend preset using a non-serializable runtime symbol (`Symbol('one-dark-pro')` via `isOneDark`) and mark `document.documentElement.dataset.hermesPalette = 'one-dark'`. User-installed themes that happen to share the `one-dark` name or prefix are never hijacked.
2. **Closed CSS token policy (`oneDarkProperties`):** Publish exact Atom One Dark Pro tokens when the preset is active, clearing all previously set properties before applying any subsequent theme so no stale values leak across theme switches:
   - Surfaces: chrome/sidebar `#282c34`, editor/card/popover/terminal `#21252b`, input/composer `#1d1f23`, muted/bubble `#2c313a`, borders `#181a1f`.
   - Foreground & semantic colors: primary text `#abb2bf`, muted `#7f848e`, accent/destructive `#e06c75`, success `#98c379`, warning `#d19a66`, links `#61afef`.
3. **Unlayered renderer stylesheet (`one-dark.css`):** Keep renderer-only surface overrides scoped to `:root[data-hermes-palette='one-dark']`:
   - Continuous `#181a1f` sidebar/content seam (`--sidebar-edge-border`).
   - Opaque `#1d1f23` inputs/composer/search fields and `#21252b` floating menus.
   - Markdown typography: headings `#e06c75`, strong `#d19a66`, links `#61afef`, list markers (`li::marker`, `--tw-prose-bullets`, `--tw-prose-counters`) `#abb2bf`, checkboxes `#c678dd`.
   - Text selection (`::selection`): `background: rgba(97, 175, 239, 0.4)` and `color: #e6e6e6`.
   - Live "Thinking" indicator wave (`@keyframes od-sun`) with `prefers-reduced-motion: reduce` fallback (`#ffd23f`, static).
4. **High-contrast xterm palette (`terminal` and `darkTerminal`):** Define an explicit One Dark Pro ANSI palette on both `terminal` and `darkTerminal` against `--ui-terminal-surface-background: #21252b`, ensuring WCAG AA contrast (>= 4.5:1) for all readable foreground/ANSI colors (primary foreground `#d7dae0` at 10.99:1, `brightBlack` `#8a919e` at 4.86:1, `brightWhite` `#e6e6e6` at 12.33:1), preventing light-mode fallback from painting dark text on a dark terminal surface.

## Consequences

- One Dark Pro renders with exact Atom palette fidelity across chat, sidebar, inputs, popovers, markdown, and the integrated xterm terminal.
- Switching away from One Dark Pro cleanly removes `data-hermes-palette` and all inline overrides.
- All behavior is covered by Vitest unit and CSSOM tests (`one-dark.test.ts`, `one-dark-input.test.tsx`, `skin.test.ts`, `context.test.tsx`, `retint.test.ts`, `persist-accent.test.tsx`, `backend-sync.test.ts`).
