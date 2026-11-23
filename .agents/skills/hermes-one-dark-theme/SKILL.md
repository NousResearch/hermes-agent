---
name: hermes-one-dark-theme
description: Safely edit Atom One Dark Pro theme tokens (apps/desktop/src/themes/one-dark.ts) and unlayered renderer rules (apps/desktop/src/themes/one-dark.css) under Tailwind v4, preserving Symbol('one-dark-pro') provenance, data-hermes-palette='one-dark', clean teardown, and CSSOM/unit test updates.
---

# Hermes One Dark Theme Skill

Based on: https://skills.sh/wshobson/agents/tailwind-design-system (MIT License)

This skill guides safe modifications to the Atom One Dark Pro theme implementation in Hermes Desktop (`apps/desktop/`).

## How to Edit One Dark Pro

1. **Maintain Runtime Provenance Guard (`one-dark.ts`)**
   - Use `Symbol('one-dark-pro')` via `isOneDark` to identify the built-in preset.
   - Set `document.documentElement.dataset.hermesPalette = 'one-dark'` when active so renderer selectors target the exact palette without hijacking user-installed custom themes.

2. **Closed CSS Token Policy (`oneDarkProperties`)**
   - Publish exact Atom One Dark Pro tokens when active.
   - Clear all previously set properties before applying subsequent themes to prevent token leakage across theme switches.
   - Standard surfaces: chrome/sidebar `#282c34`, editor/card/popover/terminal `#21252b`, input/composer `#1d1f23`, muted/bubble `#2c313a`, borders `#181a1f`.
   - Foreground and semantic colors: primary text `#abb2bf`, muted `#7f848e`, accent/destructive `#e06c75`, success `#98c379`, warning `#d19a66`, links `#61afef`.

3. **Unlayered Renderer Stylesheet (`one-dark.css`)**
   - Keep overrides scoped to `:root[data-hermes-palette='one-dark']`.
   - Style continuous sidebar/content seams, opaque input/composer/search fields, floating menus, and markdown typography (`#e06c75` headings, `#d19a66` strong, `#61afef` links, `#abb2bf` list markers/counters, `#c678dd` checkboxes).
   - Configure text selection (`::selection` background `rgba(97, 175, 239, 0.4)` / color `#e6e6e6`) and the live "Thinking" indicator wave with `prefers-reduced-motion: reduce` fallback.

4. **Testing and Verification**
   - Update and run unit and CSSOM tests (`one-dark.test.ts`, `one-dark-input.test.tsx`, `skin.test.ts`, `context.test.tsx`, `retint.test.ts`, `persist-accent.test.tsx`, `backend-sync.test.ts`).

## What this skill does NOT do
- Never alters user-installed custom themes.
- Never breaks light or other non-One-Dark presets.
- Never touches repository root README files.
