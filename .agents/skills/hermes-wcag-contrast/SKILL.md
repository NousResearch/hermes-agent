---
name: hermes-wcag-contrast
description: Deterministic WCAG 2.1 contrast verification (>= 4.5:1 normal text, >= 3:1 UI/large text) for One Dark Pro surfaces and xterm ANSI terminal/darkTerminal palettes against --ui-terminal-surface-background (#21252b), plus focus-visible, keyboard navigation, and prefers-reduced-motion checks.
---

# Hermes WCAG Contrast & Accessibility Skill

Based on: https://skills.sh/vercel-labs/agent-skills/web-design-guidelines (license not verified — inspired only)

This skill provides deterministic contrast and accessibility verification for Hermes Desktop UI surfaces and integrated terminals.

## How to Verify Accessibility and Contrast

1. **Deterministic Contrast Auditing**
   - Enforce WCAG 2.1 AA contrast ratios: `>= 4.5:1` for normal text and `>= 3:1` for UI components and large text.
   - Audit One Dark Pro surfaces against `--ui-terminal-surface-background` (`#21252b`):
     - Primary foreground (`#d7dae0`): `>= 10.99:1`
     - Bright black (`#8a919e`): `>= 4.86:1`
     - Bright white (`#e6e6e6`): `>= 12.33:1`
   - Ensure dark-mode terminal palettes never fall back to dark text on dark backgrounds.

2. **Keyboard Navigation & Focus States**
   - Verify `focus-visible` outlines are clearly visible against surface backgrounds.
   - Ensure interactive elements support full keyboard operation (Tab, Shift+Tab, Enter, Escape).

3. **Motion & Reduced Motion**
   - Check that animated indicators (such as the "Thinking" wave) provide static fallbacks under `@media (prefers-reduced-motion: reduce)`.

## What this skill does NOT do
- Never relies on visual guessing without computed contrast ratios.
- Never lowers contrast below WCAG AA thresholds.
