---
version: alpha
name: Hermes Agent Desktop
description: Calm, dense developer tool UI — near-white chrome, single blue accent, muted semantic palette. Prioritizes glanceability and focus over decoration.
colors:
  primary: "#0053FD"
  foreground: "#17171A"
  background: "#F8FAFF"
  sidebar: "#F3F7FF"
  card: "#FFFFFF"
  muted: "#F3F3F3"
  border: "#E2E6F0"
  destructive: "#CF2D56"
  warm: "#CF806D"
  green: "#1F8A65"
  yellow: "#C08532"
  orange: "#DB704B"
  cyan: "#4C7F8C"
  purple: "#9E94D5"
  blue: "#0053FD"
  red: "#CF2D56"
  primaryForeground: "#FCFCFC"
  secondary: "#E8F0FF"
  accentSoft: "#EBF2FF"
  bubble: "#EFF4FF"
typography:
  body:
    fontFamily: "'Segoe WPC', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif"
    fontSize: 1rem
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: 0
  mono:
    fontFamily: "Menlo, Monaco, 'SF Mono', 'Courier Prime', monospace"
    fontSize: 0.875rem
    fontWeight: 400
    lineHeight: 1.5
  small:
    fontFamily: "'Segoe WPC', 'Segoe UI', -apple-system, system-ui, sans-serif"
    fontSize: 0.8125rem
    fontWeight: 400
    lineHeight: 1.4
rounded:
  xs: 2px
  sm: 8px
  md: 10px
  lg: 12px
  xl: 16px
spacing:
  xs: 4px
  sm: 8px
  md: 12px
  lg: 16px
  xl: 24px
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.primaryForeground}"
    rounded: "{rounded.sm}"
    padding: 8px 16px
  button-primary-hover:
    backgroundColor: "#0046DB"
  button-secondary:
    backgroundColor: "{colors.secondary}"
    textColor: "{colors.primary}"
    rounded: "{rounded.sm}"
    padding: 8px 16px
  card-default:
    backgroundColor: "{colors.card}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.md}"
    padding: 16px
  input-field:
    backgroundColor: "{colors.card}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.sm}"
    padding: 8px 12px
  badge-destructive:
    backgroundColor: "{colors.destructive}"
    textColor: "#FFFFFF"
    rounded: "{rounded.sm}"
    padding: 2px 8px
---

## Overview

Hermes Agent Desktop is a **dense developer tool**, not a marketing site. The UI is a
workspace: near-white neutral chrome with one blue accent that marks interactivity.
Everything serves glanceability — operators scan message threads, logs, and file trees
for hours. Decoration that doesn't carry state is noise and must not exist.

Surface archetype: **Monitor/Operate**. Density and glanceable hierarchy beat heroes,
gradients, or feature cards. There is no "hero" anywhere in this product.

## Colors

- **Primary (#0053FD):** The single interaction color — buttons, links, focus rings,
  active states. Nothing else may use blue.
- **Foreground (#17171A):** Nearly-black ink for primary text. Never pure #000.
- **Background (#F8FAFF):** Cool near-white chrome. Sidebar is a touch bluer (#F3F7FF).
- **Card (#FFFFFF):** Pure white for elevated surfaces (editor, popovers).
- **Muted (#F3F3F3):** Tertiary fills, disabled states, neutral chrome.
- **Destructive (#CF2D56):** The ONLY red — errors, deletions, destructive actions.
- **Semantic accents:** green (#1F8A65) success/diff-add, orange (#DB704B) warnings,
  yellow (#C08532) caution, cyan (#4C7F8C) info, purple (#9E94D5) special.
  These are used sparingly for status — never as decoration.

## Typography

Segoe WPC/Segoe UI stack for everything (system-native, renders crisp at any DPI).
Menlo/Monaco for code and logs. Base 1rem, line-height 1.5, no letter-spacing.
Small labels 0.8125rem. Use **weight and size for hierarchy, not color or boxes**.

## Layout & Spacing

4px base grid, spacing scale 4/8/12/16/24. Dense by default — 8px is the common
inter-element gap. Panels use 12-16px padding. Never center content for hierarchy;
align left and let density communicate structure.

## Elevation & Depth

Flat surfaces; separation via background tint, not shadows. Subtle 1px borders
(`#E2E6F0`) separate panels. Shadows only for floating surfaces (popovers, menus,
composer) at 1-2px offset with ≤5% black. No glassmorphism, no blur.

## Shapes

Radius scale: xs 2px (chips, tiny), sm 8px (buttons, inputs), md 10px (cards),
lg 12px (panels), xl 16px (large dialogs). Small radii — this is a tool, not a toy.

## Components

- **Button primary:** blue fill, white text, 8px radius. The only high-emphasis action.
- **Button secondary:** 7% blue-on-white tint, blue text. Default for most actions.
- **Cards:** white on chrome, 10px radius, 1px border, 16px padding.
- **Inputs:** white, 8px radius, 1px border, 8px 12px padding.
- **Badges:** small radius, muted fill for neutral, destructive fill for errors.
- **Chat bubbles (user):** 6% blue tint (#EFF4FF), thin border — not a filled blue bubble.

## Do's and Don'ts

**Do:**
- Use blue for ONE thing: interactivity. If it's not clickable, it's not blue.
- Keep surfaces flat; separate with tint and 1px borders.
- Prefer density: more information per screen, aligned left.
- Use semantic colors only for status (success/warning/error), never aesthetics.
- Let typography weight/size carry hierarchy.
- Use small radii; the product is a precision tool.

**Don't:**
- Don't use gradients, glassmorphism, or blur.
- Don't add accent lines/strips under titles — whitespace and weight do that job.
- Don't build feature-tile grids (icon + heading + text × 3) — this is not a landing page.
- Don't center-stack content for "balance".
- Don't use purple/indigo as a default accent — blue is the brand accent.
- Don't invent icons as decoration; only where they aid scanning.
- Don't use oversized numbers or monument stats.
- Don't add emoji to UI copy.
