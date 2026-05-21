---
name: ui-ux-pro-max
description: Use when building, designing, reviewing, or fixing any UI/UX. Includes 67 UI styles, 161 color palettes, 57 font pairings, 99 UX guidelines, 25 chart types, and 15+ tech stack guidelines. Searchable via embedded Python scripts with BM25 ranking. Auto-detects domain from query.
version: 1.0.0
author: Hermes Agent (adapted from NextLevelBuilder/ui-ux-pro-max-skill)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [ui, ux, design, design-system, color, typography, accessibility, frontend, css, tailwind, shadcn]
    related_skills: [design-md, popular-web-designs, claude-design]
---

# UI UX Pro Max — Design Intelligence

A searchable design intelligence database embedded as Python scripts with CSV data. Provides style recommendations, color palettes, font pairings, UX guidelines, chart types, landing page patterns, and tech-stack-specific best practices across 15+ frameworks.

## Overview

This skill wraps the **UI UX Pro Max** dataset (by NextLevelBuilder) into Hermes Agent. It contains:

- **67 UI styles** (glassmorphism, minimalism, brutalism, claymorphism, etc.)
- **161 color palettes** (shadcn token format)
- **57 font pairings** (with Google Fonts imports)
- **99 UX guidelines** (accessibility, touch, performance, forms, navigation)
- **25 chart types** with accessibility grading
- **161 product types** with reasoning rules
- **15+ tech stack guidelines** (React, Next.js, Vue, Svelte, Tailwind, shadcn, Flutter, SwiftUI, etc.)

All data lives in `scripts/` (Python) and `data/` (CSV) under this skill directory. Zero external dependencies — pure Python stdlib.

## When to Use

### Must Use
- Designing new pages (landing page, dashboard, SaaS, admin panel, mobile app)
- Creating or refactoring UI components (buttons, modals, forms, tables, charts)
- Choosing color schemes, typography, spacing, or layout systems
- Reviewing UI code for UX, accessibility, or visual consistency
- Implementing navigation, animations, or responsive behavior
- Making product-level design decisions (style, brand expression, information hierarchy)

### Recommended
- UI looks "not professional enough" but the reason is unclear
- Pre-launch UI quality optimization
- Aligning cross-platform design (Web / iOS / Android)
- Building design systems or reusable component libraries

### Skip
- Pure backend logic, API design, database design
- Infrastructure or DevOps work
- Non-visual scripts or automation

**Decision criteria**: If the task changes how a feature **looks, feels, moves, or is interacted with**, use this skill.

## Quick Start

### 1. Domain Search (auto-detect or explicit)

```bash
python3 scripts/search.py "glassmorphism dark mode" -d style
python3 scripts/search.py "saas dashboard" -d product
python3 scripts/search.py "color palette for fintech" -d color
python3 scripts/search.py "font pairing for modern tech" -d typography
python3 scripts/search.py "bar chart comparison" -d chart
python3 scripts/search.py "landing page conversion" -d landing
python3 scripts/search.py "accessibility touch target" -d ux
```

**Available domains:**

| Domain | CSV | What it covers |
|---|---|---|
| `style` | styles.csv | 67 UI styles + AI prompts + CSS keywords |
| `color` | colors.csv | 161 palettes (shadcn tokens: primary, secondary, accent, etc.) |
| `typography` | typography.csv | 57 font pairings + Google Fonts URLs |
| `product` | products.csv | 161 product type recommendations |
| `chart` | charts.csv | 25 chart types + accessibility + library recs |
| `landing` | landing.csv | Page structure + CTA strategies |
| `ux` | ux-guidelines.csv | 99 UX best practices + anti-patterns + code examples |
| `react` | react-performance.csv | React performance guidelines |
| `web` | app-interface.csv | Web interface best practices |
| `google-fonts` | google-fonts.csv | Google Fonts family search |
| `icons` | icons.csv | Icon library recommendations |

**Auto-detection**: omit `-d` and the engine detects the domain from keywords:

```bash
# auto-detects "style"
python3 scripts/search.py "minimalist glassmorphism"
# auto-detects "color"
python3 scripts/search.py "blue palette hex"
# auto-detects "chart"
python3 scripts/search.py "pie chart visualization"
```

### 2. Stack Search

```bash
python3 scripts/search.py "react performance" -s react
python3 scripts/search.py "component best practice" -s shadcn
python3 scripts/search.py "routing" -s nextjs
python3 scripts/search.py "state management" -s vue
python3 scripts/search.py "list rendering" -s flutter
```

**Available stacks:** `html-tailwind`, `react`, `nextjs`, `vue`, `svelte`, `astro`, `swiftui`, `react-native`, `flutter`, `nuxtjs`, `nuxt-ui`, `shadcn`, `jetpack-compose`, `threejs`, `angular`, `laravel`

### 3. Design System Generation (Full Recommendation)

```bash
# Complete design system with product + style + color + typography + landing
python3 scripts/search.py "SaaS analytics dashboard" -ds -p "MyProject"

# Persist to design-system/MASTER.md (Master + Overrides pattern)
python3 scripts/search.py "SaaS analytics dashboard" -ds -p "MyProject" --persist

# Persist + page-specific override
python3 scripts/search.py "SaaS analytics dashboard" -ds -p "MyProject" --persist --page "dashboard"
```

## Design Rule Categories (Priority 1→10)

When making UI decisions, follow these rules in priority order. Use `--domain <domain>` to query details.

| # | Category | Priority | Domain | Key Rules |
|---|---|---|---|---|
| 1 | Accessibility | CRITICAL | `ux` | 4.5:1 contrast, focus rings, ARIA, keyboard nav, screen reader support |
| 2 | Touch & Interaction | CRITICAL | `ux` | 44×44pt min touch target, 8px spacing, tap feedback, safe areas |
| 3 | Performance | HIGH | `react` | Lazy loading, bundle splitting, virtualize lists, image optimization |
| 4 | Style Selection | HIGH | `style` | Match style to product, consistency, SVG icons, platform-adaptive |
| 5 | Layout & Responsive | HIGH | `ux` | Mobile-first, viewport meta, 16px body min, spacing scale, max-width |
| 6 | Typography & Color | MEDIUM | `typography`, `color` | 1.5-1.75 line-height, 65-75 chars/line, semantic tokens, dark mode |
| 7 | Animation | MEDIUM | `ux` | 150-300ms, transform/opacity only, reduced-motion support |
| 8 | Forms & Feedback | MEDIUM | `ux` | Visible labels, inline validation, error placement, loading states |
| 9 | Navigation | HIGH | `ux` | Bottom nav ≤5 items, predictable back, deep linking |
| 10 | Charts & Data | LOW | `chart` | Legends, tooltips, accessible colors, never color-only meaning |

## Essential UX Rules (Quick Reference)

### Accessibility (CRITICAL)
- **Contrast**: 4.5:1 minimum for normal text, 3:1 for large text
- **Focus**: Visible focus rings on all interactive elements (2-4px)
- **ARIA**: aria-label for icon-only buttons; logical reading order
- **Keyboard**: Tab order matches visual order; skip links
- **Headings**: Sequential h1→h6, no level skip
- **Motion**: Respect prefers-reduced-motion
- **Dynamic Type**: Support system text scaling

### Touch & Interaction (CRITICAL)
- **Touch targets**: Min 44×44pt (Apple) / 48×48dp (Material)
- **Spacing**: Min 8px between touch targets
- **Loading**: Disable button during async; show spinner
- **Feedback**: Visual feedback on press within 100ms
- **Safe areas**: Keep targets away from notch, gesture bar, edges

### Performance (HIGH)
- **Images**: WebP/AVIF, srcset/sizes, lazy load below-fold
- **Fonts**: font-display: swap; preload only critical fonts
- **Lists**: Virtualize lists with 50+ items
- **Scripts**: async/defer third-party scripts
- **Layout**: Reserve space for async content (avoid CLS)
- **Frame budget**: <16ms per frame for 60fps

### Layout & Responsive (HIGH)
- **Viewport**: width=device-width, initial-scale=1 (never disable zoom)
- **Mobile-first**: Design mobile-first, scale up
- **Body text**: Min 16px on mobile (avoids iOS auto-zoom)
- **Line length**: 35-60 chars mobile, 60-75 desktop
- **Spacing**: 4pt/8dp incremental scale
- **Container**: Consistent max-width on desktop (max-w-6xl/7xl)
- **Z-index**: Define layered scale (0/10/20/40/100/1000)
- **Dvh**: Prefer min-h-dvh over 100vh on mobile

### Typography & Color (MEDIUM)
- **Line-height**: 1.5-1.75 for body text
- **Font pairing**: Match heading/body font personalities
- **Font scale**: Consistent type scale (12/14/16/18/24/32)
- **Semantic colors**: Define tokens (primary, secondary, error) not raw hex
- **Dark mode**: Desaturated/lighter variants, not inverted colors
- **Tabular figures**: Use for data columns, prices, timers

### Animation (MEDIUM)
- **Duration**: 150-300ms micro-interactions; ≤400ms complex
- **Properties**: transform/opacity only; never animate width/height/top/left
- **Loading states**: Skeleton/shimmer when loading >300ms
- **Easing**: ease-out for entering, ease-in for exiting
- **Max animated**: 1-2 key elements per view
- **Interruptible**: User input cancels in-progress animation

### Forms & Feedback (MEDIUM)
- **Labels**: Visible labels (not placeholder-only)
- **Errors**: Show below related field; include recovery path
- **Validation**: On blur, not on keystroke
- **Input types**: Use semantic types (email, tel, number) for mobile keyboards
- **Multi-step**: Show progress indicator; allow back navigation
- **Empty states**: Helpful message + action

## Recommended Workflow

### Step 1: Generate Design System (REQUIRED for new projects)
```bash
python3 scripts/search.py "<product type and style>" -ds -p "<Project Name>"
```

This searches across product → style → color → landing → typography domains and synthesizes recommendations using reasoning rules from `ui-reasoning.csv`.

### Step 2: Supplement with Targeted Searches
```bash
# Get specific style details
python3 scripts/search.py "glassmorphism" -d style -n 3

# Get UX guidelines for specific interaction
python3 scripts/search.py "form validation error" -d ux -n 3

# Get stack-specific best practices
python3 scripts/search.py "component performance" -s react -n 3
```

### Step 3: Apply Rules
When writing HTML/CSS/JSX:
1. Check `references/ui-rules.md` for the priority rules above
2. Search for specific patterns: `python3 scripts/search.py "button states" -d ux`
3. Apply the style/color/typography from Step 1's design system output

## Pre-Delivery Checklist

### Visual Quality
- [ ] Color palette consistent across all components
- [ ] Typography hierarchy clear (heading sizes, body text, labels)
- [ ] Spacing follows 4pt/8dp scale
- [ ] Icons are SVG (Heroicons, Lucide), not emojis
- [ ] Shadows/elevation follow a consistent scale

### Interaction
- [ ] Hover/pressed/disabled states visually distinct
- [ ] Loading buttons show spinner and are disabled during async
- [ ] Error messages near the problem field
- [ ] Touch targets ≥44×44pt with 8px spacing

### Light/Dark Mode
- [ ] Both modes designed (not just inverted)
- [ ] Contrast meets 4.5:1 in both modes
- [ ] Colors use desaturated variants in dark mode

### Layout
- [ ] Mobile-first with proper viewport meta
- [ ] No horizontal scroll on mobile
- [ ] Body text ≥16px on mobile
- [ ] Container max-width on desktop
- [ ] Fixed elements have safe padding

### Accessibility
- [ ] Contrast ratios meet WCAG AA (4.5:1)
- [ ] Focus rings visible on all interactive elements
- [ ] aria-labels on icon-only buttons
- [ ] Keyboard navigation works (tab order matches visual)
- [ ] Headings sequential (h1→h2→h3)
- [ ] Meaning not conveyed by color alone

## File Structure

```
skills/creative/ui-ux-pro-max/
├── SKILL.md                      # This file
├── scripts/
│   ├── search.py                 # CLI entry point
│   ├── core.py                   # BM25 search engine + domain auto-detect
│   └── design_system.py          # Design system generator with reasoning
├── data/
│   ├── styles.csv                # 67 UI styles
│   ├── colors.csv                # 161 color palettes
│   ├── typography.csv            # 57 font pairings
│   ├── products.csv              # 161 product types
│   ├── charts.csv                # 25 chart types
│   ├── landing.csv               # Landing page patterns
│   ├── ux-guidelines.csv         # 99 UX guidelines
│   ├── ui-reasoning.csv          # Reasoning rules for design system generation
│   ├── react-performance.csv     # React performance guidelines
│   ├── app-interface.csv         # Web interface best practices
│   ├── google-fonts.csv          # Google Fonts families
│   ├── icons.csv                 # Icon library recommendations
│   └── stacks/                   # 16 tech stack guidelines
│       ├── react.csv, nextjs.csv, vue.csv, svelte.csv, ...
└── references/                   # (optional) extracted reference docs
```

## Common Pitfalls

1. **Using emojis as icons** — Use SVG icon libraries (Lucide, Heroicons, Feather). Emojis look unprofessional and render inconsistently across platforms.
2. **Raw hex values in components** — Define semantic color tokens (primary, secondary, error, surface) and reference them.
3. **Placeholder-only form labels** — Always use visible `<label>` elements; placeholders disappear on input and break accessibility.
4. **Hover-only interactions** — Don't rely on hover for primary actions on mobile; use click/tap.
5. **No loading states** — Buttons should disable and show spinners during async operations.
6. **Inconsistent shadows** — Use a defined elevation scale, not random box-shadow values.
7. **100vh on mobile** — Use `min-h-dvh` instead; mobile browsers have dynamic toolbars.
8. **Font size too small** — Minimum 16px body text on mobile; iOS auto-zooms smaller inputs.
9. **Horizontal scroll** — Always test on mobile viewport width (375px).
10. **No dark mode consideration** — Design light and dark variants together from the start.

## Verification Checklist

- [ ] Design system generated via `-ds` before writing UI code
- [ ] Style matches product type (confirmed via `-d product` or `-d style`)
- [ ] Color palette from `-d color` applied as CSS custom properties or Tailwind config
- [ ] Typography pairing from `-d typography` with correct Google Fonts import
- [ ] UX guidelines checked for the specific interaction pattern
- [ ] Stack guidelines applied (`-s react`, `-s shadcn`, etc.)
- [ ] Pre-delivery checklist completed
- [ ] Accessibility: contrast, focus, labels, keyboard tested
