# Design System: Bugatti


> **Hermes Agent — Implementation Notes**
>
> Source analysis adapted from [`VoltAgent/awesome-design-md`](https://github.com/VoltAgent/awesome-design-md) at commit `664b3e78`; see `../references/ATTRIBUTION.md` for the MIT notice.
>
> Bugatti's proprietary typography is a strict three-family system. Preserve each role separately:
> - **Display:** `Saira Condensed` at weight 400 with +0.05em tracking
> - **Editorial body:** `Cormorant Garamond` (or EB Garamond)
> - **Buttons, captions, and nav:** `JetBrains Mono`
> - **Display stack:** `font-family: 'Saira Condensed', Impact, sans-serif;`
> - **Body stack:** `font-family: 'Cormorant Garamond', Georgia, serif;`
> - **Mono stack:** `font-family: 'JetBrains Mono', ui-monospace, monospace;`
> ```html
> <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Saira+Condensed:wght@400&display=swap" rel="stylesheet">
> ```
> Use `write_file` to create HTML and open the rendered artifact with `browser_navigate`.
> Verify visual accuracy with `browser_vision` after generating.

## Overview

Bugatti's marketing surface is the most austere interface in luxury automotive: a near-pure black canvas (`{colors.canvas}` — #000000) holding white uppercase **letterspaced** display type and full-bleed automotive photography. The system has no accent color, no surface card decoration, no shadows, no gradients, no chrome — only **photography, typography, and the brand wordmark**. Every other luxury auto site in this category (BMW M, Aston Martin, Lamborghini) uses some form of accent color or signature element; Bugatti uses nothing. The empty space, the photograph, and the precisely-tracked Bugatti Display headline ARE the brand.

The system runs **three custom Bugatti typefaces**: **Bugatti Display** (display headlines, the "BUGATTI" wordmark, all caps with wide tracking), **Bugatti Text Regular** (body paragraphs, a serif text face), and **Bugatti Monospace** (button labels, navigation, captions, dates — anywhere precision and machined feel matters). The split is deliberate and unbreakable: never use Bugatti Text in a button, never use Bugatti Monospace in a paragraph.

Display sizes use weight 400 (regular) — never bold. Visual emphasis comes from **size and tracking**, not weight. Letter-spacing on the wordmark is 6px; on display headlines 2-4px; on uppercase labels 2-2.5px. Tight tracking is a brand violation. The wide spacing creates the "engineered precision" feel that no other luxury maker matches.

**Key Characteristics:**
- Pure black canvas (`{colors.canvas}` — #000000) with white type. The system does not have a light mode.
- Three custom Bugatti typefaces: **Display** (uppercase headlines + wordmark), **Text Regular** (body serif), **Monospace** (buttons, captions, nav).
- All display headlines are UPPERCASE with wide letter-spacing (2-4px). Body copy stays sentence-case at standard tracking.
- No accent color. The only non-monochrome color anywhere on the site is `{colors.link}` (#c3d9f3) — a desaturated ice-blue used on inline anchor links, and even that appears rarely.
- Buttons are pill-shaped (`{rounded.pill}`) with **transparent background** and a 1px white outline. Bugatti is the only luxury-auto brand whose primary CTA is fully transparent.
- Photography is the only depth element. No drop shadows. No gradients. No card surfaces. Surface cards are `{colors.surface-card}` (#141414) at most — a barely-different-from-black tone.
- Section rhythm is generous — `{spacing.section}` (120px) between major bands, longer than most marketing sites because Bugatti's pages are mostly photography with minimal text density.

## Colors

### Brand & Accent
- **Primary** (`{colors.primary}` — #ffffff): The single brand color. White type and white CTA outlines on the black canvas.
- **Link** (`{colors.link}` — #c3d9f3): The only non-monochrome color in the system — a desaturated ice-blue used on inline anchor links and rarely on focus states. Bugatti's brand discipline is so tight that this single token is essentially the entire chromatic vocabulary outside black-and-white.

### Surface
- **Canvas** (`{colors.canvas}` — #000000): The default page floor across every surface. Pure black.
- **Surface Soft** (`{colors.surface-soft}` — #0d0d0d): A barely-different-from-black tone used for spec table rows and dense data sections.
- **Surface Card** (`{colors.surface-card}` — #141414): Cards (career callout, newsroom article container, occasional content cards). Even card surfaces stay nearly-black — no contrast jump.
- **Surface Elevated** (`{colors.surface-elevated}` — #1f1f1f): One step further from black, used for nested cards on rare dense pages.
- **Hairline** (`{colors.hairline}` — #262626): The 1px divider tone. Visible but quiet. Used on table rows, between body sections, around card outlines.
- **Hairline Strong** (`{colors.hairline-strong}` — #3a3a3a): A heavier divider used on the underside of input fields (input fields have no border — only an underline hairline).

### Text
- **Ink / On Dark** (`{colors.on-dark}` — #ffffff): All headline and primary text on dark canvas.
- **Body** (`{colors.body}` — #cccccc): Default running-text color (slightly cooler than pure white). Used in body paragraphs.
- **Body Strong** (`{colors.body-strong}` — #e6e6e6): Emphasized body / lead paragraph.
- **Muted** (`{colors.muted}` — #999999): Footer links, dates, captions, secondary metadata. Dembrandt's frequency analysis confirms this as palette-2 (count 6, medium confidence).
- **Muted Soft** (`{colors.muted-soft}` — #666666): A second-tier muted for very-secondary text (legal disclaimer, copyright line).

### Semantic
- **Warning** (`{colors.warning}` — #d4a017): Reserved for technical-warning callouts (specifications, recall notices). Almost never appears on marketing surfaces.
- **Success** (`{colors.success}` — #5fa657): Order confirmation states (rare on marketing pages).

## Typography

### Font Family
The system runs **three custom Bugatti typefaces** as a rigid trinity:
1. **Bugatti Display** — All display headlines (h1, h2, h3), the "BUGATTI" wordmark, model name plates. Uppercase, wide-tracked. The default for any visual emphasis.
2. **Bugatti Text Regular** — A serif text face used exclusively for running body copy, lead paragraphs, model descriptions. Standard sentence-case, no letter-spacing.
3. **Bugatti Monospace** — Button labels, navigation, captions, dates, monospace-precision contexts. Always uppercase with 2-2.5px tracking.

The split is functional and absolute. Bugatti Display in a button breaks the "machined precision" voice; Bugatti Monospace in a paragraph breaks the "engineered elegance" voice; Bugatti Text in a button is unthinkable.

The fallback stack walks `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif` for Bugatti Display, `Garamond, "Times New Roman", serif` for Bugatti Text Regular, and `ui-monospace, "SF Mono", "Cascadia Mono", monospace` for Bugatti Monospace.

### Hierarchy

| Token | Size | Weight | Line Height | Letter Spacing | Use |
|---|---|---|---|---|---|
| `{typography.display-xl}` | 64px | 400 | 1.1 | 4px | Hero h1 ("THE BUGATTI F.K.P. HOMMAGE", "TOURBILLON") — Bugatti Display, uppercase, wide-tracked |
| `{typography.display-lg}` | 48px | 400 | 1.15 | 3px | Section heads — Bugatti Display, uppercase |
| `{typography.display-md}` | 32px | 400 | 1.2 | 2px | Sub-section heads, model names — Bugatti Display |
| `{typography.display-sm}` | 24px | 400 | 1.3 | 1.5px | Card titles — Bugatti Display |
| `{typography.wordmark}` | 14px | 400 | 1.0 | 6px | The "BUGATTI" brand wordmark in the top nav — Bugatti Display, the widest tracking in the system |
| `{typography.title-md}` | 20px | 400 | 1.3 | 1px | Career listing titles, intro paragraphs — Bugatti Display |
| `{typography.title-sm}` | 16px | 400 | 1.3 | 1.5px | Mid-tier headlines, callout cards |
| `{typography.caption-uppercase}` | 11px | 400 | 1.4 | 2px | Photo captions, metadata, "EXPLORE OUR OPPORTUNITIES" — Bugatti Monospace, uppercase |
| `{typography.body-md}` | 16px | 400 | 1.5 | 0 | Default body — Bugatti Text Regular (a serif face), sentence case, no tracking |
| `{typography.body-sm}` | 14px | 400 | 1.5 | 0 | Footer body, fine-print legal — Bugatti Text Regular |
| `{typography.button}` | 14px | 400 | 1.0 | 2.5px | All button labels — Bugatti Monospace, uppercase, 2.5px tracking |
| `{typography.nav-link}` | 12px | 400 | 1.4 | 2px | Top-nav menu items ("MENU", "STORE") — Bugatti Monospace |

### Principles
The system NEVER uses bold weight. Every Bugatti typeface is set at weight 400 (regular). Visual emphasis comes from:
1. **Size** — 64px hero vs 16px body is a 4× hierarchy
2. **Letter-spacing** — 6px wordmark vs 0px body
3. **Case** — Uppercase display vs sentence-case body
4. **Family contrast** — Display vs Text Regular vs Monospace

Going to weight 700 anywhere would break the "modest engineering" feel and make Bugatti read like a generic luxury template.

The serif Bugatti Text Regular sets the brand apart from the all-sans luxury crowd (BMW, Aston Martin, Lamborghini all use sans-serif body type). Bugatti's serif body voice signals literary, considered, slow-reading prose — which is the brand's editorial philosophy.

### Note on Font Substitutes
If Bugatti Display, Bugatti Text Regular, and Bugatti Monospace are unavailable, the closest open-source substitutes are:
- **Bugatti Display** → **Saira Condensed** (variable, weight 400) at +0.05em letter-spacing
- **Bugatti Text Regular** → **Cormorant Garamond** (regular) or **EB Garamond**
- **Bugatti Monospace** → **JetBrains Mono** or **IBM Plex Mono** (regular weight)

The substitution preserves the three-family split, which is more important than exact typeface match.

## Layout

### Spacing System
- **Base unit:** 4px.
- **Tokens:** `{spacing.xxs}` 4px · `{spacing.xs}` 8px · `{spacing.sm}` 12px · `{spacing.md}` 16px · `{spacing.lg}` 24px · `{spacing.xl}` 40px · `{spacing.xxl}` 64px · `{spacing.section}` 120px.
- **Section padding:** `{spacing.section}` (120px) — longer than most marketing sites because Bugatti's bands are mostly photography with minimal text. The empty space frames the cars.
- **Card internal padding:** `{spacing.lg}` (24px) for newsroom and content cards; `{spacing.md}` (16px) for the career callout card; `{spacing.xxl}` (64px) inside hero photo bands.
- **Gutters:** `{spacing.xl}` (40px) between cards in 2-up grids — wider than typical because Bugatti's grids are sparse.

### Grid & Container
- **Max content width:** ~1280px centered. Hero photo bands bleed full-width with no max.
- **Editorial body:** Single 12-column grid; photo bands are full-bleed.
- **Newsroom layout:** 2-up article grid at desktop, 1-up at tablet+mobile.
- **Career listings:** Single column with 80px row spacing.

### Whitespace Philosophy
Bugatti uses whitespace more aggressively than any luxury-auto competitor. The homepage hero is mostly photography + huge whitespace + a single sentence + a single button. The empty black space below the photograph is intentional — it lets the car breathe. Compressing the whitespace to "fit more content" breaks the brand's fundamental contract: that less is more.

## Elevation & Depth

| Level | Treatment | Use |
|---|---|---|
| Flat | No shadow, no border | Body, top nav, footer, photo bands |
| Soft hairline | 1px `{colors.hairline}` border | Section dividers, table rows |
| Card surface | `{colors.surface-card}` background — no shadow | Career callout, newsroom article container |
| Photographic depth | Full-bleed photography with edge-to-edge crop | Hero bands, model showcases — depth via subject + lens, not chrome |

The system uses no shadows, no glassmorphism, no gradients. Depth comes entirely from photography (lighting, lens, subject framing) and from the contrast between black canvas and minimally-elevated `{colors.surface-card}`.

### Decorative Depth
- None. Bugatti is the only luxury-auto brand without a single decorative element. There is no stripe, no badge, no heritage emblem on the marketing site outside the wordmark itself.

## Shapes

### Border Radius Scale

| Token | Value | Use |
|---|---|---|
| `{rounded.none}` | 0px | All cards, photo containers, inputs, spec cells — the dominant radius |
| `{rounded.pill}` | 9999px | All buttons (the only rounded element in the system) |
| `{rounded.full}` | 9999px / 50% | Circular icon buttons, avatar surfaces |

The radius hierarchy is binary: rectangular for everything except buttons, which are pills. No 4px, no 8px, no 12px in between — those would feel "designed" rather than "engineered."

### Photography Geometry
Hero photography fills full-width with no rounding. Photo cards inside grids retain `{rounded.none}` (0px) corners, edge-to-edge images. Model detail shots use 16:9 or wider cinema-aspect ratios. Newsroom thumbnails use 16:9 with 0px corners. There are no avatars or rounded photo crops anywhere on the marketing site.

## Components

### Top Navigation

**`top-nav`** — A 56px-tall transparent nav bar overlaid on the hero photo at the top of every page. No fill, no border. Carries "MENU" at left, the centered **wordmark-display** ("BUGATTI" in 14px Bugatti Display with 6px tracking), and "STORE" at right with a small bag icon. All labels in `{typography.nav-link}` (Bugatti Monospace, 12px, 2px tracking, uppercase).

**`wordmark-display`** — The "BUGATTI" wordmark itself. Bugatti Display at 14px, weight 400, 6px letter-spacing. The widest tracking in the system. Centered in the nav bar at every breakpoint.

### Buttons

**`button-primary`** — The signature primary CTA. Background **transparent**, text `{colors.on-dark}` (white), 1px white outline, rounded `{rounded.pill}` (9999px), padding 14px × 32px, height 44px. Type `{typography.button}` — Bugatti Monospace, uppercase, 14px, 2.5px tracking. The transparent fill is unique to Bugatti — every other luxury-auto brand uses a filled or outlined-with-text-shift button. Bugatti's transparent pill IS the button.

**`button-icon`** — Circular icon buttons (carousel arrows, share, language switcher). 40 × 40px, transparent background, white outline 1px, rounded `{rounded.full}`. Same outline-only treatment as the primary button.

**`text-link`** — Inline body links in `{colors.link}` (#c3d9f3, the only non-monochrome color in the system). Underlined by default. Type inherits `{typography.body-md}` (Bugatti Text Regular, serif).

### Cards & Containers

**`hero-photo-band`** — Full-width black band with full-bleed automotive photography. The h1 in `{typography.display-xl}` sits center-aligned over the photo near the top, often paired with a small Bugatti Monospace caption (`{typography.caption-uppercase}`) below the headline and a single `{component.button-primary}` further down. Vertical padding 96px-200px depending on photo height.

**`career-callout-card`** — A small right-aligned card that floats over the hero photo on the homepage with a recruiting prompt ("Are you ready for a new adventure?"). Background `{colors.surface-card}`, rounded `{rounded.none}` (0px), padding `{spacing.md}` (16px), width 320px. Carries a small thumbnail at top, body line, and a `{typography.caption-uppercase}` link ("EXPLORE OUR OPPORTUNITIES").

**`model-photo-card`** — Used in model showcases (Tourbillon page, model lineup grid). Background `{colors.canvas}` (no card surface — just photo on black), rounded `{rounded.none}`. Top: 16:9 or 21:9 hero shot of the model. Below: model name in `{typography.display-md}` (32px Bugatti Display, 2px tracking), short specs line in `{typography.caption-uppercase}` (11px Bugatti Monospace), a `{component.text-link}` ("DISCOVER").

**`newsroom-article-card`** — Used on the newsroom page (newsroom.bugatti.com). Background `{colors.canvas}` with hairline border, rounded `{rounded.none}`, padding `{spacing.lg}` (24px). Carries a 16:9 thumbnail, a `{component.date-pill}` ("12. NOVEMBER 2025"), a `{typography.title-md}` headline, and a body excerpt in `{typography.body-md}` (Bugatti Text Regular serif).

**`career-listing-row`** — Each row of the careers page job listing. Transparent background, padding 24px vertical, hairline divider between rows. Job title in `{typography.title-md}` (Bugatti Display 20px) at left; location + department in `{typography.caption-uppercase}` at right; chevron arrow (→) at far right.

**`spec-cell`** — Vehicle technical-spec display on model-detail pages (Tourbillon engine specs). Transparent background with hairline dividers between cells (not between cells inside a card). Each spec shows a value in `{typography.title-md}` at top and a label in `{typography.caption-uppercase}` below. Padding 24px vertical.

### Inputs & Forms

**`text-input`** — Standard text input on dark canvas. Background **transparent**, text `{colors.on-dark}`, 1px hairline-strong bottom border only (no top, left, right border), padding 12px × 0px, height 44px. Type `{typography.body-md}` (Bugatti Text Regular). Placeholder in `{colors.muted}`. Focus thickens the bottom border to white.

### Tags & Captions

**`caption-overlay`** — Photo-overlay caption (e.g., "HONORING THE OEYRON AND ITS VISIONARY CREATOR"). Centered or left-aligned over photography in `{typography.caption-uppercase}` (Bugatti Monospace, 11px, 2px tracking, white).

**`category-tag`** + **`date-pill`** — Both render as transparent inline labels in `{typography.caption-uppercase}`, color `{colors.muted}`. No background fill, no border. The "tag" is the type itself.

### CTA / Footer

**`cta-band-photo`** — A pre-footer "Discover Bugatti" band with full-bleed photography of a Bugatti car at speed and a centered headline in `{typography.display-md}` + a `{component.button-primary}` below. Vertical padding 80px. Inherits the editorial gravity of the hero through full-bleed photography.

**`footer`** — Black footer that closes every page. Background `{colors.canvas}`, text `{colors.muted}`. 4-column link list at desktop covering Bugatti / Models / Heritage / Connect. Vertical padding 64px. Bottom row carries the copyright line in `{typography.body-sm}` (Bugatti Text Regular). The wordmark sits center-aligned at the very bottom. The footer never inverts.

## Do's and Don'ts

### Do
- Anchor every page with full-bleed automotive photography. The cars are the brand voltage; chrome backs off entirely.
- Keep all display headlines in UPPERCASE Bugatti Display with 2-4px letter-spacing. The wordmark gets 6px.
- Use Bugatti Display for headlines, Bugatti Text Regular (serif!) for body, Bugatti Monospace for buttons + captions + nav. The trinity is unbreakable.
- Keep `{component.button-primary}` transparent with a 1px white outline. The transparent pill IS the brand button.
- Use weight 400 everywhere. Bold breaks the brand voice — the system has no bold weight role.
- Use `{spacing.section}` (120px) between major editorial bands. The whitespace is part of the brand.
- Reserve `{colors.link}` (#c3d9f3) for inline anchor links only. It's the system's only non-monochrome color.

### Don't
- Don't introduce any accent color outside `{colors.link}`. Bugatti's brand discipline is total monochrome + photography. Adding a brand-blue or brand-red breaks the contract.
- Don't bold any type. The system has no bold weight — every typeface stays at 400.
- Don't fill primary buttons. Transparent + outline only. A solid white button reads as off-brand.
- Don't compress whitespace between sections. The 120px rhythm is part of the editorial pacing.
- Don't use rounded corners outside buttons. Cards, photos, inputs all stay at 0px. Rounded cards read as consumer-tech, not luxury-engineered.
- Don't tighten letter-spacing on display headlines. 2-4px tracking on Bugatti Display is non-negotiable.
- Don't use Bugatti Display in a button (use Bugatti Monospace) or Bugatti Monospace in a paragraph (use Bugatti Text Regular). The trinity split is the brand voice.

## Responsive Behavior

### Breakpoints

| Name | Width | Key Changes |
|---|---|---|
| Mobile | < 768px | Hamburger nav; hero h1 64→32px; career callout card hides; photo bands stay full-bleed; footer 4 cols → 1 |
| Tablet | 768–1024px | Top nav stays minimal (MENU + wordmark + STORE); 2-up newsroom grid; career rows full-width |
| Desktop | 1024–1440px | Full minimal top-nav; 2-up newsroom grid; spec tables 4-up |
| Wide | > 1440px | Same as desktop with more breathing room; max content 1280px |

### Touch Targets
- `{component.button-primary}` renders at minimum 44 × 44px (matches WCAG AAA).
- `{component.button-icon}` is exactly 40 × 40px.
- `{component.text-input}` height is 44px.
- Career listing rows have 24px vertical padding; effective tap area meets 44px+ with surrounding spacing.

### Collapsing Strategy
- Top nav stays minimal at all breakpoints (MENU label + wordmark + STORE label). On mobile the labels hide behind a hamburger but the wordmark stays centered.
- Hero photography stays full-bleed at every breakpoint. Photo crops adjust — wider crops at desktop, vertical crops on mobile.
- The career callout card on the homepage hides at < 768px (it's a desktop-only floating element).
- 2-up newsroom grid collapses to 1-up at < 768px.
- Spec cells reflow from 4-up to 2-up to 1-up; values stay at the same display size regardless of column count.

### Image Behavior
- Hero photography crops responsively — wider crops at desktop, vertical crops on mobile. Bugatti cars are always shown in motion or at-angle (never flat profiles).
- Newsroom thumbnails retain 16:9 ratio and 0px corners.

## Iteration Guide

1. Focus on ONE component at a time. Reference its YAML key (`{component.hero-photo-band}`, `{component.career-callout-card}`).
2. New components default to `{rounded.none}` (0px). Only `{component.button-primary}` and `{component.button-icon}` use pill / full radius.
3. Variants live as separate entries in `components:`.
4. Use `{token.refs}` everywhere — never inline hex.
5. Never document hover. Default and Active/Pressed states only.
6. Display headlines stay UPPERCASE Bugatti Display 400 with 2-4px tracking. Body stays sentence-case Bugatti Text Regular (serif). Button labels stay Bugatti Monospace 2.5px tracking. The trinity does not blur.
7. When in doubt about emphasis: bigger photography before bigger type.

## Known Gaps

- The dembrandt frequency analyzer captured only 3 colors at root level (`#000000`, `#999999`, `#c3d9f3`). The white text (#ffffff) and dark surface tones (`#0d0d0d`, `#141414`, `#1f1f1f`) were inferred from screenshot — Bugatti's pages are so monochrome that the frequency-based analyzer didn't surface body text or surface tones as distinct palette entries.
- The three Bugatti typefaces (Display, Text Regular, Monospace) are licensed to Bugatti and not available as web fonts publicly. Substitutes are documented in the typography section.
- Animation and transition timings (photo carousel transitions, hover-reveal of menu, configurator animations) are not in scope.
- Form validation states beyond the underline-only `{component.text-input}` are not extracted — error / success states are inferred from general standards, not from the analyzed surfaces.
- The configurator surface (vehicle build pages with custom paint / interior pickers) was not in the analyzed URL set; its swatch grid, customization controls, and price-summary card are not documented here.
- The German-language newsroom (newsroom.bugatti.com/de) shares the system with the English Bugatti.com surfaces — no design-system-level differences observed, only language localization.
- The actual Tourbillon page rendered as a sparse minimal page in the captured screenshot, suggesting either lazy-loaded content or an interactive configurator-style UI that doesn't render fully in static screenshots; engine-spec layout is documented from general luxury-auto patterns informed by the captured spec cell tokens.
