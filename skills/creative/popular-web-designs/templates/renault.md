# Design System: Renault


> **Hermes Agent — Implementation Notes**
>
> Source analysis adapted from [`VoltAgent/awesome-design-md`](https://github.com/VoltAgent/awesome-design-md) at commit `664b3e78`; see `../references/ATTRIBUTION.md` for the MIT notice.
>
> Renault's NouvelR is proprietary. Keep a single geometric, semi-condensed family across all roles:
> - **Display, body, and UI:** `Inter Tight` at weights 400/600/700
> - **Primary stack:** `font-family: 'Inter Tight', Manrope, 'Arial Narrow', sans-serif;`
> ```html
> <link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;600;700&display=swap" rel="stylesheet">
> ```
> Clamp display line-height to 0.95 explicitly; substitute families otherwise render too loosely.
> Use `write_file` to create HTML and open the rendered artifact with `browser_navigate`.
> Verify visual accuracy with `browser_vision` after generating.

## Overview

Renault's Turkish marketing surfaces are unapologetically high-contrast: a
white canvas for browsing, a black canvas for product storytelling, and a
single Sunlight Yellow accent (`{colors.primary}` — `#ffed00`) reserved for
the most consequential actions. The brand's modernised flat diamond logo sets
the tone — geometric, confident, slightly industrial — and the system follows
suit. Square corners dominate, hairline borders are rare, and elevation is
expressed through colour blocking rather than shadow.

The typography is monolithic. Every text on the site is set in **NouvelR**,
Renault's bespoke display family, with a strong preference for weight 700 at
display sizes (with a tight `lineHeight: 0.95`) and weight 400 for body. There
is no secondary serif, no decorative italic, no script — the discipline is
the signature.

Page rhythm cycles between three surface modes: a **white catalogue mode** for
listings and configurators (`{colors.canvas}` with hairline-thin
`{colors.hairline}` dividers), a **black storytelling mode** for hero
sections, lifestyle imagery, and the lower half of campaign pages, and brief
**yellow accent moments** (`{colors.primary}` tiles, "NEW" badges, R5-coded
yellow paint shots) that punctuate the otherwise neutral palette.

**Key Characteristics:**
- Two-tone canvas system — `{colors.canvas}` (white) for browsing, `{colors.surface-dark}` (black) for storytelling — switched in full-bleed bands rather than subtle gradations.
- A single brand accent — `{colors.primary}` Sunlight Yellow — used scarcely on primary CTAs, "NEW" badges, R5 hero photography, and configurator dot indicators.
- **NouvelR everywhere**, with `{typography.display-xl}` headlines at 56px / weight 700 / `lineHeight: 0.95` so condensed multi-line headlines stack cleanly.
- Square geometry: `{rounded.xs}` (2px) on buttons, `{rounded.none}` on tiles and product cards, `{rounded.pill}` reserved exclusively for sub-nav chips and decorative badges.
- Photography-first product tiles — vehicle photos full-bleed inside otherwise neutral cards, with copy stacked beneath rather than overlaid.
- Page-level rhythm cycles white → black → yellow accent → black, with the wordmark and footer always closing on `{colors.surface-dark}`.

## Colors

### Brand & Accent
- **Sunlight Yellow** (`{colors.primary}` — `#ffed00`): the brand accent. Reserved for primary CTAs, "NEW" / "yeni" badges, configurator dot indicators, and full-bleed promotional tiles. Never decorative.
- **Sunlight Yellow Pressed** (`{colors.primary-deep}` — `#e6d200`): the active/pressed state of `{colors.primary}` buttons and tiles.
- **On-Primary** (`{colors.on-primary}` — `#000000`): label colour on top of `{colors.primary}` surfaces. Yellow always pairs with black text — never white.

### Surface
- **Canvas** (`{colors.canvas}` — `#ffffff`): the default page background and card surface.
- **Surface Soft** (`{colors.surface-soft}` — `#f7f7f7`): subtle elevation step for grouped configurator rows and inactive form fields.
- **Surface Dark** (`{colors.surface-dark}` — `#000000`): the alternate canvas, used for hero bands, footer, and full-bleed storytelling sections.
- **Surface Deep** (`{colors.surface-deep}` — `#111111`): a one-step-up elevation inside `{colors.surface-dark}` regions for inset cards and form panels.
- **Hairline** (`{colors.hairline}` — `#f2f2f2`): the soft 1px divider between rows on white surfaces.
- **Hairline Strong** (`{colors.hairline-strong}` — `#000000`): full-strength dividers on white, plus all card / button outlines.
- **Divider Dark** (`{colors.divider-dark}` — `rgba(255,255,255,0.16)`): the corresponding low-contrast divider used inside `{colors.surface-dark}` regions.

### Text
- **Ink** (`{colors.ink}` — `#000000`): primary text colour on white surfaces. The same value also drives logos, icons, and outline borders — black is structural, not decorative.
- **Body** (`{colors.body}` — `#222222`): secondary body text where pure black would feel too heavy in long paragraphs.
- **Charcoal** (`{colors.charcoal}` — `#333333`): captions, metadata, and small labels.
- **Mute** (`{colors.mute}` — `#666666`): supporting text and inactive nav labels.
- **Ash** (`{colors.ash}` — `#8a8a8a`): placeholder text, disabled labels.
- **Stone** (`{colors.stone}` — `#c4c4c4`): disabled-state foreground.
- **On-Dark** (`{colors.on-dark}` — `#ffffff`): primary text on `{colors.surface-dark}` surfaces.
- **On-Dark Mute** (`{colors.on-dark-mute}` — `rgba(255,255,255,0.72)`): secondary text in dark regions; preserves the brand's high-contrast feel without resorting to mid-grey.

### Semantic
- **Error** (`{colors.error}` — `#be6464`): muted desaturated red used for inline form errors. Notably warmer than typical pure-red error states.
- **Warning** (`{colors.warning}` — `#f0ad4e`): amber alert.
- **Success** (`{colors.success}` — `#8dc572`): muted green confirmation.
- **Info** (`{colors.info}` — `#337ab7`): a desaturated mid-blue used in informational chips.
- **Link** (`{colors.link}` — `#0000ee`): the unstyled-anchor default kept for fallback inline text links — production links inherit `{colors.ink}` and rely on underline/weight rather than colour.

## Typography

### Font Family

The entire system is set in **NouvelR**, Renault's proprietary display
family, used across navigation, headlines, body, captions, and button
labels. The family carries a slightly geometric, semi-condensed personality
with tall x-heights and squared apexes that pair naturally with the diamond
logomark.

When NouvelR cannot be licensed, suitable open-source substitutes include
**Inter Tight**, **Manrope**, or **HK Grotesk Semi Condensed** — all share
the geometric-with-warmth feel and adapt cleanly to weights 400 / 600 / 700.
Tighten `lineHeight` on display sizes to ~0.95 to match the original; do not
relax it.

### Hierarchy

| Token | Size | Weight | Line Height | Letter Spacing | Use |
|---|---|---|---|---|---|
| `{typography.display-xl}` | 56px | 700 | 0.95 | 0 | Hero headlines, campaign titles ("E-TECH ELEKTRİKLİ", "REVOLUTION"). |
| `{typography.display-lg}` | 40px | 700 | 0.95 | 0 | Secondary section titles. |
| `{typography.display-md}` | 32px | 700 | 0.95 | 0 | Page-level H1 on sub-pages and configurator panels. |
| `{typography.heading-lg}` | 24px | 700 | 0.95 | 0 | Section headers, card titles. |
| `{typography.heading-md}` | 20px | 700 | 0.95 | 0 | Sub-section headers, prominent labels. |
| `{typography.heading-sm}` | 18px | 700 | 1.0 | 0 | Tile titles, list group headers. |
| `{typography.subtitle}` | 19.2px | 600 | 1.3 | 0 | Lead paragraphs, hero subtitles. |
| `{typography.body-lg}` | 18px | 400 | 1.5 | 0 | Long-form body. |
| `{typography.body-md}` | 16px | 400 | 1.4 | 0 | Default body and form fields. |
| `{typography.body-sm}` | 14px | 400 | 1.57 | 0 | Captions, metadata. |
| `{typography.button-lg}` | 16px | 700 | 1.0 | 0 | Large CTAs in hero bands. |
| `{typography.button-md}` | 14.4px | 700 | 1.0 | 0.144px | Default button label across the system. |
| `{typography.button-sm}` | 13px | 600 | 1.2 | 0.13px | Sub-nav pills, small in-card actions. |
| `{typography.caption}` | 12px | 400 | 1.4 | 0 | Footer disclosure, regulatory text. |
| `{typography.overline}` | 10px | 700 | 1.45 | 0 | Short uppercase labels above titles. |

### Principles
- Display sizes always weight 700, always at `lineHeight: 0.95`. The tightness is what makes the brand feel confident rather than corporate.
- Body copy stays at weight 400 — never 500. The contrast between body and display is part of the system.
- Button labels carry a tiny positive letter-spacing (`0.144px` on `{typography.button-md}`) — almost imperceptible, but it adds the small bit of mechanical precision the brand wants on CTAs.
- No italics, no script, no decorative ligatures.

### Note on Font Substitutes

NouvelR is licensed; substitutes (Inter Tight / Manrope / HK Grotesk Semi
Condensed) preserve the geometric character but typically render with
slightly looser line heights at display sizes — clamp display
`lineHeight` to 0.95 explicitly to match the source.

## Layout

### Spacing System
- **Base unit**: 4px, with the working scale built on multiples of 4 and 8.
- **Tokens**: `{spacing.xxs}` 4px · `{spacing.xs}` 8px · `{spacing.sm}` 12px · `{spacing.md}` 16px · `{spacing.lg}` 20px · `{spacing.xl}` 24px · `{spacing.xxl}` 32px · `{spacing.xxxl}` 40px · `{spacing.section}` 80px.
- Section padding (full-bleed band → next band): `{spacing.section}` (80px) on desktop, collapsing to `{spacing.xxxl}` (40px) on mobile.
- Promo-tile internal padding: `{spacing.xxl}` (32px) all sides on desktop.
- Configurator row vertical padding: `{spacing.xl}` (24px) top/bottom with hairline divider between rows.

### Grid & Container
- **Max content width** ≈ 1440px. Beyond that, content remains centred and the dark/light bands extend full-bleed.
- **Promo grid** on the homepage: a 2-column tile grid on desktop, dropping to 1-up on mobile. Each tile is square-cornered (`{rounded.none}`) with the photography or solid colour as the background.
- **Vehicle range grids**: 3 to 4 cars per row at desktop, collapsing 2-up at tablet and 1-up at small mobile.
- **Configurator** uses a fixed left visualisation pane (~60% width) with a right-hand scrolling option list (~40% width) on desktop.

### Whitespace Philosophy
- Whitespace is structural, not decorative. Sections are separated by colour-blocking (white → black) rather than soft padding ramps.
- Inside cards and configurator rows, padding is generous but never airy — the brand is mass-market, so density is acceptable.
- Hairline `{colors.hairline}` dividers on white surfaces create the sense of catalogue precision; on dark surfaces, `{colors.divider-dark}` carries the same role.

## Elevation & Depth

| Level | Treatment | Use |
|---|---|---|
| 0 — flat | No shadow, no border | Default page surface, full-bleed bands. |
| 1 — outline | 1px solid `{colors.hairline-strong}` or `{colors.hairline}` | Promo tiles on light, vehicle cards, configurator panels. |
| 2 — colour-blocked elevation | Surface colour shift (e.g. `{colors.canvas}` card sitting inside a `{colors.surface-soft}` band) | Configurator detail cards, related-content rows. |
| 3 — dark inversion | Card swaps to `{colors.surface-dark}` against a `{colors.canvas}` band | "Ticari araç" hero promo tiles, lifestyle storytelling cards. |

Drop shadows are extracted from the system but rarely visible on the marketing
surfaces. When they appear, they are very subtle (~10% opacity, 2–4px blur)
and used on floating elements like the configurator's sticky summary bar.

### Decorative Depth
- The R5 hero band uses an atmospheric mesh-gradient backdrop — purple-to-pink-to-yellow glow behind the car silhouette — that acts as the only true atmospheric depth in the system. Everywhere else, depth is structural (colour-blocking + outlines), not atmospheric.
- E-TECH electric powertrain pages use a luminous magenta-to-violet gradient behind cutaway diagrams, reserved for the electric sub-brand.

## Shapes

### Border Radius Scale

| Token | Value | Use |
|---|---|---|
| `{rounded.none}` | 0px | Tiles, vehicle cards, dividers, banner bands, full-bleed images. |
| `{rounded.xs}` | 2px | Default buttons (primary yellow, secondary black, outline). |
| `{rounded.sm}` | 3px | Tab panels, small chips. |
| `{rounded.md}` | 4px | Form labels, inline tags. |
| `{rounded.pill}` | 46px | Sub-nav pills, "yeni" / "NEW" badges, decorative carousel chips. |
| `{rounded.full}` | 9999px | Configurator colour swatches, avatar dots. |

### Photography Geometry
- Vehicle photography is **always square-cornered** (`{rounded.none}`). No rounded vehicle hero shots, no soft-edged car cards.
- Aspect ratios cluster around **16:9** (hero bands), **1:1** (square promo tiles), and **4:3** (vehicle range cards). Lifestyle imagery sometimes runs **2:1 wide** for full-bleed bands.
- Avatars and small profile cues — when present — use `{rounded.full}` circles to contrast with the otherwise square geometry.

## Components

### Buttons

**`button-primary`** — yellow CTA
- Background `{colors.primary}`, label `{colors.on-primary}`, type `{typography.button-md}`, padding `14px 24px`, `rounded: {rounded.xs}`.
- The single most important action on a page (e.g. "Hemen randevu al", "Hesapla", "Konfigüratörü başlat").
- Pressed state lives in `button-primary-pressed` (background `{colors.primary-deep}`).

**`button-secondary-dark`** — solid black CTA
- Background `{colors.surface-dark}`, label `{colors.on-dark}`, type `{typography.button-md}`, `rounded: {rounded.xs}`.
- Equal-weight secondary action paired with `{component.button-primary}`, or the primary action when used on a yellow tile background.

**`button-outline-dark`** — outlined CTA on light
- Background `{colors.canvas}`, label `{colors.ink}`, 1px solid `{colors.hairline-strong}`, type `{typography.button-md}`, `rounded: {rounded.xs}`.
- Tertiary action; appears alongside primary/secondary for "Detayları gör", "Modeller", etc.

**`button-outline-light`** — outlined CTA on dark
- Background `{colors.surface-dark}`, label `{colors.on-dark}`, 1px solid `{colors.on-dark}`, type `{typography.button-md}`, `rounded: {rounded.xs}`.
- The dark-canvas counterpart to `{component.button-outline-dark}`.

**`button-pill`** — sub-nav chip
- Background `{colors.canvas}`, label `{colors.ink}`, 1px solid `{colors.hairline-strong}`, type `{typography.button-sm}`, `rounded: {rounded.pill}`, height 36px.
- The only place the system uses a pill — for top-level filter chips ("Servis & randevu", "Sahiplik dönemi geçişi", "Kampanyalar") and configurator tab switches.

**`button-icon-square`** — small icon button
- Background `{colors.canvas}`, 1px solid `{colors.hairline-strong}`, `rounded: {rounded.xs}`, 40×40px square.
- Carousel arrows, share, language switcher.

### Cards & Containers

**`promo-tile-light`** — white promo tile
- Background `{colors.canvas}`, text `{colors.ink}`, type `{typography.heading-lg}`, padding `{spacing.xxl}`, `rounded: {rounded.none}`.
- Used in the homepage 2-up grid for "Hybrid araç modelleri", "binek araç satış kampanyaları" tiles.

**`promo-tile-dark`** — black promo tile
- Background `{colors.surface-dark}`, text `{colors.on-dark}`, type `{typography.heading-lg}`, padding `{spacing.xxl}`, `rounded: {rounded.none}`.
- Lifestyle / commercial-vehicle storytelling tiles ("ticari araç satış kampanyaları").

**`promo-tile-yellow`** — accent promo tile
- Background `{colors.primary}`, text `{colors.on-primary}`, type `{typography.heading-lg}`, padding `{spacing.xxl}`, `rounded: {rounded.none}`.
- The single "PARLAK SARI" / "Sunlight Yellow" attention tile that anchors a campaign band. Reserved — usually one per page maximum.

**`vehicle-card`** — car listing card
- Background `{colors.canvas}`, full-bleed product photography on top, text below, `rounded: {rounded.none}`, no outer border.
- Includes vehicle name (`{typography.heading-md}`), variant subtitle (`{typography.body-sm}`), and a single trailing arrow icon.

**`hero-banner`** — full-bleed hero
- Background `{colors.surface-dark}` with full-bleed photo or atmospheric gradient, content stacked left, type `{typography.display-xl}` for the title.
- "SCENIC E-TECH ELEKTRİKLİ" hero on the homepage.

### Inputs & Forms

**`text-input`** — default input
- Background `{colors.canvas}`, text `{colors.ink}`, type `{typography.body-md}`, 1px bottom border `{colors.hairline-strong}`, `rounded: {rounded.none}`, padding `{spacing.sm} {spacing.md}`, height 48px.
- Inputs intentionally minimal — borderless on top and sides, single hairline at the bottom — keeping the catalogue feel.

### Configurator

**`configurator-row`** — option list row
- Background `{colors.canvas}`, separator hairline `{colors.hairline}` between rows, padding `{spacing.xl}` top/bottom, type `{typography.body-md}`.
- Right-side scrolling list on the configurator: "kasa tipi", "motor seçimi", "versiyon seçimi", "renk seçenekleri", etc.

**`configurator-swatch`** — circular colour pick
- Background `{colors.surface-soft}` (or the actual car colour), `rounded: {rounded.full}`, 56×56px.
- Used for paint colour selection. Active state adds a 1px solid `{colors.hairline-strong}` ring.

### Navigation

**`nav-bar`** — top nav (desktop)
- Background `{colors.canvas}`, type `{typography.button-md}`, height 60px, hairline `{colors.hairline}` bottom border.
- Left: diamond logomark. Centre: top-level nav ("Modeller", "Hizmetler", "Renault Yaşamı", "MyRenault"). Right: language switcher + login icon.

**`nav-bar`** (mobile)
- Same height 60px, collapses centre nav into a hamburger icon. Logo stays left, login stays right.

**`sub-nav-pill`** — pill-style sub-nav
- Pill chips set in a horizontal scroll bar between hero and content body (e.g. "Servis & randevu", "Sahiplik dönemi geçişi", "Kampanyalar"), `{component.button-pill}` styling.

### Signature Components

**`badge-new`** — "yeni" badge
- Background `{colors.primary}`, label `{colors.on-primary}`, type `{typography.button-md}`, `rounded: {rounded.full}`, padding `6px 14px`.
- Anchored on the bottom-left of new vehicle cards.

**`footer`** — global footer
- Background `{colors.surface-dark}`, text `{colors.on-dark}`, type `{typography.body-sm}`, padding `64px 24px`.
- Three-column legal/quick-links/locale grid above a single-line copyright row separated by `{colors.divider-dark}`.

## Do's and Don'ts

### Do
- Reserve `{colors.primary}` exclusively for primary CTAs, "yeni"/"NEW" badges, and at most one accent promo tile per band — `{component.promo-tile-yellow}` is intentionally rare.
- Pair `{colors.primary}` only with `{colors.on-primary}` text. Yellow + white is forbidden.
- Set everything in **NouvelR** — no secondary serif, no script, no decorative italic.
- Hold display headlines at `{typography.display-xl}` weight 700 with `lineHeight: 0.95` so they stack tightly on multi-line wraps.
- Use `{rounded.xs}` (2px) on every standard button — the near-flat corner is part of the brand.
- Switch full bands between `{colors.canvas}` and `{colors.surface-dark}` for storytelling rhythm. Avoid mid-greys as section backgrounds.
- Show vehicle photography full-bleed inside `{component.vehicle-card}` with copy stacked beneath, never overlaid.
- Use `{component.sub-nav-pill}` (`{rounded.pill}`) only for sub-nav and small filter rows — never for primary CTAs.

### Don't
- Don't introduce a secondary accent colour. Yellow is the only brand accent; semantic colours (`{colors.error}`, `{colors.success}`, `{colors.warning}`) are functional, not decorative.
- Don't round vehicle cards or promo tiles. Square-cornered photography is core to the brand expression.
- Don't soften body weights to 500 or 600 — the system relies on the 400 / 700 contrast.
- Don't apply `{colors.primary}` to body text or large surfaces beyond the single accent tile per band.
- Don't add atmospheric gradient washes outside the dedicated R5 / E-TECH hero contexts.
- Don't pair light grey text on white. Body text steps through `{colors.body}`, `{colors.charcoal}`, `{colors.mute}` — `{colors.ash}` and `{colors.stone}` are reserved for placeholders and disabled states.
- Don't add drop shadows to vehicle cards or promo tiles — the system is shadow-free at the catalogue level.

## Responsive Behavior

### Breakpoints

| Name | Width | Key Changes |
|---|---|---|
| Desktop XL | ≥ 1440px | Full max-width container, 3–4 column vehicle grid, 2-up promo tile grid. |
| Desktop | 1280–1439px | Same layout, container shrinks to viewport with `{spacing.xl}` side padding. |
| Tablet Large | 1024–1279px | Vehicle grid drops to 3-up, configurator left/right panes resize to 55/45. |
| Tablet | 768–1023px | Promo tile grid collapses to 2-up, sub-nav pills become horizontal scroll. |
| Mobile Large | 426–767px | Vehicle grid 2-up, configurator switches to stacked panes (visualisation on top, options below), nav collapses to hamburger. |
| Mobile | ≤ 425px | All grids 1-up, hero `{typography.display-xl}` clamps to ~40px, section padding `{spacing.section}` collapses to `{spacing.xxxl}`. |

### Touch Targets
- All buttons ship at minimum 44×44px on mobile; default `{component.button-primary}` is 48px tall, comfortably exceeding WCAG AAA.
- `{component.sub-nav-pill}` (36px) is bumped to 40px tall on mobile via increased vertical padding.
- `{component.button-icon-square}` (40px) sits at the WCAG AA minimum and remains tappable, but should grow to 44px when used as a primary navigation control.

### Collapsing Strategy
- Top-level nav collapses to hamburger at < 1024px; the logo and login icon stay anchored.
- 2-up promo grid collapses to 1-up at < 768px; tile padding shrinks from `{spacing.xxl}` to `{spacing.lg}`.
- Configurator switches from side-by-side to stacked at < 1024px, with the visualisation pinned to the top of the viewport on scroll.
- Display headlines clamp: `{typography.display-xl}` 56px → 40px → 32px across the breakpoint ladder.
- Sub-nav pills convert from a wrap row to a horizontal scroll-rail at < 768px.

### Image Behavior
- Vehicle photography is served at 1.5× and 2× DPR; below 768px, the system swaps to a portrait-oriented composition where art direction allows.
- Hero atmospheric gradients (R5, E-TECH) load lazily after primary content; they are not blocking.
- Lifestyle / commercial photography in `{component.promo-tile-dark}` keeps the same 16:9 framing across breakpoints, cropping inward rather than letterboxing.

## Iteration Guide

1. Focus on ONE component at a time. Most components share `{rounded.xs}`, `{colors.canvas}` / `{colors.surface-dark}`, and NouvelR — only the role-specific tokens (`{colors.primary}`, `{component.promo-tile-yellow}`) shift between variants.
2. Reference component names and tokens directly (`{colors.primary}`, `{component.button-primary-pressed}`, `{rounded.pill}`) — do not paraphrase.
3. Run `npx @google/design.md lint DESIGN.md` after edits; the orphaned-tokens warning will catch unused entries before they ship.
4. Add new variants as separate entries (`-pressed`, `-disabled`, `-outline`) — do not bury them in prose.
5. Default body type to `{typography.body-md}`; reach for `{typography.subtitle}` only on hero subtitles and lead paragraphs.
6. Keep `{colors.primary}` scarce — if more than one yellow element appears per viewport, ask whether one of them should drop to `{colors.surface-dark}` or `{colors.canvas}` instead.

## Known Gaps

- Active/pressed visual states are not consistently observable in static surfaces; `button-primary-pressed` documents the extracted darkened-yellow value, but no other component has a pressed variant promoted to the YAML.
- Drop-shadow values exist in the extracted tokens but are rarely surfaced visually; only the configurator's sticky summary bar uses them on the captured pages.
- The MyRenault application surfaces (logged-in product) are out of scope for this extraction — only the public marketing canvas is documented.
- Form-field focus styling is not extracted; the system likely relies on a thicker bottom border at `{colors.ink}`, but this is not visually confirmed on the captured pages.
