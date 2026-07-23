# Design System: Vodafone


> **Hermes Agent — Implementation Notes**
>
> Source analysis adapted from [`VoltAgent/awesome-design-md`](https://github.com/VoltAgent/awesome-design-md) at commit `664b3e78`; see `../references/ATTRIBUTION.md` for the MIT notice.
>
> The original site uses proprietary or legacy fonts. For self-contained HTML output, use these CDN substitutes:
> - **Primary:** `Roboto` | **Mono:** `Roboto Mono`
> - **Font stack (CSS):** `font-family: Roboto, system-ui, -apple-system, 'Segoe UI', sans-serif;`
> - **Mono stack (CSS):** `font-family: 'Roboto Mono', ui-monospace, SFMono-Regular, Menlo, monospace;`
> ```html
> <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@400;500;700&display=swap" rel="stylesheet">
> ```
> Use `write_file` to create HTML and open the rendered artifact with `browser_navigate`.
> Verify visual accuracy with `browser_vision` after generating.

## Overview

Vodafone is a global telecom super-brand and its web surface delivers exactly that posture: heroic editorial photography, occasionally cropping a portrait so tight only an eye line and a phone hand are visible, with a single colossal uppercase headline floating on top in the brand's proprietary heavy display weight. The page reads like a campaign poster more than a corporate site, then breaks into a calmer content rhythm of light-canvas story cards and a single red marker (the iconic speechmark logo) drawing the eye to the brand's centre of gravity. There is no second accent colour competing — the entire decorative palette is `{colors.primary}` Vodafone red, near-black `{colors.ink}`, and the surrounding white and grayscale neutrals.

Type is the second decisive voice. Vodafone's custom display sans (extracted as `Vodafone`) carries every headline at impossibly heavy weight 800 in upper case for hero scale (`{typography.display-hero}` 144 px, `{typography.display-xxl}` 126 px) and at lighter weight 300 for the sub-displays that follow. Body text stays in the same family at weight 400 with neutral tracking. The contrast between display weight 800 and display weight 300 IS the brand's typographic story: a shout, then a calm sentence.

Every interactive CTA renders as a generously rounded pill (`{rounded.pill-lg}` 60 px) — Vodafone has never used a square button on its marketing surface in years, and the brand's pill scale ladder runs from 32 px (badge pills) through 60 px (CTA pills) up to 9999 px (icon circular containers). Cards stay gentler at `{rounded.card}` 6 px.

**Key Characteristics:**
- A single primary CTA color `{colors.primary}` (`#e60000`) — Vodafone Red. Pill-filled for primary, pill-outline for secondary. No third button variant.
- Massive uppercase display weight 800 (`{typography.display-hero}` and siblings) is the brand's signature. The 300-weight headline siblings (`{typography.display-lg}` / `{typography.display-md}`) handle calmer secondary moments.
- The `speechmark-logo-orb` — a red square hosting Vodafone's quotation-mark icon — is the only piece of decorative chrome that's not a CTA. It anchors the brand's centre of every page.
- Pill geometry on every interactive shape — `{rounded.pill-lg}` 60 px for buttons, `{rounded.pill-md}` 32 px for inline badges. Card chrome stays at `{rounded.card}` 6 px.
- A two-band page rhythm — `{colors.ink}` dark hero / `{colors.canvas}` light content. No mid-band greys; the brand uses surface contrast, not soft neutrals, for elevation.
- Editorial photography (real portraits, real cities, real cabling) is the only consistent decorative system — no illustration, no atmospheric gradients.

## Colors

### Brand & Accent
- **Vodafone Red** (`{colors.primary}` — `#e60000`): The single brand accent. Every primary CTA pill, every speechmark logo, every conversion target. The most iconic red in telecom — never desaturated, never used at scale for body fills (reserved for high-attention surfaces).

### Surface
- **Canvas** (`{colors.canvas}` — `#ffffff`): The default light content background.
- **Canvas Soft** (`{colors.canvas-soft}` — `#f2f2f2`): A near-white tint used as the badge-chip background.
- **Ink** (`{colors.ink}` — `#25282b`): The brand's near-black surface — used as the dark hero band, the nav background, and the footer fill. Doubles as the primary text color on light surfaces.

### Text
- **Ink** (`{colors.ink}` — `#25282b`): Every heading and body paragraph on light surfaces.
- **Body** (`{colors.body}` — `#7e7e7e`): Secondary body text on light surfaces — captions, metadata, supporting copy.
- **Mute** (`{colors.mute}` — `#bebebe`): The lowest-priority text color — placeholder text, low-key footer links.
- **On Dark** (`{colors.on-dark}` — `#ffffff`): All text on `{colors.ink}` surfaces (hero, footer, nav).

### Semantic
The brand does not maintain a separate semantic palette. The primary red doubles as validation / destructive signal when needed; success / warning use are reserved for in-product contexts and are not part of the documented marketing system.

## Typography

### Font Family
A single custom face carries the entire system: **Vodafone**, the brand's proprietary display sans. The face spans weights 300 (light), 400 (regular), 600, 700, and 800 — every role in the system pulls from this one family. There is no mono companion; technical labels (rare on the marketing surface) borrow the same face at smaller sizes.

The icomoon icon-font is loaded for proprietary glyphs but does not render as a typographic role.

### Hierarchy

| Token | Size | Weight | Line Height | Letter Spacing | Use |
|---|---|---|---|---|---|
| `{typography.display-hero}` | 144px | 800 | 114px | -1px | The hero stencil (`"STAY CONNECTED"`) — uppercase, ultra-tight tracking, brand's signature size. |
| `{typography.display-xxl}` | 126px | 800 | 113px | -1px | Slightly smaller hero variant. |
| `{typography.display-xl}` | 90px | 800 | 84px | 0 | Mid-hero scale. |
| `{typography.display-lg}` | 48px | 300 | 52px | 0 | Section-headline sub-display in the lighter weight. |
| `{typography.display-md}` | 40px | 300 | 44px | 0 | Sub-section displays. |
| `{typography.display-sm}` | 32px | 700 | 40px | 0 | Card headings, story-card titles. |
| `{typography.display-xs}` | 24px | 700 | 24px | 0 | Inline display micro-headings. |
| `{typography.eyebrow-uppercase}` | 16px | 800 | 24px | 0 | Uppercase eyebrow tags above section headlines. |
| `{typography.body-lg}` | 22px | 400 | 24px | 0 | Lead body paragraphs. |
| `{typography.body-md}` | 18px | 400 | 28px | 0 | Default paragraph body. |
| `{typography.body-md-strong}` | 18px | 600 | 28px | 0 | Bolded inline body. |
| `{typography.body-sm}` | 16px | 400 | 20px | 0 | Secondary body. |
| `{typography.body-sm-strong}` | 16px | 700 | 22px | 0 | Bolded short body. |
| `{typography.caption}` | 14px | 400 | 16px | 0 | Captions, fine print. |
| `{typography.caption-strong}` | 14px | 700 | 21px | 0 | Bold captions, badge labels. |
| `{typography.caption-uppercase}` | 12px | 600 | 16px | 0.57px | Uppercase metadata, footer eyebrows. |
| `{typography.button-md}` | 18px | 400 | 28px | 0 | Default button label. |

### Principles
- **Weight 800 + uppercase = hero voice.** This is the entire reason the brand reads as a billboard rather than a tech site.
- **Weight 300 = the calmer secondary voice.** Used at 40 – 48 px for sub-displays; never below 24 px to keep legibility.
- **Single family throughout.** The brand never mixes a serif or a mono into the typographic system. Consistency is the calm.
- **Tracking stays tight at display sizes.** `-1px` at 144 px is the brand's calibration; reverting to neutral tracking softens the stencil look.

### Note on Font Substitutes
The Vodafone display sans is proprietary. Open-source substitutes:
- **Display sans** — *Inter* weight 800 at hero scale with `letter-spacing: -1px` is the closest free match. *Geist* weight 700–800 is the second-best.
- **Lighter display weight (300)** — *Inter* weight 300 holds its line-height well at 48 px display sizes.

## Layout

### Spacing System
- **Base unit**: 4 px (mostly multiples of 4; a few 5/7 px appear inside icon-padding compensation).
- **Tokens**: `{spacing.xxs}` 2 px · `{spacing.xs}` 4 px · `{spacing.sm}` 8 px · `{spacing.md}` 12 px · `{spacing.lg}` 16 px · `{spacing.xl}` 20 px · `{spacing.2xl}` 24 px · `{spacing.3xl}` 32 px.
- **Section padding**: hero bands and content bands use `{spacing.3xl}` 32 px gutters; vertical spacing inside hero is fluid (fill-the-band).
- **Card interior padding**: story cards use `{spacing.lg}` 16 px around image + headline.
- **Inline gap**: button rows and chip rows use `{spacing.md}` 12 px between siblings.

### Grid & Container
- Marketing content uses a wide container (effectively edge-to-edge with `{spacing.3xl}` gutters on desktop, shrinking on mobile).
- Story-card grids: 2-up at desktop, 1-up at mobile.
- Hero photography fills the viewport; the headline overlays at the top-left.

### Whitespace Philosophy
The hero's massive display headline owns the whole top of the page; whitespace below is generous to let the second band breathe. Inside content cards, headline and copy hug close (`{spacing.sm}` 8 px gap), then a wider gap (`{spacing.3xl}`) before the next card. The footer band is dark and dense.

### Responsive Strategy

#### Breakpoints

| Name | Width | Key Changes |
|---|---|---|
| Mobile | < 600px | Hero display scales down to ~64 px; story-card grid drops to 1-up; nav collapses to hamburger. |
| Tablet | 600–1023px | Story-card grid 2-up; display headlines drop to 90 – 110 px. |
| Desktop | 1024–1399px | Full display headline at 126 – 144 px; 2-up story grid. |
| Ultra-wide | ≥ 1400px | Container caps at ~1400 px; bands stay edge-to-edge in colour. |

#### Touch Targets
The `button-primary` pill renders at ~52 px tall (12 px vertical padding + 28 px line-height). All buttons comfortably meet WCAG AAA at every breakpoint.

#### Collapsing Strategy
- **Nav**: full link row at desktop. Collapses to a hamburger menu at mobile; the menu opens as a dark overlay with the same link list stacked.
- **Hero**: the massive display headline scales fluidly. At mobile, the photography crop tightens to the figure's face only.
- **Story-card grid**: 2-up → 1-up at the breakpoint above.
- **Speechmark logo orb**: stays at consistent size relative to surrounding content; never shrinks below ~48 px.

#### Image Behavior
- **Hero photography**: full-bleed 16:9 or 4:3 portraits at desktop; tighter crops at mobile.
- **Story-card thumbnails**: 16:9 landscape inside `{rounded.card}` 6 px chrome.
- **Speechmark orb**: always rendered as the red SVG quote-mark icon, never substituted.
- **Logo bar (if present on partner pages)**: monochrome SVGs at consistent height.

## Elevation & Depth

| Level | Treatment | Use |
|---|---|---|
| Level 0 — Flat | No shadow, no border. | Default for most cards and panels — surface contrast does the elevation work. |
| Level 1 — Hairline | 1 px solid `{colors.ink}` border. | Form inputs, the `divider-on-dark` between footer columns. |
| Level 2 — Border on Dark | 1 px solid `{colors.on-dark}` border on `{colors.ink}` surfaces. | Outline buttons sitting on the dark hero band. |

The brand does not use soft drop shadows; depth comes from polarity-flip between `{colors.ink}` and `{colors.canvas}` bands.

### Decorative Depth
- **Editorial photography**: the hero photo (real-person portrait or environment shot) is the brand's only true atmospheric effect.
- **Speechmark logo orb as visual anchor**: the red orb hosting the quote-mark icon acts as a single point of focal-depth in the centre of the otherwise-flat content rhythm.

## Shapes

### Border Radius Scale

| Token | Value | Use |
|---|---|---|
| `{rounded.none}` | 0px | Full-bleed hero bands, footer, banner strips. |
| `{rounded.xs}` | 1px | Tightest inline indicator (rarely used). |
| `{rounded.sm}` | 6px | The brand's canonical content radius — applied to images and inputs. |
| `{rounded.card}` | 6px | Card chrome and image frames (alias for `sm`). |
| `{rounded.pill-md}` | 32px | Badge / chip pills. |
| `{rounded.pill-lg}` | 60px | The brand's signature CTA pill — every primary and secondary button. |
| `{rounded.full}` | 9999px | Circular icon containers (e.g., video play/pause). |

### Photography Geometry
- Hero portraits: edge-to-edge 16:9 or 4:3 with no internal frame.
- Story-card thumbnails: 16:9 landscape inside `{rounded.card}` chrome.
- Speechmark logo orb: square with `{rounded.sm}` corners (visually a tilted-square mark; the SVG mark itself fills the orb).

## Components

### Buttons

**`button-primary`** — the red pill CTA, brand's primary conversion target.
- Background `{colors.primary}`, text `{colors.on-primary}`, 1 px solid `{colors.primary}` border, label set in `{typography.button-md}`, padding `{spacing.md} {spacing.2xl}`, shape `{rounded.pill-lg}` 60 px.

**`button-outline-red`** — the secondary pill, red-text-on-white with red border.
- Background `{colors.canvas}`, text `{colors.primary}`, 1 px solid `{colors.primary}` border, same label / padding / shape as `button-primary`.

**`button-outline-dark`** — the tertiary pill, ink-text-on-white with ink border.
- Background `{colors.canvas}`, text `{colors.ink}`, 1 px solid `{colors.ink}` border, same label / padding / shape.

**`button-icon-circular`** — the round white icon button (video play / pause / chevron).
- Background `{colors.canvas}`, ink icon, 1 px solid `{colors.canvas}` outline (effectively borderless), shape `{rounded.full}`.

### Cards & Containers

**`card-content`** — the default story-card chrome.
- Background `{colors.canvas}`, text `{colors.ink}`, padding `{spacing.lg}`, shape `{rounded.card}` 6 px. Hosts a 16:9 thumbnail at the top + headline + caption.

**`card-hero`** — the slightly larger card variant used as the lead story.
- Same chrome as `card-content` but headline scales to `{typography.display-sm}`.

### Inputs & Forms

**`text-input`** — the canonical text input.
- Background `{colors.canvas}`, text `{colors.ink}`, 1 px solid `{colors.ink}` border, body in `{typography.body-sm}`, padding `{spacing.md} {spacing.lg}`, shape `{rounded.sm}` 6 px.

### Navigation

**`nav-bar`** — the dark top nav, fixed to the page top.
- Background `{colors.ink}`, text `{colors.on-dark}`, padding `{spacing.lg} {spacing.3xl}`. Layout: logo left, link row right with login / search affordances.

**`nav-link`** — the link items inside `nav-bar`.
- Text `{colors.on-dark}`, set in `{typography.body-sm}`.

**`footer`** — the dark footer band.
- Background `{colors.ink}`, text `{colors.on-dark}`, padding `{spacing.3xl} {spacing.3xl}`. Body in `{typography.body-sm}`; column eyebrows in `{typography.caption-uppercase}`.

### Signature Components

**`hero-band-dark`** — the dark navy/ink hero band hosting the massive display headline.
- Background `{colors.ink}` with full-bleed editorial photography overlay at reduced opacity; text `{colors.on-dark}`; padding `{spacing.3xl} {spacing.3xl}`. Headline in `{typography.display-hero}` (uppercase, weight 800).

**`hero-band-red`** — the rare full-bleed red hero used on signature campaigns.
- Background `{colors.primary}`, text `{colors.on-primary}`, padding `{spacing.3xl} {spacing.3xl}`. Headline in `{typography.display-xl}`.

**`content-band-light`** — the white content band that follows every hero.
- Background `{colors.canvas}`, text `{colors.ink}`, padding `{spacing.3xl} {spacing.3xl}`. Section headline in `{typography.display-md}` or `{typography.display-lg}` (weight 300).

**`speechmark-logo-orb`** — the red square hosting Vodafone's quotation-mark icon. The brand's visual anchor.
- Background `{colors.primary}`, white icon glyph, shape `{rounded.sm}`. Acts as a focal element between content bands, often near the centre of long pages.

**`badge-chip`** — the inline metadata pill used for category tags inside story cards.
- Background `{colors.canvas-soft}`, text `{colors.ink}`, label in `{typography.caption-strong}`, padding `{spacing.xs} {spacing.md}`, shape `{rounded.pill-md}` 32 px.

**`divider-on-dark`** — the 1 px hairline used between sections / columns on dark surfaces.
- 1 px solid `{colors.on-dark}` (often at 25 % opacity at the component level).

### Examples (illustrative)

> Auto-derived kit-mirror demonstration surfaces (`scripts/derive-examples-block.mjs`). Each `ex-*` entry references brand-native primitives so downstream consumers (`/preview-design`, `/generate-kit`) re-skin the same 10 surfaces consistently. `TO_FILL` markers indicate missing primitives — resolve in the LLM judgment pass.

**`ex-pricing-tier`** — Default Pricing tier card. Re-uses feature-card chrome with brand canvas-soft surface.
- Properties: `backgroundColor`, `textColor`, `borderColor`, `rounded`, `padding`

**`ex-pricing-tier-featured`** — Featured/highlighted tier — polarity-flipped surface (dark fill + light text in light mode, light fill + dark text in dark mode).
- Properties: `backgroundColor`, `textColor`, `rounded`, `padding`

**`ex-product-selector`** — What's Included summary card — re-purposed for SaaS / B2B verticals (NOT a literal product gallery).
- Properties: `backgroundColor`, `rounded`, `padding`

**`ex-cart-drawer`** — Subscription summary — re-purposed for SaaS / B2B (line items per add-on, not literal cart).
- Properties: `backgroundColor`, `rounded`, `padding`, `item-divider`

**`ex-app-shell-row`** — Sidebar nav row inside the App Shell example. Active state uses brand primary as the indicator.
- Properties: `backgroundColor`, `activeIndicator`, `rounded`, `padding`

**`ex-data-table-cell`** — Default data-table th + td chrome. Header uses mono-caps eyebrow typography; body uses body-sm.
- Properties: `headerBackground`, `headerTypography`, `bodyTypography`, `cellPadding`, `rowBorder`

**`ex-auth-form-card`** — Sign-in / sign-up card. Re-uses feature-card chrome with text-input primitives inside.
- Properties: `backgroundColor`, `rounded`, `padding`

**`ex-modal-card`** — Modal dialog surface — same chrome as feature-card with elevated shadow.
- Properties: `backgroundColor`, `rounded`, `padding`

**`ex-empty-state-card`** — Empty-state illustration frame.
- Properties: `backgroundColor`, `rounded`, `padding`, `captionTypography`

**`ex-toast`** — Toast notification surface — feature-card shape + medium shadow.
- Properties: `backgroundColor`, `rounded`, `padding`, `typography`


## Do's and Don'ts

### Do
- Reserve `{colors.primary}` Vodafone Red for primary CTAs and the `speechmark-logo-orb`. Every conversion target uses the red pill.
- Set hero headlines in `{typography.display-hero}` weight 800 UPPERCASE with tight `-1px` tracking. That stencil look IS the brand voice.
- Use `{rounded.pill-lg}` 60 px on every interactive pill. The brand never uses square corners on CTAs.
- Cycle page surfaces in `{colors.ink}` dark hero → `{colors.canvas}` light content → `{colors.ink}` footer. Surface contrast is the depth cue.
- Pair editorial portrait photography with the massive display headline overlay — that combination IS the brand's signature.
- Render the speechmark logo orb at consistent size relative to surrounding content — it's the brand's centre of gravity on every page.

### Don't
- Don't introduce a second accent colour. The brand operates with red + ink + grayscale only.
- Don't render headlines in sentence case at hero scale. Hero display IS uppercase weight 800.
- Don't render the primary CTA as a square rectangle. The 60 px pill is non-negotiable.
- Don't drop a soft drop-shadow on cards. The brand relies on surface-colour contrast, not shadow.
- Don't substitute the speechmark logo orb with a wordmark or a different shape. The orb is the iconic mark.
- Don't pair the weight 800 display face with letter-spacing 0 at 144 px — the `-1px` tracking is part of the brand's calibration.
