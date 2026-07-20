# Design System: Slack


> **Hermes Agent — Implementation Notes**
>
> Source analysis adapted from [`VoltAgent/awesome-design-md`](https://github.com/VoltAgent/awesome-design-md) at commit `664b3e78`; see `../references/ATTRIBUTION.md` for the MIT notice.
>
> The original site uses proprietary or legacy fonts. For self-contained HTML output, use these CDN substitutes:
> - **Primary:** `Lato` | **Mono:** `JetBrains Mono`
> - **Font stack (CSS):** `font-family: Lato, system-ui, -apple-system, 'Segoe UI', sans-serif;`
> - **Mono stack (CSS):** `font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;`
> ```html
> <link href="https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
> ```
> Use `write_file` to create HTML and open the rendered artifact with `browser_navigate`.
> Verify visual accuracy with `browser_vision` after generating.

## Overview

Slack's design language centers on a deep aubergine primary (`{colors.primary}`) — the brand's most enduring visual asset — applied as the dominant button color, the footer band, the featured pricing tier, and the brand wordmark. Around that aubergine the system stages an unusually delicate ecosystem: cream-lavender hero canvases with soft pastel-mesh gradients (peachy oranges, lavenders, dusty greens) that pulse behind floating product UI mockups, with the actual interface chrome rendered in fine detail at 3:2 aspect.

Typography splits between two proprietary humanist sans families. The display tier runs at 700 weight at sizes 32–64px with negative letter-spacing for tight optical density on hero headlines. The UI tier uses the second family at 400–700 with slightly relaxed leading (1.55) — the brand's body copy reads quietly without competing with the aubergine moments.

Buttons are pill-shaped at 90px radius with an unusual amount of horizontal padding (28–30px), giving them a distinctly comfortable, almost over-padded feel. The primary aubergine pill is the only filled button in most contexts; secondary actions use a soft lavender pill (`{colors.canvas-lavender}`) which reads as a gentler echo of the primary surface. Inline links shift to a saturated blue (`{colors.link-blue}`) — the brand's only chromatic departure from the aubergine-and-cream world.

**Key Characteristics:**
- Single aubergine primary (`{colors.primary}`) reused across CTAs, the featured pricing tier, the footer band, and the wordmark — the brand's chromatic monotheism.
- Cream-lavender hero canvas (`{colors.canvas-cream}` / `{colors.canvas-lavender}`) with diffused pastel-mesh atmospheric gradients and floating UI mockups composited above.
- Pill buttons at `{rounded.pill}` (90px radius) with generous 28–30px horizontal padding — over-padded by SaaS-default standards, deliberately so.
- Tight negative letter-spacing on display sizes (-0.768px on 64px hero) for editorial-density headlines.
- Blue inline links (`{colors.link-blue}`) — the only non-aubergine chromatic accent in body type.
- Pastel-mesh gradient atmospherics: every hero band has a subtle peach-lavender-dusty-green wash behind it; product UI sits on top, never inside, the gradient.
- Statistics cards rendered in massive aubergine display type (90% / 43 / 87%) on white — quantitative emphasis through scale alone.

## Colors

> **Source pages:** home (`/`), `/features/channels`, `/pricing`, `/contact-sales`.

### Brand & Accent
- **Aubergine** (`{colors.primary}` — `#4a154b`): The brand's primary surface and CTA color. Deep, warm purple with a hint of ruby — used on filled buttons, the featured pricing tier, the footer band, and the brand wordmark.
- **Aubergine Deep** (`{colors.primary-deep}` — `#481a54`): A near-identical sibling of `{colors.primary}` extracted from a different surface; treat as functionally equivalent.
- **Aubergine Press** (`{colors.primary-press}` — `#611f69`): Pressed-state lift of the primary, slightly lighter and warmer.
- **Aubergine Tint** (`{colors.primary-tint}` — `#592466`): Border accent on aubergine-on-aubergine surfaces.
- **Link Blue** (`{colors.link-blue}` — `#1264a3`): Inline link color — saturated, slightly warm blue. The only chromatic alternative to aubergine in body type.
- **Link Hover** (`{colors.link-hover}` — `#3860be`): A more saturated blue used on link hover state.

### Surface
- **Canvas White** (`{colors.canvas}` — `#ffffff`): Default content surface.
- **Canvas Cream** (`{colors.canvas-cream}` — `#f4ede4`): Warm off-white used on hero gradients and feature bands. Adds editorial warmth.
- **Canvas Lavender** (`{colors.canvas-lavender}` — `#f9f0ff`): Pale lavender tint used as the secondary-button surface and as a soft section band.
- **Surface Aubergine** (`{colors.surface-aubergine}` — `#4a154b`): The primary aubergine reused as a surface — featured pricing tier, footer, dark feature bands.
- **Hairline** (`{colors.hairline}` — `#e6e6e6`): 1px borders on cards and table dividers.

### Text
- **Ink** (`{colors.ink}` — `#1d1d1d`): Primary body text on light surfaces. Just shy of pure black.
- **Ink Mute** (`{colors.ink-mute}` — `#696969`): Secondary text, captions, helper copy.
- **On Primary** (`{colors.on-primary}` — `#ffffff`): Text on aubergine surfaces and filled CTAs.
- **On Aubergine Mute** (`{colors.on-aubergine-mute}` — `#d9bdde`): Secondary text on aubergine surfaces — a desaturated mauve that reads as muted-light.

### Semantic
- **Error** (`{colors.semantic-error}` — `#cc4117`): Form error and destructive-action color.
- **Success** (`{colors.semantic-success}` — `#007a5a`): Inline success indicators.

## Typography

### Font Family

The display tier is **Salesforce Avant Garde** — a proprietary humanist sans with broad apertures and a slightly geometric character. When unavailable, fall back to the system font stack (`system-ui, -apple-system, BlinkMacSystemFont`).

The UI tier is **Salesforce Sans** — a separate proprietary face used for body, captions, and button labels. Same fallback chain.

Both faces are proprietary and not freely available. Substitute with **Inter** (open-source via Google Fonts) at matching weights for both display and body — Inter is the closest open analogue across both tiers.

### Hierarchy

| Token | Size | Weight | Line Height | Letter Spacing | Use |
|---|---|---|---|---|---|
| `{typography.display-xxl}` | 64px | 700 | 1.12 | -0.768px | Marketing hero headline |
| `{typography.display-xl}` | 58px | 600 | 1.25 | -0.464px | Section openers |
| `{typography.display-lg}` | 50px | 700 | 1.12 | -0.6px | Statistics callouts |
| `{typography.display-md}` | 32px | 700 | 1.25 | -0.256px | Card / feature titles |
| `{typography.heading-lg}` | 24px | 700 | 1.33 | -0.096px | Pricing tier names |
| `{typography.heading-md}` | 22px | 600 | 1.4 | 0 | Sub-section heading |
| `{typography.heading-sm}` | 18px | 600 | 1.56 | -0.0216px | Compact card title |
| `{typography.body-lg}` | 18px | 400 | 1.55 | -0.0216px | Marketing body lead |
| `{typography.body-md}` | 16px | 400 | 1.55 | 0 | Default UI body |
| `{typography.body-strong}` | 16px | 700 | 1.5 | 0.16px | Emphasized body |
| `{typography.button-lg}` | 18px | 700 | 1.0 | 0 | Hero pill button label |
| `{typography.button-md}` | 16px | 700 | 1.38 | 0.2px | Standard pill button label |
| `{typography.button-cap}` | 14.4px | 700 | 1.0 | 0.144px | Compact pill label |
| `{typography.caption}` | 14px | 400 | 1.43 | 0.1px | Helper, footnote |
| `{typography.micro-cap}` | 12px | 700 | 1.0 | 0.96px | All-caps eyebrow |

### Principles
- **Tight tracking on display.** Negative letter-spacing across 32–64px sizes; the proprietary face is wide by default, the negative tracking pulls it into editorial density.
- **Body at 1.55 leading.** Slightly relaxed for marketing readability without crossing into airy / 1.7+ territory.
- **Caps for eyebrows.** All eyebrows render uppercase with positive 0.96–0.144px tracking depending on size.

### Note on Font Substitutes
Use **Inter** (open-source Google Fonts) for both display and UI tiers — Inter at 700 weight with `-0.768px` letter-spacing closely approximates the brand's display behavior. For maximum brand fidelity, **Lato** is a softer humanist alternative that pairs well at body sizes. Avoid System UI fonts on the body — the brand's subtle warmth disappears at default weights.

## Layout

### Spacing System
- **Base unit**: 8px (with 4 / 12 / 16 / 20 / 24 / 28 sub-tokens for fine vertical rhythm).
- **Tokens**: `{spacing.xs}` 4px · `{spacing.sm}` 8px · `{spacing.md}` 12px · `{spacing.lg}` 16px · `{spacing.xl}` 20px · `{spacing.xxl}` 24px · `{spacing.huge}` 28px.
- **Section padding**: 64–96px on marketing surfaces; tightens to 48px on transactional pages.
- **Card internal padding**: 32px on pricing cards; 48px on aubergine band cards.

### Grid & Container
- Marketing pages center in a ~1240px container with edge-bleeding pastel-mesh gradients escaping the container.
- Pricing collapses 4-up → 2-up → 1-up at 992 / 768 breakpoints.
- Statistics row: 3-column grid with massive 50px aubergine display numerals.

### Whitespace Philosophy
The pastel-mesh gradients fill most of the negative space on marketing pages — sections feel expansive without being literally empty. On transactional pages the gradients drop, and whitespace reverts to traditional 48px-section breathing room.

## Elevation & Depth

| Level | Treatment | Use |
|---|---|---|
| 0 | Flat | Default surface |
| 1 | `box-shadow: rgba(0,0,0,0.1) 0 5px 20px 0` | Floating buttons on hero |
| 2 | `box-shadow: rgba(0,0,0,0.1) 0 0 32px 0` | Product UI mockup composites |
| 3 | `box-shadow: rgba(0,0,0,0.2) 0 1px 10px 0` | Toast / notification chrome |
| 4 | `box-shadow: rgb(97,31,105) 0 0 0 1px inset` | Aubergine inset border (button focus, special chrome) |

### Decorative Depth
The brand's depth language is the **pastel-mesh gradient** — peach, lavender, dusty green stops blurred together at large radii to create soft atmospheric backdrops behind product UI screenshots. The gradient is the brand's flavor of "depth without shadows": the eye perceives the product mockup as floating above a luminous backdrop without any literal lift.

## Shapes

### Border Radius Scale

| Token | Value | Use |
|---|---|---|
| `{rounded.xs}` | 2px | Hairline tags, status pills (rare) |
| `{rounded.sm}` | 4px | Form inputs |
| `{rounded.md}` | 8px | Compact card chrome, video frames |
| `{rounded.lg}` | 12px | Mid-size cards, secondary surface |
| `{rounded.xl}` | 16px | Pricing cards, feature cards |
| `{rounded.xxl}` | 48px | Stat badge backdrops |
| `{rounded.pill}` | 90px | All buttons |

### Photography Geometry
The brand uses **product UI screenshots** more than photography. UI mockups sit on top of pastel-mesh gradients at roughly 4:3 aspect, with no shadow but with the gradient providing the "lift" the eye expects. Real photography appears in customer-logo strips and the occasional case-study card, treated as full-bleed inside `{rounded.xl}` containers.

## Components

### Buttons

**`button-primary-pill`** — the dominant CTA system-wide.
- Background `{colors.primary}`, text `{colors.on-primary}`, type `{typography.button-md}`, padding `14px 28px`, rounded `{rounded.pill}` 90px.
- Pressed state `button-primary-pill-pressed` shifts background to `{colors.primary-press}`.

**`button-secondary-pill`** — the soft lavender alternative.
- Background `{colors.canvas-lavender}`, text `{colors.ink}`, padding `10px 30px`, same pill geometry. Used as the second action beside the primary aubergine pill.

**`button-outline-aubergine`** — outline variant on white surfaces.
- Background `{colors.canvas}`, text `{colors.primary}`, 2px solid `{colors.primary}` border, same pill shape.

**`button-outline-on-aubergine`** — outline on aubergine canvas.
- Background `{colors.surface-aubergine}` (transparent over the surface), text `{colors.on-primary}`, 2px solid `{colors.on-primary}` border, same pill shape.

### Cards & Containers

**`card-pricing`** — standard pricing tier card.
- Background `{colors.canvas}`, padding `{spacing.xxl}+` (32px), rounded `{rounded.xl}` 16px, 1px `{colors.hairline}` border. Title in `{typography.heading-lg}`, price in `{typography.display-md}`, body in `{typography.body-md}`, CTA pinned to bottom as `button-primary-pill`.

**`card-pricing-featured`** — the inverted aubergine featured tier.
- Background `{colors.surface-aubergine}`, text `{colors.on-primary}`, otherwise identical to `card-pricing`. The aubergine fill is the brand's signature featured-tier choice.

**`card-feature-cream`** — feature explanation card on the cream track.
- Background `{colors.canvas-cream}`, text `{colors.ink}`, rounded `{rounded.xl}`, padding 32px.

**`card-aubergine-band`** — large horizontal band card with aubergine fill, often containing the closing CTA of a marketing page.
- Background `{colors.surface-aubergine}`, text `{colors.on-primary}`, padding 48px, rounded `{rounded.xl}` 16px.

**`card-stat`** — statistics callout card.
- Background `{colors.canvas}`, text `{colors.primary}` rendered in `{typography.display-lg}` (50px aubergine numeral). Holds a single percentage/number with a small caption underneath.

### Inputs & Forms

**`text-input`** — standard form field.
- Background `{colors.canvas}`, text `{colors.ink}`, type `{typography.body-md}`, padding `10px 12px`, rounded `{rounded.sm}` 4px, 1px `{colors.hairline}` border.

### Navigation

**`nav-bar-light`** — top nav across all marketing pages.
- Background `{colors.canvas}`, text `{colors.ink}`, padding `{spacing.lg} {spacing.xxl}`. Logo wordmark on the left, nav items center, two pill buttons on the right (`button-secondary-pill` for "Sign In", `button-primary-pill` for "Try For Free").

### Pills, Tags, and Chips

**`pill-cap-shade`** — small all-caps pill used as eyebrow above pricing-tier titles.
- Background `{colors.canvas-cream}`, text `{colors.ink}`, type `{typography.micro-cap}`, padding `4px 12px`, rounded `{rounded.pill}`.

### Signature Components

**Pastel-Mesh Gradient Backdrop** — peach (`#fff0e6`-ish) + lavender (`#e9d8ff`) + dusty green stops blurred together behind hero bands. Implemented as a CSS radial-gradient stack, not a single image. Provides the brand's depth/luminosity without literal shadows.

**Floating Product UI Mockup** — product screenshots framed in `{rounded.lg}` (12px) containers, positioned above the pastel-mesh gradient with no border or shadow. The gradient does the lifting.

**Aubergine Footer Band** — every marketing page closes with a full-bleed `card-aubergine-band` containing a closing CTA in white type. The band height is generous (~480–600px on desktop) and reads as the page's signature.

**`link-on-light`** — inline links in body copy on light surfaces.
- Text `{colors.link-blue}` rendered in `{typography.body-md}`. No underline by default; underline appears on hover via the link-hover behavior.

**`link-on-aubergine`** — links inside aubergine surfaces.
- Text `{colors.on-primary}` with persistent underline.

**`footer-aubergine`** — site-wide footer.
- Background `{colors.surface-aubergine}`, text `{colors.on-primary}` rendered in `{typography.caption}`, padding `{spacing.huge}+ {spacing.xxl}` (32px 24px). Holds 4–5 columns of `{colors.on-aubergine-mute}` link groups, social icons, and a small legal/copyright row at the bottom.

## Do's and Don'ts

### Do
- Reserve `{colors.primary}` aubergine for filled CTAs, the featured pricing tier, and the closing aubergine band — it's the brand's chromatic monotheism.
- Use `{rounded.pill}` (90px) for every button across the system — never a rounded-rectangle button.
- Pair display tiers with negative letter-spacing (`-0.768px` at 64px); the proprietary face needs the tracking pull.
- Compose hero bands with pastel-mesh gradient backdrop + floating product UI mockup; the gradient is the depth.
- Use `{colors.link-blue}` for inline links — it's the only chromatic departure from aubergine and is part of the brand voice.

### Don't
- Don't add a third accent color to the system — the aubergine + blue link combination is exhaustive.
- Don't shrink button padding below `14px 28px` — the over-padded pill is part of the brand feel.
- Don't render display tiers at default tracking (0) — without negative letter-spacing the headlines read loose and unedited.
- Don't put product UI screenshots inside cards — they sit ABOVE the pastel-mesh gradient, never inside chrome.
- Don't use aubergine for body text — it's a surface and CTA color, not a type color at body sizes.
- Don't replace the pill shape with a square button anywhere.

## Responsive Behavior

### Breakpoints

| Name | Width | Key Changes |
|---|---|---|
| Wide | ≥ 1440px | Full-bleed pastel-mesh hero; pricing 4-up |
| Desktop | 1024–1440px | Default content max-width; pricing 4-up |
| Tablet | 768–1023px | Pricing 2-up; product UI mockups crop to focal panel |
| Mobile | < 768px | Pricing 1-up; hamburger nav; display-xxl drops 64 → 40px |

### Touch Targets
- Pill buttons hit ≥ 48×48px due to the over-padded geometry. WCAG AAA compliant.
- Form fields stay at the 44px minimum height.

### Collapsing Strategy
- Display tiers stair-step 64 → 50 → 32 → 28 → 24 across breakpoints.
- Pastel-mesh gradients re-tile on mobile to prevent the wash from disappearing entirely.
- Floating product UI mockups crop to the most actionable inner panel on mobile.
- Pricing tiers stair-step 4 → 2 → 1; aubergine featured tier stays distinguished.
- Top nav collapses to hamburger below 768px; menu inherits canvas color.

### Image Behavior
Product UI mockups use `srcset` for desktop / tablet / mobile crops; the mobile crop centers on the most actionable inner panel rather than scaling the whole composite down.

## Iteration Guide

1. Focus on ONE component at a time.
2. Reference component names and tokens directly (`{colors.primary}`, `{button-primary-pill}-pressed`, `{rounded.pill}`).
3. Run `npx @google/design.md lint DESIGN.md` after edits.
4. Add new variants as separate entries.
5. Default body to `{typography.body-md}`; reserve `{typography.body-lg}` for marketing leads.
6. Keep aubergine scarce — one filled aubergine button per viewport.
7. Pair every hero band with the pastel-mesh gradient backdrop; bare-canvas heroes read as off-brand.
