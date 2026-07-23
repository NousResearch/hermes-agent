# Design System: WIRED


> **Hermes Agent — Implementation Notes**
>
> Source analysis adapted from [`VoltAgent/awesome-design-md`](https://github.com/VoltAgent/awesome-design-md) at commit `664b3e78`; see `../references/ATTRIBUTION.md` for the MIT notice.
>
> WIRED's three proprietary faces have distinct editorial roles; do not collapse them into one fallback stack:
> - **Display serif:** `Playfair Display` at weight 400
> - **Article body serif:** `Lora` at weights 400/700
> - **Metadata, labels, and buttons:** `Inter` at weights 400/700
> - **Display stack:** `font-family: 'Playfair Display', Georgia, serif;`
> - **Body stack:** `font-family: Lora, 'Times New Roman', serif;`
> - **UI stack:** `font-family: Inter, Arial, sans-serif;`
> ```html
> <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Lora:wght@400;700&family=Playfair+Display:wght@400&display=swap" rel="stylesheet">
> ```
> Keep serif for narrative and sans-serif for structure; never use the UI sans for article body or display.
> Use `write_file` to create HTML and open the rendered artifact with `browser_navigate`.
> Verify visual accuracy with `browser_vision` after generating.

## Overview

Wired is the flagship technology-magazine brand under Condé Nast — and the web surface refuses to dress itself as a SaaS marketing site. The page is unmistakably an editorial product: a white canvas, a strict black wordmark in the brand's proprietary `WiredDisplay` (a tall, narrow, high-contrast serif used at 64 px), and stacked story cards that read as a printed magazine grid ported to the screen. There is no atmospheric gradient, no decorative chrome, no chromatic accent — the brand's only colour beyond the black-and-white duet is the small `{colors.link}` (`#057dbc`) used for inline body links inside long-form articles.

Type carries the entire identity. Three families ladder the system: `WiredDisplay` (the proprietary high-contrast serif) for hero / section headlines; `BreveText` (a humanist serif) for long-form body and bylines; and `Apercu` (a humanist sans) for metadata, captions, eyebrow tags, and buttons. The pairing is editorial-grade: serifs for narrative, sans for navigation and structural labels.

Buttons are square. The brand uses `{rounded.none}` 0 px corners across the entire UI — newsletter sign-ups, login forms, "Read more" CTAs all render as sharp rectangles. The only circular shape is the `button-icon-circular` used for social-share affordances. There are no soft drop-shadows; the brand uses hairline borders for elevation when needed.

**Key Characteristics:**
- A strict black-and-white duet with no chromatic accent except the inline link blue `{colors.link}`. The brand reads as a printed magazine.
- Three-face typographic system — `WiredDisplay` serif for display, `BreveText` serif for body, `Apercu` sans for metadata and buttons.
- Square buttons (`{rounded.none}`) — the brand never softens corners on interactive elements.
- A magazine-style story grid: large feature card at top, two-up secondary, then a vertical stack of bylined story rows separated by `{colors.hairline}` 1 px dividers.
- The brand's only signature decorative move is the **masthead band** — a thin black strip with the wordmark centred, no other decoration.
- A near-black `{colors.ink}` (`#000000`) footer band, no graphics, just text columns and the wordmark repeating.

## Colors

### Brand & Accent
- **Ink Black** (`{colors.primary}` — `#000000`): The brand's only "accent." Used for wordmark, headlines, CTAs, footer fill. Pure black, never softened.

### Surface
- **Canvas** (`{colors.canvas}` — `#ffffff`): The default page background.
- **Canvas Soft** (`{colors.canvas-soft}` — `#f5f5f5`): Rare tint used for the comment-section background and search-result row hover states (not in the main page rhythm).
- **Hairline** (`{colors.hairline}` — `#e0e0e0`): 1 px dividers between story rows. The brand's only "line."

### Text
- **Ink** (`{colors.ink}` — `#000000`): Every headline, every body paragraph in BreveText.
- **Ink Soft** (`{colors.ink-soft}` — `#1a1a1a`): A near-black variant used for caption-strong / footer link emphasis.
- **Body** (`{colors.body}` — `#757575`): Secondary metadata — bylines, timestamps, supporting body lines.

### Semantic
The brand operates with one inline link colour and no separate error / success / warning palette in its marketing surface. Validation cues on form pages use the ink black + body grey hierarchy.

- **Link** (`{colors.link}` — `#057dbc`): The inline body-link blue. Used only inside long-form article body copy, never on UI buttons or navigation.

## Typography

### Font Family
Three families ladder the system:
1. **WiredDisplay** — the proprietary tall-narrow high-contrast serif used exclusively for display headlines (64 px hero, scaling down to 26 px sub-display). The brand's most-recognisable typographic signature.
2. **BreveText** — the proprietary humanist serif used for long-form body, bylines, and editorial captions. Used at 16 – 19 px line-height 1.45 – 1.50 for comfortable reading density.
3. **Apercu** — a humanist sans used for nav, button labels, category eyebrows, metadata, and captions. Weights 400 / 700.

Inter is loaded as a fourth fallback face for embedded utility surfaces (the comment section, account pages) but does not appear on the main marketing / article surface.

### Hierarchy

| Token | Size | Weight | Line Height | Letter Spacing | Use |
|---|---|---|---|---|---|
| `{typography.display-hero}` | 64px | 400 | 59.5px | -0.5px | Cover-story headline. |
| `{typography.display-lg}` | 48px | 400 | 50.4px | -0.4px | Major section / feature headlines. |
| `{typography.display-md}` | 32px | 400 | 35.2px | -0.3px | Story-card large variant. |
| `{typography.display-sm}` | 26px | 400 | 28px | 0 | Sub-display headings inside long-form articles. |
| `{typography.display-xs}` | 20px | 700 | 24px | -0.28px | Sans (Apercu) display micro-headings for category callouts. |
| `{typography.body-serif-lg}` | 19px | 400 | 27.93px | 0.108px | Lead paragraph of an article (BreveText). |
| `{typography.body-serif-md}` | 16px | 400 | 24px | 0.09px | Default article body (BreveText). |
| `{typography.body-md}` | 17px | 400 | 20px | 0 | Sans body (Apercu) for nav / metadata. |
| `{typography.body-md-strong}` | 17px | 700 | 22px | -0.144px | Bold sans body. |
| `{typography.body-sm}` | 14px | 400 | 18px | 0.4px | Secondary sans body. |
| `{typography.body-sm-strong}` | 14px | 700 | 18px | 0.4px | Bold caption / nav-link. |
| `{typography.byline}` | 12.73px | 700 | 28px | 0.108px | Article byline (BreveText). |
| `{typography.caption}` | 12px | 400 | 16px | 0 | Fine print, image captions. |
| `{typography.button-md}` | 16px | 700 | 20px | 0.3px | Button label. |

### Principles
- **Serif for narrative, sans for structure.** The serif faces never carry button labels or nav text; the sans face never carries article body.
- **Display weight 400** — the proprietary WiredDisplay reads as elegant by virtue of its thin-tall-narrow design at default weight, not via weight 700+.
- **Bylines use BreveText weight 700 with relaxed line-height 2.2.** The vertical breathing is part of the editorial signature.

### Note on Font Substitutes
The three proprietary faces have no exact substitutes. Best open-source approximations:
- **WiredDisplay** — *Playfair Display* weight 400 at large display sizes captures the high-contrast didone feel, though wider than the brand's tall-narrow proportions.
- **BreveText** — *Lora* or *Source Serif Pro* at 16 – 19 px.
- **Apercu** — *Inter* or *Manrope* weights 400 / 700.

## Layout

### Spacing System
- **Base unit**: 4 px.
- **Tokens**: `{spacing.xxs}` 2 px · `{spacing.xs}` 4 px · `{spacing.sm}` 8 px · `{spacing.md}` 12 px · `{spacing.lg}` 16 px · `{spacing.xl}` 20 px · `{spacing.2xl}` 24 px · `{spacing.3xl}` 32 px · `{spacing.4xl}` 48 px.
- **Section padding**: hero / story grid use `{spacing.4xl}` 48 px top/bottom on desktop.
- **Story row padding**: `{spacing.lg}` 16 px vertical between bylined story rows.

### Grid & Container
- Marketing content uses a wide container (~1400 px max).
- Cover-story grid: 1 large hero + 2-up secondary stories + vertical stack.
- Story-row stack: full-width single column with hairline dividers.

### Responsive Strategy

#### Breakpoints

| Name | Width | Key Changes |
|---|---|---|
| Mobile | < 768px | Cover hero 64→40 px; all grids 1-up; hamburger nav. |
| Tablet | 768–1023px | 2-up secondary story grid. |
| Desktop | ≥ 1024px | Full magazine grid. |

#### Touch Targets
Button-primary renders ~44 px tall (12 vertical padding + 20 line). WCAG AAA at all widths.

#### Collapsing Strategy
- Nav: full link row + Subscribe CTA at desktop. Hamburger at mobile.
- Magazine grid: hero stays full-width; 2-up secondary drops to 1-up at mobile.
- Story rows: stay single-column at all viewports.

#### Image Behavior
- Cover images: full-bleed 16:9 hero / 4:3 secondary.
- Article body images: full-width inside the article column.
- Author avatars: small inline circular crops next to bylines.

## Elevation & Depth

| Level | Treatment | Use |
|---|---|---|
| Level 0 — Flat | No shadow, no border. | Default — almost every surface lives at this level. |
| Level 1 — Hairline | 1 px solid `{colors.hairline}` border. | Story-row dividers, input borders. |
| Level 2 — Heavy Black Border | 2 px solid `{colors.ink}` border. | Subscribe CTA on certain campaign moments. |

The brand uses no drop-shadows. Surface contrast and hairline borders carry all visual hierarchy.

## Shapes

### Border Radius Scale

| Token | Value | Use |
|---|---|---|
| `{rounded.none}` | 0px | Every interactive shape — buttons, inputs, cards. The brand's signature square geometry. |
| `{rounded.full}` | 9999px | Circular icon containers only (social-share, account avatar). |

### Photography Geometry
- Cover stories: 16:9 hero, edge-to-edge.
- Secondary story cards: 4:3 thumbnails.
- Article body images: native aspect, full column width.
- Bylines / avatars: circular `{rounded.full}` 28 px crops.

## Components

### Buttons

**`button-primary`** — the square black CTA.
- Background `{colors.primary}`, text `{colors.on-primary}`, label `{typography.button-md}` (Apercu 16 px / 700 / 0.3 px tracking), padding `{spacing.md} {spacing.xl}`, shape `{rounded.none}` 0 px.

**`button-outline`** — the white outline CTA.
- Background `{colors.canvas}`, text `{colors.ink}`, 1 px solid `{colors.ink}` border, same typography / padding / shape.

**`button-icon-circular`** — the circular share-icon button.
- Background `{colors.canvas}`, ink icon, shape `{rounded.full}`.

### Cards & Containers

**`story-card-large`** — the cover-story card with `{typography.display-md}` headline.
- Background `{colors.canvas}`, text `{colors.ink}`, padding `{spacing.lg}`. No border — the card lives on the canvas with the photo doing the work.

**`story-card`** — the secondary story card.
- Background `{colors.canvas}`, text `{colors.ink}`, padding `{spacing.md}`. Photo at top, sans display heading + body lead below.

**`story-row`** — the bylined story list row.
- Background `{colors.canvas}`, text `{colors.ink}`, 1 px solid `{colors.hairline}` bottom border, body in `{typography.body-md-strong}`, padding `{spacing.lg}` 0.

### Inputs & Forms

**`text-input`** — the standard text input.
- Background `{colors.canvas}`, text `{colors.ink}`, 1 px solid `{colors.ink}` border, body in `{typography.body-md}`, padding `{spacing.md} {spacing.lg}`, shape `{rounded.none}`.

### Navigation

**`nav-bar`** — the top nav, light by default.
- Background `{colors.canvas}`, text `{colors.ink}`, padding `{spacing.md} {spacing.xl}`. Layout: hamburger left, masthead centre, Subscribe right.

**`nav-link`** — link items inside nav.
- Text `{colors.ink}`, set in `{typography.body-sm-strong}` (Apercu 14 / 700).

**`footer`** — the black footer band.
- Background `{colors.primary}`, text `{colors.on-primary}`, padding `{spacing.4xl} {spacing.xl}`. Body in `{typography.body-sm}` (Apercu 14 / 400). Footer column eyebrows in `{typography.body-sm-strong}`.

### Signature Components

**`hero-band`** — the white hero band hosting the cover-story.
- Background `{colors.canvas}`, text `{colors.ink}`, padding `{spacing.4xl} {spacing.xl}`. Cover headline in `{typography.display-hero}` (WiredDisplay 64 px).

**`masthead-band`** — the thin top band with the wordmark.
- Background `{colors.canvas}`, text `{colors.ink}`, padding `{spacing.md} {spacing.xl}`. The wordmark sits centred; flanked by section nav.

**`category-eyebrow`** — the small uppercase category label above story headlines.
- Text `{colors.ink}`, set in `{typography.body-sm-strong}` (Apercu 14 / 700 — though some campaigns use ALL CAPS via CSS).

**`byline-row`** — the article byline strip.
- Background `{colors.canvas}`, text `{colors.body}`, body in `{typography.byline}` (BreveText 12.73 / 700 / line-height 2.2). Includes an author avatar + name + date.

**`hairline-divider`** — the 1 px line between story rows.
- 1 px solid `{colors.hairline}`.

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
- Reserve `{colors.primary}` black for the wordmark, every CTA, and the footer fill. The brand IS the strict black-on-white duet.
- Set hero headlines in `{typography.display-hero}` (WiredDisplay 64 px weight 400). The proprietary serif IS the brand's typographic signature.
- Use `{rounded.none}` 0 px on every button and form input. The brand reads as a printed magazine — square corners are non-negotiable.
- Pair WiredDisplay (serif display) with BreveText (serif body) and Apercu (sans labels). Three faces, three roles.
- Render story rows with `{colors.hairline}` 1 px dividers — the brand's only elevation cue.

### Don't
- Don't introduce a chromatic brand accent. The link blue is reserved for inline body links inside articles only.
- Don't round button corners. The brand never softens its rectangular geometry.
- Don't drop a soft drop-shadow on cards. Surface contrast and hairlines carry elevation.
- Don't substitute the proprietary serif faces with a generic sans for display. The serif voice is the brand.
- Don't promote display weight beyond 400. The brand's elegance is in the typeface design, not bold weight.
