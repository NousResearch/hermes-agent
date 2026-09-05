# Design System: Nike


> **Hermes Agent — Implementation Notes**
>
> Source analysis adapted from [`VoltAgent/awesome-design-md`](https://github.com/VoltAgent/awesome-design-md) at commit `664b3e78`; see `../references/ATTRIBUTION.md` for the MIT notice.
>
> Nike separates campaign display typography from its UI/body system:
> - **Campaign display:** `Bebas Neue` at 96px, 0.9 line-height, uppercase
> - **UI and body:** `Inter` at weights 400/500/700
> - **Display stack:** `font-family: 'Bebas Neue', Anton, 'Arial Narrow', sans-serif;`
> - **UI/body stack:** `font-family: Inter, Helvetica, Arial, sans-serif;`
> ```html
> <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;700&display=swap" rel="stylesheet">
> ```
> Tighten substitute campaign tracking by about -0.5% to approximate Nike Futura ND.
> Use `write_file` to create HTML and open the rendered artifact with `browser_navigate`.
> Verify visual accuracy with `browser_vision` after generating.

## Overview

Nike's commerce system is built on a single, almost violently simple idea: photography speaks, the chrome doesn't. Every page reads as an athletic editorial — towering uppercase Futura display lockups (`{typography.display-campaign}`) burned into full-bleed campaign imagery, with everything else (nav, filters, buttons, cards, footer) reduced to neutral typography and pill geometry on `{colors.canvas}` and `{colors.soft-cloud}`. There is no decorative gradient, no soft shadow nostalgia, no accent color used for "tone" — the system saves all chromatic energy for product photography and the small handful of moments that actually need to signal (sale price `{colors.sale}`, success `{colors.success}`, swatch dots).

The result is a layout that feels physical — campaign hero, product grid, sport tile, footer — stacked like a printed catalog rather than animated like a typical SaaS landing page. Density is high but never crowded, because the system relies on three relentless devices: square or near-square 1:1 product imagery on `{colors.soft-cloud}`, pill-shaped black CTAs (`{rounded.full}`) anchoring every actionable surface, and a tight 8px-base spacing scale that keeps cards and filters mathematically aligned across PLP, PDP, and editorial pages.

Across `/men`, the trail-running listing, the Zegama PDP, `/membership`, and Jordan Golf, the same chrome appears in identical proportions — only the photography and copy change. That is the system's signature: maximum editorial expression in the imagery, maximum mechanical restraint everywhere else.

**Key Characteristics:**
- Editorial campaign hero with `{typography.display-campaign}` (Nike Futura ND, 96px, line-height 0.9, uppercase) burned directly into full-bleed photography
- Pure black/white/single-gray UI palette: `{colors.ink}`, `{colors.canvas}`, and `{colors.soft-cloud}` carry ~95% of the chrome surface area
- Pill geometry everywhere: every CTA, search field, filter chip, and badge uses `{rounded.full}` (30px) or `{rounded.md}` (24px) — there are no sharp-cornered buttons in the system
- Product cards have zero radius, zero shadow, sit directly on `{colors.soft-cloud}` swatch backgrounds — the photograph is the card
- Two-tone CTA hierarchy: `{component.button-primary}` (black on anything light) versus `{component.button-secondary}` (`{colors.soft-cloud}` on anything bright) — never both at once on the same surface
- 8px spacing system with section rhythm at `{spacing.section}` (48px) creating consistent vertical breathing across PLP, PDP, and editorial pages
- Sale signaling is the only place a non-neutral color appears in retail chrome: `{colors.sale}` price + strike-through original price, no badge background

## Colors

> **Source pages:** `/men` (primary), `/w/mens-acg-trail-running-shoes-…`, `/t/acg-zegama-…`, `/membership`, `/w/jordan-golf-…`. The chrome palette is identical across all five — only photography varies.

### Brand & Accent
- **Nike Black** (`{colors.ink}` — `#111111`): The brand's only "color." It is the primary CTA, the swatch dot, the active filter chip, the campaign overlay, the headline color, and the body text. When Nike wants to assert anything, it goes black.
- **Pure White** (`{colors.on-primary}`, `{colors.canvas}` — `#ffffff`): Equal partner to black. Carries every page background, the on-image CTA, and the inverse text on `{colors.ink}` surfaces.

### Surface
- **Soft Cloud** (`{colors.soft-cloud}` — `#f5f5f5`): The most-used non-white surface in the entire system. Product card image backgrounds, search pill, secondary CTA, utility bar, sport-category swatch tiles. It is the "color" of every product photograph's stage.
- **Hairline** (`{colors.hairline}` — `#cacacb`): 1px dividers between filter rows, footer columns, and PDP disclosure rows.
- **Hairline Soft** (`{colors.hairline-soft}` — `#e5e5e5`): Inset 1px shadow under sticky bars and tab strips, the only "shadow" the system uses.

### Text
- **Ink** (`{colors.ink}` — `#111111`): Primary text on light surfaces — headlines, product names, prices, nav.
- **Charcoal** (`{colors.charcoal}` — `#39393b`): Slightly softer body where ink is too heavy.
- **Ash** (`{colors.ash}` — `#4b4b4d`): Disabled secondary border on dark surfaces and very low-emphasis utility text.
- **Mute** (`{colors.mute}` — `#707072`): Product category subtitles ("Men's Trail Running Shoes"), footer link text, secondary metadata.
- **Stone** (`{colors.stone}` — `#9e9ea0`): Inverse secondary text on dark surfaces and lowest-emphasis utility text.

### Semantic
- **Sale** (`{colors.sale}` — `#d30005`): Discounted price color and "% off" copy — the only red in the entire retail chrome.
- **Sale Deep** (`{colors.sale-deep}` — `#780700`): Sale price hover/pressed and dark-mode sale anchor.
- **Success** (`{colors.success}` — `#007d48`): Confirmation messages, in-stock indicators, eligibility ticks.
- **Success Bright** (`{colors.success-bright}` — `#1eaa52`): Inverse success on dark surfaces.
- **Info** (`{colors.info}` — `#1151ff`): Informational link/badge accent in member-experience callouts.
- **Info Deep** (`{colors.info-deep}` — `#0034e3`): Pressed state for info accent.

### Category Accents (sport / collection chips)
These appear sparingly — almost exclusively as small chip backgrounds, swatch dots, or category illustrations in editorial tiles. They are never used as text or primary CTA color.
- **Accent Pink** (`{colors.accent-pink}` — `#ed1aa0`): SKIMS / women's collection moments.
- **Accent Pink Soft** (`{colors.accent-pink-soft}` — `#ffb0dd`): Soft tinting on member-experience tiles.
- **Accent Purple Soft** (`{colors.accent-purple-soft}` — `#beaffd`): Editorial swatch dot, soft category chip.
- **Accent Purple Pale** (`{colors.accent-purple-pale}` — `#d6d1ff`): Lightest soft-tile fill.
- **Accent Teal** (`{colors.accent-teal}` — `#0a7281`): Trail / outdoor / ACG editorial accent in lockups.
- **Accent Pink Deep** (`{colors.accent-pink-deep}` — `#4c012d`): Deepest editorial overlay tint, used as wash on heritage / Jordan tiles.

## Typography

### Font Family
- **Nike Futura ND** (display campaign only) — proprietary geometric sans for the towering uppercase headlines burned into campaign hero photography. Falls back to Helvetica Now Text Medium → Helvetica → Arial.
- **Helvetica Now Display Medium** (headings 16–32px) — modern Helvetica cut tuned for display sizes; carries every section title, PDP product name, and dialog headline.
- **Helvetica Now Text Medium** (UI 12–16px) — buttons, captions, swatch labels, badge text. The system's UI workhorse.
- **Helvetica Now Text** (body and links) — long-form body and underlined inline links.
- **Neue Frutiger Arabic** — RTL pairing for Arabic locales at `{typography.heading-lg}` and caption sizes.
- **Helvetica Neue 9px** — legal-fine-print utility row only (`{typography.utility-xs}`).

When substituting on systems without proprietary Nike fonts: pair **Inter** (Display 700 for body chrome, Display 500 for buttons) with **Bebas Neue** or **Anton** at 96px/0.9 line-height for the campaign headline tier. Tighten letter-spacing slightly (-0.5%) on the substitute to approximate Futura ND's optical weight.

### Hierarchy

| Token | Size | Weight | Line Height | Letter Spacing | Use |
|---|---|---|---|---|---|
| `{typography.display-campaign}` | 96px | 500 | 0.9 | 0 | Editorial campaign headline burned into hero photography (uppercase) |
| `{typography.heading-xl}` | 32px | 500 | 1.2 | 0 | Section headers — "FEATURED FOOTWEAR", "LATEST IN CLOTHING", PDP product title block |
| `{typography.heading-lg}` | 24px | 500 | 1.2 | 0 | Subsection / member-benefit card title, large CTA label, PDP price |
| `{typography.heading-md}` | 16px | 500 | 1.75 | 0 | Card title, FAQ row label, filter group header |
| `{typography.body-md}` | 16px | 400 | 1.5 | 0 | Body copy, search-pill placeholder, product description |
| `{typography.body-strong}` | 16px | 500 | 1.5 | 0 | Product card name, filter row label, primary nav link |
| `{typography.button-lg}` | 24px | 500 | 1.2 | 0 | Pressed-letter campaign CTA inside hero blocks |
| `{typography.button-md}` | 16px | 500 | 1.5 | 0 | Standard pill CTAs across the system |
| `{typography.button-sm}` | 14px | 500 | 1.5 | 0 | Compact pill CTA, badge label, geo-selector button |
| `{typography.link-md}` | 16px | 500 | 1.75 | 0 | Underlined inline link, "View Product Details" |
| `{typography.caption-md}` | 14px | 500 | 1.5 | 0 | Product subtitle ("Men's Trail Running Shoes"), filter count, footer link |
| `{typography.caption-sm}` | 12px | 500 | 1.5 | 0 | Filter chip label, badge text, color count |
| `{typography.utility-xs}` | 9px | 500 | 1.75 | 0 | Legal copyright / fine-print row at the very bottom |

### Principles
The system runs on extreme typographic contrast: a single 96px uppercase display tier reserved for editorial campaign moments, and a quiet 12–16px Helvetica Now Text/Medium tier carrying everything else. There is almost no middle ground — the jump from `{typography.heading-xl}` (32px) directly to `{typography.body-strong}` (16px) is intentional and creates the "billboard above, catalog below" effect across every page. Letter-spacing is left at 0 (Futura ND and Helvetica Now are both cut for tight optical fit at scale).

### Note on Font Substitutes
The closest open-source path to Nike's display tier is **Bebas Neue** (free, geometric condensed) at 96px / 0.9 / uppercase / 500. For UI text, **Inter** is the safest substitute — match weights 400/500 and the system reads almost identically at button and caption sizes.

## Layout

### Spacing System
- **Base unit:** 8px
- **Tokens (front matter):** `{spacing.xxs}` (2px) · `{spacing.xs}` (4px) · `{spacing.sm}` (8px) · `{spacing.md}` (12px) · `{spacing.lg}` (18px) · `{spacing.xl}` (24px) · `{spacing.xxl}` (30px) · `{spacing.section}` (48px+)
- **Universal rhythm:** every page in the set uses `{spacing.section}` (48px) as the vertical gap between major content blocks (campaign hero → trending row → featured row → shop-by-sport → latest-in-clothing → footer). PLP card grids use `{spacing.sm}` (8px) gutters. PDP disclosure rows are stacked at `{spacing.xl}` (24px) vertical padding.
- **Card internal padding:** product cards use 0px internal padding — image is full-bleed; metadata rows sit directly below with `{spacing.sm}` (8px) gap between name, subtitle, and price.

### Grid & Container
- **Max width:** ~1440px content area with edge gutters that grow to ~80px at 1920px (the system lets very wide viewports breathe rather than stretch).
- **Column patterns:** PLP listing uses 3-up at desktop, collapsing to 2-up at 1023px and 1-up at 599px. The men's home `/men` mixes a 2-up campaign hero row, a 3- or 4-up "Trending Now" row, a horizontal-scroll "Shop by Sport" rail, and a 4-up "Latest in Clothing" thumbnail grid.
- **Filter sidebar:** ~220px fixed-width left rail on PLP at desktop, collapsing into a `Hide Filters` toggle button at narrow widths.

### Whitespace Philosophy
Whitespace is a tool for separation, not for breath. Sections butt directly against each other vertically with `{spacing.section}` rhythm, and product photos tile edge-to-edge inside their grid — there is no padding wrapped around the product image itself. The "air" comes from the `{colors.soft-cloud}` background of the photograph, not from layout margin. Headlines do not have decorative whitespace above them; they sit immediately under the section divider line.

## Elevation & Depth

| Level | Treatment | Use |
|---|---|---|
| 0 — Flat | No shadow, no border | Default for cards, buttons, sections — the dominant treatment |
| 1 — Hairline divider | 1px solid `{colors.hairline}` | Filter row separators, footer column borders, PDP disclosure-row separators |
| 2 — Inset bottom-line | `box-shadow: inset 0 -1px 0 {colors.hairline-soft}` | Sticky utility/sub-nav bar bottom edge, tab strip underline |

The system has no drop-shadow elevation in its retail chrome at all. Cards do not lift on the page. The only depth cue is the 1px inset hairline on sticky strips and the contrast between full-bleed photography and `{colors.soft-cloud}` product backdrops.

### Decorative Depth
Depth in Nike's system comes entirely from photography, not from CSS effects:
- **Editorial campaign tiles** create depth via cinematic perspective — a runner on a trail, a model in a courtyard — with the Futura display headline overlaid in white or `{colors.ink}` directly on the image.
- **Product card photography** is shot on flat `{colors.soft-cloud}` to remove any background depth, so the product itself is the only thing with form on the page.
- **Sport-category tiles** on the home page are full-bleed cinematic photography with a small `{component.button-outline-on-image}` pill anchored at the bottom-left, giving a moment of crisp white pill against atmospheric image.

## Shapes

### Border Radius Scale

| Token | Value | Use |
|---|---|---|
| `{rounded.none}` | 0px | Cards, campaign tiles, product imagery, navigation, footer — every container in the system |
| `{rounded.sm}` | 18px | Avatar / icon container in member-benefit lockups |
| `{rounded.md}` | 24px | Search pill, search submit, filter input |
| `{rounded.lg}` | 30px | Every CTA pill — primary, secondary, on-image, filter chip, geo-selector, "Notify Me" |
| `{rounded.full}` | 9999px | Color swatch dots and circular icon buttons (back, share, favorite, carousel paddle) |

### Photography Geometry
- **Product cards:** consistent 1:1 square or near-square (~4:5 portrait on tall product crops), full-bleed within the card with no padding, sitting on `{colors.soft-cloud}` backdrop.
- **Editorial campaign hero:** ~16:9 or wider cinematic crop, full-bleed across the content max-width, with the Futura display headline burned into the lower-left or upper-left third.
- **Sport-category rail:** 4:5 portrait full-bleed thumbnails with a small CTA pill anchored bottom-left.
- **PDP main image:** square primary image with vertical thumbnail rail to its left (~5–7 thumbnails stacked at small size), enabling rapid color/angle browsing without leaving the page.
- **Avatar / category icon cards:** centered illustrated icon at ~80–96px on `{colors.canvas}` with `{typography.caption-md}` label below.

## Components

> **No hover states documented** per system policy. Each spec covers Default and Active/Pressed only; variants live as separate `components:` entries in the front matter.

### Buttons

**`button-primary`** — the universal Nike CTA
- Background `{colors.ink}`, text `{colors.on-primary}`, type `{typography.button-md}`, padding `16px 32px`, height `{spacing.section}` (48px), rounded `{rounded.lg}` (30px pill).
- Used on every primary action in the system: "Sign Up", "Notify Me", "Buy", "Türkiye" geo-confirm, "Shop" overlay on sport tiles, "Continue".
- Pressed state lives in `button-primary-active` — the bg stays `{colors.ink}` while the surface shrinks to `scale(0.5)` with `opacity: 0.5` (Nike's signature "tap collapse" feedback that's extracted across all five pages).

**`button-secondary`** — soft alternative on light surfaces
- Background `{colors.soft-cloud}`, text `{colors.ink}`, type `{typography.button-md}`, padding `16px 32px`, rounded `{rounded.lg}`.
- Used as the lower-emphasis alternate when a primary CTA already exists, e.g., "United States" geo-decline next to the black "Türkiye" confirm; "Cancel" or "Discover More" on light cards.

**`button-outline-on-image`** — overlay CTA on photography
- Background `{colors.canvas}`, text `{colors.ink}`, type `{typography.button-md}`, padding `12px 24px`, rounded `{rounded.lg}`.
- The crisp white pill that anchors the bottom-left of every full-bleed sport-category and editorial campaign tile.

**`button-icon-circular`** — chrome icon controls
- Background `{colors.soft-cloud}` or transparent, icon `{colors.ink}`, rounded `{rounded.full}`, size 40px.
- Used for back-arrow, carousel paddle (left/right), wishlist heart, share, and "Hide Filters" toggle.

**`filter-chip`** + **`filter-chip-active`**
- Default: background `{colors.canvas}`, text `{colors.ink}`, 1px hairline `{colors.hairline}`, type `{typography.button-md}`, rounded `{rounded.lg}`, padding `8px 16px`.
- Active: background `{colors.ink}`, text `{colors.on-primary}` — the chip flips fully inverted when selected. No middle state.

### Inputs & Forms

**`search-pill`** + **`search-pill-focused`**
- Default: background `{colors.soft-cloud}`, text `{colors.ink}`, type `{typography.body-md}`, rounded `{rounded.md}` (24px), padding `8px 16px`, height `40px`. Anchored to the right of the primary nav with a small magnifier icon.
- Focused: background `{colors.canvas}`, 2px solid border `{colors.ink}`, with a 12px outer halo of `{colors.soft-cloud}` (the system's only "focus ring" effect). The pill shape stays `{rounded.md}` so the halo reads as a soft glove, not a hard outline.

### Cards & Containers

**`product-card`**
- Container: background `{colors.canvas}`, rounded `{rounded.none}`, padding 0, no shadow.
- Image area: `{component.product-card-image}` — full-bleed product photo on `{colors.soft-cloud}` square.
- Below image (in this order with `{spacing.sm}` between): swatch dot row (3–6 dots at 12px circular), promo badge if applicable (`{component.badge-promo}` "Just In", "Coming Soon", "Recycled Materials"), product name `{typography.body-strong}` `{colors.ink}`, category subtitle `{typography.caption-md}` `{colors.mute}`, price row.
- Price row: regular price `{typography.body-strong}` `{colors.ink}`. If on sale: discounted price `{colors.sale}` followed by strike-through original `{colors.mute}` followed by "% off" in `{colors.sale}`.

**`campaign-tile`** — the brand's signature editorial unit
- Full-bleed photography with `{typography.display-campaign}` headline burned in (uppercase, 96px, line-height 0.9).
- Headline color is whichever of `{colors.canvas}` or `{colors.ink}` reads against the underlying image — not parameterized; chosen per-asset.
- A single `{component.button-outline-on-image}` pill anchored bottom-left of the tile carries the call-to-action.

**`category-icon-card`**
- Container: background `{colors.canvas}`, rounded `{rounded.none}`.
- Centered category illustration (~80px) + label `{typography.caption-md}` `{colors.ink}` directly below. Used in the "Latest in Clothing" 4–8-up icon strip on `/men`.

**`member-benefit-card`**
- Full-bleed photographic card on a dark image background; copy slot at the bottom-left with `{typography.heading-lg}` headline `{colors.on-primary}` and a `{component.button-outline-on-image}` "Explore" pill below.
- Used in the `/membership` "Member Benefits" 3-up grid.

**`swatch-dot`** + **`swatch-dot-active`**
- 12px circle, rounded `{rounded.full}`, no border in default state. Renders the colorway options on every product card and PDP color picker.
- Default: filled with the colorway's actual product color (extracted at runtime from the product image), 1px subtle outer ring in `{colors.hairline}` for white/light colorways so they remain visible on `{colors.canvas}`.
- Active: identical fill with a 2px `{colors.ink}` outer ring and 2px white interior gap, creating Nike's signature concentric-ring "selected" state. No size change between default and active.

**`badge-promo`**
- Background `{colors.canvas}` with 1px hairline `{colors.hairline}`, text `{colors.ink}`, type `{typography.caption-sm}`, rounded `{rounded.lg}`, padding `4px 12px`.
- Sits on top of product imagery (top-left of card) with copy like "Just In", "Coming Soon", "Recycled Materials", "Member Exclusive".

**`badge-sale-text`**
- Inline price-row text in `{colors.sale}` with no background — the only "badge" in the system that has no container.

### Navigation

**`utility-bar`** — top utility strip
- Background `{colors.soft-cloud}`, text `{colors.ink}`, type `{typography.caption-sm}`, height ~36px, rounded `{rounded.none}`.
- Right-aligned cluster: "Find a Store · Help · Join Us · Sign In". Always present; never collapses.

**`primary-nav`** — main navigation
- Background `{colors.canvas}`, text `{colors.ink}`, type `{typography.body-strong}` for nav links, height 56–64px, rounded `{rounded.none}`.
- Layout: Nike swoosh logo at left (32×32), centered nav row ("New & Featured · Men · Women · Kids · Jordan · Nike SKIMS · Sport"), right cluster (search pill, wishlist heart icon, bag icon).
- The active section gets a 2px-bottom underline in `{colors.ink}` — no background fill.

**Sub-nav strip** (PLP) — appears under primary nav with breadcrumb + sort + hide-filters controls.
- Same `{colors.canvas}` background with a 1px inset hairline-soft bottom edge.
- Left: breadcrumb in `{typography.caption-md}` `{colors.mute}` separated by " / ".
- Right: "Hide Filters" toggle + "Sort By: …" dropdown — both in `{typography.button-md}` with chevron icons.

**Top Nav (Mobile)**
- Hamburger menu icon (left), Nike swoosh (center), search + bag icons (right).
- Search pill collapses into an icon-only button at narrow widths; tapping expands a full-width overlay search pill with `{rounded.md}`.
- Primary nav collapses into a full-height drawer that slides in from the left, listing nav rows top-down with `{spacing.xl}` vertical padding.

### Signature Components

**`pdp-disclosure-row`** — PDP information accordion rows
- Stacked rows for "View Product Details", "Shipping & Returns", "Reviews (n)" with `{spacing.xl}` vertical padding and a 1px `{colors.hairline}` divider below each.
- Label `{typography.body-strong}` `{colors.ink}` left-aligned; chevron `{colors.ink}` right-aligned.

**`faq-row`** — `/membership` FAQ accordion
- Identical pattern to `pdp-disclosure-row` but with `{typography.heading-md}` label weight; 1px `{colors.hairline}` divider below each.

**`filter-sidebar`** — PLP left rail
- Container `{colors.canvas}`, rounded `{rounded.none}`.
- Section headers `{typography.body-strong}` `{colors.ink}` with `{spacing.lg}` (18px) vertical gap between groups.
- Active filters get a 1px ink underline; counts in parentheses use `{colors.mute}`.

**`footer`**
- Background `{colors.canvas}` with a single 1px `{colors.hairline}` top divider.
- Four columns: Resources / Help / Company / Promotions & Discounts, each with column header `{typography.body-strong}` `{colors.ink}` and link list `{typography.caption-md}` `{colors.mute}`.
- Below the columns: a horizontal rule, then a fine-print row with `{typography.utility-xs}` `{colors.mute}` (copyright, locale switcher, terms, privacy, supply-chain act).

## Do's and Don'ts

### Do
- Reserve `{typography.display-campaign}` exclusively for editorial campaign hero lockups — never use 96px Futura for section headers or product titles.
- Use `{component.button-primary}` (`{colors.ink}` pill) as the single primary action per viewport. Pair it at most with `{component.button-secondary}` (`{colors.soft-cloud}` pill) for a soft alternative.
- Stage every product photograph on `{colors.soft-cloud}` — the gray is the system's "studio."
- Keep all CTAs pill-shaped at `{rounded.lg}` (30px). Never introduce a square or `{rounded.sm}` button.
- Use `{colors.sale}` only on price rows — never on backgrounds, badges, or chrome.
- Stack content sections at `{spacing.section}` (48px) rhythm with no decorative dividers between them; the photography's bleed-edge is the divider.
- Anchor on-image CTAs with `{component.button-outline-on-image}` (white pill) at bottom-left — the system's universal "shop this image" position.

### Don't
- Don't introduce drop shadows or card elevation. Cards sit flat on the page; the only depth cue is the 1px inset hairline on sticky bars.
- Don't use any of the category accent colors (`{colors.accent-pink}`, `{colors.accent-purple-soft}`, `{colors.accent-teal}`) for primary chrome — they belong to swatch dots, soft tile fills, and editorial moments only.
- Don't replace `{colors.ink}` with a near-black gray like `{colors.charcoal}` for a CTA — Nike's primary pill is true `#111111`.
- Don't pad inside product cards. The image is full-bleed; metadata sits directly below with `{spacing.sm}` (8px) between rows.
- Don't put two campaign-tile lockups in the same row at the same scale — Nike alternates a single full-bleed editorial tile with a 2-up or 4-up product/category grid.
- Don't underline anything other than `{typography.link-md}` inline links and the active primary-nav indicator. Buttons, headings, and prices stay un-underlined.
- Don't introduce a third button shape. Pill or icon-circular — that's the entire button shape vocabulary.

## Responsive Behavior

### Breakpoints

| Name | Width | Key Changes |
|---|---|---|
| ultrawide | 1920px+ | Content max-width holds at ~1440px; outer gutters grow to ~80px on each side |
| desktop-large | 1440px | Default desktop layout — 3-up product grid, 4-up clothing strip, full primary nav |
| desktop | 1200px | Same as large with slightly narrower outer gutters |
| desktop-small | 1024px | Filter sidebar starts compressing; sport rail shows ~3 visible tiles |
| tablet | 1023–961px | 3-up PLP collapses to 2-up; "Hide Filters" becomes a default toggle |
| tablet-narrow | 960–640px | Primary nav center cluster collapses to a hamburger drawer; search pill becomes icon-only |
| mobile-landscape | 639–600px | 2-up PLP collapses to 1-up; product cards become full-width with image and metadata stacking |
| mobile | 599–320px | Single-column everything; campaign tiles render at full screen width with shorter Futura sizes (~64px) |

### Touch Targets
All interactive elements meet WCAG AAA (44×44px minimum). Pills (`{component.button-primary}`, `{component.button-secondary}`) sit at 48px height with 32px horizontal padding. Icon-circular buttons (`{component.button-icon-circular}`) sit at 40px — Nike's PDP carousel paddle and wishlist heart sit just under AAA but above AA at 40×40, with hit-target padding extending the tappable area to 48px+. Filter-chip pills are 40px height with 16px padding.

### Collapsing Strategy
- **Primary nav:** desktop center cluster → mobile drawer triggered by hamburger at left of the swoosh.
- **PLP grid:** 3-up → 2-up → 1-up at 1023, 599, and below; gutters drop from `{spacing.sm}` to `{spacing.xs}` on mobile.
- **Filter sidebar:** 220px fixed → "Hide Filters" toggle → off-canvas full-screen filter drawer at mobile.
- **Sport rail:** desktop horizontal scroll with ~5 visible → mobile horizontal scroll with ~1.5 visible (peek-next-card pattern).
- **Section spacing:** `{spacing.section}` 48px desktop → 32px tablet → 24px mobile to keep editorial rhythm tight on small screens.
- **Editorial campaign headline:** desktop 96px → tablet 64px → mobile 48px, line-height stays at 0.9 across all sizes.

### Image Behavior
- Product imagery is responsive at the same 1:1 ratio across all breakpoints — the image scales, the ratio doesn't.
- Editorial campaign tiles use art-direction crops: a 16:9 wide hero on desktop swaps to a 4:5 portrait on mobile so the figure stays centered and the headline still has burn-in space.
- All non-critical product imagery is lazy-loaded as the user scrolls into the next grid row.

## Iteration Guide

1. Focus on ONE component at a time. Pull its YAML entry from the front matter and verify every property resolves.
2. Reference component names and tokens directly (`{colors.ink}`, `{component.button-primary-active}`, `{rounded.lg}`) — do not paraphrase color names or radii in prose.
3. Run `npx @google/design.md lint DESIGN.md` after edits — `broken-ref`, `contrast-ratio`, and `orphaned-tokens` warnings flag issues automatically.
4. Add new variants as separate component entries (`-active`, `-disabled`, `-focused`) — do not bury them inside prose. Nike's pressed state (`scale(0.5) opacity 0.5`) is intentional and must be its own entry, not a hover stand-in.
5. Default body to `{typography.body-md}`; reach for `{typography.body-strong}` for product names and primary nav links; reserve `{typography.display-campaign}` strictly for hero campaign lockups.
6. Keep `{colors.ink}` scarce per viewport — if more than one solid-black pill or block appears in the same fold, neutralize one to `{component.button-secondary}` or `{component.button-outline-on-image}`.
7. When introducing a new component, ask whether it can be expressed with the existing pill + flat-card + photography-on-`{colors.soft-cloud}` vocabulary before adding new tokens. The system's strength is that it almost never needs new ones.

## Known Gaps

- **Mobile screenshots not captured** — responsive behavior described above synthesizes Nike's known mobile pattern (hamburger drawer, 1-up grid, headline downscale) from desktop evidence and the breakpoint list extracted from tokens.
- **Hover states not documented** by system policy — Nike's CSS uses `--pds-color-element-hover` and `--pds-color-text-hover` tokens but these are not included here.
- **Dialog / modal styling** beyond the geo-selector and the country-confirmation pill pair could not be confirmed from the captured surfaces; bag, wishlist, and login overlays are not documented.
- **Form field styling** for checkout, sign-up, and address forms is not present in the captured surfaces — only the search pill is documented.
- **Bag and wishlist** icon-state variants (filled count badges) not visible in the captured pages.
