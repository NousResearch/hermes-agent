# Design System: BMW M


> **Hermes Agent — Implementation Notes**
>
> Source analysis adapted from [`VoltAgent/awesome-design-md`](https://github.com/VoltAgent/awesome-design-md) at commit `664b3e78`; see `../references/ATTRIBUTION.md` for the MIT notice.
>
> BMW Type Next Latin is proprietary. Preserve its light/regular split with these open substitutes:
> - **Display and body:** `Inter` at weights 300/400/700
> - **Optional condensed display:** `Saira Condensed`
> - **Primary stack:** `font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;`
> - **Condensed display stack:** `font-family: 'Saira Condensed', Inter, sans-serif;`
> ```html
> <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&family=Saira+Condensed:wght@300;400;700&display=swap" rel="stylesheet">
> ```
> Use -0.5px tracking on large Inter headlines to approximate BMW Type's tighter spacing.
> Use `write_file` to create HTML and open the rendered artifact with `browser_navigate`.
> Verify visual accuracy with `browser_vision` after generating.

## Overview

BMW M's marketing surface is a near-pure black canvas (`{colors.canvas}` — #000) holding white BMW Type Next Latin headlines in **confident UPPERCASE**. The system has no decorative voltage of its own; brand energy comes from **full-bleed automotive photography** — cars cornering at speed, carbon-fiber wheel detail, driver cockpit shots, motorsport pit lanes — placed as edge-to-edge content that fills entire bands. UI chrome around the photography stays minimal: thin sans-serif copy, dividers as 1px hairlines (`{colors.hairline}`), all-caps button labels with no fill until hovered.

The **M tricolor stripe** — `{colors.m-blue-light}` (#0066b1) → `{colors.m-blue-dark}` (#1c69d4) → `{colors.m-red}` (#e22718) — appears sparingly as the brand's signature accent, used on the M wordmark, motorsport chrome, vehicle-tech callouts, and model badges. It is never a CTA color and never used as a background fill — the tricolor is exclusively a brand-identity marker.

Type voice runs **BMW Type Next Latin** in two cuts: regular for display + nav labels and Light for body + secondary copy. Display sizes use weight 700 (BMW's signature heavy-but-tight setting), while body type drops to weight 300 (Light). The contrast between heavy display and light body is the system's editorial signature.

**Key Characteristics:**
- Near-pure black canvas (`{colors.canvas}` — #000) with white type. The system inverts almost nothing — there is no light-mode marketing surface.
- Display headlines in UPPERCASE BMW Type Next Latin at weight 700. Sub-heads stay sentence-case at lighter weight.
- M tricolor (`{colors.m-blue-light}` / `{colors.m-blue-dark}` / `{colors.m-red}`) used as 4px brand-stripe dividers, M-wordmark accents, and motorsport chrome — never as buttons or fills.
- Photography fills entire bands edge-to-edge. Cars are always the visual subject; UI chrome backs off to small white labels overlaid on photography.
- Buttons are flat with `{rounded.none}` (0px) corners and uppercase letterspaced labels. The "industrial precision" rectangular silhouette IS the brand.
- Border radius is mostly zero across the system. The few exceptions: `{rounded.full}` on circular icon buttons (carousel arrows, chatbot launcher) and `{rounded.sm}` on a handful of small toggle pills.
- Spacing is generous and grid-aligned: `{spacing.section}` (96px) between major bands; `{spacing.xxl}` (64px) inside hero photo bands; `{spacing.xl}` (40px) inside content cards.

## Colors

### Brand & Accent
- **Primary** (`{colors.primary}` — #ffffff): The system's primary type and CTA color. Used for h1/h2/h3 display, body text on dark, and primary button labels (the buttons themselves are transparent or canvas-colored — the white text + outline IS the button).
- **M Blue Light** (`{colors.m-blue-light}` — #0066b1): The first stop in the M tricolor stripe. Used on M-badge accents and motorsport chrome.
- **M Blue Dark** (`{colors.m-blue-dark}` — #1c69d4): The middle stop. The same hex as `{colors.bmw-blue}` — BMW's heritage corporate blue, repurposed as the middle band of the M stripe.
- **M Red** (`{colors.m-red}` — #e22718): The third stop. The signature M-power red, used in the stripe and on motorsport-pace callouts.
- **Electric Blue** (`{colors.electric-blue}` — #0653b6): A separate electric-vehicle accent used on M xDrive electric model pages. Distinct from the heritage blue — feels colder, more digital.

### Surface
- **Canvas** (`{colors.canvas}` — #000000): The default page floor across every marketing surface. True black.
- **Surface Soft** (`{colors.surface-soft}` — #0d0d0d): A barely-different-from-black used for spec table cells and footer-adjacent strips.
- **Surface Card** (`{colors.surface-card}` — #1a1a1a): Cards, secondary buttons, icon-button backgrounds.
- **Surface Elevated** (`{colors.surface-elevated}` — #262626): One step lighter, used for nested cards inside dark bands.
- **Carbon Gray** (`{colors.carbon-gray}` — #2b2b2b): Carbon-fiber-inspired surface tone used on technical-spec cards.

### Hairlines & Borders
- **Hairline** (`{colors.hairline}` — #3c3c3c): The 1px divider tone on dark surfaces. Used between body sections, between table rows, around card outlines.
- **Hairline Strong** (`{colors.hairline-strong}` — #262626): Same hex as `{colors.surface-elevated}` — borders feel like one-step elevations rather than ink lines.

### Text
- **Ink / On Dark** (`{colors.on-dark}` — #ffffff): All headline and primary text on dark canvas.
- **Body** (`{colors.body}` — #bbbbbb): Default running-text color (slightly cooler than pure white). Used for body paragraphs and secondary metadata.
- **Body Strong** (`{colors.body-strong}` — #e6e6e6): Emphasized body / lead paragraph.
- **Muted** (`{colors.muted}` — #7e7e7e): Footer links, breadcrumbs, captions.

### Semantic
- **Warning** (`{colors.warning}` — #f4b400): Used very sparingly on technical-warning callouts.
- **Success** (`{colors.success}` — #0fa336): Order-confirmation states (rare on marketing surfaces).

## Typography

### Font Family
**BMW Type Next Latin** is BMW's licensed display + body typeface. The system uses two cuts: regular and Light. The fallback stack walks `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`.

The split is a deliberate weight-pair:
- Display (700) for headlines, navigation labels, button text, and category labels — the "stamped" voice
- Light (300) for body paragraphs, descriptive copy, and secondary metadata — the "engineered" voice

The contrast between heavy display and light body is BMW's editorial signature — never blur it by using regular (400) display or medium (500) body.

### Hierarchy

| Token | Size | Weight | Line Height | Letter Spacing | Use |
|---|---|---|---|---|---|
| `{typography.display-xl}` | 80px | 700 | 1.0 | 0 | Hero h1 ("THE ULTIMATE", "MORE BMW M.") |
| `{typography.display-lg}` | 56px | 700 | 1.05 | 0 | Section heads ("MORE FROM BMW M MAGAZINE.") |
| `{typography.display-md}` | 40px | 700 | 1.1 | 0 | Sub-section heads, model names |
| `{typography.display-sm}` | 32px | 700 | 1.15 | 0 | CTA-band heads, category page titles |
| `{typography.title-lg}` | 24px | 700 | 1.3 | 0 | Card titles in 3-up grids |
| `{typography.title-md}` | 20px | 400 | 1.4 | 0 | Card sub-titles, lead paragraphs |
| `{typography.title-sm}` | 18px | 400 | 1.4 | 0 | Spec callouts, intro paragraphs |
| `{typography.label-uppercase}` | 14px | 700 | 1.3 | 1.5px | Category tabs, "VIEW MORE" inline labels |
| `{typography.body-md}` | 16px | 300 (Light) | 1.5 | 0 | Default body — BMW Type Next Latin Light |
| `{typography.body-sm}` | 14px | 300 (Light) | 1.5 | 0 | Footer body, cookie consent, fine print |
| `{typography.caption}` | 12px | 400 | 1.4 | 0.5px | Photo captions, image-credit lines |
| `{typography.button}` | 14px | 700 | 1.0 | 1.5px | All button labels — uppercase, letterspaced |
| `{typography.nav-link}` | 14px | 400 | 1.4 | 0.5px | Top-nav menu items |

### Principles
The system contrasts heavy headlines (700) against very light body (300) at all times — the gap is the editorial signature. Letter-spacing is non-trivial: button labels and category labels carry 1.5px tracking that makes them feel "machined" rather than "typed." Display headlines stay at 0 letter-spacing — BMW Type's natural cap-height handles spacing on large sizes.

UPPERCASE display is the default voice for h1/h2 — sentence case appears on body and intro paragraphs but rarely on headlines. The all-caps treatment is a brand-voice signal, not a stylistic choice.

### Note on Font Substitutes
If BMW Type Next Latin is unavailable, **Inter** (variable) at 700/300 is the closest open-source substitute. Adjust display headline tracking to -0.5px to match BMW Type's tighter spacing at large sizes. **Saira Condensed** is an alternative for headlines if a slightly more compressed feel is desired.

## Layout

### Spacing System
- **Base unit:** 4px.
- **Tokens:** `{spacing.xxs}` 4px · `{spacing.xs}` 8px · `{spacing.sm}` 12px · `{spacing.md}` 16px · `{spacing.lg}` 24px · `{spacing.xl}` 40px · `{spacing.xxl}` 64px · `{spacing.section}` 96px.
- **Section padding (vertical):** `{spacing.section}` (96px) between major editorial bands.
- **Hero photo bands:** `{spacing.xxl}` (64px) internal vertical padding around the hero h1 + sub-headline pair.
- **Card internal padding:** `{spacing.lg}` (24px) for content and model cards; `{spacing.xl}` (40px) for spec-cell tables.
- **Gutters:** `{spacing.lg}` (24px) between cards in 3-up grids; `{spacing.md}` (16px) inside footer columns.

### Grid & Container
- **Max content width:** ~1440px centered on marketing pages — wider than typical SaaS to give photography breathing room.
- **Editorial body:** Single 12-column grid; photo bands bleed full-bleed (no max-width).
- **Card grids:** 3-up at desktop, 2-up at tablet, 1-up at mobile.
- **Footer:** 4-column link list at desktop, 2-up at tablet, 1-up at mobile.

### Whitespace Philosophy
BMW M trusts photography to do the visual work. Whitespace around photography is restrained — the cars fill the frame, and copy sits below or beside them in tightly-aligned columns. Where whitespace appears (between body sections, around CTAs), it's always uniform `{spacing.section}` (96px). The system never adds atmospheric backdrops, gradients, or decoration — empty space stays as empty black canvas.

## Elevation & Depth

| Level | Treatment | Use |
|---|---|---|
| Flat | No shadow, no border | Body sections, top nav, footer, photo bands |
| Soft hairline | 1px `{colors.hairline}` border | Section dividers, card outlines, table rows |
| Card surface | `{colors.surface-card}` background over canvas — no shadow | Feature photo cards, magazine cards, chatbot launcher |
| Photographic depth | Full-bleed photography with edge-to-edge crop | Hero bands, motorsport features — depth via subject matter, not chrome |

The system uses no drop shadows and no layered chrome. Depth comes entirely from photography (subject + lens + lighting) and the contrast between black canvas and slightly-elevated `{colors.surface-card}`.

### Decorative Depth
- **M Stripe Divider** (`{component.m-stripe-divider}`): A 4px-tall horizontal divider carrying the M tricolor (`{colors.m-blue-light}` → `{colors.m-blue-dark}` → `{colors.m-red}`). Used on motorsport chrome, model-detail headers, and brand-identity moments. The stripe is the system's only true "decorative" element — used sparingly to mark significance.
- **Carbon-fiber surfaces**: The technical-spec page uses `{colors.carbon-gray}` (#2b2b2b) cells with subtle texture overlay. This is a single-page treatment, not a system-wide pattern.
- **Photographic depth**: Full-bleed cars are the depth. Lighting in the photography (track lights, sunset rim-light) does the elevation work that drop shadows would do in a SaaS system.

## Shapes

### Border Radius Scale

| Token | Value | Use |
|---|---|---|
| `{rounded.none}` | 0px | All buttons, cards, photo containers, spec cells, inputs — the dominant radius |
| `{rounded.xs}` | 2px | Almost no use — reserved for legal CTAs |
| `{rounded.sm}` | 4px | Small toggle pills on configurator surfaces |
| `{rounded.md}` | 6px | Rare — small dropdown menu items |
| `{rounded.full}` | 9999px / 50% | Circular icon buttons, carousel arrows, chatbot launcher |

The radius hierarchy is "almost always 0, sometimes circular." This binary radius decision is a deliberate brand-language choice — sharp rectangles read as engineered precision; circles read as functional controls. Nothing in between.

### Photography Geometry
Hero photography fills full-width with no rounding. Photo cards inside grids retain `{rounded.none}` corners, edge-to-edge images. Carbon-wheel detail shots and motorsport-pit photos use 16:9 or 21:9 cinema-aspect ratios. Driver portraits in racing-team grids use 4:5 portrait crops, also with sharp corners.

## Components

### Top Navigation

**`top-nav`** — Black nav bar pinned to the top of every page. 64px tall, `{colors.canvas}` background. Carries the BMW M logo at left (M tricolor + BMW roundel + "M" wordmark), primary horizontal menu (Models, Topics, Magazine, Configurator, Fastlane), right-side cluster with language selector, search icon, account icon. Menu items render in `{typography.nav-link}` with sentence-case labels.

### Buttons

**`button-primary`** — The signature primary CTA. Background `{colors.canvas}` (or transparent over photography), text `{colors.on-dark}` (white), 1px white border outline, rounded `{rounded.none}` (0px), padding 16px × 32px, height 48px. Type `{typography.button}` — uppercase 14px / 700 / 1.5px tracking. The rectangular silhouette and uppercase letterspaced label IS the brand button.

**`button-primary-outline`** — Same shape as primary but with transparent background and white outline only. Used over photography where a filled button would clash with the image.

**`button-on-light`** — Used on rare light-surface contexts (configurator, account dialogs). Background `{colors.canvas}`, text `{colors.on-dark}` — black button with white text, inverted from the dark-canvas default.

**`button-icon`** — Circular icon buttons (carousel controls, share, favorite). 48 × 48px, background `{colors.surface-card}`, white icon centered, rounded `{rounded.full}`. The only non-rectangular button shape in the system.

**`carousel-arrow`** — Specific 48 × 48 circular arrow used in photo carousels. Same shape as `{component.button-icon}` with chevron glyph.

**`text-link`** — Inline uppercase letterspaced links ("VIEW ALL MODELS", "READ MORE"). `{typography.label-uppercase}`, white on dark, no underline. The chevron arrow → glyph appears next to most link labels.

### Cards & Containers

**`hero-photo-band`** — Full-width black band with full-bleed automotive photography filling most of the frame. The h1 uses `{typography.display-xl}` (80px / 700) and sits left-aligned over the photo, often with a small subtitle in `{typography.body-md}` below. Vertical padding `{spacing.xxl}` (64px). No card frame — the photo IS the band.

**`feature-photo-card`** — Used in 3-up grids for "MORE FROM BMW M MAGAZINE" and similar editorial sections. Background `{colors.surface-card}`, rounded `{rounded.none}`, internal padding `{spacing.lg}` (24px). Top half of the card is a 16:9 photo (full-bleed within the card); below the photo, a category tag in `{typography.label-uppercase}`, a `{typography.title-lg}` title, and a short body description.

**`model-card`** — Used in the "MORE NEW M MODELS" 3-up grid. Background `{colors.canvas}` (no card surface — just photo on black), rounded `{rounded.none}`. Top: 16:10 hero shot of the model. Below: model name in `{typography.display-md}` (40px / 700), short specs line in `{typography.body-sm}`, a `{component.text-link}` ("EXPLORE THIS MODEL").

**`magazine-article-card`** — A more text-forward card variant used on the magazine overview page. Background `{colors.canvas}` with hairline border, rounded `{rounded.none}`. Carries a small thumbnail at top, a category label in `{typography.label-uppercase}`, headline in `{typography.title-lg}`, and a body excerpt.

**`spec-cell`** — Technical specification cells used on model-detail pages (engine specs, weight, top speed, 0-100 time). Background `{colors.surface-soft}` (#0d0d0d), rounded `{rounded.none}`, padding `{spacing.lg}` (24px). Each cell holds a value in `{typography.display-sm}` (32px / 700) at top and a label in `{typography.label-uppercase}` below.

**`motorsport-photo-card`** — Edge-to-edge photo cards used in the racing-team / motorsport sections. No card surface — just a full-bleed photograph with a small overlay caption in white text at the bottom-left. The photography IS the brand here.

**`chatbot-launcher`** — A right-side card-style entry point ("BMW M CHATBOT") on the homepage. Background `{colors.surface-card}`, rounded `{rounded.none}`, padding `{spacing.lg}` (24px). Carries an h3 title, a short prompt, and a `{component.button-primary}` to launch.

**`category-tab`** + **`category-tab-active`** — The category selector tabs used on the magazine and topics pages (e.g., "ALL · MAGAZINE · MODELS · LIFESTYLE · MOTORSPORT"). Tabs render as text-only labels in `{typography.label-uppercase}`. Active state changes text color from `{colors.body}` to `{colors.on-dark}` and adds a 2px white underline below the label. No background fill, no rounded corners.

### Inputs & Forms

**`text-input`** — Standard text input on dark surfaces. Background `{colors.surface-card}`, text `{colors.on-dark}`, type `{typography.body-md}`, rounded `{rounded.none}` (0px), padding 12px × 16px, height 48px. 1px hairline border. Focus state thickens the border to white.

**`cookie-consent-card`** — A right-side cookie-banner card visible on the homepage. Background `{colors.canvas}` with 1px hairline, rounded `{rounded.none}`, padding `{spacing.lg}` (24px). Body text in `{typography.body-sm}` (14px / 300) — Light weight even for legal text. Two buttons stacked at bottom: primary outline + text-link.

### Signature Components

**`m-stripe-divider`** — The 4px horizontal stripe carrying the M tricolor (`{colors.m-blue-light}` → `{colors.m-blue-dark}` → `{colors.m-red}`). Used as a divider on motorsport chrome, between brand-identity sections, and as a hover-state indicator on category tabs. The most distinctive non-typographic element in the system.

**`cta-band-photo`** — A pre-footer "Drive an M" CTA band carrying full-bleed photography of a car cornering on a track, with a centered headline in `{typography.display-md}` and a `{component.button-primary-outline}` below. Vertical padding 80px. The CTA inherits the editorial gravity of the rest of the page through full-bleed photography rather than chrome.

### Footer

**`footer`** — Black footer that closes every page. Background `{colors.canvas}`, text `{colors.body}`. 4-column link list at desktop covering BMW M Models / BMW M Lifestyle / Owners / Company. Vertical padding 64px. Bottom row carries the BMW corporate disclaimer in `{typography.caption}` and language selector. The footer never inverts — it stays black even when the body might transition.

## Do's and Don'ts

### Do
- Anchor every page with full-bleed automotive photography. The cars are the brand voltage; chrome backs off.
- Use UPPERCASE display headlines in `{typography.display-xl}` or `{typography.display-lg}`. Sentence-case display reads as off-brand.
- Pair heavy display (700) with light body (300). The weight contrast is the editorial signature.
- Reserve the M tricolor stripe for brand-identity moments — wordmark accents, motorsport chrome, model badges. Never as a button fill or surface.
- Use `{rounded.none}` (0px) by default. Reserve `{rounded.full}` for circular icon buttons only.
- Letter-space all-caps labels at 1.5px. The "machined" feel is non-negotiable.
- Use `{spacing.section}` (96px) between major editorial bands for grid-aligned vertical rhythm.

### Don't
- Don't introduce a brand color outside the M tricolor (`{colors.m-blue-light}` / `{colors.m-blue-dark}` / `{colors.m-red}`) and the heritage `{colors.bmw-blue}`.
- Don't bold body type. Body stays at 300 (Light) — bumping to 400 or 500 makes the page feel marketing-bombastic instead of European-engineered.
- Don't use rounded buttons. The rectangular silhouette IS the brand. Rounded corners read as consumer-tech, not motorsport.
- Don't put gradient backdrops behind hero type. The hero IS the photography — the page floor stays pure black, and the photo provides the depth.
- Don't repeat the same surface mode in two consecutive bands. Rhythm: photo band → spec table → photo band → magazine grid → photo band. Two text-only bands in a row read as a corporate site.
- Don't use the M stripe as a button fill. The stripe is a divider / accent — never an action surface.
- Don't bold uppercase tracking under 1.5px on button labels — the spacing is what makes them feel "machined."

## Responsive Behavior

### Breakpoints

| Name | Width | Key Changes |
|---|---|---|
| Mobile | < 768px | Hamburger nav; hero h1 scales 80→48px; demo grid 1-up; photo cards stack full-width; footer 4 cols → 1 |
| Tablet | 768–1024px | Top nav stays horizontal but tightens; 2-up card grids; spec tables 2-up |
| Desktop | 1024–1440px | Full top-nav; 3-up card grids; spec tables 4-up |
| Wide | > 1440px | Same as desktop with more breathing room; max content 1440px |

### Touch Targets
- `{component.button-primary}` renders at 48 × 48px minimum — meets WCAG AAA.
- `{component.button-icon}` and `{component.carousel-arrow}` are exactly 48 × 48 — comfortably above the 44 × 44 minimum.
- `{component.text-input}` height is 48px.
- Category tabs render as text-only labels with 12px vertical padding; effective tap area meets 44px with surrounding spacing.

### Collapsing Strategy
- Top nav collapses to a hamburger sheet at < 768px; the menu opens as a full-screen black overlay with the M tricolor stripe at the top.
- Photography stays full-bleed at every breakpoint — never collapses to a margin'd container.
- Card grids reduce columns rather than scaling cards down; photography retains its native aspect ratio.
- Spec tables collapse from 4-up to 2-up to 1-up; spec values stay at `{typography.display-sm}` regardless of column count.
- The M-stripe divider stays at 4px height across all breakpoints.

### Image Behavior
- Hero photography crops responsively — wider crops at desktop, vertical crops on mobile.
- Lifestyle and motorsport photos retain native aspect ratios; the system never letterboxes or pillarboxes.
- The M wordmark + tricolor logo scales proportionally with viewport width.

## Iteration Guide

1. Focus on ONE component at a time. Reference its YAML key (`{component.hero-photo-band}`, `{component.spec-cell}`).
2. New components default to `{rounded.none}` (0px). Only use `{rounded.full}` if it's a circular icon button.
3. Variants (`-active`, `-disabled`) live as separate entries in `components:`.
4. Use `{token.refs}` everywhere — never inline hex.
5. Never document hover states. Default and Active/Pressed only.
6. Display headlines stay UPPERCASE 700; body stays sentence-case 300. Never blur the contrast.
7. The M tricolor is brand-identity-only — never extend it to system tokens for "primary action."
8. When in doubt about emphasis: bigger photography before bigger type.

## Known Gaps

- The dembrandt frequency analyzer captured the white text (count 955) as the highest-frequency token. The black canvas was inferred from screenshot — dembrandt's body-background sampling didn't surface it as a top palette entry, but the page is unambiguously black-on-white-text.
- The exact M tricolor stops are documented from public BMW brand guidelines; the screenshots show the stripe as a small element but pixel-sampling at this resolution doesn't reliably distinguish #0066b1 from #1c69d4. Treat the documented stops as canonical based on BMW Design Works' published brand spec.
- BMW Type Next Latin weight axis values beyond Light (300) and regular (700) are not documented — only the static weights observed in screenshots.
- Animation and transition timings (photo carousel transitions, hover-reveal effects, configurator interactions) are not in scope.
- Form validation states beyond `{component.text-input}` defaults are not extracted — error / success input variants would need a configurator or order flow to confirm.
- The configurator surface (vehicle build pages with color / wheel / interior pickers) was not in the analyzed URL set; its swatch grid, comparison panels, and price-summary card are not documented here.
- The cookie consent overlay obscured part of the homepage hero in the captured screenshot; secondary hero treatments (different car models cycling through the hero band) may carry variations not captured.
