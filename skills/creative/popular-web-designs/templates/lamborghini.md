# Design System: Lamborghini


> **Hermes Agent — Implementation Notes**
>
> Source analysis adapted from [`VoltAgent/awesome-design-md`](https://github.com/VoltAgent/awesome-design-md) at commit `664b3e78`; see `../references/ATTRIBUTION.md` for the MIT notice.
>
> The original site uses proprietary or legacy fonts. For self-contained HTML output, use these CDN substitutes:
> - **Primary:** `Barlow Condensed` | **Mono:** `JetBrains Mono`
> - **Font stack (CSS):** `font-family: 'Barlow Condensed', system-ui, -apple-system, 'Segoe UI', sans-serif;`
> - **Mono stack (CSS):** `font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;`
> ```html
> <link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
> ```
> Use `write_file` to create HTML and open the rendered artifact with `browser_navigate`.
> Verify visual accuracy with `browser_vision` after generating.

## 1. Visual Theme & Atmosphere

Lamborghini's website is a cathedral of darkness — a digital stage where jet-black surfaces stretch infinitely and every element emerges from the void like a machine under a spotlight. The page is almost entirely black. Not dark gray, not near-black — true, uncompromising black (`#000000`) that saturates the viewport and refuses to yield. Into this abyss, white type and Lamborghini Gold (`#FFC000`) are deployed with surgical precision, creating a visual language that feels like walking through a nighttime motorsport event where every surface absorbs light except the things that matter.

The hero is a full-viewport video — dark, cinematic, immersive — showing event footage or vehicle reveals with the Lamborghini bull logo floating ethereally above. The navigation is minimal: a centered bull logo, a "MENU" hamburger on the left, and search/bookmark icons on the right, all rendered in white against the black canvas. There are no borders, no visible nav containers, no background color on the header — just white marks floating in darkness. The overall mood is nocturnal luxury: exclusive, theatrical, and deliberately intimidating. Each section transition is a scroll through darkness into the next revelation.

Typography is the voice of this darkness. LamboType — a custom Neo-Grotesk typeface created by Character Type and design agency Strichpunkt — is used for everything from 120px uppercase display headlines to 10px micro labels. Its distinctive 12° angled terminals are inspired by the aerodynamic lines of Lamborghini's super sports cars, and its proportions range from Normal to Ultracompressed width. Headlines SHOUT in uppercase at enormous scales with tight line-heights (0.92 at 120px), creating dense blocks of text that feel stamped from steel. The typeface carries hexagonal geometric DNA — constructed from hexagons, three-armed stars, and circles — that echoes throughout the interface in the hexagonal pause button and UI icons. Built on Bootstrap grid with 68 Element Plus/UI components, the technical infrastructure is substantial beneath the theatrical surface.

**Key Characteristics:**
- True black (`#000000`) dominant surfaces with white and gold as the only relief colors
- LamboType custom Neo-Grotesk font with 12° angled terminals inspired by aerodynamic car lines
- Lamborghini Gold (`#FFC000`) as the sole accent color — used exclusively for primary CTA buttons
- All-uppercase display typography at extreme scales (120px, 80px, 54px) with tight line-heights
- Full-viewport video heroes with cinematic event/vehicle content
- Zero border-radius on buttons — sharp, angular, uncompromising rectangles
- Hexagonal motifs in UI elements (pause button, icon system) echoing brand geometry
- Bootstrap grid system + Element Plus/UI 68 components underneath
- Transparent ghost buttons with white borders at 50% opacity as the secondary CTA pattern

## 2. Color Palette & Roles

### Primary
- **Lamborghini Gold** (`#FFC000`): The signature accent color — a warm, saturated amber-gold (rgb 255, 192, 0) used exclusively for primary action buttons ("Discover More", "Tickets", "Start Configuration"). The only chromatic color in the entire interface, it ignites against the black canvas like a headlight cutting through night
- **Pure White** (`#FFFFFF`): Primary text color on dark surfaces, logo rendering, nav elements, and light-mode button fills — the voice that speaks from the darkness

### Secondary & Accent
- **Dark Gold** (`#917300`): Hover/pressed state for gold buttons — a deep amber (rgb 145, 115, 0) that darkens the gold to signal interaction
- **Gold Text** (`#FFCE3E`): Slightly lighter gold variant (rgb 255, 206, 62) used for inline text accents and highlighted labels
- **Cyan Pulse** (`#29ABE2`): Electric blue-cyan (rgb 41, 171, 226) appearing as an informational accent and interactive element highlight
- **Link Blue** (`#3860BE`): Medium blue (rgb 56, 96, 190) used universally for link hover states across all text colors

### Surface & Background
- **Absolute Black** (`#000000`): The dominant surface color — used for page background, hero sections, header, footer, and most containers
- **Charcoal** (`#202020`): Elevated dark surface (rgb 32, 32, 32) — the primary "dark gray" for cards, panels, and text containers sitting above the black canvas
- **Dark Iron** (`#181818`): Subtle surface variant (rgb 24, 24, 24) — barely distinguishable from black, used for footer and deep sections
- **Overlay Black** (`rgba(0,0,0,0.7)`): Semi-transparent overlay for modals and video dimming
- **Near White** (`#F8F8F8`): Rare light surface (rgb 248, 248, 248) for content blocks in white-mode sections
- **Mist** (`#E6E6E6`): Light gray surface for secondary light-mode containers

### Neutrals & Text
- **Pure White** (`#FFFFFF`): Primary text on dark backgrounds — headlines, body, nav labels
- **Smoke** (`#F5F5F5`): Secondary text on dark surfaces — slightly softer than pure white
- **Graphite** (`#494949`): Dark gray text on light surfaces (rgb 73, 73, 73)
- **Ash** (`#7D7D7D`): Mid-range gray for muted text, timestamps, and metadata (rgb 125, 125, 125)
- **Steel** (`#969696`): Lighter gray for disabled text and subtle labels (rgb 150, 150, 150)
- **Slate** (`#666666`): Alternative mid-gray for secondary content
- **Iron** (`#555555`): Dark mid-gray for body text variants
- **Shadow** (`#313131`): Very dark gray for text on dark surfaces where white is too strong

### Semantic & Accent
- **Cyan Pulse** (`#29ABE2`): Used for informational highlights and interactive feedback
- **Link Blue** (`#3860BE`): Universal hover state for all hyperlinks
- **Teal Action** (`#1EAEDB`): Button hover background for transparent/ghost variants (rgb 30, 174, 219)

### Gradient System
- No explicit gradients in the color palette — the dark-to-light progression is achieved through surface layering: `#000000` → `#181818` → `#202020` → `#494949` → `#7D7D7D`
- Video heroes use natural atmospheric gradients from the content itself
- Top-of-page gradient: subtle dark-to-darker fade at the edges of full-bleed imagery

## 3. Typography Rules

### Font Family
- **Display & UI**: `LamboType`, Roboto, Helvetica Neue, Arial — custom Neo-Grotesk typeface by Character Type for Lamborghini's 2024 brand refresh. Available in widths from Normal to Ultracompressed and weights from Light (300) to Black. Features 12° angled terminals inspired by aerodynamic car geometry, hexagonal construction logic, and support for 200+ languages including Latin, Cyrillic, and Greek
- **Fallback/UI**: `Open Sans` — used for some button/form contexts as system fallback
- **No italic variants** observed on the marketing site — the brand voice is always upright

### Hierarchy

| Role | Size | Weight | Line Height | Letter Spacing | Notes |
|------|------|--------|-------------|----------------|-------|
| Hero Display | 120px (7.50rem) | 400 | 0.92 | normal | LamboType, uppercase, maximum impact |
| Display 2 | 80px (5.00rem) | 400 | 1.13 | normal | LamboType, uppercase, major section titles |
| Section Title | 54px (3.38rem) | 400 | 1.19 | normal | LamboType, uppercase |
| Sub-section | 40px (2.50rem) | 400 | 1.15 | normal | LamboType, uppercase |
| Feature Heading | 27px (1.69rem) | 400 | 1.37 | normal | LamboType, uppercase |
| Card Title | 24px (1.50rem) | 400 | — | normal | LamboType |
| Body Large | 18px (1.13rem) | 400 | 1.56 | normal | LamboType, mixed case and uppercase variants |
| Body / UI | 16px (1.00rem) | 400/700 | 1.50 | normal/0.16px | LamboType, primary body text |
| Button Large | 16px (1.00rem) | 400 | 1.50 | normal | Gold CTA buttons |
| Button Standard | 14.4px (0.90rem) | 300/700 | 1.00 | 0.14–0.2px | LamboType, uppercase, ghost buttons |
| Button Small | 13px (0.81rem) | 300/500 | 1.20 | 0.13–0.2px | LamboType, compact button variant |
| Caption | 14px (0.88rem) | 600/700 | 1.14–1.50 | -0.42px | LamboType, uppercase, negative tracking |
| Label | 12px (0.75rem) | 400/500 | 1.83 | 0.96px | LamboType, uppercase badges and micro labels |
| Micro | 10px (0.63rem) | 400 | 1.00–2.00 | 0.225px | LamboType, uppercase, smallest text |

### Principles
- **ALL-CAPS is the default voice**: Display and feature headings are universally uppercase. This creates a shouting, commanding tone that matches the brand's aggression
- **Extreme scale range**: From 120px heroes to 10px micro labels — a 12:1 ratio that creates dramatic visual hierarchy
- **Tight line-heights at scale**: Display sizes use 0.92-1.19 line-height, creating dense, compressed blocks of type that feel stamped rather than typeset
- **Weight 400 dominates**: Unlike many design systems that use bold for emphasis, Lamborghini's regular weight carries the headlines — the typeface itself is so distinctive it doesn't need weight variation
- **Negative tracking on captions**: -0.42px letter-spacing on 14px captions creates a compressed, technical aesthetic
- **Positive tracking on micro text**: +0.225px at 10px ensures legibility at the smallest sizes
- **Single typeface discipline**: LamboType handles everything — the 12° angled terminals and hexagonal geometry provide visual coherence across all sizes

## 4. Component Stylings

### Buttons
All buttons use **zero border-radius** — sharp, angular rectangles that echo the aggressive lines of Lamborghini vehicles.

**Gold Accent CTA** — The primary action:
- Default: bg `#FFC000` (Lamborghini Gold), text `#000000`, padding 24px, fontSize 16px, fontWeight 400, borderRadius 0px, no border
- Hover: bg `#917300` (Dark Gold), darkens significantly
- Class: `btn-accent btn-large`
- Used for: "Discover More", "Tickets", "Start Configuration"

**Transparent Ghost** — The secondary action on dark backgrounds:
- Default: bg transparent, text `#FFFFFF`, border 1px solid `#FFFFFF`, padding 16px, opacity 0.5
- Hover: bg `#1EAEDB` (Teal Action), text white, opacity 0.7
- Focus: bg `#1EAEDB`, border 1px solid `#000000`, outline 2px solid `#000000`
- Used for: secondary CTAs on hero sections and dark panels

**White Filled** — Light-mode primary:
- Default: bg `#FFFFFF`, text `#202020`, no border
- Used for: CTAs on dark sections where gold isn't appropriate

**Black Filled** — Dark filled variant:
- Default: bg `#000000`, text `#202020`
- Used for: Inverted CTA on light sections

**Gray Neutral** — Subtle action:
- Default: bg `#969696`, text `#202020`
- Used for: secondary/tertiary actions, badge-like buttons

### Cards & Containers
- Background: `#202020` (Charcoal) on black canvas, or `#000000` on lighter sections
- Border: `0px 1px solid #202020` bottom borders for section dividers
- Border-radius: 0px (completely sharp corners)
- Shadow: minimal, uses overlay opacity for depth
- Content: full-bleed photography + overlaid text in white

### Inputs & Forms
- Minimal form presence on the marketing site
- Switch elements: border-radius 20px (the only rounded element), border 1px solid `#DDDDDD`
- Cookie banner input style: white text on black with `#7D7D7D` borders

### Navigation
- **Desktop**: Centered bull logo, "MENU" hamburger with icon on left, search icon + bookmarks icon on right
- **Background**: Transparent (inherits black page background)
- **Sticky**: Fixed to top, floats above content
- **No visible borders or shadows** — elements float in the darkness
- **"MENU" label**: White text at 14px weight 400, uppercase, accompanies hamburger icon
- **Hexagonal motifs**: Pause button on hero sections uses hexagonal outline shape

### Image Treatment
- **Hero**: Full-viewport video sections (100vh) with cinematic event/vehicle footage
- **Event photography**: Full-bleed aerial shots of Lamborghini Arena events
- **Vehicle imagery**: High-contrast studio shots on dark backgrounds, full-width
- **Aspect ratios**: Predominantly 16:9 and wider for cinematic feel
- **Dark gradient overlays**: Subtle darkening at top/bottom edges of video to ensure text legibility

### Distinctive Components
- **Hexagonal Pause Button**: Video control uses a hexagonal outline (matching the brand's geometric DNA from the typeface), positioned bottom-right of hero sections
- **Progress Bar**: Thin white line at bottom of hero sections indicating video/slide progress
- **Badge/Tag**: bg `#969696`, text white, padding 8px, fontSize 10px, borderRadius 2px — tiny metallic pills

## 5. Layout Principles

### Spacing System
- **Base unit**: 8px
- **Full scale**: 2px, 4px, 5px, 8px, 10px, 12px, 15px, 16px, 20px, 24px, 32px, 40px, 48px, 56px
- **Button padding**: 16px (ghost), 24px (gold accent)
- **Section padding**: 48–56px vertical, 40px horizontal
- **Small spacing**: 2–5px for fine adjustments (badge padding, border spacing)

### Grid & Container
- **Framework**: Bootstrap grid system (container + row + col)
- **Max width**: 1440px (largest breakpoint)
- **Columns**: Standard 12-column Bootstrap grid
- **Full-bleed**: Hero sections break out of grid to fill viewport edge-to-edge
- **Content areas**: Centered within 1200px max-width containers

### Whitespace Philosophy
Lamborghini uses darkness as whitespace. The generous black expanses between content blocks serve the same function as white space in a light design — creating breathing room that elevates each element to the status of exhibit. A model name floating in the middle of a black viewport has the same visual weight as a gallery piece on a white wall. The absence of color IS the design.

### Border Radius Scale
| Value | Context |
|-------|---------|
| 0px | Default for everything — buttons, cards, containers, images |
| 1px | Subtle span elements |
| 2px | Badges, close buttons, cookie elements — barely perceptible |
| 20px | Toggle switches only — the sole rounded element |

## 6. Depth & Elevation

| Level | Treatment | Use |
|-------|-----------|-----|
| Level 0 (Abyss) | `#000000` flat | Page background, deepest layer |
| Level 1 (Surface) | `#181818` or `#202020` | Cards, content panels, elevated sections |
| Level 2 (Overlay) | `rgba(0,0,0,0.7)` | Modal backdrops, video dimming |
| Level 3 (Fog) | `rgba(0,0,0,0.5)` | Lighter overlays, hover states |
| Level 4 (Mist) | `rgba(0,0,0,0.25)` | Subtle depth hints |

### Shadow Philosophy
Lamborghini achieves depth through surface color layering rather than shadows. On a black canvas, traditional drop shadows are invisible — instead, the system creates elevation by shifting from absolute black to progressively lighter dark grays: `#000000` → `#181818` → `#202020` → `#494949`. This "darkness gradient" approach means that elevated elements are literally lighter than their surroundings, inverting the traditional shadow model.

### Decorative Depth
- Full-bleed video provides atmospheric depth through cinematic lighting
- The hexagonal pause button floats with a thin white outline stroke
- Progress bars at hero section bottoms create a subtle horizon line
- No gradients, glows, or blur effects on UI elements — the photography provides all visual richness

## 7. Do's and Don'ts

### Do
- Use absolute black (`#000000`) as the primary background — never dark gray as a substitute
- Apply Lamborghini Gold (`#FFC000`) exclusively for primary CTA buttons — never for decorative purposes
- Set all display headings in uppercase with LamboType — the brand voice is always SHOUTING
- Use zero border-radius on buttons and cards — sharp angles are non-negotiable
- Maintain tight line-heights (0.92–1.19) on display type to create dense, architectural text blocks
- Use the transparent ghost button (white border, 50% opacity) as the secondary CTA on dark backgrounds
- Let full-viewport video/photography carry emotional weight — UI is infrastructure, not decoration
- Reserve hexagonal geometry for UI icons and the video control button
- Use weight 400 (regular) for headlines — the typeface is distinctive enough without bold emphasis
- Keep the gray palette achromatic — all neutrals are pure gray without color tinting

### Don't
- Introduce additional accent colors beyond gold — the monochrome-plus-gold system is sacred
- Apply border-radius to buttons or cards — curved edges contradict the angular vehicle aesthetic
- Use LamboType in italic or decorative styles — the brand is always upright and direct
- Add gradients to buttons or surfaces — depth comes from surface layering, not blending
- Use light backgrounds as the primary canvas — darkness is the default state, light is the exception
- Mix lowercase into display headings — the uppercase convention communicates authority and power
- Add hover animations with scale or translate — interactions should be color-only (background/opacity shifts)
- Use Open Sans for display text — LamboType must handle all visible typography
- Create busy layouts with many small elements — Lamborghini's design is about singular, bold statements
- Apply shadows to elements — on a black canvas, shadows are meaningless; use surface color shifts instead

## 8. Responsive Behavior

### Breakpoints
| Name | Width | Key Changes |
|------|-------|-------------|
| Mobile Small | <425px | Single column, reduced type scale, stacked buttons |
| Mobile | 425-576px | Single column, hamburger nav, hero text ~40px |
| Tablet Small | 576-768px | 2-column grid begins, padding adjusts |
| Tablet | 768-1024px | 2-column layout, expanded hero, vehicle cards side-by-side |
| Desktop | 1024-1280px | Full navigation, 3+ column grids, display text at 80px |
| Desktop Large | 1280-1440px | Full layout, hero at 120px display, max-width containers |
| Wide | >1440px | Content centered, margins expand, hero fills viewport |

### Touch Targets
- Gold CTA buttons: 48px+ minimum height with 24px padding (exceeds WCAG 44×44px)
- Ghost buttons: 48px+ with 16px padding
- Hamburger menu: large touch target (~48px square)
- Hexagonal pause button: approximately 48px diameter

### Collapsing Strategy
- **Navigation**: Always hamburger-based ("MENU" + icon) — no horizontal nav expansion on any breakpoint
- **Hero video**: Maintains full-viewport height across all breakpoints, adjusting object-fit
- **Display type**: Scales from 120px (desktop) → 80px (tablet) → 54px/40px (mobile)
- **Button layout**: Side-by-side on desktop, stacks vertically on mobile
- **Grid columns**: 3-column → 2-column → 1-column progression
- **Section spacing**: Reduces from 56px → 40px → 24px vertical padding

### Image Behavior
- Hero videos use `object-fit: cover` to maintain cinematic framing at all sizes
- Vehicle images scale within their containers with maintained aspect ratios
- Event photography crops to viewport width on narrow screens
- Background images darken at edges to maintain text contrast on all viewports

## 9. Agent Prompt Guide

### Quick Color Reference
- Primary CTA: "Lamborghini Gold (#FFC000)"
- Background: "Absolute Black (#000000)"
- Surface: "Charcoal (#202020)"
- Heading text: "Pure White (#FFFFFF)"
- Body text: "Ash (#7D7D7D)"
- Link hover: "Link Blue (#3860BE)"
- Accent: "Cyan Pulse (#29ABE2)"
- Border: "Pure White (#FFFFFF) at 50% opacity"

### Example Component Prompts
- "Create a hero section with a full-viewport black background, the model name 'TEMERARIO' in LamboType at 120px uppercase weight 400 white text with 0.92 line-height, centered vertically, with a Lamborghini Gold (#FFC000) 'Discover More' button below — sharp corners, 0px radius, 24px padding, black text"
- "Design a transparent ghost button with 1px solid white border at 50% opacity, white text at 14.4px uppercase with 0.2px letter-spacing, padding 16px, on a black background — hover state changes to Teal Action (#1EAEDB) background with 70% opacity"
- "Build a navigation bar with zero visible background on absolute black, a centered bull logo, 'MENU' text label with hamburger icon on the left, and search + bookmark icons on the right — all in white, sticky position"
- "Create a news card grid on charcoal (#202020) background with white headlines at 27px uppercase, body text in #7D7D7D at 16px, and a white underlined 'Read More' link that turns #3860BE on hover"
- "Design a section divider using a 1px solid bottom border in #202020 on a black canvas — the elevation difference is purely through surface color shift, not shadow"

### Iteration Guide
When refining existing screens generated with this design system:
1. Focus on ONE component at a time — Lamborghini's system is extreme and every element must feel aggressive
2. Reference specific color names and hex codes from this document — the palette has only about 5 active colors
3. Use natural language descriptions, not CSS values — "sharp-cut golden rectangle" not "border-radius: 0px; background: #FFC000"
4. Describe the desired "feel" alongside specific measurements — "floating in total darkness" communicates the black canvas better than "background: #000000"
5. Remember that UPPERCASE IS THE DEFAULT — if text isn't uppercase at display sizes, it probably should be
