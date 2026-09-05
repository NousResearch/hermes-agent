# Design System: The Verge


> **Hermes Agent — Implementation Notes**
>
> Source analysis adapted from [`VoltAgent/awesome-design-md`](https://github.com/VoltAgent/awesome-design-md) at commit `664b3e78`; see `../references/ATTRIBUTION.md` for the MIT notice.
>
> The Verge uses four proprietary type roles; keep their substitutes separate:
> - **Condensed display:** `Anton` (Manuka substitute)
> - **UI and secondary headlines:** `DM Sans` (PolySans substitute)
> - **Uppercase labels/timestamps:** `JetBrains Mono` (PolySans Mono substitute)
> - **Editorial serif:** `Newsreader` (FK Roman substitute)
> - **Display stack:** `font-family: Anton, Impact, sans-serif;`
> - **UI stack:** `font-family: 'DM Sans', Helvetica, Arial, sans-serif;`
> - **Mono stack:** `font-family: 'JetBrains Mono', 'Courier New', monospace;`
> - **Editorial stack:** `font-family: Newsreader, Georgia, serif;`
> ```html
> <link href="https://fonts.googleapis.com/css2?family=Anton&family=DM+Sans:wght@300;500;700&family=JetBrains+Mono:wght@500;600&family=Newsreader:wght@400;500;600&display=swap" rel="stylesheet">
> ```
> Raise substituted Manuka display line-height from 0.80 to about 0.95 to prevent collisions.
> Use `write_file` to create HTML and open the rendered artifact with `browser_navigate`.
> Verify visual accuracy with `browser_vision` after generating.

## 1. Visual Theme & Atmosphere

The Verge's 2024 redesign feels like somebody wired a Condé Nast magazine to a chiptune soundboard. The canvas is almost-black (`#131313`), the headlines are built from a brutally heavy display face (Manuka) that runs up to 107px, and the whole page is peppered with acid-mint `#3cffd0` and ultraviolet `#5200ff` that behave less like brand colors and more like hazard tape. Story tiles are not quiet gray cards — they're saturated, full-bleed color blocks (yellow, pink, orange, blue, purple) that feel like pasted-up rave flyers arranged into a timeline. The mood is "developer console meets club night meets tech tabloid": serious enough to cover a congressional hearing, loud enough to review a synthesizer.

What makes this system unmistakable is the **StoryStream** timeline: a vertical feed where every post is a rounded rectangle — often 20–40px radius — filled edge-to-edge with color, framed by a thin border, and marked by a mono-uppercase timestamp on its left rail. Stories don't float on a grid; they stack on a dashed vertical rule like commits in a git log. Above that, a massive **"The Verge" wordmark** dominates the masthead in Manuka at hero scale, letting the reader know before any headline loads that this is editorial territory, not a template.

There is no "light mode" on the homepage — the dark canvas is the product, and the only time the palette inverts is when a single story tile takes a mint or yellow fill. The depth is almost entirely flat: **hairline 1px borders** (`#ffffff`, `#3cffd0`, or `#5200ff`) do the work that shadows would do on a Material-flavored site. Every container is either `#131313` with a 1px outline, a fully saturated accent block, or a slate-gray `#2d2d2d` secondary surface.

**Key Characteristics:**
- Near-black editorial canvas (`#131313`) as the default surface — no light mode on the homepage
- Acid-mint `#3cffd0` + ultraviolet `#5200ff` as hazard-tape accents, never quiet background wash
- Massive Manuka display headlines up to 107px — the single loudest type move in mainstream tech media
- Rounded pill-card everything: 20/24/30/40px corner radii, never square
- Fully saturated color-block story tiles (mint, purple, yellow, pink, orange, electric blue) on a dark page
- Timeline "StoryStream" feed with mono uppercase timestamps rather than a traditional magazine grid
- Flat depth — 1px borders in white, mint, purple do the work that shadows would do elsewhere

## 2. Color Palette & Roles

### Primary (Brand Hazards)
- **Jelly Mint** (`#3cffd0`): The Verge's signature acid-mint accent. Used as CTA button fill, link underlines, active tab borders, and high-attention story-tile backgrounds. Treat it as the visual equivalent of neon safety paint — applied sparingly to the most important element on screen.
- **Verge Ultraviolet** (`#5200ff`): The complementary brand hazard. Used for secondary color-block tiles, promotional spans, and the occasional outlined button. Often applied at 0.9 alpha to soften its cathode intensity.

### Secondary & Accent
- **Console Mint Border** (`#309875`): A darker variant of the jelly mint used on card outlines and button borders where pure mint would over-saturate.
- **Deep Link Blue** (`#3860be`): The link *hover* color — the one moment blue appears on the site. It replaces mint/white/black on hover across every link style.
- **Focus Cyan** (`#1eaedb`): Reserved for button focus rings. Never shown outside a keyboard-focus state.
- **Purple Rule** (`#3d00bf`): A darker ultraviolet variant used as the vertical border on StoryStream `<li>` items.

### Surface & Background
- **Canvas Black** (`#131313`): The default dark surface for the entire homepage. Almost-but-not-quite pure black — has just enough warmth to feel like a printed newsprint negative rather than an OLED void.
- **Surface Slate** (`#2d2d2d`): Secondary card background, used when a story tile doesn't need to be a saturated color block.
- **Image Frame** (`#313131`): The 1px border that wraps inline imagery.
- **Hazard White** (`#ffffff`): Used as story-tile fill, button border, and primary text. When white appears as a large block, it's an editorial decision — a "spotlight" on that tile.
- **Absolute Black** (`#000000`): Reserved for text on the mint/yellow/white tiles — the only place it appears.

### Neutrals & Text
- **Primary Text** (`#ffffff`): Headlines and display text on the canvas.
- **Secondary Text** (`#949494`): Bylines, timestamps, photo credits. The mid-gray that anchors the metadata layer.
- **Muted Text** (`#e9e9e9`): Button text on dark slate buttons. Slightly off-white to reduce screen glare.
- **Inverted Text** (`#131313`): Used only on accent tiles (mint, yellow, white) to keep contrast legible.

### Semantic & Accent
- **Focus Ring** (`#1eaedb`): Keyboard focus only.
- **Overlay Black** (`rgba(0, 0, 0, 0.33)`): Subtle 1px ring used as the quiet shadow alternative on stacked cards.
- **Dim Gray** (`#8c8c8c`): Active/pressed button background — the "pressed down" state.

### Gradient System
The Verge uses **zero decorative gradients**. The only gradient-like treatment is the transition from a saturated accent story tile (mint/purple/yellow) back to the `#131313` canvas between rows. Color is applied in solid blocks, not as washes. This is a deliberate choice — the site's hazard-tape visual identity would dissolve if anything faded.

## 3. Typography Rules

### Font Family
- **Manuka** (Klim Type Foundry) — fallback: Impact, Helvetica. The signature display face for The Verge wordmark and feature headlines. A heavy-weight (900) industrial sans-serif with a condensed, almost-athletic stance. Runs at 60–107px on the homepage, never smaller.
- **PolySans** (PanGram Pangram / Nikolas Wrobel) — fallback: Helvetica, Arial. The UI and secondary headline workhorse. Covers weights 300 / 500 / 700 across the system — everything from kicker captions to body decks.
- **PolySans Mono** — fallback: Courier New, Courier. The monospaced sibling, used exclusively for ALL-CAPS labels: kickers, timestamps, category tags, button labels. This mono-uppercase usage is the second-most-identifiable Verge detail after Manuka.
- **FK Roman Standard** (Florian Karsten) — fallback: Georgia. A serif used sparingly for specific body/caption treatments (article excerpts, certain review pulls). Adds a "print-magazine" counterpoint to the PolySans stack.
- **Roboto** — fallback: `-apple-system`, `system-ui`. Utility UI font for widgets and legacy modules.

### Hierarchy

| Role | Font | Size | Weight | Line Height | Letter Spacing | Notes |
|---|---|---|---|---|---|---|
| Hero Wordmark / Display | Manuka | 107px / 6.69rem | 900 | 0.80 | 1.07px | The top-of-page "The Verge" logo and feature headlines |
| Secondary Display | Manuka | 90px / 5.63rem | 900 | 0.80 | — | Section-level feature headlines |
| Tertiary Display | Manuka | 60px / 3.75rem | 900 | 0.80 | — | Inline feature callouts |
| Large Headline | PolySans | 34px / 2.13rem | 700 | 1.00 | — | Section and module headlines |
| Heading Wide | PolySans | 32px / 2.00rem | 400 | 1.10 | 0.32px | Sub-heroes, promotional units |
| Heading Medium | PolySans | 24px / 1.50rem | 700 | 1.00 | — | Story tile headlines in the main feed |
| Heading Small | PolySans | 20px / 1.25rem | 700 | 1.00 | — | Compact tile headlines |
| Light Capitalized Label | PolySans | 19px / 1.19rem | 300 | 1.20 | 1.9px | Thin-weight capitalized eyebrows — a distinctive Verge move |
| All-Caps Label XL | PolySans | 18px / 1.13rem | 400 | 1.10 | 1.8px | UPPERCASE section kickers |
| Bold Body | PolySans | 16px / 1.00rem | 700 | 1.00 | — | Emphasis within decks |
| Body Relaxed | PolySans | 16px / 1.00rem | 500 | 1.60 | — | Long-form reading body |
| Inline Label | PolySans | 15px / 0.94rem | 400 | 1.20 | 0.15px | UI labels and secondary headlines |
| Body Compact | PolySans | 13px / 0.81rem | 400 | 1.60 | — | Secondary captions and decks |
| Eyebrow All-Caps | PolySans | 12px / 0.75rem | 400 | 1.30 | 1.8px | UPPERCASE kicker above tile headlines |
| Tag Label | PolySans | 12px / 0.75rem | 400 | 1.20 | 0.72px | UPPERCASE category tag |
| Caption Micro | PolySans | 11px / 0.69rem | 400 | 1.20 | 1.1px | UPPERCASE bylines |
| Meta Nano | PolySans | 10px / 0.63rem | 500 | 1.40 | 1.5px | UPPERCASE timestamp microtext |
| Mono Button Label | PolySans Mono | 12px / 0.75rem | 600 | 2.00 | 1.5px | UPPERCASE button text, very open leading |
| Mono Timestamp | PolySans Mono | 11px / 0.69rem | 500/600 | 1.20 | 1.1–1.8px | UPPERCASE StoryStream timestamps |
| Serif Body | FK Roman Standard | 16px / 1.00rem | 400 | 1.30 | -0.16px | Review decks, print-voice excerpts |
| Serif Caption | FK Roman Standard | 20px / 1.25rem | 400 | 1.20 | — | Magazine-style pull quotes |

### Principles
- **Manuka is always the hero, never the UI.** If you see Manuka below 60px you're looking at a bug. It exists to *shout the brand*, not to label a button.
- **PolySans is the workhorse, PolySans Mono is its uniformed sibling.** Mono is used exclusively for UPPERCASE labels, timestamps, tags, and certain buttons. Lowercase mono doesn't exist in this system.
- **Thin-weight (300) capitalized headlines** are a signature Verge move. The 19–20px weight-300 with 1.9px tracking creates a "fashion magazine whisper" that contrasts with the 107px Manuka shout above it. This whisper-vs-shout contrast is the typographic fingerprint.
- **Letter-spacing has two registers**: positive (0.72–1.9px) for ALL-CAPS mono and sans labels, negative (`-0.16px`) for the rare serif appearances, barely-positive (0.32px, 1.07px) for massive display. Plain 0 letter-spacing is rare.
- **FK Roman Standard is the editorial exception**, not the rule. Reserve it for long-form print-voice moments — reviews, critic pulls, masthead essays. Never use it in UI.
- **Line heights are tight** (0.80–1.30) for every display and label, relaxed (1.60–2.00) only for reading body and mono button labels. The leading jump is intentional — it gives the page a "telegraph ticker" rhythm.

### Note on Font Substitutes
The 0.80 line-height on Manuka display (107px, 90px, 60px) assumes the **proprietary Manuka face from Klim Type Foundry**, which has aggressively tight vertical metrics designed for athletic stance at large sizes. If you substitute with wide-metric open-source condensed displays like **Anton**, **Oswald**, **Bebas Neue**, or **Archivo Black**, loosen display line-heights by approximately **+0.10 to +0.15** to prevent ascender/descender collisions (e.g., 0.80 → 0.95). PolySans substitutes (Space Grotesk, DM Sans, Hanken Grotesk) work at the token values without adjustment — their metrics are close enough. PolySans Mono substitutes (Space Mono, JetBrains Mono) and FK Roman substitutes (Newsreader, Literata) also work without adjustment.

## 4. Component Stylings

### Buttons

**Primary — Jelly Mint Pill**
- Background: `#3cffd0` (Jelly Mint)
- Text: `#000000` (Absolute Black), PolySans 16px / 700 or PolySans Mono 12px / 600 UPPERCASE
- Border: none (pure fill)
- Border radius: `24px` — fully rounded pill
- Padding: `10px 24px`
- Outline: `none` at rest
- Hover: background shifts to `rgba(255, 255, 255, 0.2)` (translucent white), text stays black, adds a 1px `#c2c2c2` ring shadow
- Active: background `rgba(140, 140, 140, 0.87)`, opacity `0.5`, ring shadow `#8c8c8c`
- Focus: background `#1eaedb`, white text, 1px solid `#0500ff` border, translucent white focus ring
- Transition: ~180ms ease on background and shadow

**Secondary — Dark Slate Pill**
- Background: `#2d2d2d` (Surface Slate)
- Text: `#e9e9e9` (Muted Text), PolySans 16px / 400
- Border: none
- Border radius: `24px`
- Padding: `10px 24px`
- Outline: `rgb(233, 233, 233) none 0px`
- Hover: same translucent white invert as primary — `rgba(255, 255, 255, 0.2)` bg, black text, 1px `#c2c2c2` ring
- Focus: same cyan focus treatment as primary

**Tertiary — Outlined Mint**
- Background: transparent
- Text: `#3cffd0`, PolySans Mono 12px / 600 UPPERCASE, 1.5px tracking
- Border: `1px solid #3cffd0`
- Border radius: `40px` — larger pill for secondary outline style
- Padding: ~`10px 20px`
- Hover: inverts to mint fill, black text
- Transition: 150ms ease

**Outlined Ultraviolet (Promotional)**
- Background: transparent
- Text: `#5200ff` or `#ffffff`
- Border: `1px solid #5200ff`
- Border radius: `30px`
- Used for "Subscribe" or "Join the Stream" style promotional callouts

**Pill Tag (Non-interactive)**
- Background: saturated accent (`#3cffd0`, `#5200ff`, yellow, etc.)
- Text: black or white depending on background luminance
- Border radius: `20px` (tighter radius than buttons — this is the *text pill*)
- Font: PolySans Mono 11px / 600 UPPERCASE, 1.8px tracking
- Padding: ~`4px 10px`

### Cards & Containers

**StoryStream Tile**
- Background: either `#131313` + 1px white border, OR a saturated accent fill (mint, purple, yellow, pink, orange, white)
- Border radius: `20px` (standard) or `24px` (feature)
- Border: `1px solid #ffffff` (on dark) or `0px 0px 1px solid #3cffd0` (on mint) or nothing (on saturated fill)
- Padding: ~24–32px interior
- Hover: no lift, no scale — the headline text color transitions from white to `#3860be` (deep link blue)
- Transition: 150ms ease on color only

**Feature Card (Top Story)**
- Background: `#131313` with 1px hairline border, OR full-bleed color accent
- Border radius: `24px`
- Padding: 32px+
- Image inside: clipped to match the outer radius (`3px` or `4px` inner radius when nested)
- Hover: text color shift only; the image remains static

**StoryStream Rail (Timeline)**
- A vertical dashed or solid rule (1px `#3d00bf` or `#ffffff`) runs along the left edge of each item, marking the timeline spine
- Timestamps sit on the left rail in PolySans Mono 11px / 500 / UPPERCASE / 1.1px tracking
- Each entry is a pill-cornered rectangle separated from its neighbors by 12–16px vertical gap

### Inputs & Forms
- **Default**: `#131313` background, 1px solid `#ffffff` or `#949494` border, `2px` border radius (tight, newspaper-form feel), PolySans 15px text in `#ffffff`, placeholder in `#949494`.
- **Focus**: border transitions to `#3cffd0` (jelly mint) with optional `1px solid #5200ff` inner ring on deep focus. No glow.
- **Error**: border turns `#5200ff` (ultraviolet — used as error/alert accent here, not the usual red).
- **Transition**: ~150ms ease on border-color.

### Navigation

- **Top nav**: thin `#131313` bar with the Verge wordmark (Manuka) left-aligned, a search icon and a few UPPERCASE mono category links (12–14px, PolySans Mono, 1.5–1.8px tracking), and a single mint-pill CTA (usually "Subscribe") pinned right.
- **Wordmark**: massive on first scroll — the homepage treats the "The Verge" logo as a hero element, not a 32px corner logo.
- **Hover**: every link transitions from `#ffffff` to `#3860be` (deep link blue). No underline — it's a color-only response.
- **Active section**: marked by a 1px mint underline (inset box-shadow `0px -1px 0px 0px inset #3cffd0`)
- **Mobile**: the wordmark shrinks, category nav collapses into a hamburger drawer. Inside the drawer, links are mono-uppercase and stack with 16–20px gaps.

### Image Treatment

- **Aspect ratios**: 16:9 dominates for hero and feature images, 4:3 for mid-feed, 1:1 for thumbnails and author avatars.
- **Corners**: always rounded to match the parent card — `3px`, `4px`, or inherit `20px` / `24px` from the tile.
- **Frame**: 1px `#313131` or `#ffffff` hairline around photography, giving a "contained Polaroid" feel.
- **Full-bleed**: only within the color-block tiles, where the image runs to the padded edge of the accent fill.
- **Hover**: static — no zoom, no scale, no opacity shift. The headline below is the only interactive response.
- **Lazy loading**: `loading="lazy"` on everything below the first fold; eager on the masthead hero only.

### StoryStream Timeline Item (Distinctive)

- Vertical rail line (1px `#3d00bf` or `#ffffff` on `#131313`)
- Mono timestamp on the left in PolySans Mono 11px / UPPERCASE
- Pill-cornered body card (20px radius) with kicker, headline, and optional deck
- Stacked vertically with 12–16px gap, the rail continuing between them
- Often interleaved with full-bleed accent tiles that "break" the timeline rhythm for emphasis

## 5. Layout Principles

### Spacing System
- **Base unit**: 8px.
- **Scale**: 1, 2, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 20, 24, 25px.
- **Section padding**: 32–64px vertical between major feed sections. StoryStream items themselves are tighter — 12–16px gaps.
- **Card padding**: 20–32px interior. Feature cards expand to 40–48px.
- **Inline spacing**: kickers sit ~6–10px above headlines; headlines sit ~10–14px above decks; timestamps sit ~6–8px below decks.
- **Micro-scale**: The 2/4/5/6/9/10px values are used inside buttons, pills, and tight label clusters, not in the editorial grid.

### Grid & Container
- **Max width**: ~1280–1300px (dembrandt detected breakpoints at 1200/1280/1300).
- **Column patterns**: a 12-column underlying grid that resolves into 3-column hero + 1-column StoryStream rail + feature panels. The homepage feels freeform because color-block tiles frequently span 2–3 columns on a whim.
- **Container padding**: 24px mobile / 48px desktop on the outer edges.
- **Gutters**: 16–24px between columns, tighter (8–12px) inside StoryStream items.

### Whitespace Philosophy
The Verge treats whitespace like a club DJ treats silence — as a dramatic reset between loud moments. The canvas is so dark and the accents are so saturated that even 32px of empty `#131313` between two tiles acts as a palette cleanser. The page is not airy like Apple or Stripe; it's **paced**, with loud hazard-color blocks interrupting stretches of near-black. Whitespace carries the rhythm, not the elegance.

### Border Radius Scale
- **2px** — inputs, small badges (feels like a typewriter tag)
- **3px** — inline images (just enough to soften against the canvas)
- **4px** — nested card images and small button variants
- **20px** — standard pill cards and color-block tiles
- **24px** — feature tile radius and primary button pill
- **30px** — large promotional buttons
- **40px** — outlined CTA pills (the loudest pill in the system)
- **50%** — avatar circles, icon buttons, and certain round badges

Eight discrete radius values — a **lot** for a single site. This is deliberate: the rhythm between 2px typewriter tags, 20px pill cards, and 40px outlined buttons creates a "nested scale" feel where every component announces its hierarchy through its corners.

## 6. Depth & Elevation

| Level | Treatment | Use |
|---|---|---|
| 0 | No border, no shadow | Default `#131313` canvas text |
| 1 | `rgba(0,0,0,0) 0px 0px 0px 0px inset` (placeholder) | Reset state for interactive elements |
| 2 | `1px solid #ffffff` or `#313131` hairline | Image frames and quiet card outlines |
| 3 | `1px solid #3cffd0` hairline | Active button outlines, focused story tiles |
| 4 | `1px solid #5200ff` hairline | Promotional/alternate state outlines |
| 5 | `rgba(0, 0, 0, 0.33) 0px 0px 0px 1px` | The single "atmospheric" ring — applied to layered cards |
| 6 | `0px -1px 0px 0px inset` (mint/black/white) | Active tab underline — a signature Verge move |
| 7 | Saturated accent fill (`#3cffd0`, `#5200ff`, white, yellow, pink) | Story-tile elevation via color, not shadow |

The Verge's depth philosophy is **color-as-elevation**. When something needs to stand out, it doesn't get a shadow — it gets a mint fill or a 1px hazard-color border. There are 14 shadow entries in the extracted tokens, but all of them are either inset underlines (0px -1px inset) or near-transparent 1px rings — none of them are traditional elevation shadows. The `#131313` canvas stays perfectly flat throughout, and hierarchy is carried by color saturation.

### Decorative Depth
- **1px inset underline** on active tabs/nav links (mint, black, or white depending on context)
- **Subtle `rgba(0, 0, 0, 0.33)` 1px ring** on stacked cards — the only effect that faintly resembles a shadow
- **No gradients, no glows, no atmospheric blurs** anywhere. The hazard-tape aesthetic would break if anything faded softly.

## 7. Do's and Don'ts

### Do
- **Do** use `#131313` as the canvas for every view. There is no light mode.
- **Do** use Jelly Mint (`#3cffd0`) and Verge Ultraviolet (`#5200ff`) as hazard accents — buttons, borders, active states, and saturated color-block tiles.
- **Do** use Manuka exclusively at 60px+ for hero headlines. Treat anything smaller as a bug.
- **Do** round everything: 20px for cards, 24px for feature cards, 30–40px for pill buttons.
- **Do** use PolySans Mono for UPPERCASE labels, timestamps, kickers, and button text. Lowercase mono doesn't exist here.
- **Do** apply 1.5–1.9px letter-spacing to every ALL-CAPS label — this is a Verge signature.
- **Do** use saturated color-block tiles (mint, purple, yellow, pink, orange, white) to elevate a story — never a drop shadow.
- **Do** use `#3860be` (deep link blue) as the hover color on every link, regardless of base color.
- **Do** apply the StoryStream timeline rail (1px dashed/solid `#3d00bf` or white) on feed views.
- **Do** use thin-weight (300) PolySans at 19–20px with 1.9px tracking for "fashion-whisper" capitalized eyebrows — the contrast with the 107px Manuka shout is the whole voice.

### Don't
- **Don't** use a light background. The dark canvas is the product.
- **Don't** add `box-shadow` for elevation. Use 1px borders or saturated accent fills instead.
- **Don't** use square corners. Every interactive and content container is rounded.
- **Don't** use Manuka for UI, buttons, or body copy. It's strictly display.
- **Don't** use lowercase mono. PolySans Mono is always UPPERCASE.
- **Don't** let mint and ultraviolet appear as background washes — they're hazard accents, not canvas tints.
- **Don't** use gradients anywhere. The system is solid color blocks only.
- **Don't** introduce new accent colors outside the declared mint / purple / yellow / pink / orange tile palette.
- **Don't** pair Manuka with FK Roman Standard in the same headline cluster — Manuka is the only display shout, serif pulls are reserved for body moments.
- **Don't** use `#3cffd0` text on a `#131313` background at under 16px — the contrast vibrates at small sizes.

## 8. Responsive Behavior

### Breakpoints

| Name | Width | Key Changes |
|---|---|---|
| Small Mobile | <400px | Single column, Manuka hero scales down to ~48–54px, StoryStream rail collapses to inline timestamps |
| Mobile | 400–549px | Single column, color-block tiles stack full-width, nav is a hamburger drawer |
| Large Mobile | 550–767px | Still single column but padding opens up, tile radii stay at 20px |
| Tablet | 768–1023px | 2-column StoryStream with feature card spanning, wordmark shrinks ~50% |
| Small Desktop | 1024–1179px | Full 3–4 column editorial grid, mint pill CTA restored to nav |
| Desktop | 1180–1299px | Max padding, Manuka wordmark at full hero scale |
| Large Desktop | ≥1300px | Container caps at ~1280–1300px, whitespace expands at the margins, no further scaling |

The dembrandt sweep detected 26 intermediate breakpoints (1300 → 1280 → 1200 → 1181 → 1180 → 1179 → 1024 → 1023 → 901 → 900 → 897 → 896 → 890 → 769 → 768 → 620 → 605 → 600 → 550 → 549 → 530 → 426 → 425 → 400 → 320). The Verge tunes its grid at virtually every major device boundary — an unusually aggressive responsive strategy.

### Touch Targets
- Primary pill buttons are ~44px minimum height (10px vertical padding + 16px text + 2px border) — meets WCAG AA.
- Mono uppercase nav links are smaller (~28–32px tall) — for derivative work, pad to 44px on mobile.
- Circle icon buttons are 40–44px circles, touch-friendly.

### Collapsing Strategy
- **Nav**: wordmark scales from hero (Manuka 60–107px) to ~24–32px on mobile. Category links collapse to a hamburger drawer below 900px.
- **Grid**: 4-col → 3-col → 2-col → 1-col. Feature cards that span 2 columns on desktop reflow to full-width single-column on mobile.
- **Spacing**: section padding tightens from 64px → 32px → 20px. Tile interior padding tightens from 32px → 20px.
- **Type**: Manuka hero scales from 107px to ~48–54px on mobile. PolySans headlines scale from 34px → 24px. Mono labels stay pinned at 11–12px (they don't shrink further or they become unreadable).
- **Color tiles**: accent story blocks never lose saturation on mobile — they just reflow to full width.

### Image Behavior
- Responsive raster via `srcset`, aspect ratios preserved.
- No art-direction swaps — same crop scales across all viewports.
- `loading="lazy"` on everything below the fold, `eager` on the masthead hero.
- Images inside color-block tiles inherit the tile's inner radius (4px or 20px nested).

## 9. Agent Prompt Guide

### Quick Color Reference
- **Primary CTA**: "Jelly Mint (`#3cffd0`)"
- **Background (Canvas)**: "Canvas Black (`#131313`)"
- **Accent (Secondary Hazard)**: "Verge Ultraviolet (`#5200ff`)"
- **Heading Text**: "Hazard White (`#ffffff`)"
- **Body Text**: "Hazard White (`#ffffff`)" (primary) or "Muted Text (`#e9e9e9`)"
- **Secondary Text / Metadata**: "Secondary Text (`#949494`)"
- **Card Border**: "Hazard White (`#ffffff`)" hairline on dark, "Console Mint Border (`#309875`)" on mint variants
- **Link Hover**: "Deep Link Blue (`#3860be`)"

### Example Component Prompts
1. *"Create a StoryStream timeline item on a `#131313` canvas: a 20px-radius rectangle with a 1px solid `#ffffff` border, a PolySans Mono 11px / 600 / UPPERCASE / 1.1px tracking timestamp on the left rail, a 12px PolySans UPPERCASE kicker in mint (`#3cffd0`), and a 24px / 700 PolySans headline in white below. No shadow, no lift — hover only shifts the headline color to `#3860be`."*
2. *"Design a primary subscribe button with a Jelly Mint (`#3cffd0`) fill, black text in PolySans Mono 12px / 600 / UPPERCASE / 1.5px tracking, 24px border radius, 10px × 24px padding. Hover state shifts to `rgba(255, 255, 255, 0.2)` background with a 1px `#c2c2c2` ring shadow, 180ms ease."*
3. *"Build a feature hero with a 107px Manuka 900 headline in white with 1.07px letter-spacing and 0.80 line-height, a thin-weight 300 PolySans 20px capitalized kicker above with 1.9px tracking, on a `#131313` canvas with 64px vertical padding."*
4. *"Create a color-block accent tile filled with Verge Ultraviolet (`#5200ff`) at 0.9 alpha, 24px border radius, white text, a PolySans Mono 11px UPPERCASE category label with 1.5px tracking at the top, and a 32px PolySans 400 capitalized headline with 0.32px tracking below."*
5. *"Design a dark slate secondary button with a `#2d2d2d` background, `#e9e9e9` PolySans 16px text, 24px radius pill shape, 10px × 24px padding. Hover matches the primary button — translucent white `rgba(255, 255, 255, 0.2)` bg with black text."*

### Iteration Guide
When refining existing screens generated with this design system:
1. **Audit the canvas.** If you see a light background anywhere on the homepage, flatten it to `#131313`. There is no light mode.
2. **Audit corners.** Every rectangle should land on 2/3/4/20/24/30/40px or 50%. Square corners break the voice.
3. **Audit shadows.** Strip every `box-shadow` that isn't a 1px inset underline or a 1px hazard-color border. The Verge uses color for elevation, not shadow.
4. **Audit type roles.** Manuka only ≥60px. PolySans Mono only UPPERCASE. PolySans 300 at 19–20px should have 1.9px tracking. FK Roman only for body/magazine moments, never UI.
5. **Audit accent usage.** Mint and ultraviolet should appear as hazard accents — buttons, 1px borders, active underlines, saturated tile fills. If they're appearing as background washes or gradient fades, correct to solid blocks.
6. **Audit labels.** Every kicker, timestamp, category tag, and button label should be ALL CAPS with 1.1–1.9px letter-spacing. Missing tracking = missing voice.
7. **Audit link hover.** Every link, regardless of its base color, should hover to `#3860be` deep link blue with no underline. Any other hover color is drift.
