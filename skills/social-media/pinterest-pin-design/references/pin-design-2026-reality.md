# Pin design 2026 — verified

*Volatile. Re-verify quarterly (formats + feed rendering move).*

## The POV: a search result in a two-column mobile feed
A pin competes as a **thumbnail in a search grid**, not a full-screen post. Most usage is **mobile (~85%)**,
where the feed is **two columns** — a pin renders at roughly **half a phone screen wide** (~236px wide in
the desktop grid). Everything below follows from that: one focal point, big overlay text, high contrast.
And unlike IG/TikTok, the pin **works for months** (~4-month lifespan, some 3–6) — design it like an asset,
not a post.

## Format: 2:3 (1000×1500) is the rule
- **2:3 aspect ratio, 1000×1500 px recommended** — Pinterest's official guidance; **other ratios are
  penalized** in distribution. Square (1:1) survives but wastes feed height; **pins taller than 2:3 get
  truncated in the feed** (the extra height is cropped until closeup), so extra-tall "giraffe pins" no
  longer earn extra real estate — anything critical in the bottom of a tall pin disappears.
- **Video pins: 9:16, 6–15s**, designed to read **without sound** (overlay text carries it). Roughly ~2×
  engagement vs static, but **static pins drive most outbound traffic.**

## Text overlay: a ranking surface, not decoration
- **Pinterest OCR-reads the text on the image** — the overlay keyword is indexed. This is the pairing with
  **pinterest-seo**: the **keyword goes in the overlay AND the title.**
- **Legibility at feed size:** ≤~6 words, **bold weight**, large enough to read at ~236px wide — if you
  can't read it on a phone feed without tapping, it's decoration. High contrast against the photo; a
  **solid or semi-transparent band** behind the text when the image is busy. No thin scripts for the
  keyword line (a script accent word is fine; the load-bearing words stay bold).
- **Placement: top third or center.** Not only in the bottom (tall-pin truncation + closeup UI). The
  overlay states the **promise** ("Small Pantry Organization Ideas"), it doesn't duplicate the description.

## Imagery: lifestyle and outcome, faces optional
- **In-context / lifestyle imagery outperforms sterile product-on-white** — Pinterest users are planning
  (projects, purchases, occasions), so show the **outcome**: the styled shelf, the finished braid, the
  plated recipe.
- **Faces are optional** — unlike YouTube thumbnails, the product/idea is usually the hero; faceless pins
  are normal and perform fine. Use a person when the content is about people (hair, fashion, fitness);
  never an unconsented real person's face.
- **Vertical composition:** use the height — subject, overlay band, breathing room. Don't center a
  horizontal photo in a vertical canvas with dead bars.

## Safe zones + branding
- Keep **~5–8% margins**; nothing critical at the exact edges (grid rounding + crops).
- **Logo/domain: subtle, and NOT in the bottom-right corner** — Pinterest overlays its own icons (visual
  search / product) there. Top-left, top-center, or a bottom-left/center footer bar are safe.
- **Brand consistency lifts recognition on a crowded search page** — locked fonts/palette/overlay position
  (a template family via **design-and-templates**). But a *family*, not clones: near-identical pins defeat
  the fresh-pin test and read as spam.

## Fresh pins: the design economics
- Pinterest **heavily favors fresh content**: a new pin gets an early **few-day test window** that shapes
  its long-term reach, and **fresh pin designs drive ~90% of website traffic** (saves/repins now carry
  minimal ranking weight).
- **"Fresh" = a new image + layout + text overlay for the same URL** — a repin is not fresh, and a color
  swap is not fresh. Make **3–5 visually distinct designs per destination**: different photo, different
  layout, different overlay angle/keyword (a listicle pin, a single-hero pin, a before/after pin for the
  same post). Volume without genuinely distinct design/copy gets **suppressed.**

## Idea pins / formats (moving target)
- Pinterest has been **folding Idea Pins into one unified pin format.** Treat **multi-slide/video-first
  pins** as an **awareness/saves play** — historically **no outbound link** — and the **standard static pin
  with a link as the traffic workhorse.** Collage/product pins serve shopping. **Verify quarterly**; this
  is the fastest-moving part of the platform.

## Board covers + profile aesthetics
- Boards are the **shelf** your pins sit on: a consistent **board-cover style** (same template, one title
  word, brand palette) makes the profile read as a curated catalog and signals topical focus. Board
  **names/descriptions are keyword surfaces** — that's **pinterest-seo**'s call; this skill makes the
  covers match.

## Honesty is enforced by the ranking model
- **Long clicks (30+ sec on the destination) are a top signal; quick bounces are negative** — so a pin that
  over-promises what the page delivers loses over its whole multi-month life. **Pin → Board → Page must
  match.** Disclose **GenAI imagery** (users can filter it); **FTC-disclose** affiliate pins.
