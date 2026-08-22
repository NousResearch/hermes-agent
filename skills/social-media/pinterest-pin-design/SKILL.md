---
name: pinterest-pin-design
description: >-
  The Pinterest pin design skill -- the visual half that pairs with pinterest-seo's keyword layer.
  Use when someone wants to design a pin, a Pinterest pin image, a pin text overlay, fresh pin
  variations for an existing URL, or board covers. Designs and briefs the 2:3 (1000x1500) vertical
  pin: one focal point, a keyword text overlay readable at feed size, brand-consistent template,
  safe zones, and 3-5 visually distinct FRESH designs per destination URL. Uses the SAVED framework.
  Reads brand-profile + the destination + the pinterest-seo keyword first. A HUMAN or an
  image/design tool (ideogram/canva/nano-banana/a designer) creates the final raster; WoopSocial
  publishes the finished pin image + title + description + board + link. No misleading pins, no
  repin-as-fresh dupes, engagement never fabricated. Distinct from pinterest-seo (the keyword layer
  -- they PAIR), pinterest-growth (the strategy), design-and-templates (the template system), and
  ideogram/canva (render the raster this briefs).
version: 1.0.0
---

# pinterest-pin-design

A **design-craft skill** — design the pin's concept + composition + overlay, produce the **brief**, and plan
the **fresh-pin variants**. The agent **designs + briefs + critiques**; a **human or image tool makes the
raster**; **WoopSocial publishes the finished pin** (image + title + description + board + link).

## The POV: a search result, not a post — designed to be found, saved, and clicked
A pin is a **search result in a two-column mobile feed**, competing at roughly **half a phone screen wide**
(~236px in the desktop grid). Its job is the **save and the outbound click**, and it keeps doing that job for
**months** (~4-month lifespan) — so a pin is an **asset, not a post.** Two rules anchor everything: the
**visual and the keyword are one unit** (this skill owns the visual; **pinterest-seo** owns the keyword — the
keyword goes in the **overlay AND the title**), and **fresh beats frequent reposting** — a "fresh pin" is a
**new image/layout/overlay for the same URL**, not a repin, and fresh designs drive **~90% of Pinterest
website traffic.** Design **mobile-first at feed size**, always in **2:3 (1000×1500)** — other ratios are
penalized.

## Read these first
1. **brand-profile** — palette/fonts/style (pins are a brand surface that must stay recognizable).
2. the **destination** — the page the pin links to (the visual must promise what the page delivers).
3. the **keyword** — from **pinterest-seo** (it goes in the overlay and the title).

## The framework: SAVED
(Depth: `references/the-saved-framework.md`.)
- **S — Stop the scroll on a 2:3 canvas:** 1000×1500, 2:3 mandatory (other ratios penalized/cropped); one
  focal point; vertical composition; lifestyle/in-context imagery beats sterile product-on-white; parseable
  at feed size.
- **A — Anchor the keyword overlay:** the pinterest-seo keyword rendered large on the image (Pinterest
  OCR-reads it); ≤~6 words, bold, high contrast, readable at ~236px; top-third or center; not a caption
  duplicate.
- **V — Visual brand system:** a pin template family (from **design-and-templates**) — consistent fonts/
  palette/layout; subtle logo/URL, **never bottom-right** (Pinterest's own icons overlay that corner).
- **E — Every URL gets fresh variants:** 3–5 **visually distinct** designs per destination (different image
  + layout + overlay angle — not a color swap); each is a new chance to rank.
- **D — Decide the format + hand off:** static pin = the traffic workhorse (has the link); video pin (9:16,
  6–15s) = engagement; multi-slide/idea-style = awareness (historically no outbound link); board covers set
  the shelf aesthetic; brief → image tool → WoopSocial publishes.

## The reality (verify-quarterly)
Feed-size mobile-first (two-column feed, ~236px, ~85% mobile); 2:3 1000×1500 mandatory (other ratios
penalized, tall pins truncated); overlay text OCR-read (keyword on the image); lifestyle/in-context imagery;
faces optional (product/idea is usually the hero — unlike YouTube thumbnails); safe zones (margins, logo not
bottom-right, nothing critical in the bottom of extra-tall pins); fresh-pin economics (~90% of traffic,
3–5 distinct designs/URL, few-day test window); video 9:16 6–15s; idea pins folding into the unified format;
brand consistency without clone-stamping: `references/pin-design-2026-reality.md`. The pin brief template,
design checklist, spec block, fresh-variant plan + worked examples: `references/design-and-variant-recipes.md`.

## Honest scope (never violate)
- **The agent** designs the concept/composition/overlay text, writes the **brief**, plans the **fresh
  variants**, and can **critique a mockup** — but a **HUMAN or an image/design tool** (`ideogram` for
  in-image text / `canva` / `nano-banana` / a designer) **creates the final raster.**
- **WoopSocial publishes** the finished pin — **image + title + description + board (`pinterestBoardId`) +
  link** — it does **NOT** design, render, or generate the visual, and analytics are **native Pinterest
  Analytics**, not WoopSocial.
- **No misleading pins** — the visual must promise what the page delivers (**Pin → Board → Page must
  match**; long clicks rank you, bounces punish you). **No repin-as-fresh dupes** (a color swap is not a
  fresh pin). **Disclose AI imagery** (Pinterest lets users filter GenAI). **Never fabricate saves/clicks/
  impressions.** (Scope, distinctions + connections: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**pinterest-pin-design (this)** = the pin's VISUAL craft + brief + fresh-variant plan · **pinterest-seo** =
the keyword/search-RANKING layer (they **PAIR** — the keyword goes in the overlay AND the title) ·
**pinterest-growth** = the account strategy/cadence (boards, 72h same-URL spacing, seasonal timing) ·
**design-and-templates** = the brand template SYSTEM (the pin template is one asset in it) · **ideogram**/
**canva**/**nano-banana** = render the raster this briefs · **thumbnail-design** = the same craft for YouTube's
16:9 click decision (different job: click-to-watch vs save-and-click-out).

## Where this connects
Reads first: **brand-profile** + the **destination** + the keyword from **pinterest-seo**. Pairs with:
**pinterest-seo** (keyword in the overlay + title), **image-prompt** → **ideogram** (in-image overlay text)/
**nano-banana**/**flux** (backgrounds), **canva** (template production), **ai-image-editing** (crops/cleanup),
**design-and-templates** (the pin template in the brand system). Routes to/from: **pinterest-growth**
(strategy/cadence), **content-recycling** (which URLs deserve fresh variants), **seasonal-and-moment-marketing**
(45–90-day seasonal lead). Publishes via: **scheduling-and-queue → WoopSocial** (image + title + description
+ board + link) + **platform-specs-and-validation**. Measures with: native **Pinterest Analytics** +
**analytics-and-reporting** (saves, outbound clicks, impressions). The raster stays with the image tools/
designer; the words stay with pinterest-seo.

## Definition of done
A 2:3 (1000×1500) vertical concept with one focal point in lifestyle/in-context imagery, parseable at feed
size; the pinterest-seo keyword rendered as a bold, high-contrast overlay (≤~6 words, OCR-readable, no
caption duplicate); brand-consistent template (fonts/palette/layout) with the logo out of the bottom-right;
safe margins and nothing critical lost to tall-pin truncation; 3–5 visually distinct fresh designs planned
per destination URL (new image + layout + overlay angle, never a repin/color swap); the right format chosen
(static for traffic, video 9:16 6–15s for engagement, multi-slide for awareness) and board covers consistent;
the raster briefed to an image tool/designer and the finished pin published via WoopSocial (image + title +
description + board + link); the visual honest to the destination; AI imagery disclosed; no fabricated
engagement; correctly paired with pinterest-seo and distinguished from pinterest-growth, design-and-templates,
and the image tools.
