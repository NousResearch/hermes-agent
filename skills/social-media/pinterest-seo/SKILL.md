---
name: pinterest-seo
description: >-
  The Pinterest SEO / search-ranking skill -- the keyword layer that pairs with pinterest-pin-design's visual.
  Use when someone wants pins to rank in Pinterest search, needs keyword research, or wants
  keyword-optimized pin titles/descriptions/boards. Pinterest is a visual search engine where keywords
  and relevance rank you, not followers. Uses the FOUND framework. Reads brand-profile + the pin's
  destination first. The agent researches real queries (autocomplete + Guided Search + Trends; never
  fabricates volume), writes the keyword-led title + a natural description (no stuffing), specs the
  on-pin overlay keyword, and advises boards + link. WoopSocial publishes the pin's title + description
  + board + link; profile/bio, board setup, alt text, Rich Pins, and Trends/Analytics stay native/human.
  No stuffing; Pin->Board->Page must match; fresh pins, not dupes. Distinct from pinterest-pin-design
  (the visual), pinterest-growth (the strategy), and social-seo (cross-cutting search).
version: 1.0.0
---

# pinterest-seo

The **keyword / search-ranking layer** — research the queries and write the keyworded words that make a pin
rank. The agent **researches + writes the title/description + specs the on-pin keyword + advises the board/
link**; **WoopSocial publishes the pin's title + description + board + link**; **profile, board setup, alt
text, Rich Pins, and the research tools are native/advised.**

## The POV: a visual search engine — keywords rank you, not followers
Pinterest is **"Google with pictures"** (631M MAU Q1 2026, **DA 94**, pins also rank in Google image results),
where **~97% of top searches are unbranded** and **follower count barely matters — keywords + relevance +
engagement rank you.** Pins have a **~4-month lifespan**, so optimization **compounds.** The craft is
Pinterest-specific: research via **autocomplete + Guided Search bubbles + Trends**, layer **long-tail
keywords** across profile → board → pin, align **Pin → Board → Page**, and optimize for **saves + long
clicks.** WoopSocial publishes the pin's **title + description + board + link**; the rest is native/advised.
**Pairs with `pinterest-pin-design`** (the keyword goes in the overlay AND the title).

## Read these first
1. **brand-profile** — niche/voice.
2. the **destination** — the page the pin links to.

## The framework: FOUND
(Depth: `references/the-found-framework.md`.)
- **F — Find the real queries:** Pinterest autocomplete + Guided Search bubbles + Trends/Predicts; long-tail
  (3+ words); think like a searcher. *Advise the method; never fabricate volume.*
- **O — Optimize the pin's words:** title leads with the primary keyword (`[Keyword] | [Benefit] | [Context]`,
  ≤100); a natural keyword-rich description (~100–200 chars, no stuffing) — the **publishable** surfaces; spec
  the on-pin overlay keyword → `ideogram`.
- **U — Unify Pin → Board → Page:** the pin, its semantically-precise board, and the destination page
  reinforce the **same keyword phrase**; claim the site + Rich Pins; the link delivers the promise.
- **N — Nest keywords across every surface:** profile (broad) → board titles/descriptions (topic) → pin
  title/description/alt/file-name (long-tail). *Most native/advised; WoopSocial publishes the pin's title +
  description + board + link.*
- **D — Deploy fresh, in clusters, on a seasonal clock:** fresh keyword-optimized pins (not dupes); related
  clusters; seasonal 45–90 days ahead; optimize for saves + long clicks.

## The reality (verify-quarterly)
Visual search engine / DA 94 / ~97% unbranded / followers barely matter / ~4-month compounding; ranking
(domain+pin+pinner+relevance; saves + long-clicks top signals); keyword research (autocomplete + Guided
Search bubbles + Trends/Predicts; long-tail); keyword layering (profile/board/pin/alt/file-name); title format
`[KW] | [benefit] | [context]`; description natural 100–200 chars no-stuffing; board semantic alignment
(precise > generic); Pin → Board → Page alignment; Rich Pins + claimed site; hashtags deprioritized; fresh
pins (not dupes) + clusters + seasonal 45–90 days ahead; over-optimization suppressed:
`references/pinterest-seo-2026-reality.md`. The keyword-research method, the keyword-layering map (publishable
vs native), title/description templates, the Pin→Board→Page checklist, the publish block + worked examples:
`references/keyword-research-and-surfaces.md`.

## Honest scope (never violate)
- **The agent** researches keywords (**advises the method**, **never fabricates volume**), writes the
  **keyword-led title (≤100)** + **description** (`content.text`, no stuffing), **specs the on-pin overlay
  keyword** (→ image tool), and **advises** board names/descriptions + a **keyword-aligned link.**
- **WoopSocial publishes** the pin's **title + description + board (`pinterestBoardId`) + link (≤2048)** — the
  publishable SEO surfaces.
- **WoopSocial does NOT:** set **profile/bio** keywords, **create/rename/describe boards**, set **alt text** or
  **file names**, **claim the site** or enable **Rich Pins**, query **Pinterest Trends/autocomplete/
  Analytics**, or render the **on-pin text** — all native/human (advised).
- **No keyword stuffing / over-optimization** (suppressed); **Pin → Board → Page must match** (no misleading);
  **fresh pins, not dupes**; **never fabricate volume or rankings.** (Scope, distinctions + connections:
  `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**pinterest-seo (this)** = the keyword/search-RANKING layer · **pinterest-pin-design** = the VISUAL craft
(they **PAIR** — keyword in the overlay AND the title) · **pinterest-growth** = the account/visual-search
STRATEGY · **social-seo** = cross-cutting platform search (this is Pinterest-specific depth) ·
**ai-search-optimization** = AI engines · **hashtag-strategy** = Pinterest deprioritizes hashtags.

## Where this connects
Reads first: **brand-profile** + the **destination**. Pairs with: **pinterest-pin-design** (keyword in the
overlay + title), **caption-writer** (the description), **ideogram** (the on-pin keyword overlay). Routes to/
from: **pinterest-growth** (strategy/cadence), **content-recycling** (fresh variations), **social-seo**/
**ai-search-optimization** (the cross-platform search picture), **link-in-bio-and-traffic** (the destination).
Publishes via: **scheduling-and-queue → WoopSocial** (title + description + board + link) +
**platform-specs-and-validation**. Measures with: native **Pinterest Analytics** + **analytics-and-reporting**
(saves, long clicks, search impressions). Profile, board setup, alt text, Rich Pins + the research tools stay
native/human.

## Definition of done
Real queries researched via Pinterest autocomplete + Guided Search bubbles + Trends (long-tail, method
advised, no fabricated volume); a keyword-led pin title (`[Keyword] | [Benefit] | [Context]`, ≤100) and a
natural keyword-rich description (no stuffing) written as the publishable surfaces; the on-pin overlay keyword
specced to an image tool; a semantically-precise board and a Pin → Board → Page that all reinforce the same
phrase (with Rich Pins + claimed site advised natively); fresh pins (not dupes), clustered, and seasonal
content timed 45–90 days ahead; the pin's title + description + board + link published via WoopSocial while
profile/board/alt-text/Rich-Pins/research-tools stay native; no keyword stuffing, no misleading pages, nothing
fabricated; correctly paired with pinterest-pin-design and distinguished from pinterest-growth and social-seo.
