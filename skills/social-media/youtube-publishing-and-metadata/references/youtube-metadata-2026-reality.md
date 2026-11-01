# YouTube metadata 2026 — verified

*Volatile. Re-verify quarterly (algorithm + policy move; COPPA rules are legal).*

## The POV: YouTube is a search engine; metadata is the launchpad
YouTube is the **2nd-largest search engine**; **~27% of traffic from search**, **~70% from
recommendations** — both **fed by metadata**, which YouTube reads **before user behavior kicks in.** Good
metadata is the **launchpad**, and a well-optimized video **compounds for years** ("owned media; the rest
is rented"). **But retention/watch time keeps it ranking** — metadata gets you discovered, retention keeps
you there. Don't oversell metadata as a guarantee.

## Title (`title`, required — the single most important field)
- **Primary keyword early** (first third / **first ~40 chars** — mobile truncates ~35–40). Best length
  **50–60 chars** (max **100**; search shows ~60–70).
- Does **two jobs at once**: **rank** (keyword) + **click** (clarity/curiosity). **Clarity over
  cleverness** — if viewers can't tell what it is, they won't click.
- **Power words** ("ultimate/complete/proven/surprising") ≈ **+8.3% CTR**. Use **YouTube autocomplete** for
  proven phrases; **long-tail** for smaller channels. **A/B testing is native YouTube Studio** (not the
  API): **Test & Compare** now covers **titles as well as thumbnails** — up to 3 title variants or
  title+thumbnail combos, winner by watch-time share (global since Dec 2025; needs Advanced Features).

## Description (= `content.text`)
- **Above-the-fold** (first ~150 chars / 2–3 lines, before "Show more") is weighted: **line 1 = value/
  hook**, **lines 2–3 = primary keyword** naturally. **300–500 words** total (max **5,000 chars**), **natural integration**
  (semantic; **stuffing penalized** — natural ≈ **+23% impressions**).
- **Chapters/timestamps** with **keyword-rich names** (not "Part 1") — indexed by YouTube + Google, boost
  search-snippet eligibility. **3–7 links** (internal playlists keep session time; external strategic);
  **affiliate links require disclosure** (FTC).

## Tags (`tags[]`)
- **Less influential than in 2020** — YouTube understands content from title/description/captions. But still
  aid **search matching/disambiguation.** **First tag = exact primary keyword** (weighted most). Then
  secondary + niche. **Relevant-only** — YouTube's AI checks tag-content match; irrelevant tags can signal spam.

## Category + privacy
- **`category`** — pick the **accurate** category so YouTube groups/recommends correctly (validate catches
  invalid ones). **`privacy`**: **`public`** (reach), **`unlisted`** (link-only, not in search/browse),
  **`private`** (you + invited). Pick per goal.

## madeForKids (`madeForKids` — a COPPA/FTC LEGAL flag, not an optimization)
- Every video must declare whether it's **"made for kids"**, determined by whether the **content is
  child-directed** (characters/toys/kids' songs/etc.) — **not** merely whether kids might watch.
- If **made-for-kids**, YouTube **disables** comments, personalized ads, notifications, end screens, cards,
  Save-to-playlist, etc. **Set it truthfully** — **mislabeling has legal consequences** (FTC). Never flip
  it to keep comments/ads on genuinely kids' content. This is compliance, not a reach lever.

## WoopSocial gaps (honest)
**No thumbnail field** in the YouTube fields → the **thumbnail is a native Studio step** (and it's a major
CTR lever, so don't skip it). **No A/B title testing** (native Studio Test & Compare). **No update endpoint** → edit =
delete+recreate before publish; **post-publish metadata edits happen in Studio.** Shorts vs long-form: a
vertical or square video **up to 3 minutes** posts as a **Short** (the ≤60s rule ended Oct 2024); SEO
weights differ slightly but the metadata fields are the same.
