# LinkedIn 2026 — verified

*Volatile. Re-verify quarterly. LinkedIn publishes less than X does; figures below are the widely
cited practitioner consensus, not open-sourced weights.*

## The engine: dwell time + meaningful comments
LinkedIn's two dominant ranking signals are **dwell time** (a long read = a good post; the
"…see more" tap on mobile at ~**140 characters** is the first dwell signal) and **early, meaningful
comments** — substantive ones (~15+ words) far outweigh likes, and the **author replying to comments
in the first 1–2 hours** measurably extends reach. Reactions barely move distribution; **engagement
bait ("Comment YES", "Agree? 👇") is explicitly down-ranked.** Saves/sends are rising signals on
carousels and reference content.

## Feed structure: people, not logos
- **Personal profiles get ~65% of feed allocation; company pages ~1–5%** (page organic reach fell
  60–66% since 2024 — see `linkedin-company-pages`). Grow on a **person**.
- Median per-post reach has declined as LinkedIn shifted to **relevance over recency**: it shows
  posts to people likely to find them useful, in and out of network, via **suggested posts**. Net
  effect: **posts have a long shelf life** — a strong post keeps surfacing for days or weeks, so
  quality compounds and volume doesn't.
- The algorithm classifies you by **topic consistency**: it must be able to say what you're the
  person for. Multi-topic scatter suppresses out-of-network distribution (→ `social-seo` for the
  keyword layer in headline/About/posts).

## Format hierarchy (2026)
- **Document/PDF carousels** — highest engagement + saves per post; the dwell power format
  (slides → `carousel-writer`; created natively, not via API tools).
- **Native video** — LinkedIn's current push: vertical, captioned, under ~90s; video watch time grew
  ~36% y/y and gets a discovery feed of its own. Uploaded natively, never a YouTube link
  (→ `talking-head-and-piece-to-camera`).
- **Text posts** — still the workhorse; ~1,800–2,100 chars is the oft-cited sweet spot **if it earns
  the read**; formatted for phones (1–2-line paragraphs).
- **Text + image, polls** — solid; polls engage but convert follows weakly.
- **External links in the body suppress reach ~25–60%.** Put the link in the **first comment** or
  post link-free. Native beats linked, always.
- **Hashtags:** 3–5 max, end of post — a weak lever now, not a strategy.

## Creator mode: retired, folded in
The **creator-mode toggle was retired (2024)** — its features (Follow as an option, profile links,
topic hashtags) became **defaults for all profiles**. The remaining live decision is
**Connect-primary vs Follow-primary**: switch the profile button to **Follow** once inbound interest
outpaces your outbound connecting (typically a few thousand followers), so strangers can subscribe
without burning your connection cap.

## Connections vs followers
- **Connections:** hard cap **30,000**; invites throttled to roughly **100/week**; unanswered
  invites pile up as a spam signal (withdraw stale ones). Every connection is also a follower.
- **Followers:** unlimited — the scalable audience number.
- **Invite hygiene:** no pitch in the request; a short specific note when context isn't obvious;
  connect with people whose conversations you're already in (comment first, connect second).
  Mass-connect automation risks account restriction.

## Native growth surfaces
- **Newsletters:** subscribers get an in-app notification **and an email**; LinkedIn pushes
  subscription prompts to your network at launch. The strongest owned-audience feature on the
  platform (pairs with `email-and-newsletter` for the off-platform list).
- **Audio events / LinkedIn Live:** small reach, high trust; good for warm-audience depth, not
  discovery.
- **Collaborative articles:** low ROI for growth now (the gold "Community Top Voice" badge was
  retired); the invite-only blue **Top Voice** badge can't be applied for — ignore both as goals.
- **SSI (Social Selling Index):** free at linkedin.com/sales/ssi; four pillars × 25 (brand, right
  people, insights, relationships). Useful as a **directional health check** — never a KPI to chase.

## Commenting = the afterburner
A substantive comment on a **bigger account in your niche** is shown to *their* audience and to your
network — repeat visibility with the exact people you want, at zero posting cost. Practitioner
consensus: **5–10 real comments/day on 10–20 target accounts** grows profile visits faster than
extra posts. Early comments on fresh posts from large accounts get the most eyeballs.

## Cadence + the AI flood
- **3–5 posts/week, max 1/day** — a second post the same day cannibalizes the first's golden hour.
  Consistency beats bursts; the golden hour (first **60–90 min**) decides initial distribution.
- Studies peg **~half of long-form LinkedIn posts as AI-written**; the feed is saturated with
  generic competence. Specific, first-person, lived content is now the visible minority — and the
  algorithm's relevance push + human skepticism both reward it.
- **Penalized/corrosive:** engagement bait, pods (detected via mutual-engagement patterns),
  mass-tagging, bought followers, reposting hourly, broetry theatre.
