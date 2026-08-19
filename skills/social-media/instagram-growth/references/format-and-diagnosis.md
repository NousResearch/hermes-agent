# Format Strategy & Diagnosis

This is the orchestration core: which format does which growth job, and — when growth is stuck — how
to diagnose the **broken stage** and route to the skill that fixes it. Instagram-growth doesn't
re-do content; it decides *what's wrong* and *which skill to call.*

## Format strategy — map each format to its surface's signal

Don't treat formats the same; each wins a different signal on a different surface:

- **Reels → discovery / growth.** The primary non-follower reach engine (~2.35× the reach of static
  posts). Optimize for watch time + sends. This is the top of the funnel. → `reels-script`,
  `tiktok-script` (repurpose), `hook-writer`.
- **Carousels → saves + authority + follower reach.** High watch time and save rates; great for
  frameworks/how-tos/reference (up to ~20 slides). Mid-funnel depth and credibility. →
  `carousel-writer`.
- **Stories → retention / relationship.** Followers only; relationship-driven (replies, closeness).
  Deepen the warm audience, drive DMs and profile/link taps. Stickers (poll, question, slider) lift
  engagement; smaller accounts get a Stories reach boost. Not a reach tool — a *retention* tool.
- **Photos → low discovery.** Fine for the feed/aesthetic, weakest for reaching non-followers.

The strongest accounts run **Reels for reach + carousels for saves/authority + Stories for
retention** as one system, not interchangeable posts.

## Niche clarity is now a ranking edge

Instagram must be able to **categorize** your account to recommend it. With user-facing controls
("Your Algorithm") letting people tune topics, **multi-niche accounts get throttled** — the system
can't tell who to show you to. Define a **tight lane**. Reinforce the topic with **keywords in the
caption, on-screen text, and spoken audio** (Instagram SEO; route to `hashtag-strategy`/`social-seo`
for the keyword layer). Clear niche → easier reach + easier every other signal.

## Cadence — sustainable, consistent

Consistency beats volume. A commonly cited healthy rhythm is **~2 Reels/week + a few feed posts**,
held sustainably; posting once a week or less reduces priority, and high volume at low quality
dilutes each post's signals. **Match cadence to capacity** and build the recurring system in
`content-calendar` — don't prescribe a heroic schedule that gets abandoned.

## The diagnosis — symptom → broken stage → fix

Growth is stuck at a **specific stage** of the loop. Diagnose it, then route:

| Symptom | Broken stage | Primary fix (skill) |
|---|---|---|
| Low reach / few non-follower views | **Reach** — weak hook/watch time, not enough Reels, not original, watermarked, unclear niche | `reels-script`, `hook-writer`, niche clarity, originality |
| Decent views but **few follows** | **Follow-conversion** — the profile doesn't convert | `profile-optimization` |
| Views but content **doesn't travel** | **Sends** — not share-worthy | `the-sends-lever.md`, `viral-reverse-engineering` |
| Followers **don't engage / drift** | **Retention** — weak Stories/community/relationship | Stories + broadcast channel + community/engagement routine |
| Inconsistent posting / burnout | **Sustainability** — no system | `content-calendar` |
| Not found in search | **Discoverability** — weak keywords/niche | `hashtag-strategy` / `social-seo`, `profile-optimization` |
| Reach collapsed suddenly | **Eligibility** — watermarks/reposts/strikes | originality, remove watermarks, check guidelines |

Most "Instagram isn't growing" problems are **one** of these stages — find which, fix that, don't
spray generic tips.

## What doesn't work (skip these)

- **Buying followers / engagement pods / follow-for-follow** — fake engagement hurts reach and risks
  the account.
- **Reposting watermarked TikToks** — throttled by the originality gate.
- **Chasing follower count** — vanity; optimize reach/sends/saves/profile-visits/leads.
- **#fyp-style generic tags** — see `hashtag-strategy`.
- **Grid-aesthetic obsession** — a pretty grid isn't a growth lever in a discovery-first feed.
