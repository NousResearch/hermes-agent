# Reels Mechanics — Instagram's discovery engine

Reels are how Instagram distributes content to **non-followers** — most Reel reach comes from people
who don't follow you, decided per post by how viewers respond. Understanding what the Reels ranking
system rewards (and what disqualifies a Reel before ranking even starts) is the whole game — and it
differs enough from TikTok that a TikTok reposted to Reels usually underperforms.

> Specs, lengths, and features change fast — re-verify quarterly. The principles are durable.

## Discovery-first, not follower-first

The vast majority of Instagram distribution is AI recommendation, not the follower graph — reach is
earned per Reel, not borrowed from follower count. A small account can out-reach a big one on a
single Reel. That also means **every Reel has to earn its reach from scratch**, and follower count
is a vanity number next to reach, sends, and saves.

## The signals that expand reach

Instagram has been unusually explicit about its ranking signals. In rough order of weight:

1. **Watch time / watch-through** — the single most important signal. The **first ~3 seconds**
   decide whether a Reel gets pushed wider or throttled. Completion and **rewatches** are strong.
2. **Sends per reach (DM shares)** — the dominant signal for reaching *new* people, weighted
   roughly **3–5× more than likes** for discovery. A Reel people forward travels.
3. **Likes and comments per reach** — still count, but matter more for existing followers than for
   discovery.

The practical combination that travels: **a hook that holds (watch time) + an idea worth forwarding
(sends).** Everything in `hooks-and-retention.md` exists to maximize those two.

## The eligibility gates (before ranking happens)

A Reel must clear these to be recommended to non-followers at all:

- **Original content** — reposts get far less distribution; heavy reposting can exclude the whole
  account from recommendations.
- **No other-platform watermarks** — a TikTok or CapCut logo is a known, avoidable reach killer.
  Export clean; never repost the watermarked file.
- **Has audio, under the length cap, safe/suitable** for a broad audience.
- **Clear niche** — the system has to be able to categorize you to know who to show you to.

Made-for-Instagram, original, watermark-free is the price of entry.

## Sound-off is the default viewing state

Most Reels viewers watch **on mute** (feed and DM contexts especially). So the script is written
**muted-first**: the on-screen text must carry the hook and the story on its own; the audio adds
energy and nuance for the minority who unmute. Burned-in captions lift retention — always on, kept
inside the safe zone (see `instagram-reels-publishing`).

## Length tiers

- **~15–35s** — the reach sweet spot: easiest to hold watch-through and loop.
- **~35–90s** — earns its length for tutorials, storytelling, and transformation arcs; re-hook in
  the middle.
- **Up to ~3 min** — still eligible for non-follower recommendations (longer uploads exist via
  third-party tools but lose recommendation reach — verify current caps).
- **Let the idea pick the length.** Padding kills watch-through; dead air kills everything.

## The Reel lives in three places — plan the frame

A Reel renders full **9:16** in the Reels tab, cropped in the feed, and cropped again on the
**profile grid** — so keep hook text, faces, and CTAs centered, and plan the **cover frame** at
script time. The cover is chosen at publish and **can't be edited after**; exact specs, safe zones,
and the cover brief live in `instagram-reels-publishing`.

## Trial Reels — the free test

Instagram's **Trial Reels** show a Reel **only to non-followers** first. If it performs cold, roll it
out to followers; if not, you've learned at zero cost to your follower-facing stats. Use trials for
unproven hooks, formats, and topics — it's the cheapest experiment on the platform.

## Cadence, measurement & disclosure

- **Consistency over bursts** — regular Reels give the system more tests; treat posting-time advice
  as general guidance, not analytics (WoopSocial has no analytics surface — judge via native
  Instagram Insights: watch time, sends-per-reach, saves, profile visits).
- **AI-generated media** — follow Meta's AI-labeling rules and the brand's policy; disclosure on
  Instagram is native/manual (see `veo-prompt-pack.md`). Never fabricate testimonials or stats.
