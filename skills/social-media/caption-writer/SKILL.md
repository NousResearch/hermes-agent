---
name: caption-writer
description: >-
  Use to write platform-native social media captions (post copy) in the brand's voice — the
  words that accompany a post on Instagram, LinkedIn, TikTok, Facebook, X/Twitter, Threads,
  Pinterest, YouTube, or Bluesky. Run when the user says "write a caption," "caption this,"
  "post copy," "write the copy for this photo/video/Reel," "IG caption," "what should I write
  for this post," or needs the words to go with a visual. Reads brand-profile and voice first
  so captions sound like the brand, not generic AI. For long-form LinkedIn thought-leadership
  posts use linkedin-post-writer; for X/Threads multi-post threads use thread-writer; for video
  scripts use the video skills. This skill writes the caption itself.
metadata:
  version: 1.0.0
license: MIT
---

# Caption Writer

A caption is not a description of the post. It's the part of the post that does the *work* —
it stops the scroll, earns attention, and drives one action. Done well it carries the
post; done generically it sinks even great visuals under emoji and hashtag noise.

Two principles run through everything here:

1. **The first line is the whole game.** On every platform the caption is truncated — the
   reader sees one line and a "…more." If that line doesn't earn the tap, the rest is never
   read. Write the first line like its only job is to win the second line.
2. **Every caption serves one goal.** Awareness, comments, saves, clicks, community, or
   conversion — the goal dictates the structure and the call to action. A caption "to get
   engagement" is a caption with no goal.

## Step 0 — Read the foundation first (required)

Load `brand-profile.md` and `voice.md` before writing a word. The voice fingerprint, the
"never" list, and the brand guardrails are non-negotiable. A caption that doesn't sound like
them is a failure even if it's clever. If neither exists, run `brand-profile` (and
`voice-builder` if samples exist) first.

## Step 1 — Establish four inputs

Don't write until you know these. Infer what you can; ask only what you can't:

- **Platform** — the mechanics are different on each (see `references/platforms.md`).
- **The asset** — what's in the photo/video/carousel? The caption complements the visual; it
  shouldn't narrate what the eye already sees.
- **The goal** — awareness, comments, saves, clicks/traffic, community, or conversion.
- **The topic / point** — what this specific post is about, and the one idea it lands.

If the user gives you only "write a caption for this," ask at most one or two questions
(usually platform + goal) and proceed.

## Step 2 — Choose the format, then the goal-driven structure

Pick the caption **format** that fits the asset and point — story, value/how-to, hot take,
question, behind-the-scenes, social proof, announcement, relatable (see
`references/formats.md`). Then shape it with the **goal-driven structure** and CTA from
`references/frameworks.md`. Format gives it a recognizable shape; goal gives it a job.

## Step 3 — Write the first line first

Spend disproportionate effort here. Draft 3–5 candidate opening lines and pick the one that
most makes a stranger need the next line — within the platform's preview cutoff. Never waste
the first line on a greeting, a hashtag, "Happy Monday!", or a restatement of the image. See
`references/first-line.md`.

## Step 4 — Build the body and one CTA

- **Body:** deliver the value or story the first line promised, in the brand's voice, at the
  right length for the platform and goal (short isn't always better — match it). Use line
  breaks and whitespace where the platform rewards scannability.
- **CTA:** exactly one, matched to the goal ("save this for your next launch," "what would you
  add?", "link in bio"). Two CTAs split the action and kill both. Respect any "never hard-sell
  here" guardrails from the brand profile.

## Step 5 — Mechanics: hashtags, links, formatting

Apply the platform's current norms from `references/platforms.md`: how many hashtags (and
whether to use any), where links go and whether they hurt reach, line-break rendering, and the
ideal length. Getting mechanics wrong makes a good caption underperform.

## Step 6 — Deliver options, recommend one

Give the user **2–3 distinct caption options** (genuinely different angles or formats, not the
same caption reworded), then recommend one with a one-line rationale tied to the goal. People
choose faster from real alternatives than from a single take.

## Quality bar — self-check before delivering

- Does the **first line** stop the scroll on its own, inside the preview cutoff?
- Does it **sound like them** — fingerprint hit, "never" list respected, banned words absent?
- Is there **one** clear CTA matched to the goal?
- Does the caption **complement** the asset rather than describe it?
- Are hashtags/links/length **correct for the platform**?
- Would a real person stop, read, and act — or scroll past?

If the first line is weak, rewrite it before anything else. It's 80% of the result.

## Edge cases

- **No asset described:** ask what's in it, or write a caption that works for the stated topic
  and flag that the first line may need tuning to the visual.
- **Multiple platforms requested:** don't paste the same caption everywhere. Adapt the hook,
  length, hashtags, and CTA to each platform's mechanics (see `cross-platform-repurposing` for
  doing this at scale).
- **Regulated/sensitive brand:** apply the brand profile's compliance guardrails — no
  forbidden claims, required disclaimers, careful CTAs.
- **Carousel / multi-image:** the caption supports the swipe — the first line should pull
  people *into* slide 1; consider a "keep reading / swipe" nudge only if it fits the voice.
- **Link-driven goal on a no-link platform** (e.g., Instagram feed): route to "link in bio"
  and say so; don't write a clickable-link CTA where links aren't clickable.
- **Very short-form platforms** (X, Threads, Bluesky): the caption *is* the whole post — lead
  with the point, cut everything else.

## Related skills

- `brand-profile`, `voice-builder` — read first; supply voice and guardrails.
- `hook-writer` — for deeper work on the opening hook.
- `linkedin-post-writer`, `thread-writer` — specialized long-form / thread copy.
- `hashtag-strategy` — for a fuller hashtag system.
- `cross-platform-repurposing` — adapt one caption across many platforms.
- `scheduling-and-queue` — push the finished caption to WoopSocial to schedule.

## References

- `references/platforms.md` — per-platform caption mechanics (limits, fold, hashtags, links).
- `references/frameworks.md` — caption structure, goal→structure mapping, length, CTA library.
- `references/formats.md` — caption formats/archetypes and when to use each.
- `references/first-line.md` — opening-line craft; winning the "…more" tap.
- `references/examples.md` — worked captions across platforms and goals.
