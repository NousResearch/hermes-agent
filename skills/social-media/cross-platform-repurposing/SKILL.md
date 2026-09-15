---
name: cross-platform-repurposing
description: >-
  Use to turn one piece of content into multiple platform-native posts — adapt, never
  copy-paste. Run when the user says "repurpose this," "turn this into posts for every platform,"
  "cross-post," "make this work on TikTok/LinkedIn/X too," "one idea into a week of content,"
  "turn this blog/video/podcast into social posts," or wants to fan content out. Reads
  brand-profile and voice first, extracts the core idea, then re-natives it for each platform
  (different hook, format, length, mechanics, CTA) using the content skills, and hands the
  fan-out to scheduling-and-queue. This is the multiplier that makes a content operation
  sustainable — but every version must stand on its own and feel native, never duplicated.
metadata:
  version: 1.0.0
license: MIT
---

# Cross-Platform Repurposing

This skill turns one idea into many posts — the single biggest lever for producing content
sustainably. One blog post becomes a week of social; one strong insight becomes five
platform-native posts. It's how a small team posts like a big one.

The principle that makes it work — and the trap that ruins most attempts:

> **Repurposing is not reposting.** Pasting the same caption onto every platform is the lazy
> version: it fits none of them natively, it looks like spam to anyone who follows you in two
> places, and several platforms quietly suppress obviously cross-posted content. The craft is
> **atomize → adapt**: extract the core idea, then rebuild it natively for each target — a
> different hook, format, length, and CTA, same truth and voice.

## When to use this

- Fan one post out across platforms (one → many).
- Atomize a long piece (blog, video, podcast, newsletter) into many short posts (long → short).
- Convert one format into others (a written insight → a carousel, a Reel, a thread).
- Inside `batch-content-plan`, to stretch limited raw material into a full calendar.

**When NOT to use it:** when content is genuinely platform-specific (an in-joke, a
platform-native trend, a reply) — forcing it elsewhere reads wrong. And when the audience
overlaps almost entirely on two platforms, vary the *angle*, don't just relocate the post. Know
when a piece shouldn't travel.

## Step 0 — Read the foundation first

Load `brand-profile.md` and `voice.md`. Voice stays constant across platforms; **register
shifts** per platform (use the voice tone-map — LinkedIn more formal, TikTok looser). Carry
guardrails everywhere.

## Step 1 — Atomize: find the core idea

Extract the reusable **atom** from the source — the point, the insight, the story, the stat, the
moment. For a single post that's usually one atom; for long-form (a blog, a 20-minute video) it's
**several** atoms, each strong enough to be its own post. Don't summarize a long piece into one
weak post — pull its best 5–10 atoms. See `references/repurposing-model.md`.

## Step 2 — Pick targets and direction

Decide the **platforms** and the **direction** of the repurpose (one→many, long→short,
format→format, video↔text — see `references/source-types.md`). Direction determines the
mechanics.

## Step 3 — Re-native per platform (don't translate, rebuild)

For each target, rebuild the atom natively — composing the right skills:

- **A new hook per platform** (`hook-writer`) — what hooks on a LinkedIn feed differs from a
  TikTok first frame.
- **The right format and length** for that platform.
- **The copy** (`caption-writer`), the **script** if video (`reels-script`), or the **thread**
  (`thread-writer`).
- **Native mechanics** — hashtags, links (bio vs clickable vs comments), CTA.

What stays the same: the **atom** and the **voice**. What changes: hook, format, length,
structure, CTA, mechanics, register. See `references/platform-transforms.md`.

## Step 4 — Quality-check: does each stand alone and feel native?

Every version must pass two tests:

1. **Stands alone** — someone who only sees this one post gets full value (no "as I said on
   LinkedIn…").
2. **Feels native** — it looks built *for* this platform, not relocated. If you can tell it was
   copy-pasted, rebuild it.

If two platforms share most of the audience, change the *angle* between them, not just the
formatting.

## Step 5 — Hand off to scheduling

Pass the finished, per-platform versions to `scheduling-and-queue` to schedule/fan out (it
confirms and validates first; WoopSocial applies per-platform fields). Space them out — don't
fire the same idea everywhere at the same minute.

## Quality bar — self-check

- Did you **atomize** (extract the core idea) rather than translate the wrapper?
- Is each version **genuinely native** — different hook, format, length, CTA — not copy-paste?
- Does each **stand on its own**?
- Is the **voice constant** with the **register shifted** per platform?
- For long-form, did you pull **multiple atoms**, not one summary?
- Did you recognize anything that **shouldn't** be repurposed?

If two versions are the same text with different hashtags, you reposted — go back to Step 3.

## Edge cases

- **Heavy audience overlap across platforms** → vary the angle/format between them so followers
  don't see the same post twice.
- **Platform-specific content** (trends, replies, in-jokes) → don't force it elsewhere; say so.
- **Long-form with one idea** → it's one atom; don't manufacture filler to hit a post count.
- **Video → text** → pull the spoken insights/quotes as written atoms; don't just transcribe.
- **Text → video** → hand the atom to `reels-script` to build it as a video, don't read the
  caption aloud.
- **Regulated/sensitive** → carry compliance guardrails into every version.

## Related skills

- `brand-profile`, `voice-builder` — voice + register + guardrails.
- `hook-writer` — a fresh hook for each platform.
- `caption-writer`, `reels-script`, `thread-writer` — build each native version.
- `content-recycling` — repurposing across **time** (refresh evergreen later).
- `captions-and-clipping`, `opus-clip` — cutting clips from long video (podcasts, webinars) at depth.
- `batch-content-plan` — uses this to stretch material into a calendar.
- `scheduling-and-queue` — schedules/fans out the finished versions.

## References

- `references/repurposing-model.md` — atomize → adapt; what stays vs changes; the directions.
- `references/platform-transforms.md` — how one idea becomes each platform's native post.
- `references/source-types.md` — repurposing by source/direction (long-form, fan-out, cross-medium).
- `references/examples.md` — one idea → five native versions, and a long-form atomization.
