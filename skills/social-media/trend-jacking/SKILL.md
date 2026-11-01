---
name: trend-jacking
description: >-
  Use to ride a trend — a trending sound, format, meme, hashtag, news moment, or cultural
  conversation — to earn reach by attaching the brand to something already getting attention (a.k.a.
  trendjacking / newsjacking). Run when the user says "trend," "trending," "jump on this trend,"
  "trendjack," "newsjack," "is this trend worth doing," "make this go viral," or wants to react to a
  current moment. Reads brand-profile, voice, and audience first, then runs three gates — FIT,
  SAFETY, and TIMING — before executing a fast, on-brand remix and handing to the right content
  skill (tiktok-script, reels-script, caption-writer) and scheduling-and-queue. Refuses to trendjack
  tragedies or divisive moments and won't force irrelevant trends; live trend discovery needs
  current scanning — this skill vets and executes. For meme-format craft, cultural fluency, and meme
  IP/likeness judgment ("make a meme," "is this meme safe"), route to meme-and-culture.
metadata:
  version: 1.0.0
license: MIT
---

# Trend-Jacking

Riding a trend can earn reach a brand could never buy — or get it dragged. The difference is
discipline. Trend-jacking works only when a trend is **a fit, safe to touch, and still rising**, and
most brand failures come from skipping one of those checks: forcing an irrelevant trend, touching a
sensitive moment, or arriving late.

So this skill is mostly **judgment**. Before anything gets made, a trend runs three gates:

1. **FIT** — can you add a genuine angle, or are you just slapping your logo on someone's moment?
2. **SAFETY** — would a brand inserting itself here look opportunistic, tone-deaf, or exploitative?
3. **TIMING** — is the trend still on the way up, or already saturated?

**Any single fail = don't do it.** Fit and safety are checked *before* timing — a fast post on an
unfit or unsafe trend is just a fast mistake. (Full framework: `references/the-three-gates.md`.)

> Honest scope: this skill **vets and executes**. It does **not** know what's trending right now —
> live trends change hourly and can't be recalled from training data. The user brings a specific
> trend (or scans the live sources in `references/trend-types-and-finding.md`); the skill judges and
> rides it.

## Step 0 — Read the foundation first

Load `brand-profile.md`, `voice.md`, and `audience.md` — they decide FIT (does it match the brand +
audience?), the voice of the remix, and the SAFETY guardrails.

## Step 1 — Gate 1: FIT

Is there an **authentic connection** and a **real angle** your audience would enjoy? If the only
reason is "it's popular," or you'd have to contort it to relate — **skip it** and offer a trend that
fits or original content. Not every trend is your trend.

## Step 2 — Gate 2: SAFETY

Screen for sensitivity and brand-safety. **Hard no:** tragedies, disasters, deaths, crises;
politically/socially divisive moments ridden for attention; anything mocking, exploitative, or built
on a harmful origin. When in doubt, sit it out. (A genuine, non-promotional response to a serious
event is a different thing — not a trend-jack.) See `references/the-three-gates.md`.

## Step 3 — Gate 3: TIMING

Where is the trend in its lifecycle (emerging → rising → peak → saturated → dead)? **Rising = go;
saturated/dead = skip** (late reads as out-of-touch). Once fit and safety pass, move fast — the
window is short.

## Step 4 — The remix (your angle, your voice)

Take the trend's **structure/sound/format** and fill it with **your niche's specific truth** — not a
logo-slap copy. Keep the brand voice (no forced slang). Add wit/value/relatability so it earns the
spot. See `references/the-remix.md`.

## Step 5 — Execute fast + schedule

Hand execution to the right content skill so it's still well-made — `tiktok-script` / `reels-script`
(short video), `caption-writer` (text), `carousel-writer` (carousel) — then publish via
`scheduling-and-queue` while the window's open. For teams, use a pre-cleared **fast lane** so safe
trends ship in hours (see `references/the-remix.md`).

## Quality bar — self-check

- Did the trend clear **all three gates**, checked in order (fit, safety, then timing)?
- Is there a **real brand angle**, not a logo-slap copy?
- Is it **in the brand voice**, not forced trend-speak?
- Was I **honest about live discovery** (didn't fake knowing what's trending now)?
- Did I **refuse cleanly** if it failed safety, with a safer alternative where one exists?
- Is this a **portion of the mix** (reactive), not the whole identity?
- **No analytics claims** WoopSocial can't back?

## Edge cases & refusals

- **Tragedy/disaster/death** → refuse; explain the exploitation risk; genuine non-promotional help
  only, or stay out.
- **Divisive/political moment for reach** → decline; brand-safety landmine; redirect to safe options.
- **Forced irrelevant trend** → fit fails; skip or find a fitting format.
- **Dead/saturated trend** → flag the window's closed; find a current one.
- **"Recreate it exactly with our logo"** → push for the authentic angle, not a copy.
- **"Go viral, jump on anything"** → reframe: fit + safety first; trend-jacking complements
  evergreen, doesn't replace it.
- **"What's trending right now?"** → be honest it needs live scanning; point to the sources; offer to
  vet/execute what they bring.

## Related skills

- `brand-profile`, `voice-builder`, `audience-research` — fit, voice, guardrails.
- `meme-and-culture` — meme-format craft + cultural-fluency + IP/likeness judgment; pairs with this
  (this skill owns the speed/timing mechanics).
- `tiktok-script`, `reels-script`, `caption-writer`, `carousel-writer` — execute the remix.
- `content-calendar` — owns the reactive slots this fills; `hook-writer` — the hook.
- `scheduling-and-queue` — ship it fast; `viral-reverse-engineering` — why things spread.

## References

- `references/the-three-gates.md` — FIT, SAFETY, TIMING: the decision framework and the hard-no list.
- `references/trend-types-and-finding.md` — trend types, how each is ridden, where to find them, the live-scope honesty.
- `references/the-remix.md` — execution: the angle, voice, speed, the fast lane, don't-over-chase, honest measurement.
- `references/examples.md` — worked trend-jacks and refusals (tragedy, forced, dead, divisive).
