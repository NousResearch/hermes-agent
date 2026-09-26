---
name: social-strategy
description: >-
  Use to set a brand's overall social media strategy — which channels to focus on (and which to
  ignore), what to achieve, the angle to win, and a realistic operating cadence. The top-level
  plan that sits above content pillars and the calendar. Run when the user says "social
  strategy," "social media strategy," "which platforms should I be on," "what's my plan,"
  "should I be on TikTok/LinkedIn," or is starting or resetting their social presence. Reads
  brand-profile, audience-research, and content-pillars first. Makes real choices and
  trade-offs — especially what NOT to do — ties goals to the business not vanity metrics, and
  is honest that WoopSocial has no analytics (measurement is the platforms' native analytics).
  Produces a social-strategy.md. It names the objective; to turn it into SMART targets, KPIs,
  and North Star metrics, use goals-and-kpis. Works for any business.
metadata:
  version: 1.0.0
license: MIT
---

# Social Strategy

This is the top-level plan: which channels, what for, how to win, and at what cadence. It sits
above `content-pillars` (what themes) and `batch-content-plan` (what posts) and tells them where
to point. Everything downstream inherits the choices made here.

The core idea, and what separates strategy from a to-do list:

> **Strategy is choices and trade-offs — above all, what *not* to do.** "Be on every platform,
> post consistently, engage your audience" is not a strategy; it's a list of platitudes that
> spreads a brand thin and wins nowhere. A real strategy picks the few channels where the
> audience is and the brand can win, deliberately drops the rest, and commits to one clear
> objective and a differentiated angle.

## When to use this

- Starting a social presence from scratch, or resetting a scattered one.
- Deciding which platforms to be on (and which to quit).
- Before `content-pillars` / `batch-content-plan`, to set the objective and channels they serve.

**When NOT to use it:** if a current `social-strategy.md` exists and is sound, load it and move
on unless the user wants to revisit.

## Step 0 — Read the foundation first

Load `brand-profile.md`, `audience-research` (`audience.md`), and `content-pillars` if present.
Strategy is downstream of *who you are*, *who you're for*, and *what you talk about* — without
those it's guesswork. Run the missing ones first where it matters.

## Step 1 — Set one primary objective (tied to the business)

Pick **one** primary objective social must serve right now — awareness, audience growth,
community, authority/positioning, leads, or sales — chosen from what the **business actually
needs** (a pre-launch startup needs awareness/audience; an established one needs leads/sales).
Add at most one secondary. Define a **meaningful metric**, not a vanity one, and be honest about
measurement (see `references/goals-and-measurement.md`). To cascade the objective into SMART
targets, per-channel North Stars, and action thresholds, hand off to `goals-and-kpis` — this
skill names the objective; that one sets the numbers.

## Step 2 — Select channels (focus, don't spread)

The central decision. Score each candidate platform on four dimensions (see
`references/channel-selection.md`):

1. **Audience presence** — is the target audience actually there? (from `audience.md`)
2. **Format fit** — does the brand's natural content strength suit it? (don't lead with TikTok if
   the brand won't make video)
3. **Goal fit** — does the platform serve the objective?
4. **Capacity fit** — can the brand sustain quality there?

Pick the **1–3** that win on all four. **Name what you're deliberately dropping** (or running as
repurpose-only maintenance). Most brands should start with **one** primary channel.

## Step 3 — Define the strategic wedge

How does *this* brand win on its chosen channels, given everyone else? The wedge comes from the
brand's POV and positioning — a differentiated angle, a distinctive format, a consistent stance —
not "post good content" (that's table stakes, not a wedge). See `references/operating-model.md`.

## Step 4 — Set the operating model

A strategy you can't sustain is a wish. Define (see `references/operating-model.md`):

- **Cadence** per chosen channel — realistic, matched to capacity; consistency over volume.
- **The content engine** — how foundation → pillars → plan → produce → repurpose → schedule fit
  together.
- **Phasing** — Phase 1 nail one channel; expand only once it works. Don't launch everywhere.
- **Capacity/roles** — who makes and who engages, and how much time.

## Step 5 — Write the artifact

Produce `social-strategy.md`: the objective + metric, the chosen channels (and the dropped ones,
with reasons), the wedge, the operating model and phasing. Summarize back and invite edits.
`content-pillars` and `batch-content-plan` read this to stay aligned.

## Quality bar — self-check

- Is there **one clear objective**, tied to the business (not "grow followers" for its own sake)?
- Are **channels chosen by the fit framework**, with the dropped ones named? (Not "all of them.")
- Is there a real **wedge** — a differentiated way to win, from the brand's POV?
- Is the operating model **sustainable** and **phased** (start focused, then expand)?
- Is measurement **honest** — meaningful metrics, read from the platforms' native analytics, and
  no analytics promised that WoopSocial doesn't have?
- Would following this make the brand *choose* — or just do everything everywhere?

If the "strategy" doesn't say no to anything, it isn't one.

## Edge cases

- **"We should be on every platform"** → push back: focus beats spread; pick where the audience
  is and the brand can win; name what to drop or run as maintenance.
- **Wrong-format pressure** (wants TikTok but won't/can't make video) → match channels to the
  brand's real content strengths, or plan to build the capability first.
- **No clear business goal** → help define one from the brand's stage; a strategy without an
  objective optimizes nothing.
- **Already spread thin across 6 platforms** → recommend consolidating to the 1–2 that work; it's
  often a *subtraction* strategy.
- **B2B vs B2C** → channel and goal differ sharply (LinkedIn/X authority+leads vs IG/TikTok
  awareness); use the audience profile.
- **Measurement expectations** → be clear WoopSocial publishes; measurement is the platforms'
  native analytics (read manually) — don't promise dashboards.

## Related skills

- `brand-profile`, `audience-research`, `content-pillars` — read first; supply the inputs.
- `goals-and-kpis` — turns this strategy's objective into SMART targets, per-channel North
  Stars, and thresholds ("set our goals/KPIs" requests route there).
- `content-pillars` — the themes that serve this strategy.
- `batch-content-plan` — turns the strategy + pillars into a calendar.
- `cross-platform-repurposing` — makes a focused multi-channel strategy sustainable.
- The platform-growth skills (`instagram-growth`, `linkedin-growth`, …) — execute per channel.

## References

- `references/channel-selection.md` — the channel-fit framework and platform strengths.
- `references/goals-and-measurement.md` — objectives, meaningful vs vanity metrics, honest measurement.
- `references/operating-model.md` — the content engine, cadence, phasing, the wedge in practice.
- `references/examples.md` — worked strategies (with real trade-offs) across business types.
