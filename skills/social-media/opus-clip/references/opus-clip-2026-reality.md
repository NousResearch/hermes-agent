# The reality of OpusClip in 2026 (verify-quarterly)

Tiers, gates, and reliability details shift — attribute everything, **verify-quarterly**, verify tiers in-app.

## What it is (and its honest ceiling)
OpusClip (opus.pro; ~10M users, ~172M clips generated; SOC 2 Type II; $215M valuation after a Mar 2025 SoftBank
round — attribute) turns long video into scored, captioned, reframed vertical shorts. Its engine, **ClipAnything**,
is multimodal — visual cues + audio sentiment + facial expressions + emotional peaks — so it works beyond
talking-head (gaming, sports, wordless), plus **natural-language moment search** ("find where the comeback
starts") and **negative prompting** (exclude intros/sponsor reads). Speaker diarization drives active-speaker /
split-screen layouts; ReframeAnything keeps subjects centered in vertical; captions self-reported ~97% accurate
across 20+ languages (vendor-claimed — QA anyway). Best genre remains talk-heavy long-form; a podcast episode
reliably yields ~8–12 candidates.

## The integrity center: the Virality Score is triage, not truth
Every clip gets a 0–99 score across Hook / Flow / Engagement / Trend — a proprietary **prediction**, and the
evidence says treat it skeptically:
- **The one credible independent test (BIGVU, attributed) found ~40% of generated clips get discarded as
  unusable, and the score regularly mispredicts real performance.** Users widely report low scorers blowing up
  and high scorers flopping.
- There are **no independent benchmarks**; the "85% faster than manual" headline is self-published with no
  disclosed method. The only number you can plan around is the **~40% you'll throw away.**
- The right use: **rank a big batch to prioritize editing time** (edit the top 3–5, approve clean mid-scorers,
  discard the rest) — never an auto-post threshold, never a guarantee. Track your real retention vs the scores
  and recalibrate.

## The credit mechanics (the billing trap — attribute)
**1 credit = 1 minute of SOURCE video processed, regardless of clips out** (a 60-min upload = 60 credits whether
it yields 5 or 15). Small traps: +1 credit per direct X post; failed processing reported eating credits before
refunds; **projects reported to vanish after a subscription ends — download exports promptly** (Trustpilot
patterns; 4.0/5 with ~22% one-star reviews citing processing failures, credit mechanics, cancellation friction).
Credit hygiene: trim the source before upload; don't process what you won't clip; map monthly source-minutes to
the tier first.

## Tiers (≈, conflicts noted — verify in-app)
**Free** = evaluation only (60 min/mo, watermark, 3-day clip expiry, 9:16 only, no score/editor). **Starter ≈
$15** (150 min, score, basic posting). **Pro ≈ $29 (~$14.50 annual)** = the real production tier (300 min, all
aspect ratios, scheduler, AI B-roll, XML export to Premiere/Resolve, team workspace). **Business** = custom —
and the **API is reported Business-only** (Zapier/Make integrations exist lower). **Sources conflict on which
tier unlocks the editor** (Starter-included vs Pro-gated reports) — verify in-app before subscribing.
Operational reality: reported processing hangs ("stuck at 96%"), **no public status page or SLA** — build slack
into deadlines.

## House metrics
Real per-clip retention/watch-through + follows/inquiries from clips (native) — compared against the scores to
recalibrate; never fabricate a metric or treat a score as one.
