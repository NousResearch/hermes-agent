---
name: opus-clip
description: >-
  The OpusClip craft skill — turn long video (podcasts, webinars, interviews, streams) into publishable
  short clips with AI candidate-finding, honest triage, and a human-reviewed pipeline. Use when someone
  wants to use OpusClip, turn a podcast/webinar/VOD into shorts, understand the Virality Score, pick a
  plan, pull a moment from a long recording (ClipAnything moment search), or automate a clipping
  pipeline. Uses the CLIPS framework. Reads captions-and-clipping + the source skill + brand-profile
  first. The agent plans the pipeline, credit math, and QA; the HUMAN reviews every clip and approves;
  WoopSocial publishes the finished exports (it does not clip or edit video). Integrity spine: the
  Virality Score is a PREDICTION for triage, never truth; 1 credit = 1 SOURCE minute; no unreviewed
  auto-post pipelines; no out-of-context clips. Distinct from captions-and-clipping (general craft),
  descript (long-form edit), capcut (short-form polish), and the platform publishing skills.
version: 1.0.0
---

# opus-clip

The **clip-pipeline tool skill** — choose the source, let AI find candidates, inspect and rank honestly, polish
the top few, ship on a drip. The agent plans and QA-checks; the **human reviews every clip**; **WoopSocial
publishes**. (Ships with `tools/integrations/opus-clip.md`; pairs with the `captions-and-clipping` craft skill.)

## The POV: a first-pass clip finder, not a one-click viral machine
OpusClip genuinely compresses 3–5 hours of manual clipping into minutes of processing plus review — its
**ClipAnything** engine is multimodal (visuals + audio sentiment + facial expressions), so it finds moments even
in wordless content, and natural-language moment search pulls one moment from a 2-hour VOD without scrubbing. But
the top-1% operator holds two facts the marketing won't volunteer. **(1) The Virality Score is triage, not
truth:** the only credible independent test found **~40% of generated clips get discarded** and the score
**regularly mispredicts** real performance — low scorers blow up, high scorers flop. Use it to rank where your
editing time goes; never as an auto-post threshold or a guarantee ("85% faster" is a self-published number with
no method — the ~40% discard is the figure you can plan around). **(2) The human gate is non-negotiable:** every
clip passes three questions — does it stand alone, does it **represent the speaker fairly** (the interview
meaning-spine applies to clips), does it serve the audience — before anything ships. And the billing mechanic
that catches everyone: **1 credit = 1 minute of SOURCE video, regardless of clips out** — trim before you upload.

## Read these first
1. **captions-and-clipping** — the general clipping/repurposing craft this tool executes.
2. The **source skill** (podcast-and-audiograms / youtube-long-form) + **brand-profile/design-and-templates**
   (the clip template).

## The framework: CLIPS
(Depth: `references/the-clips-framework.md`.)
- **C — Choose the source:** conversational long-form clips best (~8–12 candidates/hour); non-verbal genres work
  via multimodal analysis with higher discard — test on Free first; clean audio, no burned-in subs, trim before
  upload (credits bill on source minutes).
- **L — Let AI find candidates:** genre + length + AI-hook configured; natural-language moment search; negative
  prompting (exclude intros/sponsor reads).
- **I — Inspect and rank honestly:** score = triage; plan for ~40% discard; the human three-question pass —
  stands alone / fair / serves; the score never overrides judgment.
- **P — Polish the top few:** edit only the top 3–5, approve clean mid-scorers, discard without guilt; fix the
  hook's first line, caption errors, reframe drift, boundaries; one brand template across the batch; auto-music
  OFF → licensed/native audio at publish.
- **S — Ship on a drip:** batch-schedule spread across the week → WoopSocial; download exports promptly; study
  real retention vs the scores monthly and recalibrate.

## The reality (verify-quarterly)
2026 OpusClip: ~10M users / ~172M clips; ClipAnything (multimodal + moment search + negative prompts); speaker
diarization layouts; ReframeAnything tracking; ~97% caption accuracy **vendor-claimed** across 20+ languages (QA
anyway); AI B-roll; XML export to Premiere/Resolve (Pro). **The score:** 0–99 across Hook/Flow/Engagement/Trend —
independently found to mispredict; ~40% discard (BIGVU test, attributed); no independent benchmarks. **Credits:**
1 = 1 source minute; +1 per direct X post; failures reported eating credits; projects reported vanishing after
subscription ends — download promptly. **Tiers (≈, conflict — verify in-app):** Free = evaluation only (watermark,
3-day expiry, no score/editor); Starter ≈ $15/150 min; Pro ≈ $29 (~$14.50 annual)/300 min = the production tier;
Business = custom, **API reported Business-only**; editor gating conflicts across sources. Reliability: reported
processing hangs, no public status page/SLA — build slack. Trustpilot 4.0 with ~22% one-star (attribute). Full
detail: `references/opus-clip-2026-reality.md`. The weekly loop, per-clip QA, credit-math worksheet, score
recalibration, and two worked examples: `references/workflows-and-templates.md`.

## Honest scope (never violate)
- **The agent** plans (pipeline, prompts, triage, QA, credit math, drip) and drafts fixes; the **human reviews
  every clip** and approves (the agent can't see video; never fabricates "that clip works"). **Automation
  honesty:** API reported Business-only; Zapier/Make exist lower; **no unreviewed auto-post pipeline on any
  tier.** **WoopSocial publishes** approved exports (its scheduler-vs-OpusClip's is a stated choice); it does
  **not** clip/edit/score video; native audio/stickers are added in-app by the human.
- **The score spine:** a prediction — triage, never truth, never quoted as a metric. **The fairness spine:**
  clips keep context and meaning; no manufactured endorsements or gotchas (deception/defamation); consent/
  likeness rules apply. **Never fabricate** tiers, gates, capabilities, or metrics — verify in-app; build slack
  for the reliability reality. (Full scope: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**opus-clip (this)** = the OpusClip-specific pipeline · **captions-and-clipping** = the general clipping craft
(read first; `tools/integrations/clipping.md` covers the multi-tool layer — `tools/integrations/opus-clip.md`
goes deep on this tool) · **descript** = the long-form talk edit (master there; its Underlord clip flags overlap) · **capcut** =
short-form styling/polish beyond the built-in editor · **short-form-video-script** = the retention craft the QA
applies · **tiktok-video-publishing / instagram-reels-publishing / youtube-shorts** = platform publish specifics.

## Where this connects
Reads first: **captions-and-clipping** + the **source skill** + **brand-profile/design-and-templates.** Pulls the
master from: **descript** (the edited episode), **livestream-and-realtime** (VODs), webinars. Feeds: **capcut**
(polish), the **platform publishing skills**, **content-recycling**, **scheduling-and-queue.** Publishes via:
approved exports → **scheduling-and-queue → WoopSocial.** Tool file: **`tools/integrations/opus-clip.md`.**
Measure with: native + **analytics-and-reporting** on per-clip retention vs score (recalibrate monthly) — never
fabricated.

## Definition of done
A clip pipeline that treats OpusClip as a first-pass finder: source chosen and trimmed for the credit mechanic
(1 credit = 1 source minute), candidates generated with genre/negative prompts set, the batch triaged by score as
ranking only with a planned ~40% discard, every clip passed through the human three-question gate (stands alone /
fair to the speaker with context intact / serves the audience), only the top 3–5 edited (hook line, caption
accuracy on names/jargon, reframe drift, boundaries) with one brand template across the batch and auto-music off
in favor of licensed/native audio, the approved set drip-scheduled and published via WoopSocial with exports
downloaded promptly, and real retention tracked against the scores monthly to recalibrate; tier/editor/API gating
verified in-app before subscribing (API automation claimed only at the reported Business tier; no unreviewed
auto-posting anywhere); **no score treated as truth, no out-of-context clips, no fabricated tiers/capabilities/
metrics**; and correctly distinguished from captions-and-clipping, descript, capcut, and the platform publishing
skills.
