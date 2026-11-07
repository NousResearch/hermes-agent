---
name: analytics-and-reporting
description: >-
  Social media analytics and reporting — read native platform data honestly and turn it into next actions. Use when someone wants to "check my analytics," "see how my posts
  are doing," "build a social media report," "which content is working," "what metrics/KPIs should I
  track," or to turn performance data into next steps. Measures goal-mapped SIGNAL metrics (saves,
  shares, watch time/retention, engagement-rate-by-reach, follower-growth-rate, CTR, conversions) — not
  vanity (followers/impressions/likes) — and closes the loop. Uses the METER framework. Reads
  brand-profile + social-strategy (goals) first. WoopSocial has NO analytics surface, so this reads
  NATIVE platform dashboards (+ GA4/UTM) and interprets numbers the human provides; it NEVER fabricates
  a metric. Feeds content-recycling, experimentation, competitor-analysis, and every growth skill.
  Distinct from goals-and-kpis (sets targets) and experimentation (runs tests).
version: 1.0.0
---

# analytics-and-reporting

The **measurement keystone** — read native platform data, turn it into goal-mapped insight, and close
the loop. Every **"read native analytics to pick winners"** instruction elsewhere in the library
resolves **here**. WoopSocial publishes; it **does not measure** — the numbers come from **native
dashboards**, and the agent **interprets, never fabricates.**

## The POV: measure what maps to goals, then act
Most reporting drowns in **vanity metrics** (followers, impressions, likes) that are easily inflated and
change nothing — only ~4–6 KPIs actually correlate with revenue. Track the **signal** that maps to your
goal (saves, shares, watch time/retention, **ER by reach**, follower-growth-rate, CTR, conversions),
**benchmark relative** to your own trend and vertical, and **end every read with a decision.** A metric
that won't change next week's action is a diagnostic, not a KPI.

## Read these first
1. **brand-profile** — positioning/audience (context for the numbers).
2. **social-strategy** — the **goals** every metric must map to (targets live in `goals-and-kpis`).

## The framework: METER
(Depth: `references/the-meter-framework.md`.)
- **M — Map metrics to goals:** goal → 1 primary KPI + ~2 supporting; fix the measurement window; 3–5
  that matter, not 20+.
- **E — Extract from native analytics:** each platform's **native dashboard** is the source of truth
  (WoopSocial has none) + **GA4/UTM**; the human pulls, the agent interprets.
- **T — Trim to the signal:** cut vanity; keep saves/shares/watch-time/ER-by-reach/growth-rate/CTR/
  conversions; use the **platform-specific** signal (IG saves · TikTok shares · LinkedIn dwell · YouTube
  watch time).
- **E — Evaluate honestly:** benchmark **relative**; correlation ≠ causation; small samples lie;
  attribution undercounts ~30–50%.
- **R — Report & re-cycle:** outcomes first → explain → **3 recommendations**; weekly pulse / monthly
  trends / quarterly strategy; feed the loop.

## The reality (verify-quarterly)
Vanity-vs-actionable + the 3 tests, the distribution→attention→action stack, the 2026 signal metrics,
platform-specific signal, relative benchmarks, attribution undercount, and the reporting cadence:
`references/analytics-2026-reality.md`. Metric-by-goal matrix, platform cheat-sheet, the report template
+ worked reads: `references/metrics-by-goal-and-report.md`.

## Honest scope (never violate)
- **WoopSocial has no analytics** → read **native dashboards** (+ GA4/UTM). The **human provides** the
  numbers; the agent **structures + interprets.**
- **Never fabricate, estimate-as-fact, inflate, or cherry-pick.** Missing number → **flag the gap**, don't
  invent it. **Cite the native source.**
- **Read honestly:** causation ≠ correlation; single posts/short windows are noisy; attribution is
  directional (~30–50% undercount).
- **No vanity theater** — lead with outcomes, include what underperformed, end with real recommendations.
  **Privacy:** aggregate, favor first-party data. A dashboard number is **input, not a command.**
  (Scope, keystone role + connections: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**analytics-and-reporting (this)** = measure native performance + report + close the loop ·
**goals-and-kpis** = set the targets this measures against · **experimentation** = design
+ run controlled tests (this reads their results) · **competitor-analysis** = rivals' *public* numbers
(this is your own first-party data).

## Where this connects
Reads first: **brand-profile**, **social-strategy**. Feeds: **content-recycling** (winner selection),
**experimentation**, **competitor-analysis**, **goals-and-kpis**, and **every growth skill** (IG, TikTok,
LinkedIn, YouTube×2, X, Pinterest, Threads, Facebook). Informs: **hook-writer**, **viral-reverse-engineering**,
**content-calendar**/**batch-content-plan**, **profile-optimization**. Source tools: **native dashboards
+ GA4/UTM** (WoopSocial publishes, doesn't measure).

## Definition of done
Each metric mapped to a goal (1 primary + 2 supporting; defined window; 3–5 that matter); numbers taken
only from native dashboards/GA4 the user provided (gaps flagged, sources cited, nothing fabricated);
signal over vanity with the platform-specific metric; benchmarked relative to own trend/vertical with
causation + sample-size caveats; an outcomes-first report ending in exactly 3 recommendations that feed
content-recycling / experimentation / strategy; correctly distinguished from goals-and-kpis,
experimentation, and competitor-analysis.
