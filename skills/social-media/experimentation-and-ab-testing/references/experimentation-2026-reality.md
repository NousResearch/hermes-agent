# Experimentation & A/B testing 2026 — verified

*Method is stable; re-verify platform specifics quarterly.*

## What it is
Publish two variants (**A** and **B**) that differ in **one deliberate way**, compare them under
**similar conditions** against a **single primary metric**. The goal is **evidence over opinion** — learn
what reliably moves your **KPI**, not chase vanity metrics. (It's the **causation** tool of the
measurement loop: `goals-and-kpis` sets the target, this isolates what moves it, `analytics-and-reporting`
reads the result.)

## The cardinal rules
- **One variable at a time.** Change only the hook *or* caption *or* time *or* format — everything else
  identical. The #1 mistake is changing several at once → you can't attribute the result to anything.
- **Set the decision rule BEFORE publishing.** Pick the **primary metric + win threshold + a guardrail**
  up front (e.g. *"B wins if reach +15% AND saves-per-reach not worse"*). No post-hoc rationalizing (HARKing).
- **Enough duration + sample.** Run **≥7 days** (daily variation); large accounts ~3–7 days, **small
  accounts (<10k) ~2–4 weeks**. Don't call it after 24h.
- **Significance, honestly.** Organic rarely reaches true statistical significance (small N, the
  algorithm intervenes) → look for a **consistent ~20%+ effect across multiple tests**, not a one-time
  winner. Paid: use platform/Optimizely/VWO significance (aim ~95%). **A tie is valuable** — that element
  doesn't move behavior, so test elsewhere.
- **Repeat.** Aim for **3–5 paired tests** before declaring a new "best practice." One winner is noise; a
  pattern is signal.
- **Control the context.** Same platform, format, similar topic, length, and posting window.
- **Document.** Keep a testing log (hypothesis, variant, result, learning) or you re-learn the same things.

## What to test (organic, in your control)
- **Creative:** the **hook / first-frame** (highest leverage on short video — first 3 seconds),
  thumbnail, caption length/style, **CTA**, visual style, format (Reels vs feed).
- **Distribution:** **posting time** (easiest to start, often high impact), frequency.
- **Offer:** CTA wording, landing-page angle, hashtag strategy (branded vs generic, count, placement).
- Test elements **within your control** — not algorithm-dependent factors. Prioritize what affects the
  **KPI**.

## The process (6 steps)
1. **Choose the test type** (creative / distribution / offer). 2. **Lock the variable** (one element).
3. **Control the context** (same platform/format/topic/length/window). 4. **Define success** (primary
metric + decision rule + guardrail, pre-set). 5. **Run repetitions** (3–5 paired). 6. **Document + scale**
(log it; roll the winner into the playbook → `content-recycling`).

## Cadence
Run a **30-day sprint**: week 1 hooks (3 paired posts, same topic/windows), week 2 first-frame visuals,
etc. **Always have at least one test running**; start monthly, increase with capacity.

## Organic reality (say it)
You **can't split the audience perfectly** and the algorithm shapes who sees what, so organic A/B isn't
lab-grade. **Compensate** with tighter controls, a single primary metric, **effect-size thresholds**, and
**repetition** — and treat results as **directional learning**, not proof.

**The one native exception — YouTube Test & Compare** (verify-quarterly): YouTube Studio runs a true
audience-split test on long-form videos of up to **3 variants — title only, thumbnail only, or
title+thumbnail** (expanded beyond thumbnail-only in the Dec 2025 global rollout; desktop Studio,
advanced features enabled; not Shorts/premieres/kids content). For YouTube title/thumbnail tests, prefer
it over sequential posts — it removes the "different audience, different day" confound. The discipline
here (decision rule first, one variable, guardrail, log it) still applies to how you use it.
