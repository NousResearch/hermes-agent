---
name: gamefi-research
description: Structured, neutral research workflow for discovering and technically reviewing early-stage Web3 gaming (GameFi) projects. Helps Hermes review public repository signals, check documentation quality, add project notes, produce risk notes, classify next steps (WATCH / TEST / CONTACT / SKIP), and generate clean, hype-free research summaries. Research only — not financial advice.
---

# GameFi Research Workflow

## Overview

`gamefi-research` is a Hermes Agent workflow skill for **structured research and technical review
of early-stage Web3 gaming projects**. It gives Hermes a repeatable process for turning public
project signals (repository activity, documentation quality, testnet / early-access markers,
community presence) into a neutral, comparable research summary.

The skill is a *workflow*, not a script. It tells Hermes how to think about a game project, what
public signals to collect, how to score them, how to classify next steps, and how to present the
result so a gaming community can act on it. Optional supporting tooling (a scanner prototype,
scheduled reports) is layered on top in later milestones — but the workflow stands on its own and
can be run by Hermes manually today.

## Problem statement

Web3 gaming communities — guilds, creators, content teams, scouts — almost always discover
early-stage projects **after** the hype starts. By then the early research window (testnet access,
whitelist spots, playtest invites, low-noise community) is gone.

Hermes has strong research and automation capabilities, but there is **no dedicated, structured
workflow** for game project research. Without a shared framework, research is ad-hoc,
inconsistent, hard to compare, and prone to hype-driven language. This skill closes that gap.

## When to use this skill

- A user or community asks Hermes to **review an early-stage game project**.
- Someone wants a **structured, comparable summary** of one or several projects.
- A guild / creator / scout wants to **organize discovery research** and triage what deserves
  deeper attention.
- You want a **clean, postable research summary** with consistent formatting.
- You are setting up **recurring discovery** (daily / weekly sweeps).

## When NOT to use this skill

- **Financial, trading, or investment questions.** This is not financial advice and produces no
  buy/sell/allocation guidance.
- **Price prediction, ROI estimation, or asset-farming strategy.**
- **Established, well-known projects** where the value is market analysis, not early discovery.
- Any task that is really "should I invest?" — decline that framing and redirect to neutral
  research.

## Expected inputs

- A **project URL** (repository, website, or docs).
- A **project name** plus whatever context the user provides.
- A **list of candidate projects** to triage.
- Raw signals already gathered (repository stats, README text, community links).

If inputs are incomplete, Hermes runs the workflow anyway and marks unknown signals as `unknown`
rather than guessing (see *Handling incomplete or uncertain information*).

## Expected outputs

- A **neutral structured research summary** per project: overview, detected public signals,
  GameFi Signal Score, classification (WATCH / TEST / CONTACT / SKIP), project notes, suggested
  research action. For a single repository-based review, use
  `templates/github-project-review.md`.
- A **daily/weekly report** aggregating projects (see `templates/daily-report.md`).
- A **postable summary** safe for a community channel.

### Templates

- `templates/github-project-review.md` — single-project, repository-based technical review.
  Use this when reviewing one project from its GitHub repo.
- `templates/daily-report.md` — aggregate daily/weekly report across multiple projects.

## Step-by-step Hermes agent workflow

Hermes performs these steps in order for each project. Each step produces a small, recorded fact —
not a judgment about the project's worth.

1. **Identify.** Capture name, URL(s), and a one-line description of what the project claims to be.
   If the claim is unclear, record that as a risk flag.
2. **Collect repository signals.** Repository age, stars, forks, primary language, topics, commit
   recency, and whether the repo looks actively maintained.
3. **Assess documentation quality.** Is there a README? Does it explain the game, the tech, the
   chain, and how to try it — or is it a placeholder?
4. **Scan for stage signals.** Look for: testnet, early access, whitelist, points, demo, playable
   build / client, gameplay footage, devnet, closed beta.
5. **Assess playability.** Is there something to actually play or test, or is it docs-only?
6. **Assess ecosystem fit.** Which chain / ecosystem, and does the tech stack look coherent with
   the claim?
7. **Record project notes & risk flags.** Short factual observations (see *Project note examples*).
8. **Score** with the GameFi Signal Score framework.
9. **Classify** next steps as WATCH / TEST / CONTACT / SKIP using the decision rules below.
10. **Write neutral output** following the output formatting rules. For a single repository-based
    review, fill in `templates/github-project-review.md`.

Treat every project claim as **unverified until manually checked**. Hermes reports public signals,
not conclusions about quality or value.

## Research checklist

Hermes should be able to answer each before classifying. Unknown is an acceptable answer — mark it.

- [ ] Project name and one-line claim captured?
- [ ] Repository URL(s) recorded?
- [ ] Repository age and last-activity date noted?
- [ ] Stars / forks / primary language / topics noted?
- [ ] README present and assessed for technical substance?
- [ ] Stage signals scanned (testnet / early access / whitelist / points / demo / beta)?
- [ ] Playability assessed (something to test vs. docs-only)?
- [ ] Ecosystem / chain identified?
- [ ] At least one project note and any risk flags recorded?
- [ ] GameFi Signal Score assigned with a one-line rationale?
- [ ] Classification chosen with a reason tied to the signals?
- [ ] Disclaimer attached to the output?

## GameFi Signal Score framework

A **neutral** score that makes projects *comparable*. It reflects *research signal strength* — how
much there is to look at and how early/active the project appears — **not** financial merit. A high
score means "worth a closer look", never "worth buying".

| Component                         | What it measures                                                       |
| --------------------------------- | ---------------------------------------------------------------------- |
| **Repository Activity**           | Stars, forks, commit recency, maintenance signals.                     |
| **Freshness**                     | How recently the project/repo was created or updated.                  |
| **Documentation Quality**         | Clarity, completeness, technical substance vs. marketing fluff.        |
| **Playability Signal**            | Presence of demo / playable build / client / gameplay footage.         |
| **Early Access / Testnet Signal** | Testnet, whitelist, points, early access, closed beta markers.         |
| **Ecosystem Fit**                 | Coherence of chain + tech stack with the stated game claim.            |
| **Risk Flags**                    | Deductions for missing info, unclear purpose, hype-without-substance.  |

Exact point weights arrive in a later milestone with the optional scanner. In the manual workflow
Hermes scores the components qualitatively and states a one-line rationale.

## Decision rules: WATCH / TEST / CONTACT / SKIP

Pick exactly one. This is a **research triage decision**, not a financial one. Apply the first rule
that matches, top to bottom:

- **SKIP** if any of: no README / empty description, unclear purpose, or strong hype language with
  no technical substance. Weak or unverifiable signals → SKIP.
- **TEST** if there is a concrete way to play or test now — playable build, demo, client, or a live
  testnet — and basic documentation exists.
- **CONTACT** if the project looks relevant for creator / guild / community outreach
  (collaboration, coverage, early-access requests) and has enough substance to justify reaching
  out, but no open test yet.
- **WATCH** (default) if there are interesting signals but not enough to test or contact yet —
  track it over time.

If two categories seem to fit, prefer the **more conservative** one (SKIP > WATCH > CONTACT > TEST
when in doubt about substance). Always state the single signal that drove the decision.

## Handling incomplete or uncertain information

- **Never fabricate signals.** If a value is unknown, write `unknown` — do not infer stars, dates,
  or a chain that isn't stated.
- **Lower confidence, don't inflate.** Missing documentation or unclear purpose pushes a project
  toward WATCH or SKIP, never toward TEST/CONTACT.
- **Separate claim from evidence.** Phrase unverified items as "claims X" or "states X", not "has
  X".
- **Note what's missing.** Add a project note like `Risk: no README / cannot confirm playability`
  so a human knows what to verify next.
- **Don't block on gaps.** Produce the summary with the signals available and flag the gaps.

## Output style rules

These keep every output neutral, research-focused, and safe to post publicly.

- **No hype language.** Avoid "moon", "huge", "next big", "guaranteed", "don't miss", "early
  alpha". Describe signals plainly.
- **No promotional or asset-related language.** Do not mention buying, selling, prices, allocation,
  returns, or farming. The skill discusses *project research*, not assets.
- **Neutral verbs.** "The repo shows…", "The README states…", "Testnet signals detected…" — not
  "amazing", "promising buy", "must-try".
- **Evidence first.** Every claim ties to a public signal. No signal → no claim.
- **Consistent structure.** Use the per-project block: Overview → Detected public signals → Score →
  Classification → Project notes → Suggested research action → Disclaimer.
- **Brevity.** A few clear lines per field. This is a research note, not marketing copy.

## Project note examples

Short, factual, signal-anchored. Good examples:

- `Active repo: 14 commits in the last 7 days; primary language TypeScript.`
- `README explains core loop and lists a public testnet URL.`
- `Risk: description empty; purpose unclear from repo alone.`
- `Playable web demo linked; could not confirm it runs (manual check needed).`
- `Mentions points + early access; no team or community links found.`

Avoid notes like: `Looks like a great opportunity`, `Strong upside`, `Could be the next big game` —
these are hype/promotional and not signal-anchored.

## Examples (good vs. bad use)

**Good use:** "Analyze this early-stage game project for community research." → Hermes runs the
workflow, records public signals, scores, classifies, and returns a neutral summary.

**Good use:** "Triage these 5 repos and tell us which are worth testing." → Hermes returns a
comparable table plus per-project notes.

**Bad use (decline / redirect):** "Is this a good investment?" / "Will the price go up?" → Hermes
declines the financial framing and offers neutral research signals instead.

**Bad use (decline / redirect):** "Tell the community to get in early." → Hermes does not produce
promotional calls to action; it reports signals only.

## Worked example

**Example input:**
> "Analyze this early-stage game project for community research."
> `https://github.com/example-org/example-onchain-game`

**Example output:**

> **Example Onchain Game** — `https://github.com/example-org/example-onchain-game`
>
> **Overview:** Repo states it is an early-stage onchain strategy game with a browser client.
> Claim is clearly described in the README.
>
> **Detected public signals:** Repo created ~18 days ago; 42 stars, 6 forks; primary language
> TypeScript; topics include `web3`, `game`, `onchain`. README documents the core loop and links a
> public testnet. A playable web demo is linked.
>
> **GameFi Signal Score:** 71 / 100 — strong freshness + documentation + a live testnet/demo;
> minor deduction for no visible team or community links.
>
> **Classification:** TEST — there is a concrete way to try it now (testnet + demo) and basic docs
> exist.
>
> **Project notes:**
> - `Active repo: recent commits; README documents core loop.`
> - `Public testnet + playable demo linked (not yet verified to run).`
> - `Risk: no team/community links found; claims unverified.`
>
> **Suggested research action:** Join the testnet, test the core loop, and confirm the demo runs.
> Re-check community presence in ~1 week.
>
> *Disclaimer: neutral research summary; not financial advice; signals unverified — confirm
> manually.*

## Writing postable (Discord-ready) summaries

When producing a summary for a community channel:

- Keep it short: project name, one-line overview, top signals, score, classification, one
  suggested action.
- Stay neutral and hype-free — apply the output style rules above.
- Never frame anything as a call to buy, invest, or "get in early".
- Always include a one-line research-only disclaimer.
- Mark unknowns explicitly; do not pad with guesses.

## Hermes integration notes

- **Scheduled runs.** Hermes can run the workflow on a schedule (daily/weekly sweep) and post
  results to a channel.
- **Future scanner prototype.** Hermes can call a dedicated scanner (later milestone) to gather
  public repository signals automatically instead of by hand.
- **Memory of previous reports.** Hermes can remember prior reports and projects already seen, so
  it doesn't re-surface the same ones and can track how a project evolves.
- **Comparison over time.** Hermes can compare a new project against older reports — flagging when
  a WATCH project gains testnet/playable signals and should be re-classified.
- **Postable summaries.** Hermes can generate consistent, hype-free summaries ready for a channel.
- **Community organization.** Hermes can help guilds and creators organize discovery research into
  shared, comparable reports rather than scattered links.

## Future automation ideas

- A scanner that searches by game keywords and filters by recency.
- An automated GameFi Signal Score calculator from raw repo + README data.
- Scheduled daily / weekly Markdown report generation.
- A reusable agent interface so Hermes, a bot, a CLI, or a cron job can trigger the same pipeline.
- Memory-backed comparison to detect newly-promising projects across runs.

---

*Disclaimer: This skill is a neutral research workflow for gaming communities. It is not financial
advice, not a trading tool, and not an investment recommendation. Verify all project claims
manually.*
