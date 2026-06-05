---
name: gamefi-research
description: Structured, neutral research workflow for discovering and assessing early-stage Web3 gaming (GameFi) projects. Helps Hermes analyze GitHub activity, README quality, testnet/early-access/community signals, produce risk notes, classify projects (WATCH / TEST / CONTACT / SKIP), and generate Discord-ready research summaries. Research only — not financial advice.
---

# GameFi Research Workflow

## Overview

`gamefi-research` is a Hermes Agent workflow skill for **structured research on early-stage
Web3 gaming projects**. It gives Hermes a repeatable process for turning raw project signals
(GitHub activity, README quality, testnet / early-access markers, community presence) into a
neutral, comparable research summary.

The skill is a *workflow*, not a script. It describes how Hermes should think about a GameFi
project, what signals to collect, how to score them, how to classify the project, and how to
present the result so a gaming community can act on it. Optional supporting tooling (a GitHub
scanner, scheduled reports) is layered on top of this workflow in later milestones — but the
workflow stands on its own and can be run by Hermes manually today.

## Problem statement

Web3 gaming communities — guilds, creators, content teams, scouts — almost always discover
early-stage projects **after** the hype starts. By the time a project is trending, the early
research window (testnet access, whitelist spots, playtest invites, low-noise community) is gone.

Hermes already has strong research and automation capabilities, but there is **no dedicated,
structured workflow** for GameFi / Web3 gaming research. Without a shared framework, research is
ad-hoc, inconsistent, hard to compare across projects, and prone to hype-driven language.

This skill closes that gap with a neutral, signal-based research process.

## When to use this skill

Use `gamefi-research` when:

- A user or community asks Hermes to **analyze an early-stage Web3 gaming project**.
- Someone wants a **structured, comparable summary** of one or several GameFi projects.
- A gaming guild / creator / scout wants to **organize discovery research** and triage which
  projects are worth deeper attention (testing, outreach, watching).
- You want to produce a **Discord-ready research summary** with consistent formatting.
- You are setting up **recurring discovery** (daily / weekly project sweeps).

## When NOT to use this skill

Do **not** use this skill for:

- **Financial, trading, or investment decisions.** This is not financial advice and produces no
  buy/sell/allocation guidance.
- **Token price prediction, ROI estimation, or airdrop farming strategy.**
- **Established, well-known projects** where the value is in market analysis rather than early
  discovery — this skill is tuned for *early-stage* signal research.
- Any task where the user is really asking "should I invest?" — decline that framing and redirect
  to neutral research.

## Expected inputs

The skill can work from any of the following:

- A **project URL** (GitHub repo, website, or docs).
- A **project name** plus whatever context the user provides.
- A **list of candidate projects** to triage.
- Raw signals already gathered (GitHub stats, README text, community links).

If inputs are incomplete, Hermes should still run the workflow and clearly mark unknown signals
as `unknown` rather than guessing.

## Expected outputs

- A **neutral structured research summary** per project containing:
  - Overview (what the project claims to be)
  - Detected signals
  - GameFi Signal Score (with component breakdown)
  - Classification: WATCH / TEST / CONTACT / SKIP
  - Risk notes
  - Suggested next research action
- A **daily/weekly report** aggregating multiple projects (see `templates/daily-report.md`).
- A **Discord-ready summary** suitable for posting in a community channel.

## Research workflow

Hermes follows these steps for each project:

1. **Identify the project.** Capture name, URL(s), and a one-line description of what it claims
   to be.
2. **Collect GitHub signals.** Repository age, stars, forks, primary language, topics, commit
   recency, and whether the repo is actively maintained.
3. **Assess README quality.** Is there a README? Does it explain the game, the tech, the chain,
   and how to try it? Or is it a marketing placeholder?
4. **Scan for stage signals.** Look for: testnet, early access, whitelist, points, rewards,
   demo, playable build / client, gameplay footage, devnet, closed beta.
5. **Assess playability.** Is there anything to actually play or test, or is it whitepaper-only?
6. **Assess ecosystem fit.** Which chain / ecosystem (Solana, Base, Ronin, Abstract, etc.), and
   does the tech stack look coherent with the claim?
7. **Record risk flags.** Missing README, empty description, unclear purpose, hype language
   without technical substance, no team/community traces, etc.
8. **Score** using the GameFi Signal Score framework below.
9. **Classify** as WATCH / TEST / CONTACT / SKIP.
10. **Write neutral output** — overview, signals, score, classification, risks, next action.

Every factual claim about a project should be treated as **unverified until manually checked**.
Hermes presents signals, not conclusions about the project's quality or value.

## GameFi Signal Score framework

A **neutral** scoring system designed to make projects *comparable*, not to rank them as good or
bad investments. The score reflects *research signal strength*, i.e. how much there is to look at
and how early/active the project appears — not financial merit.

Components:

| Component                        | What it measures                                                        |
| -------------------------------- | ----------------------------------------------------------------------- |
| **GitHub Activity**              | Stars, forks, commit recency, maintenance signals.                      |
| **Freshness**                    | How recently the project/repo was created or updated (earlier = higher research value for discovery). |
| **README Quality**               | Clarity, completeness, technical substance vs. marketing fluff.         |
| **Playability Signal**           | Presence of demo / playable build / client / gameplay footage.          |
| **Early Access / Testnet Signal**| Testnet, whitelist, points, rewards, early access, closed beta markers. |
| **Ecosystem Fit**                | Coherence of chain + tech stack with the stated GameFi/Web3 game claim. |
| **Risk Flags**                   | Deductions for missing info, unclear purpose, hype-without-substance.   |

The score is a **research signal**, not a recommendation. A high score means "worth a closer
look", never "worth buying". The exact point weights are defined in later milestones when the
optional scanner script is added; in the manual workflow Hermes uses the components qualitatively
and explains its reasoning.

## Decision framework: WATCH / TEST / CONTACT / SKIP

Every project is classified into exactly one category:

- **WATCH** — Project has interesting signals but needs more observation before action. Not enough
  yet to test or reach out, but worth tracking over time.
- **TEST** — Project shows playable / demo / testnet signals and may be worth hands-on testing by
  the community.
- **CONTACT** — Project looks relevant for creator / guild / community outreach (collaboration,
  coverage, partnership, early-access requests).
- **SKIP** — Project has weak signals, unclear information, or high risk flags. Not worth research
  attention right now.

The classification is a **research triage decision**, not a financial one.

## Safety notes

- **Not financial advice.** This skill never tells anyone to buy, sell, hold, or allocate.
- **Not a trading tool.** It produces no price targets, ROI, or trading signals.
- **Not an investment recommendation.** Classifications (WATCH/TEST/CONTACT/SKIP) are about
  *research attention*, not investment merit.
- **All project claims must be verified manually.** Signals are starting points for human
  research, not verified facts.
- **Do not encourage buying tokens or investing.** If a user asks for that, decline and redirect
  to neutral research.
- **Focus on research, project discovery, and community organization** — nothing else.

A disclaimer to this effect must appear on every generated report and summary.

## Hermes integration notes

This workflow is designed to be Hermes-centered:

- **Scheduled runs.** Hermes can run the workflow on a schedule (e.g. a daily or weekly discovery
  sweep) and post results to a community channel.
- **Future GitHub scanner tool.** Hermes can call a dedicated GitHub scanner (added in a later
  milestone) to gather raw signals automatically instead of by hand.
- **Memory of previous reports.** Hermes can remember prior reports and the projects already seen,
  so it does not re-surface the same projects and can track how a project evolves.
- **Comparison over time.** Hermes can compare a new project against older reports — flagging when
  a WATCH-tier project gains new testnet/playable signals and should be re-classified.
- **Daily Discord summaries.** Hermes can generate consistent, hype-free summaries ready to post
  in a Discord channel for a gaming community.
- **Community organization.** Hermes can help guilds and creators organize discovery research into
  shared, comparable reports rather than scattered links.

## Future automation ideas

- A GitHub scanner that searches by GameFi keywords and filters by recency.
- An automated GameFi Signal Score calculator from raw repo + README data.
- Scheduled daily / weekly Markdown report generation.
- A reusable agent interface so Hermes, a Discord bot, a CLI, or a cron job can all trigger the
  same pipeline.
- Memory-backed comparison to detect newly-promising projects across runs.

---

*Disclaimer: This skill is a neutral research workflow for gaming communities. It is not financial
advice, not a trading tool, and not an investment recommendation. Verify all project claims
manually.*
