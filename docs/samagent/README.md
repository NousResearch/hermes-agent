# SamAgent planning & implementation pack

**Status:** Phase 0 complete (Gate G0 passed; 9/9 spikes verified) + core `samagent/`, `plugins/samagent/`, Mission Control UI, and `SamBench-Web v0` implemented & tested · **Date:** 2026-09-29 · **Branch:** `arena/01a0eb9a-samagent`
**Base:** this repo (a fork of Nous Research `hermes-agent`, MIT) + ideas from the Pi harness (`earendil-works/pi`, MIT)

## Read in this order

| # | File | Answers |
|---|------|---------|
| 1 | [`01-research.md`](01-research.md) | What Arena's agent, Pi, Hermes (your repo), and the other agents actually do. What the evidence says about swarms, local models, routing, memory and speed. |
| 2 | [`02-gaps.md`](02-gaps.md) | The biggest unsolved developer/vibe-coder problems, and which ones SamAgent should attack. |
| 3 | [`03-draft-plan-v0.md`](03-draft-plan-v0.md) | My first, deliberately ambitious plan. It is written to be attacked. |
| 4 | [`04-red-team.md`](04-red-team.md) | "Kill the plan": 14 attacks on the draft, a pre-mortem, and the verdicts. |
| 5 | [`05-final-plan.md`](05-final-plan.md) | **The plan to execute.** Architecture, roadmap, exit and kill criteria, Phase 0 task list mapped to files in this repo. |
| 6 | [`ADR-001-phase0-spikes.md`](ADR-001-phase0-spikes.md) | **Gate G0 empirical results:** full `AIAgent` prefix measurements, Spikes S1–S9 verdicts (9/9 passed), and core patch elimination (down to 1 file). |
| 7 | [`CORE_PATCHES.md`](CORE_PATCHES.md) | Tracked upstream patch ledger (`pyproject.toml` only; P1 and P4 eliminated by Spikes S2 and S3). |
| 8 | [`06-benchmark-report.md`](06-benchmark-report.md) | **SamBench-Web v0 & H1–H7 Ablation Report:** `A0`–`A5` comparison, git-worktree swarm integration, and self-security verification. |

## The answer in 8 lines

1. **Do not rewrite the core.** Hermes already has about 860K lines of Python (excluding tests) covering providers, delegation, a kanban swarm kernel, verification hooks, a managed llama.cpp runtime and a JSON-RPC UI gateway. SamAgent is an **overlay**: one plugin, one small library and one new UI. Core edits are tracked in a short patch list.
2. **"Smarter" is defined as 7 falsifiable hypotheses** measured on a benchmark we build (SamBench-Web). Every component (swarm, router, memory) must beat its own ablation to stay switched on.
3. **The product is a pipeline, not a chat:** interview → executable spec → **frozen contract** → parallel workers in isolated worktrees, one owner per module → layered verification (tests, live app, security, independent judge) → delivery.
4. **The swarm is gated.** The evidence says parallel *writers* fail without a shared contract (Cognition, MAST). Fan-out happens only after the contract freeze and only where modules are independent.
5. **Local-first at task boundaries, not per turn.** Switching model mid-conversation destroys the prompt cache. Escalation means a *fresh* agent with a handoff brief.
6. **Memory lives outside the context window:** a project ledger (decisions with validity windows, attempts, task graph) in SQLite plus a git-tracked markdown mirror. Only 5–10 relevant items are injected per task.
7. **UI:** "Mission Control", a small React app (first as a dashboard plugin) on the existing `tui_gateway` contracts. Simple by default, developer depth one click away.
8. **Phase 0 is one week of measurement** before any big build. If the baselines contradict a premise, the plan changes.

## Assumptions I made (you skipped the interview)

| # | Assumption | Changes if wrong |
|---|-----------|------------------|
| A1 | Layered UI: describe-your-idea by default, dev depth on demand | UI phase gets bigger or smaller |
| A2 | One React UI, served locally and later wrapped as a desktop app | Wrapper choice |
| A3 | Mid-range hardware (≈32 GB unified RAM or ≈24 GB VRAM); one cloud key available | Local share target (H4), model picks |
| A4 | Autonomy is a per-project dial (plan-only → approve-each-step → hands-off) | Approval-gate defaults |
| A5 | First target stack is web apps (React/Next or Vite + SQLite/Postgres + auth) | Template packs, verification recipes |
| A6 | Solo builder with agent help; about 12 weeks; small eval budget | Phase lengths, benchmark size |
| A7 | Keep MIT; add your copyright line; rename product surfaces away from "Hermes" | Branding work |

## Questions I would still ask (defaults are used if you don't answer)

1. **Who is user #1?** You (a developer, dogfooding) or complete beginners? *Default: you first, beginners' UX from Phase 5.*
2. **Your machine and keys?** RAM/GPU, and which cloud providers you have. *Default: A3.*
3. **Local-only mode?** Should "never send my code to the cloud" be a supported mode, even if it lowers quality? *Default: yes, with visible warnings and more approval gates.*
4. **Where should code run?** On your machine (worktrees + optional Docker), or also in a remote cloud sandbox like Arena? *Default: local first; remote is a later adapter.*
5. **Budget?** Roughly how much can you spend on benchmark runs? *Default: a hard cap set per run and per campaign, printed before launch.*

## Honest limits of this work

- **I did not run the agent end to end.** The sandbox has 2 CPUs, 3 GB RAM, no model endpoint or keys, and Python 3.11, while Hermes officially targets **Python 3.14** (`pyproject.toml`; 3.11 is only a legacy update band). I could, however, install the dependencies by hand on 3.11 and **import the tool registry**, so exactly one thing is measured: **tool-schema size per posture** ([`measurements/tool_footprint.json`](measurements/tool_footprint.json), reproducible with `evals/samagent_bench/tool_footprint.py`; token counts are a chars/4 proxy because the tokenizer download was blocked). Everything else about Hermes is from **reading code**. There are **no measured latency, cost or pass-rate numbers** for Hermes in this pack. Getting them is Phase 0.
- **Arena's swarm, routing and caching internals are not public.** I documented what Arena publishes and marked everything else as inference.
- **Many web sources are SEO blogs** with inconsistent benchmark numbers. Every number is tagged with its evidence level (below). I preferred papers, vendor docs and Arena's own posts.
- Effort estimates are for one person and carry roughly ±50% uncertainty.

## Evidence tags used in these docs

- **[R]** read directly in this repo or in a cloned doc (primary)
- **[W1]** paper, vendor engineering post or Arena's own post (strong secondary)
- **[W2]** blog / SEO / community post (directional only)
- **[I]** my inference or design judgment (a hypothesis, not a fact)
