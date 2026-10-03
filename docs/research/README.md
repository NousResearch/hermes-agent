# Research — Aro / SamAgent

Research folder for the Aro rebrand and desktop redesign — lives in-repo at `docs/research/`. Subject: **SamAgent**, a fork of Nous Research's Hermes Agent rebranded by samjuniors as the **Aro family**. The runnable prototype lives at [`prototype/`](../../prototype/).

## Index

| Doc | What it is |
|---|---|
| [01-competitive-analysis.md](01-competitive-analysis.md) | The agent-tool landscape (Codex app, Cursor, Zed, Claude Code + 8 more) — strengths, weaknesses, and UX lessons to steal for Aro Desktop. |
| [02-gap-analysis.md](02-gap-analysis.md) | Capability matrix: Aro vs Codex App vs Cursor vs Zed vs Claude Code across 13 axes; Aro's moats, gaps, and where it can win. |
| [03-redesign-plan.md](03-redesign-plan.md) | The 8-phase (P0–P7) Aro Desktop redesign plan, plus the 3 implementation options (restyle / new frontend / ACP) with a recommendation. |
| [04-rebrand-log.md](04-rebrand-log.md) | What the surface-rebrand commit `2ed2492` changed, and the remaining deep-rebrand TODO list (`~/.aro`, `aro://`, module renames, signing, update feed). |
| [05-prototype-spec.md](05-prototype-spec.md) | Spec for the first Aro Desktop prototype (Next.js 16 single route): shell, chat canvas with tool timeline, tasks board, context inspector, ⌘K palette, live AI backend. |
| [06-prototype-vs-hermes-integration.md](06-prototype-vs-hermes-integration.md) | **Current.** The attached design system (now the Aro Workbench prototype) vs. the Hermes/SamAgent codebase: capability matrix, what to port into `apps/desktop` (phases R1–R7), what to add to the prototype next (P-A…P-F). Round 2: all §5 decisions resolved and executed — Option A confirmed with **R1 shipped** (additive design-system port at `apps/desktop/src/aro/`), and **P-A…P-F shipped** in the prototype (automations, memory graph, hunk review, live-backend toggle, onboarding, artifacts). |

Brand guidelines: [`../brand/BRAND.md`](../brand/BRAND.md).

The screenshots referenced by docs 05–06 are in [`screenshots/`](screenshots/).
