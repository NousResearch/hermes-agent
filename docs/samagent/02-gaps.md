# 02 — Gaps: what is still broken in coding agents

The question: *what are the biggest developer problems that existing agents leave unsolved, and which should SamAgent attack?* Evidence tags as in `01-research.md`.

## The gap table

| # | Gap | Evidence | Who is weak today | SamAgent answer | Already in repo? |
|---|-----|----------|-------------------|-----------------|------------------|
| G1 | **Vague ideas turn into wrong software.** Failures cluster in spec and design, not coding. | MAST: spec/design 41.8% [W1]; Spec Kit/Kiro need long manual spec work [W2] | Lovable/Bolt/Replit (little interview); Kiro/Spec Kit (prose specs that don't run) | Adaptive, skip-safe interview → PRD whose acceptance criteria compile into **executable tests** | `kanban_specify` (text only) |
| G2 | **Parallel agents disagree with each other.** | Cognition; MAST misalignment 36.9% [W1] | Most swarms | **Frozen contract** before fan-out; one owner per module; change-request protocol | `kanban_swarm` blackboard; no freeze |
| G3 | **"Done" is not verified.** Agents mark tasks done on their own say-so. | MAST verification 21.3% [W1]; Kiro criteria don't run [W2] | Nearly all | Layered verification: unit → integration → **live-app browser walkthrough** → security → independent judge against the *original brief* | `pre_verify`, verify recipes, goals judge (partial) |
| G4 | **Insecure output for non-developers.** Missing auth, broken RLS, leaked keys. | Moltbook 1.5M keys; Lovable CVE; about 45% OWASP fail [W2] | Vibe-coding platforms | Secure-by-default scaffolds, secret/dependency scans, **authz probes generated from the spec's roles**, dev/prod separation | approval system, not a build-time gate |
| G5 | **Destructive autonomy.** | Replit prod-DB deletion [W2]; Arena: users tighten control [W1] | Autonomous cloud agents | Autonomy dial, plan-only mode, hard spend/time caps, approval gates for deploy/delete/secrets | approvals, subagent auto-deny |
| G6 | **Cost is unpredictable and harness-dependent.** Up to 5× for equal success. | Arena harness tax [W1]; Replit credit complaints [W2] | Wide-tool agents, credit-based builders | Lean tool profile, cache-aware routing, **cost estimate before the run and hard caps** | `tool_search` deferral; no estimate/cap UX |
| G7 | **Context is the memory.** Long projects rot; new sessions start blank. | Context-stuffing anti-pattern; validity windows [W2] | Almost all | External **project ledger** with supersession; injection budget ≤ 2K tokens per task | curated memory (system-prompt-bound) |
| G8 | **Local models are unreliable as agents.** | Tool-call breakage [W2]; open models trail [W1] | Everything that just "adds Ollama" | Capability profiles per model, grammar/validators, per-hardware scorecards, verified escalation | `local_runtime/*` (runtime, not reliability) |
| G9 | **Simple UIs hide control; powerful UIs are too complex.** | Arena control finding [W1]; Hermes UI 608K LOC [R] | Lovable (opaque), Hermes (dense) | Layered UI: Simple by default, live preview + plain-language run graph, one-click Pro view | gateway contracts only |
| G10 | **Lock-in.** Work lives in the vendor's platform. | Replit lock-in complaints [W2] | Hosted builders | Everything in a normal git repo; `.samagent/` files are plain text | yes (git worktrees) |

## Which gaps we attack first

Ranked by **(impact on "it actually works") × (tractability with what the repo already has)**:

1. **G3 verification + G1 executable spec** — highest leverage, and the hooks exist (`pre_verify`, kanban, goals).
2. **G2 contract-gated swarm** — the user's explicit ask, but it is *only* safe after 1 and 3 exist.
3. **G6 cost/latency** — cheap wins (lean profile, cache-aware routing); also measurable in Phase 0.
4. **G7 ledger** — needed for resume, but less urgent than the first three.
5. **G4 security gates** — high value for beginners; scaffolds first, scanners next.
6. **G8 local reliability** — the user's stated preference; treat as a *measured* capability, not a promise.
7. **G9 UI** — starts early in parallel (needed for dogfooding), MVP by mid-project.
8. G5 and G10 are mostly *policy* on top of existing approvals and git.

## What we deliberately do not compete on

- **Raw model intelligence.** We do not train models. The harness-tax study says the harness moves cost far more than success. Our success gains must come from **spec, decomposition and verification**, and our cost gains from **leanness, caching and routing**. Anything we cannot measure is a slogan, not a feature.
- **IDE features** (completion, inline diff editing). You said "not an IDE".
- **Being Arena.** Arena has cloud infrastructure and a model marketplace. We copy the *mechanics* that are visible (preview, diff, structured tools, control), not the platform.
