# 05 — Final plan

Supersedes `03-draft-plan-v0.md` after the attacks in `04-red-team.md`. Evidence tags as in `01-research.md`. Everything marked *(target)* is a hypothesis to be measured, not a result.

---

## 1. Thesis, and what "smarter" means

We do not make the model smarter. We make **the system around it** deliver *working, verified software* for *less money and less time*, with the user in control.

The evidence says where that is possible [W1]: multi-agent failures are 42% spec/design, 37% coordination and 21% verification (MAST), and the harness changes cost up to 5× at similar success (Arena). So:

- **Success gains** must come from *spec, contract and verification*.
- **Cost and latency gains** must come from *leanness, caching and routing*.

### The seven hypotheses (this is the definition of "smarter")

Arms (each step must beat the one before it, or the component ships **off**):

| Arm | What it is |
|-----|-----------|
| A0 | Hermes default coding posture, single agent (baseline) |
| A1 | Pi defaults at the same model (external baseline) |
| A2 | SamAgent **lean** single agent (lean profile + ledger, no pipeline) |
| A3 | A2 + interview → spec → contract → layered verification, **no fan-out** |
| A4 | A3 + contract-gated swarm |
| A5 | A4 + local-first task router |

| # | Hypothesis | Comparison | Pass rule *(targets)* | If it fails |
|---|-----------|-----------|-----------------------|-------------|
| H1 | Spec-first raises real acceptance | A3 vs A2 | +15 points acceptance pass rate at ≤ 1.5× cost | Simplify spec; drop generated tests for hard criteria |
| H2 | Contract-gated swarm is faster without losing quality | A4 vs A3 (tier 2–3) | wall-clock ≤ 0.6×, pass rate within −2 pts, tokens ≤ 2× | Swarm ships opt-in only |
| H3 | Lean profile cuts cost | A2 vs A0 | input tokens ≤ 0.7×, pass rate within −2 pts (tool schemas alone are already ≈ 72% smaller: 14.4K → 4.0K, but caching softens the *cost* effect, so test with real runs) | Adopt only the parts that win |
| H4 | Local-first share is real | A5 vs A4 on local model | ≥ 50% of output tokens local, pass rate within −5 pts | Local becomes "helper" role; cloud-primary default |
| H5 | Ledger memory survives context loss | cold-resume eval, 10 scripted probes | ≥ 90% correct continuation, ≤ 2K injected tokens (median) | Add embeddings or simplify schema |
| H6 | Secure by default | seeded vulnerabilities (missing auth, IDOR, client-exposed secret, raw-SQL injection, missing validation) | ≥ 95% caught or prevented; false blocks < 10% | Expand scaffolds/probes before beginner release |
| H7 | Beginners can ship | 5 non-developers, unaided | 4/5 reach a working app; median time-to-first-preview ≤ 3 min | Cut features, redo Simple mode |

Benchmark: **SamBench-Web** (§13). We publish whatever the numbers say.

---

## 2. Architecture decision

| Option | Description | Verdict |
|--------|-------------|---------|
| A | Hard fork, edit core freely | Fast start, painful merges later |
| B | Fork then slim by deleting | **Rejected.** Breaks tests and merges |
| C | New TypeScript core, Pi-style | **Rejected.** Multi-year; loses solved problems; Pi's own protocol is still moving |
| D | Overlay: Hermes untouched as engine, all new logic in a plugin and library | Cleanest merges; limited by the plugin surface |
| **D+** | **D plus a tracked, minimal patch list of *generic*, upstream-able core edits** | **Chosen** |

Rules [R, from `AGENTS.md`/`plugins/AGENTS.md`]: plugins never edit core files; prompt-cache byte-stability; strict role alternation; config in `config.yaml`. We slim by **config and profile**, never by deleting upstream files.

### Diagram

```
 ┌──────────────── Mission Control (React; dashboard plugin first) ────────────────┐
 │ Brief · Plan card · Live run + preview · Review/Ship · Memory      [Simple|Pro] │
 └────────────────────────────────▲────────────────────────────────────────────────┘
                                  │ JSON-RPC/WebSocket (tui_gateway contracts) + plugin_api
 ┌────────────────────────────────┴────────────────────────────────────────────────┐
 │ Hermes engine (unchanged core)                                                  │
 │  agent loop · ~40 providers · approvals · delegate_task · kanban kernel ·       │
 │  local_runtime (llama.cpp) · tool_search · hooks (pre_verify, pre_tool_call…)   │
 │                                                                                 │
 │ plugins/samagent + samagent/ library                                            │
 │  intake ─▶ spec ─▶ contract ─▶ conductor ─▶ verify ─▶ deliver                   │
 │              ledger (memory) ◀── router (task boundary) ◀── scorecards          │
 │              bench (SamBench-Web)                                               │
 └─────────────────────────────────────────────────────────────────────────────────┘
   Workspaces: git worktree per worker · optional container · local llama.cpp · cloud
```

### Repo layout (all new; nothing existing is moved)

```
plugins/samagent/        plugin.yaml, __init__.py (hooks/tools), dashboard/ (Mission Control MVP)
samagent/                pure-Python library: intake/ spec/ contract/ conductor/ router/ verify/ ledger/ bench/
samagent/templates/      scaffold packs: web-basic, web-auth-crud (secure by construction)
evals/samagent_bench/    SamBench-Web (reuses evals/core_tool_deferral orchestrator/worker/report patterns)
docs/samagent/           this pack, ADRs, CORE_PATCHES.md
```

### Expected core patch list (target ≤ 4; anything more needs an ADR)

| Id | Patch | Why | Needed only if |
|----|-------|-----|----------------|
| P1 | Optional per-task `model`/`provider`/`profile` in `delegate_task` | Task-boundary routing; today one global `delegation.model` [R] | Kanban profiles (which carry model + toolset) cannot route workers (spike S3) |
| P2 | Add `samagent*` to packaging in `pyproject.toml` | Ship the library | Always (verify in S1) |
| P3 | Product-identity strings (CLI banner, desktop `product-identity.cjs`, `SOUL.md`) | Rebrand seam already exists [R] | Always |
| P4 | Generic hook capability to add an ephemeral user-turn block | Ledger injection | `pre_llm_call` cannot do it cache-safely (spike S2) |

Also: add a git `upstream` remote (NousResearch/hermes-agent) and rehearse a merge monthly.

---

## 3. The project record: `.samagent/`

Plain text in the user's own git repo, so nothing is locked in (G10).

```
.samagent/
  brief.md          human-readable PRD (what the user approves)
  spec.yaml         machine-readable: goals, roles, stories + acceptance, non-goals, stack,
                    budgets, autonomy level, assumptions[]
  contract/         FROZEN interfaces: openapi.yaml, db/schema.sql, types.ts, design-tokens.json,
                    ownership.yaml (module → owned globs)
  acceptance/       generated executable tests (Playwright/pytest), red before work starts
  ledger/           markdown export of decisions (committed, diffable)
  ledger.db         SQLite: facts, attempts, tasks, scorecards, repo_map  (gitignored)
  runs/<id>/        event log (JSONL), cost report, evidence (screenshots, test output)
```

Small `spec.yaml` sketch:

```yaml
goal: "Booking site for my yoga studio"
roles: [visitor, member, admin]
stories:
  - id: S1
    as: visitor
    can: see the class schedule
    accept: "GET / shows ≥1 class; no login required"
  - id: S2
    as: member
    can: book a class
    accept: "logged-in member books; second booking of same class is rejected; other members cannot see it"
assumptions:            # visible, editable, changeable by agents only via the conductor
  - {id: X1, text: "Payments are out of scope", source: default}
budget: {max_usd: 6, max_minutes: 45}
autonomy: milestones     # plan_only | milestones | hands_off
```

---

## 4. The pipeline (one run, start to finish)

1. **Intake.** Free text, plus optional screenshot, link or existing repo.
2. **Adaptive interview.** ≤ 5 questions, each with a recommended default and "use defaults". It never blocks. Every skipped answer becomes an entry in `assumptions[]`. *(Directly learned from this session: you skipped the interview and said "continue".)*
3. **Spec critique.** A separate agent lints the spec for ambiguity, untestable criteria, missing auth or data-privacy rules, and contradictions. It asks at most one clarifying question, and only for blockers.
4. **Contract freeze** by **one** strong-model agent: API, DB schema, shared types, design tokens, and the module **ownership map**. Acceptance tests are generated from the stories and must **fail for the right reasons** (red-first check).
5. **Plan card** to the user: modules, parallelism, local/cloud split, **estimated cost and time range**, risks, assumptions, autonomy dial. On approval, hands-off within the chosen autonomy level.
6. **Execute** (§5): deterministic scaffold from a template pack (no LLM), then workers.
7. **Verify** in layers (§7). Failures become targeted repair tasks with a capped budget.
8. **Deliver:** plain-language summary, acceptance checklist with *evidence*, security report, assumptions, cost report, how to run, ready-to-push branch.
9. **Change requests** re-enter at the **spec diff**: only impacted tests and modules re-run.

---

## 5. Swarm rules (built on `delegate_task` + `kanban_swarm`, not a new scheduler)

- **Fan-out gate (all must hold):** contract frozen; ≥ 2 modules with disjoint write-sets and no unresolved interface dependency; module large enough to amortise a fresh context (initial guess: > ~5 min of single-agent work, tuned in Phase 3); budget headroom. Otherwise the run stays single-threaded.
- **One owner per module.** Ownership map in `contract/ownership.yaml`. Workers run in their own **git worktree** (`tools/subagent_worktree.py`).
- **Ownership guard, two layers:**
  1. `pre_tool_call` hook denies file-tool writes outside owned globs (cheap, early);
  2. authoritative **post-hoc diff check** (`git diff --name-only` vs globs) before any merge. Terminal writes are hard to police in advance, so violations are rejected and sent back.
- **Contract change requests (CCR).** Workers cannot edit `contract/`. They file a CCR; the conductor (the only writer) versions it (`contract@vN`), records the decision in the ledger, and **steers** only the affected workers (`delegate_task action=steer`).
- **Single integrator.** One agent merges worktrees into an integration branch and resolves conflicts; workers never merge each other.
- **Independent verifiers:** fresh context, no access to the writer's summary, a different model family where available.
- **State outside context:** the DAG, task states and blackboard live in the kanban DB and the ledger, not in any agent's context.
- **Caps:** per-task and per-run token/cost/time caps; stuck detection (same error signature twice → change strategy → escalate → ask the human in plain language).

---

## 6. Router: task-boundary, static policy, honest local-first

**Route at task boundaries.** Each worker is a fresh context, so the model can differ per task with **no cache loss**. Escalation = spawn a *new* agent with a handoff brief from the ledger. Never swap models mid-conversation (Pi and Hermes docs both warn of the cache penalty) [R]. Retries are sticky.

**Static policy table for v1** (no learned router until logs and evals justify one):

| Task | Default model class | Notes |
|------|--------------------|-------|
| Interview, spec critique | strongest available | local if user chose local-only |
| Contract freeze | strongest available | the highest-leverage step |
| Scaffold from template | **no LLM** | deterministic |
| Module implementation | local-capable **if scorecard passes**, else mid cloud | scorecard threshold set in Phase 1 |
| Test/lint fix loops | local first; escalate after 2 *verified* failures | verification, not model "confidence", triggers escalation |
| Explore / summarise / log triage | small local | |
| Browser verification | vision-capable (local or cloud) | |
| Judge | strongest available, **different family from the writer** | |
| Memory extraction | local | privacy |
| Anything touching secrets or tagged private | local only, or redacted | |

**Where this deviates from your "local first", on purpose:** spec, contract and judge default to the strongest model **when a cloud key exists**, because the evidence says those phases dominate quality. With no cloud key, or with `router.policy: local_strict`, local runs everything at lower default autonomy with visible warnings. It is one config switch.

---

## 7. Verification stack and security gates (the core differentiator)

| Level | Check | Who/where |
|-------|-------|-----------|
| L0 | Lint, types, unit tests for the owned module | worker, local model OK |
| L1 | Merge, build, run **all acceptance tests** | integrator; `pre_verify` hook blocks "finish" while red |
| L2 | **Run the app; a browser agent walks each user story**, saving screenshots | verifier with browser tools (Playwright) |
| L3 | **Security:** secret scan, dependency audit, static rules, and **authz probes generated from the role matrix** (anonymous can't reach private routes; user A can't read user B's data) run against the *running* app | verifier |
| L4 | **Independent judge** compares the result to the **original brief** and assumptions, not to the writer's summary | fresh agent, different family |

Safety rules: default sandbox is worktree plus optional container; **dev/prod separation** (agents never receive production credentials); approval gates for deploy, delete, secrets and spend (reusing Hermes approvals); plan-only mode; hard caps; untrusted-content rule (repo files, web pages and specs are *data*, never instructions; ledger writes pass the existing threat-pattern scan in `tools/threat_patterns`).

---

## 8. Local runtime and tool-call reliability

Build on what exists: `hermes_cli/local_runtime/*` already supervises llama.cpp in router mode, probes hardware, estimates fit, grows the context window, and writes per-model presets including speculative-decoding flags [R]. Do not rebuild it.

Add (in `samagent/router/` and `samagent/verify/`):

- **`ModelProfile`** per model: `max_visible_tools`, `edit_format` (search/replace | patch | whole file), `grammar_mode` (none | JSON-schema | GBNF), `ctx_budget`, `vision`, `known_quirks`.
- **`bench-model` scorecard:** a ≤ 15-minute micro-suite (tool-call validity, edit-apply, multi-step tool use, fix-a-failing-test) that records prefill/decode tokens per second, time to first token, validity % and pass %. Stored in the ledger; the router reads it. *We never route on a leaderboard.*
- **Guards for weaker models:** `pre_tool_call` argument validator (path exists; patch context matches), a repeat-failure detector, and the lean profile (≤ 8 visible tools). Try llama.cpp `--jinja` tool calling and JSON-schema/grammar constraints via `request_overrides.extra_body` (config, likely no patch; spike S6).

---

## 9. Memory: the project ledger

- **Store:** SQLite + FTS5 (already used by Hermes), with a markdown mirror committed to git.
- **Tables:** `facts(scope, kind, text, source_ref, valid_from, valid_to, superseded_by, sensitivity)`, `attempts(task, approach, outcome, error_signature)` (the "tried and failed" journal), `tasks`, `scorecards`, `repo_map`.
- **Supersession, not deletion:** a changed decision closes the old fact's validity window.
- **Writes only on events:** decision made, contract version bumped, user correction, verified outcome. **No periodic "reflection" passes** (a known Hermes complaint).
- **Retrieval:** at task start, a ≤ 2K-token block of the spec excerpt + top 5–10 facts + recent attempts on the same files, placed in the **user** turn, never the system prompt (cache invariant). FTS5 first; embeddings only if the H5 eval says FTS5 is not enough.
- **Privacy tie-in:** facts tagged `private` are excluded from any cloud-routed prompt.
- **Vehicle:** the `MemoryProvider` ABC (`plugins/memory/__init__.py`) and/or a `context_engine` plugin, plus a `pre_llm_call` injection (spike S2).

---

## 10. Speed budget

Arena's internals are not public, so we set **our own measurable budget**. All are *targets* until Phase 0 and Phase 6 measure them.

| Metric | Target |
|--------|--------|
| Time to first visible agent signal | ≤ 2 s cloud, ≤ 5 s warm local |
| Agent compute to approved spec (excluding human thinking) | ≤ 60 s |
| Approval → first live preview of the scaffold | ≤ 90 s |
| Time-to-green, tier 1 app | ≤ 10 min cloud |
| Time-to-green, tier 2 app | ≤ 25 min cloud, ≤ 60 min local (32 GB) |
| Prompt-cache hit on continuation turns | ≥ 80% of input tokens |

**Levers** (each is evidenced in `01-research.md` §9): deterministic scaffolds (skip generation); byte-stable prefixes and cache-aware routing; parallel workers only where the gate allows; a warm local model (the supervisor already keeps `llama-server` up); speculative decoding presets; speculative read-only prefetch (repo map at start); incremental tests; streaming plain-language progress for perceived speed.

---

## 11. Mission Control (the simple UI)

Built **first as a dashboard plugin** (`plugins/samagent/dashboard/`, following the Kanban tab precedent [R]) on the existing typed gateway client in `apps/shared`. Size cap ≈ 15K lines. No editor, no marketplace, no 3D graph.

| Screen | What the user sees |
|--------|-------------------|
| **1. Brief** | One box: "What do you want to build?" + attach. Up to 5 interview cards, each with a *Recommended* chip and "Skip — assume defaults". |
| **2. Plan card** | Plain-language summary, modules, assumptions (editable), cost and time *range*, local/cloud split, **autonomy dial** (Plan only · Check in at milestones · Hands-off), **Approve & build**. |
| **3. Run** | Left: milestone timeline in plain words. Centre: **live preview** (or screenshots). Right: a small agent graph, cost/time meter, **Pause · Steer (type a note) · Stop**. |
| **4. Review & Ship** | "Does it do what you asked?" checklist with ✔/✖ and evidence; security report in plain words; what I assumed; how to run; **Push branch / Open PR / Deploy (gated)**. |
| **5. Memory** | "What I remember" about you and this project; edit, forget, mark private. |

A **Simple / Pro** toggle reveals tool calls, diffs, terminal logs, model per task, token/cost per task and the raw ledger. Simple mode never shows tool names or JSON. Plain-language event text comes from templates first, with a small local model only for free-text.

Fallback if the dashboard-plugin route hits limits (transport or build pipeline, spike S5): a standalone `apps/mission/` on the same client. Desktop packaging (Electron/Tauri wrapper) is a Phase 7 step, reusing the `product-identity.cjs` seam.

---

## 12. Lean profile (fixes the Hermes token complaint)

Hermes' core set is **59 tools**; the coding posture is **37**. **Measured** (tool schemas only, chars/4 proxy; `measurements/tool_footprint.json`): coding posture ≈ **14.4K tokens** (44% of a 32K window), lean-8 ≈ **4.0K** (12%), lean-5 ≈ 2.6K. The kanban tools alone are ≈ 6K. Define a `samagent-lean` profile by role (in config/profile, no deletion):

- **Worker:** `read_file`, `search_files`, `patch`, `write_file`, `terminal`, `todo_list` (+ ledger tools).
- **Orchestrator:** the above + `delegate_task` + a minimal kanban subset.
- **Verifier:** read tools + browser navigate/snapshot/click/type/console/vision.
- Everything else stays behind the existing `tool_search` bridge or is off. Skills index and background reflection are **off by default** in this profile.

Measure in Phase 0 (with `evals/prompt_footprint` and `evals/token_accounting`) before deciding what to cut. H3 decides what stays.

---

## 13. SamBench-Web

- **Tasks:** v0 = 6, v1 = 20, in three tiers. **T1** single-page UI (~10 min); **T2** CRUD + auth + DB (3 modules); **T3** multi-module with roles, mock payments, admin and realtime. Some briefs are deliberately ambiguous.
- **Grading:** hidden acceptance tests (Playwright + API) written by us, and seeded security probes. Never graded by the agent that built it.
- **Metrics:** acceptance pass rate, security probes passed, cost, wall-clock, input/output tokens, cache-hit rate, tool calls, human interventions (should be 0).
- **Models:** one strong cloud, one cheaper cloud, one local (the catalog's `qwen3.8-27b` if it fits the machine).
- **Harness:** reuse `evals/core_tool_deferral/{orchestrator,worker,report}.py` (hermetic per-cell subprocess, programmatic graders, resume-safe, per-model reports) and `evals/fanout_resource_bench.py` for swarm resource limits.
- **Size discipline:** v0 = 6 tasks × 3 arms × 2 models × 2 reps = **72 runs**. The runner prints an estimated spend before launch and enforces a hard campaign cap.

---

## 14. Roadmap

Calendar assumes one builder plus agents, about 12 weeks, **±50% uncertainty**. Tracks overlap.

| Phase | Weeks | Deliverables | Exit criteria | Kill / fallback |
|-------|-------|--------------|---------------|-----------------|
| **0 Measure and de-risk** | 1 | Running baseline, SamBench v0, spikes S1–S9, ADR-001 | **G0** baselines exist; spikes pass or have workarounds | Revisit architecture choice |
| **1 Lean core and local reliability** | 2–3 | `samagent-lean` profile, `ModelProfile`, `bench-model` scorecards, arg validators, loop detector | **G1** H3 measured; local tool-call validity ≥ 95% | Local becomes "helper" role |
| **2 Brief → spec → contract** | 3–5 | Adaptive interview, `spec.yaml` schema and `brief.md` render, spec critique, red-first acceptance tests, contract freeze, plan card with estimate, `web-basic` and `web-auth-crud` templates | **G2** ≤ 5 questions, < 10 min to approved spec; tests generated and red; **H1** measured | Simplify schema |
| **3 Conductor and verification** | 5–9 | DAG, fan-out gate, ownership guard, CCR, integrator, L0–L4 verification, repair loop with caps, task-boundary router and escalation | **G3** **H2, H4, H6** measured | Swarm opt-in; single-thread pipeline default |
| **4 Ledger** (parallel) | 5–8 | Schema, event writes, retrieval and injection, markdown mirror, sensitivity tags | **H5** ≥ 90% | Add embeddings or simplify |
| **5 Mission Control** (parallel) | 4–9 (MVP wk 7) | 5 screens, Simple/Pro, live preview, steer/approve | **G4 / H7** 4 of 5 novices | Cut features, not screens |
| **6 Speed pass** | 9–10 | Profiling, budgets met, warm-up, prefetch, cache-hit ≥ 80% | §10 budgets measured | Report honestly, fix top-2 offenders |
| **7 Harden and ship** | 11–12 | Packaging wrapper, docs, upstream-merge rehearsal, self-security review of the plugin (prompt injection via specs/repo), public benchmark report | All hypotheses reported | — |

**After v1 (not now):** deploy adapters, more stacks (mobile, desktop, games) via skills and template packs, external worker adapters (Pi over RPC, Claude Code/Codex over ACP), remote cloud sandbox runs, a learned router (only if it beats the static table on SamBench), team mode.

---

## 15. Phase 0 — first week, concrete

**Tasks**

| Id | Task | Output |
|----|------|--------|
| T0.1 | Add `upstream` remote; create `docs/samagent/CORE_PATCHES.md` with the rule "≤ 4 patches, each generic" | Merge hygiene from day 1 |
| T0.2 | Run Hermes on its supported **Python 3.14** (via `uv`) with one local llama.cpp model and one cloud provider | A working baseline install |
| T0.3 | Extend the seed `evals/samagent_bench/tool_footprint.py` (**done for tool schemas**: coding ≈ 14.4K vs lean-8 ≈ 4.0K tokens, chars/4 proxy) to the **full request**: system prompt, skills index and memory block, using a real tokenizer; probe how `evals/token_accounting` builds the prefix | Measured per-turn prefix size per posture |
| T0.4 | Build SamBench-Web v0 (6 briefs, hidden tests, seeded security probes) adapted from `evals/core_tool_deferral` | `evals/samagent_bench/` |
| T0.5 | Run baselines A0 (Hermes default) and A1 (Pi) on two models, 2 reps | A results table (cost, time, pass rate) |
| T0.6 | Run spikes S1–S9 below | One-line verdict each |
| T0.7 | Write ADR-001: confirm or revise this plan against the numbers | **Gate G0** |

**Spikes (things I believe but have *not* verified)**

| Id | Question | If false |
|----|----------|----------|
| S1 | Can `samagent/` be packaged via `pyproject.toml` with only patch P2? | Ship as a separate pip package with an entry point |
| S2 | Can `pre_llm_call` inject an ephemeral user-turn block without breaking cache byte-stability? | Patch P4 or a `context_engine` plugin |
| S3 | Do kanban profiles carry model + toolset, so workers can be routed per task with no core patch? | Patch P1 |
| S4 | Do `pre_tool_call` hooks fire inside delegated child agents (for the ownership guard)? | Rely on the post-hoc diff check only |
| S5 | What is the build pipeline for dashboard-plugin `dist/`, and can a plugin get streaming + steer transport (`plugin_api.py` vs gateway methods)? | Standalone `apps/mission/` |
| S6 | Does llama-server `--jinja` + JSON-schema/grammar via `extra_body` give ≥ 95% valid tool calls on the chosen model? | Add a text edit-format fallback (search/replace) |
| S7 | Can `pre_verify` run L1–L4 and block "finish" with a useful message? | Drive verification from the conductor via kanban tasks |
| S8 | Can the `kanban_swarm` blackboard serve as the run's task channel with the ledger as its durable store? | Own small DAG runner in `samagent/conductor` |
| S9 | Does the live agent actually defer tools via the `tool_search` bridge by default? (`get_tool_definitions` returned identical tool sets with assembly on and off in my direct call, though `session_search`/`todo_list` are on the default `defer` list.) | Enable deferral explicitly in the lean profile, or drop non-lean tools by toolset |

---

## 16. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Swarm costs far more than it saves | High | High | Gate, caps, ablation rule (H2) |
| Local model too weak on user hardware | Medium | High | Scorecards, guards, honest capability display, escalation |
| Fork drift | Medium | High | Overlay, patch list, monthly merge rehearsal |
| Generated acceptance tests are wrong or trivial | Medium | High | Red-first check, spec critique, human-visible test list, judge against brief |
| Prompt injection via repo/web/spec content | Medium | High | Untrusted-content rule, threat-pattern scan, verifiers ignore instructions in artifacts |
| UI scope creep | High | Medium | 15K-line cap, plugin first, Simple/Pro only |
| Benchmark too small or noisy | Medium | Medium | ≥ 2 reps (v0), 3 (v1), report intervals; do not over-claim |
| Solo burnout | Medium | High | Kill criteria, weekly end-to-end demo, capped parallel tracks |

---

## 17. Not building in v1

A new agent loop; a TypeScript core; an IDE, editor or completions; a plugin marketplace; a knowledge graph; periodic memory "reflection"; a learned router; permanent role-named specialist agents; mobile/game/desktop targets; built-in multi-host deploy; a hosted sandbox fleet.

## 18. Open questions (defaults apply until answered)

See the list in [`README.md`](README.md): primary user, hardware and keys, local-only mode, where code runs, and the eval budget.
