# 01 — Research

Evidence tags: **[R]** read in repo/docs · **[W1]** paper / vendor / Arena post · **[W2]** blog/community (directional) · **[I]** my inference.
See [`README.md`](README.md) for limits. Hermes claims are static (code reading), not measured.

---

## 1. Arena's agent: what is public, what is not

**Public [W1]** (`arena.ai/blog`: `agent-mode`, `coding-in-agent-mode`, `agent-arena-methodology`, `coding-agents-harness-tax`, Jun–Sep 2026):

- One orchestrating agent with built-in tools inside a **cloud sandbox per session**: clone a repo, GitHub OAuth, a live diff panel, a **preview of served ports**, the full git/PR lifecycle, and **dedicated structured tools instead of raw bash**.
- User behaviour: people delegate, then tighten control. About **twice as many users tighten as loosen** control after using it. This supports an **autonomy dial** and visible plans.
- **Harness tax study** (21 model×harness pairs): the harness barely changes *success* (about ±2–5%) but changes *cost* by up to **5×**. Claude Code cost about 2× Pi. Pi, with 4 tools, sits on the Pareto frontier.
- Open-weight models trail in Arena's aggregate ranking. The published gaps include GLM 5.1 +3.4, Kimi K2.6 −0.6, DeepSeek V4 Pro −1.9, Qwen 3.6 Plus −3.4, Gemma 4 31B −14.6. I did not verify the scale.

**Not public:** swarm or sub-agent internals, routing policy, caching strategy, warm-pool design.

**Consequence [I]:** "Arena's speed" cannot be copied, only *reproduced by design*. The levers that are evidenced elsewhere are in §9 and become the speed budget in `05-final-plan.md`. Arena's visible mechanics that we should copy are: a **live preview**, a **diff panel**, **structured tools**, a **lean single orchestrator loop**, and **control** (steer/approve).

---

## 2. Pi (`earendil-works/pi`, MIT)

- **Four tools, system prompt under 1k tokens.** No MCP, sub-agents or plan mode in the core; those are extensions, skills or packages. [R/W1]
- **JSONL session tree** (branch, fork, clone). New packages `pi-durable` and `chord`. Its own client/server protocol is still moving, so it is a fast-changing target. [R]
- **Virtual models** (`virtual-models.md`) [R]:
  - An extension registers a model whose `route(request, ctx)` runs before every request and returns `{model, thinkingLevel, state}`.
  - `request.reason` is `user | continuation | retry | direct`. Router state is stored per session branch.
  - **Sticky routing on `continuation`/`retry`** keeps the prompt cache and thinking signatures valid.
  - The example `jev-router.ts` plans on a strong model, lets it make the first edit, then switches **once** to a cheaper model, accepting one cache miss.
- **llama.cpp** (`llama-cpp.md`): use router mode with `--jinja`, and load models on demand. [R]
- **`cache-warmer.ts`:** refresh at 90% of the TTL, only if expected savings ≥ $0.05. [R]
- **What to take from Pi:** the lean tool posture, sticky routing, cache-aware switching, the session tree idea and the cache-warmer economics. **What not to take:** the code. Hermes is Python with 860K lines of solved provider and gateway problems.

---

## 3. Your repo (Hermes) — audit [R, static]

**Scale:** about 860K non-test Python lines in 2,391 files, plus about 1.26M lines of tests. `hermes_cli` 251K, `agent` 128K, `tools` 120K, `gateway` 94K, `plugins` 88K, `tui_gateway` 41K. The desktop app is about **608K lines of TS/TSX in 1,022 files**; the web dashboard is about 58K.

**Already built and reusable:**

| Need | Where | Notes |
|------|-------|-------|
| Sub-agents | `tools/delegate_tool*.py` | `leaf`/`orchestrator` roles, depth limits, batch-parallel and background modes, steer/interrupt, blocked child tools, default 10 concurrent children |
| Isolation | `tools/subagent_worktree.py` | `delegation.worktree_isolation`, branch `hermes-subagent/<id>` |
| Swarm kernel | `hermes_cli/kanban*.py`, `tools/kanban_tools.py` | `kanban_specify` (idea→spec), `kanban_decompose` (task graph routed to **profiles**), `kanban_swarm` (root → parallel specialists → verifier → synthesizer, JSON blackboard) |
| Goal loops | `hermes_cli/goals.py` | Ralph-style loop with an auxiliary judge |
| Verification | `agent/verification_stop.py`, `agent/verify/*`, plugin hook `pre_verify` | A plugin can block "finish" and force another round |
| Context | `agent/context_engine.py` | ABC for pluggable engines |
| Memory | `plugins/memory/*`, `tools/memory_tool_store.py`, `session_search` | Curated MEMORY.md/USER.md **injected into the system prompt**, bounded by chars; 7 provider plugins |
| Local models | `hermes_cli/local_runtime/*` | Managed llama.cpp supervisor in router mode, hardware probe, GGUF header reader, memory "physics check", context-window growth, per-model presets including speculative-decoding flags, a model catalog (repo lists `qwen3.8-27b`, `qwen3.6-35b-a3b`, `deepseek-v4-flash`, …) |
| Lean tool exposure | `tools/tool_search.py`, `tools.tool_search.*` config | Defers cold tools behind a 3-tool bridge (`tool_search`/`tool_describe`/`tool_call`) |
| Providers | `plugins/model-providers/*` | About 40 providers, credential pools, fallback chains |
| UI seam | `tui_gateway/` + `apps/shared` | JSON-RPC over stdio/WebSocket, Pydantic contracts generated to TypeScript (`gateway-contract.generated.ts`, `.openrpc.json`). Desktop, TUI and dashboard `/chat` all use it |
| Dashboard plugins | `plugins/kanban/dashboard/` | A tab with `manifest.json` + prebuilt `dist/index.js` + `plugin_api.py`. This is a precedent for shipping a UI *as a plugin* |
| Eval harnesses | `evals/core_tool_deferral/`, `evals/fanout_resource_bench.py`, `evals/prompt_footprint/`, `evals/token_accounting/` | Hermetic real-agent A/B batteries with programmatic graders, resume-safe orchestrators, per-model reports |
| Rebrand seam | `apps/desktop/product-identity.cjs`, `SOUL.md` | Product identity is centralised |

**Weak or missing relative to your goals:**

1. **Tool surface.** `_HERMES_CORE_TOOLS` has **59** tools (browser ×18, kanban ×14, HA ×4, …). The coding posture (`_CODING_TOOLS`) is **37**. Pi uses 4. See the measured footprint in §3a. [R + measured]
2. **No per-task model in `delegate_task`.** Children share one global `delegation.model/provider`. The signature is `goal, context, tasks, role, background, output_schema, …`. Per-task routing needs either **kanban profiles** (which already carry model + toolset) or one small generic patch. [R]
3. **Memory is context-bound.** The built-in store is a bounded snapshot in the system prompt. There is no bi-temporal validity, no supersession, and no project *decision* ledger with an "attempts that failed" journal. [R]
4. **Specs are text.** `kanban_specify` turns an idea into a written spec, but I found no step that compiles acceptance criteria into *executable* tests, or that freezes a shared contract before fan-out. Not exhaustively audited. [R]
5. **No default live-app or security verification layer.** `pre_verify`, verify recipes and goals exist. I found no built-in browser-walkthrough of user stories or authz probes. Not exhaustively audited. [R]
6. **UI complexity.** 608K lines of TSX, chat-centric, aimed at power users. [R]
7. **Hermes community complaints [W2]:** about 10–20% more tokens than a comparable agent in one day-long test; too many tools/skills on by default; a hard-to-customise system prompt; a roughly 3k-token skills index in the system prompt; background reflection passes that cost tokens; hard-coded counters (skill creation after 5 tool calls, reflection every 15 turns); flaky onboarding. Praised: desktop app, skill growth, cache hit rate.

### 3a. The one thing I measured: tool-schema footprint

Source: [`measurements/tool_footprint.json`](measurements/tool_footprint.json), produced by `evals/samagent_bench/tool_footprint.py` (imports the real tool registry; **no model calls**). Token counts are **chars/4, a rough proxy** (the tokenizer download was blocked). It covers **tool schemas only**, not the system prompt, skills index or conversation, and it ran on unsupported Python 3.11 with hand-installed dependencies.

| Posture | Tools | Schema tokens (≈) | Share of a 32K window | Share of 128K |
|---------|-------|-------------------|-----------------------|---------------|
| All core tools | 59 | 24,900 | 76% | 19% |
| **Coding posture** | 37 | **14,400** | **44%** | 11% |
| Lean-8 (read, search, patch, write, terminal, todo, delegate, clarify) | 8 | 4,000 | 12% | 3% |
| Lean-5 (no todo/delegate/clarify) | 5 | 2,600 | 8% | 2% |
| *kanban_\* (14 tools, inside core)* | 14 | 6,000 | | |
| *browser_\* (18 tools, inside core)* | 18 | 4,700 | | |

Largest single schemas: `terminal` ≈1.2K, `delegate_task` ≈1.1K, `session_search` ≈1.0K, `skill_manage` ≈0.9K, `memory` ≈0.9K, `execute_code` ≈0.8K.

What this supports, and what it does not:

- **Supports [measured + I]:** for a **32K-context local model** (the llama.cpp doc default is `-c 32768`), the coding posture alone would spend about **44% of the window on tool schemas** before any system prompt or code. That is a strong, concrete reason for a lean per-role profile on local workers.
- **Does not show:** real per-turn cost. With prompt caching, repeated schema tokens are billed at cache-read rates, so cost impact is smaller than the token count suggests; the larger effects are on **first-turn latency and small-window local models**. H3 in the final plan must be tested with real runs.
- **Open question:** in this direct call, `get_tool_definitions(...)` returned the same 17 tools with and without tool-search assembly, even though `session_search`/`todo_list` are on the default `defer` list. The live agent may apply deferral elsewhere. This is spike S9 in the final plan. Also, only 17 of 37 coding tools resolved here because backends (browser, etc.) are not available in the sandbox.

**Invariants we must respect [R]** (`AGENTS.md`, `agent/AGENTS.md`, `plugins/AGENTS.md`): prompt-cache byte-stability (only compression may mutate context); strict role alternation; synchronous agent loop; plugin-first; **plugins never touch core files**; config in `config.yaml` not env vars; no telemetry without opt-in; tool descriptions must not name tools from other toolsets; new memory backends live outside the in-tree `plugins/memory/`.

---

## 4. Other agents (landscape)

| Agent | Relevant facts | Confidence |
|-------|----------------|-----------|
| **Claude Code** | Strong at coding. Dynamic Workflows (May 2026): the model writes a JS orchestration script that holds the plan *outside* context, up to 1,000 sub-agents (16 concurrent), adversarial verifiers. A New Stack test scaffolded a **shared contract first**, then ran 5 parallel agents in 6m59s with 62 tests and about 109K sub-agent tokens (about $3–5). Agent Teams (Feb 2026) use a git-based shared workspace with task claiming. | W2 |
| **Codex CLI** | Roughly equal to Claude Code on Terminal-Bench 2.1 in secondary reports. | W2 |
| **Cursor 3 / Windsurf** | IDE-centric. Cursor has Cloud Agents. Windsurf and Qwen Code have "arena/compare" modes that use worktrees and let the user pick a winner. | W2 |
| **OpenCode** | 75+ providers, terminal-first. | W2 |
| **Replit Agent 3** | 200-minute autonomy and browser self-testing, but reviews report slow runs, unpredictable credits, infra lock-in, black-box decisions. Replit deleted a production DB during a code freeze (fixes: dev/prod DB separation, planning-only mode). | W2 |
| **Lovable / Bolt** | About 5-minute autonomy; users reach about 70% and the last 5% is hard for non-developers. Lovable CVE-2025-48757: 170 of 1,645 sampled apps had broken row-level security. | W2 |
| **Kiro / Spec Kit / OpenSpec** | Spec-driven development (requirements → design → tasks). Acceptance criteria are prose and don't execute; verification is left to the user; Spec Kit reportedly costs about 90 minutes of spec+plan per feature. | W2 |

---

## 5. Multi-agent: what the evidence actually says

- **Cognition, "Don't Build Multi-Agents" [W1]:** parallel sub-agents make conflicting *implicit* decisions (the Flappy-Bird example). They recommend one thread and sharing the full trace.
- **Anthropic's research system [W1]:** orchestrator + 3–5 parallel sub-agents beat a single agent by **+90.2%** on *research*, at about **15× the tokens**. Token spend explains about 80% of the variance. Anthropic itself says tightly interdependent work such as most coding fits poorly.
- **MAST failure taxonomy, arXiv 2503.13657 [W1]:** 14 failure modes in multi-agent systems. **Spec/design 41.8%**, **inter-agent misalignment 36.9%**, **verification 21.3%**. Multi-level verification is recommended.
- **Synthesis [I]:** parallelism pays for *read-shaped, independent* work. Coding is *write-shaped*. It needs (1) a frozen shared contract before fan-out, (2) isolated workspaces, (3) one owner per module, (4) an independent verifier, and (5) a single integrator. The interview attacks MAST's biggest bucket (spec). The contract attacks the second (misalignment). Layered verification attacks the third.

---

## 6. Local models and tool-calling reliability

- **Rankings are unreliable [W2].** Sources conflict on sizes and scores. Recurring candidates: Qwen 3.x 27B/30B-A3B/35B-A3B, Devstral Small 2 (24B), gpt-oss 20B/120B, GLM. The repo's own catalog lists `qwen3.8-27b` as "best all-round agent model". **Plan: benchmark on the user's hardware; never trust a leaderboard.**
- **Small models break on tool calls first**, especially with crowded context. [W2]
  - Grammar-constrained decoding removes syntax errors. One blog measured 68% → 91% first-try on a 7B model (weak source; directional).
  - Validate arguments (does the path exist?), keep 3–5 tools visible per step, prefer text edit formats (search/replace) over JSON edits, and detect repeat-same-fix loops.
- **Arena's harness-tax post [W1]** also reports large tool-hallucination and bash-recovery gaps for small open models.
- **The "physics" of fit** is already handled by `hermes_cli/local_runtime/estimator.py` (it refuses loads that will not fit). [R]

## 7. Routing

- RouteLLM reported up to 85% cost reduction at 95% of GPT-4 quality on MT-Bench [W1]. Databricks and AT&T report 30–56% savings on coding at about 2% quality loss, **but only after building golden-task eval suites** [W2].
- Plan-on-strong / execute-on-cheap saves about 50–70% [W2].
- Quality-triggered cascades (try cheap, verify, escalate) must live in the application layer.
- **Cache reality [R]:** Pi's router docs and Hermes' AGENTS.md both say a mid-conversation model switch loses the prompt cache. Route at **task/phase boundaries**; be sticky on retries.

## 8. Memory

- LongMemEval numbers are inconsistent across vendors and independent runs (Zep/Graphiti temporal KG about 64–71%; Mem0 about 49% self-reported vs about 32% in independent OSS runs) [W2]. **Vendor self-reports are unreliable.**
- What is consistent: **validity windows** ("true from X until superseded") beat plain vector stores for changing facts; **context-stuffing is an anti-pattern** (inject the top 5–10 items); raw verbatim stores recall well. [W2]
- For coding agents the highest-value memory is **project state**, not chat trivia: decisions and their supersession, the contract, the task graph, what was tried and failed, and per-model scorecards. [I]

## 9. Speed

- Agent turns are **prefill-dominated**, so prompt/KV-cache reuse is the largest single lever. Caching cut cost 41–80% and time-to-first-token 13–31% (arXiv 2601.06007) [W1]. Keep the prefix byte-stable and put dynamic content last (this is a Hermes invariant already [R]).
- Speculative decoding gives about 1.5–3× decode speed [W2]; Hermes' presets already carry `--spec-type` flags [R].
- Sandbox cold starts under 1 second are normal for microVM sandboxes (E2B about 150ms; Firecracker about 125ms) versus about 15s for plain Docker [W2].
- Speculatively executing read-only tools reduced task time by up to about 48% in research [W2].
- **Perceived speed** (first visible action, streaming progress, a live preview) matters as much as wall-clock. [I]

## 10. Vibe-coding failure modes

- Moltbook leaked about 1.5M API keys; Lovable's RLS CVE; Replit's prod-DB deletion; a scan of about 380k apps; about 45% of AI-generated code failing OWASP Top-10 checks; "hallucinated auth" recurs. About 63% of vibe-coding users are non-developers. [W2]
- **Gap:** no verification layer between AI output and production, and no infrastructure-level limits on what an agent may touch. [I]

---

## Sources (fetched or searched during research)

- Arena: `arena.ai/blog/agent-mode`, `/coding-in-agent-mode`, `/agent-arena-methodology`, `/coding-agents-harness-tax`
- Pi: `github.com/earendil-works/pi` (cloned; read `README`, `AGENTS.md`, `packages/coding-agent/docs/{virtual-models,how-pi-works,llama-cpp,sessions}.md`, `src/core/cache-warmer.ts`)
- Cognition: `cognition.ai/blog/dont-build-multi-agents`
- Anthropic: `anthropic.com/engineering/multi-agent-research-system`
- MAST: arXiv 2503.13657 · Prompt caching for agents: arXiv 2601.06007
- Claude Code Dynamic Workflows / Agent Teams, The New Stack test, Replit/Lovable/Bolt/Kiro/Spec Kit reviews, local-model and routing/memory blog posts, Hermes community threads: secondary web results tagged **[W2]** above
- This repo at commit `39faafb6`
