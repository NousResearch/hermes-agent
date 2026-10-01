# 01 — Competitive Analysis: Aro vs the Agent Landscape

Sep 2026 · research for the Aro family (SamAgent / Hermes Agent fork, by samjuniors).

**TL;DR:** Codex app owns the *agentic desktop* category, Cursor owns *IDE integration*, Zed owns *speed + open standards*, Claude Code owns *the terminal*. Aro's opening: none of them have a **self-improving, model-agnostic agent that lives in your messaging apps and runs serverless**. Aro doesn't need to beat them at editors — it needs Codex-app-class polish on its unique surfaces.

---

## The big four

### OpenAI Codex (app + CLI)

**What:** OpenAI's coding agent; standalone desktop app shipped **macOS Feb 2, 2026**, **Windows Mar 4, 2026**, **Linux preview Aug 2026** (alongside the ChatGPT desktop app). Local + cloud task threads running in parallel; **diff-first** review — you see changes as diffs and approve before they land.

| Strengths | Weaknesses |
|---|---|
| Parallel task threads (local + cloud) from one window | OpenAI models only |
| Diff-first review UX — the reference design | Cloud-verbose: sends repo context to OpenAI infra |
| Deep OS integration, genuinely native feel | Heavier/larger footprint than a terminal agent |
| Reviewers: outshines CLI tools for agentic coding | Young Linux support (preview) |

**Steal:** parallel task board with local+cloud mixed execution; diff-first approval (nothing lands unseen); per-task environment isolation.

### Cursor

**What:** The most polished IDE-integrated agent (VS Code fork). Agent mode, Composer (multi-file planning), Tab autocomplete, background agents (cloud), BugBot (PR review).

| Strengths | Weaknesses |
|---|---|
| Most polished agent UX in an editor | Agentic loop less mature than Claude Code's terminal flow |
| Tab autocomplete — best-in-class inline completion | Editor-first: agent is a feature of the IDE, not the product |
| Composer: multi-file orchestration with a plan surface | Model routing good but defaults push Cursor's own models |
| BugBot + background agents cover CI/PR workflows | Closed source, subscription-gated |

**Steal:** Tab-style inline completion (Aro's biggest code-surface gap); Composer-style plan artifact shown *before* edits; background work that keeps flowing when you close the window.

### Zed

**What:** Rust editor, fastest in class. Agent Panel with tool calling; **creator of the ACP (Agent Client Protocol) standard** (now adopted across editors/agents); new **Zed Delta** standalone agent app extending the brand beyond the editor.

| Strengths | Weaknesses |
|---|---|
| Performance — the fastest editing experience available | No usage/cost indicators until requests fail |
| Agent Panel with real tool calling, ACP server built-in | Agent experience still secondary to the editor core |
| ACP: open standard, now multi-editor (VS Code, JetBrains, …) | Smaller extension ecosystem; agent UX rougher than Cursor |
| Zed Delta: credible standalone agent-app bet | Telemetry transparency criticized by community |

**Steal:** visible usage/cost meters (fix their weakness); ACP compatibility (Aro already ships `hermes-acp` — brand it); performance as a feature; minimal dark chrome (Zed is a visual reference for Aro's aesthetic).

### Claude Code

**What:** Anthropic's terminal-native agent. Reviewers' daily driver, ~9.2/10 consensus. Deepest terminal + IDE integration (VS Code/JetBrains extensions), best long-session behavior (context compaction, memory files).

| Strengths | Weaknesses |
|---|---|
| Best terminal-native agent loop | Anthropic-first (other providers possible but not the path) |
| Best long-session behavior: compaction, CLAUDE.md memory, hooks | Terminal-bound: no first-class desktop app |
| Subagents, MCP, hooks, headless modes — complete toolkit | Parallel tasks = multiple terminals, no unified board |
| IDE extensions bridge terminal↔editor better than anyone | Usage/cost visibility improved but still terse |

**Steal:** long-session discipline (Aro's compressor + caching invariant is already stronger — make it visible); CLAUDE.md/SOUL.md-style persona file as a feature; hook/subagent composition; `--dangerously-skip` style explicit risk tiers.

---

## The rest of the field

| Tool | What it is | Strengths | Weaknesses | Lesson for Aro |
|---|---|---|---|---|
| **Windsurf** (ex-Codeium) | IDE with Cascade agent + Flows | Smooth multi-step agentic flows; generous free tier | Cascade quality trails Cursor/CC; company churn | Flow-style guided agent sessions (less prompt roulette) |
| **Devin** (Cognition) | Autonomous cloud engineer | Real parallel cloud VMs, Slack-native handoff, playbooks | Expensive, walled, trust gap on complex tasks | Aro's Modal/Daytona backends = same capability, bring-your-own infra |
| **Warp** | Agent-native terminal | Terminal UX reinvented around AI; blocks + agent mode | Sync/privacy concerns; agent depth < CC | Terminal can be the AI surface — validate's Aro's TUI bet |
| **GitHub Copilot Coding Agent** | Agent in Issues→PR flow | Zero-friction for GH users; assigns an issue, gets a PR | Shallow autonomy, small scope, GH-only | "Delegate to agent" affordance in existing surfaces |
| **Gemini CLI** | Open-weight-friendly terminal agent | Free generous tier, fast multimodal models | Agent loop maturity, ecosystem | Model-agnostic pricing pressure — Aro's 39 providers answer this |
| **OpenCode** | OSS terminal agent (ex-Shell GPT lineage) | OSS, provider-agnostic, modern TUI | Smaller feature surface, community support | OSS terminal agents have real pull — Aro is OSS too, lean on it |
| **Goose** (Block) | OSS local agent | Extensible MCP-first, runs anywhere local | Local-only scale, consumer-polish gaps | Local-first + extensions story resonates |
| **Aider** | Terminal pair-programmer | Git-native commit discipline, precise diffs, reproducible | Narrow: code edits, not general agent | Diff/commit discipline worth copying verbatim |
| **Trae** (ByteDance) | Free agent IDE | Aggressive free frontier models | Monetization/trust uncertainty, regional availability | Free-tier gravity works — Aro + Ollama local = free forever |

---

## Where the puck is going

1. **Agent apps decouple from editors.** Codex app and Zed Delta both bet: the agent is the app, the editor is a view. Aro Desktop is structurally ahead here (it already ships).
2. **Parallelism becomes table stakes.** One conversation = one serial thread is over. Boards of concurrent tasks with isolated envs are the new baseline.
3. **Review is diff-first.** Post-hoc chat logs lose; structured diffs with accept/reject per hunk win.
4. **Cost/usage visibility becomes mandatory.** Zed's failure mode (requests failing before you know you're out) is the anti-pattern to avoid.
5. **Open protocols (ACP, MCP) neutralize lock-in.** Aro ships both a client ecosystem (MCP) and an ACP adapter — rare and valuable.

**Aro's position:** the only product combining a self-improving loop (memory + skills), 39 model providers, ~30 messaging platforms, 7 execution backends incl. serverless, and cron automation — wrapped in a desktop app that currently looks like 2023. The redesign plan ([03](03-redesign-plan.md)) closes the polish gap; the gap analysis ([02](02-gap-analysis.md)) shows where the moats are.
