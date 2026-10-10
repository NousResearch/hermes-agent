# Context Engineering

Context is **everything the model sees at decision time**, and the model operates limited to what it receives. The harness is the machinery that decides what that context is. Writing better prompts stops scaling once an application has data, memory and governance requirements: the problem shifts from "how do I phrase the instruction" to "how do I organize the environment the agent operates in".

| Aspect | Prompt engineering | Context engineering |
| --- | --- | --- |
| Focus | Writing instructions | Orchestrating data and context |
| Scope | Isolated interactions | Integrated systems |
| Data | Limited to the prompt | Multiple sources and retrieval |
| Memory | Absent or limited | Short-term and long-term |
| Scalability | Low | High |
| Governance | Restricted | Structured and auditable |
| Result | One-off answers | Consistent, actionable decisions |

**Context sources inventory.** Six sources feed the window. Each must map to a concrete artifact — a source with no artifact is a gap, not a decision.

| Source | Harness artifact | Notes |
| --- | --- | --- |
| **Instructions** | `CLAUDE.md`, `.claude/rules/` | Always-on behaviour and guardrails |
| **State and short-term memory** | `.claude/memory/memory.md` | Current session state, rewritten on every update |
| **Long-term memory** | `.claude/memory/{YYYYMMDD}-memory.md` | Decisions, lessons and debt across sessions |
| **Retrieved knowledge** | `.claude/knowledge/`, `docs/` | Repository knowledge pulled in on demand |
| **System integrations** | MCP servers in `.mcp.json` or `.claude/settings.json` | External APIs with headers, timeouts, rate limits |
| **Structured outputs** | Sub-agent output schemas in `.claude/agents/` | Forces machine-checkable results |

Never invent a source the repository does not need — an unused knowledge base costs tokens on every session.

**Loading strategies.** Every artifact is classified into exactly one type. An artifact with no declared strategy either bloats every session or is never loaded.

| Type | When | Examples |
| --- | --- | --- |
| **Native always-on** | Loaded by Claude Code | `CLAUDE.md`, `.claude/rules/global-rules.md` |
| **Always-on via read ritual** | Read every session (Agent Loop) | `.claude/memory/memory.md`, `.claude/CONTEXT.md`, `.claude/RULES.md` |
| **Pattern-matched** | By file type | `.claude/rules/{domain}.md` with `paths: ['**/*.cs']` |
| **On-demand** | When referenced | `.claude/knowledge/*.md`, `.claude/TOOLS.md`, `.claude/WORKFLOWS.md`, `.claude/README.md`, `docs/`, long-term memory |
| **Progressive disclosure** | Large codebases | Directory map → headers → content |

**Loading priority.** Lower priority is loaded only when the higher levels leave budget.

1. `CLAUDE.md` and `.claude/rules/global-rules.md` — non-negotiable, native always-on.
2. `.claude/memory/memory.md` — current state (always-on via read ritual).
3. `.claude/CONTEXT.md` and `.claude/RULES.md` — always-on references (read ritual).
4. Rules matching the files being touched.
5. Skills relevant to the request.
6. Knowledge and `docs/` explicitly referenced.
7. Long-term memory — the 3 most recent files only.

**Token budget.** Reserve **20% of the window for output** — a plan the model cannot finish writing is worse than no plan. Always-on content stays small: `CLAUDE.md` at most 1000 lines, `memory.md` at most 100 lines. Everything else is on-demand.

**Chunking.** Files over **500 lines** are never loaded whole. Load the header and section index first, then only the sections needed. Generated artifacts respect the same limit.

**Compaction ladder.** Apply in order, escalating only when the previous step is not enough:

| Step | Action |
| --- | --- |
| **Budget reduction** | Drop on-demand content that is no longer relevant |
| **Snip** | Cut verbose tool output down to the summary lines |
| **Microcompact** | Summarize completed sub-tasks into one line each |
| **Collapse** | Replace a finished work group with its checkpoint entry |
| **Auto-compact** | Full summarization — last resort, always write memory first |

**Rule:** never let auto-compact fire before memory is written. Everything not persisted is lost at that point.

**Memory tiers.**

| Tier | Persistence | Content | Implementation |
| --- | --- | --- | --- |
| **Procedural** | Always loaded | How to work | `CLAUDE.md`, `.claude/rules/` |
| **Semantic** | On demand | Facts, patterns | `.claude/knowledge/`, `docs/` |
| **Episodic** | Cross-session | Experiences, decisions | `.claude/memory/` |

**Governance and risk controls.** Structured context is what makes an agent auditable.

| Risk | Control | Where it lives |
| --- | --- | --- |
| **Data exposure** | Read restrictions on secrets and sensitive paths | `permissions.deny` with `Read(...)` patterns |
| **Hallucination** | Answers grounded in repository sources; every knowledge entry cites its origin | `.claude/knowledge/`, `docs/`, the no-invented-context rule |
| **Stale context** | Just-in-time verification against the code before trusting a memory entry | Memory protocol below |
| **Untraceable decisions** | Append-only decision log with rationale and discarded alternatives | `.claude/memory/{YYYYMMDD}-memory.md` |
| **Uncontrolled tools** | Risk-classified permissions: allow, ask, sandboxed | `.claude/settings.json` |
| **Unbounded external calls** | Documented headers, timeouts and rate limits | MCP declaration plus `CLAUDE.md` |

Two rules apply to every control: never store sensitive data in context artifacts — reference where the value lives, never the value; and prefer computational controls over prompts — a `deny` entry cannot be talked around, a paragraph can.

**Checklist.** The generated `## Context Engineering` section is incomplete without all of these:

- [ ] Every context source mapped to an artifact, or explicitly declared unnecessary
- [ ] Which artifacts are always-on, pattern-matched, on-demand and progressive
- [ ] The loading priority hierarchy
- [ ] The token budget, including the 20% output reserve
- [ ] The chunking rule for files over 500 lines
- [ ] The compaction ladder and the write-memory-first rule
- [ ] The three memory tiers and where each lives
- [ ] The governance controls for data exposure, hallucination and traceability

