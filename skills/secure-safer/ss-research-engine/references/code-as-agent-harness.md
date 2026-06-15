# Code as Agent Harness — Applied to Secure Safer Research Engine

**Reference:** Ning et al., "Code as Agent Harness: Toward Executable, Verifiable, and Stateful Agent Systems" (2026) — UIUC, Meta, Stanford

## Core Idea

The paper proposes that agent systems should be organized around **code artifacts** rather than free-form prompts. Code is:
- **Executable** — model outputs become operations with verifiable results
- **Inspectable** — every step can be reviewed, debugged, and audited
- **Stateful** — agents persist intermediate state across turns
- **Composable** — multi-agent coordination through shared code artifacts

## Applied to Secure Safer Research Engine

### Layer 1: Harness Interface (Code for Reasoning + Acting + Environment)

The **research workflow** is the harness interface. Rather than prompting "research insurance compliance in NY," the agent executes a structured pipeline:

```
research_engine.py -> Tier 1 (trending) -> Tier 2 (sentiment) -> 
Tier 3 (industry) -> Tier 4 (competitor) -> scorer() -> report()
```

Each tier is a **verifiable code artifact** — saved to vault with timestamps, sources, and scores.

### Layer 2: Harness Mechanisms (Planning, Memory, Tool Use)

- **Planning** — linear pipeline (Tier 1 -> 2 -> 3 -> 4) with optional multi-path search (if Tier 2 finds something big, escalate to deep-dive)
- **Memory** — vault acts as working memory (_research/) and long-term memory (_architecture/, _templates/)
- **Tool use** — You.com Search/Contents/Research/News APIs via MCP or direct HTTP; their output is structured into code artifacts (markdown briefs)
- **Control** — cron jobs for recurring market scans; manual trigger for event-driven research

### Layer 3: Scaling the Harness (Multi-Agent)

Future Phase 2:
- Research Agent (this) -> Content Writer Agent -> Social Media Agent
- Coordination through shared vault state
- Different agents can read/write to _research/, _content/, _social/

## Key Takeaway

Build the research engine as **executable pipelines**, not prompt templates. Every research run produces a verifiable artifact in the vault with timestamps, source citations, and scores. This makes the engine:
1. Auditable (what was found, when, from where)
2. Repeatable (cron jobs run the same pipeline)
3. Improvable (add new tiers or sources without rewriting prompts)
