---
name: grok-x-research
description: "Use when you need powerful, persistent, citable research on X (Twitter) powered by Grok's native server-side search. Great for ongoing monitoring, sentiment tracking, trend analysis, and synthesizing real-time discourse with Hermes memory and automation."
version: 0.1.0
author: trefong (via Grok Build)
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [xai, grok, x-search, research, monitoring, sentiment, trends, citations]
    related_skills: [grok-xai-oauth, grok-build-patterns, subagent-driven-development]
---

# Grok X Research (Native Search + Hermes Persistence)

When your Hermes session is using the xAI Grok OAuth provider, you get access to Grok's best-in-class X search: server-side, real-time, with high-quality synthesis and direct post citations. This skill turns that into a first-class persistent research capability inside Hermes.

**Why this is special:** Generic web search is noisy. Grok's X search understands context, sarcasm, communities, and recency on the platform where a lot of signal lives. Combine it with Hermes' long-term memory, cron, subagents, and your custom bridges (like grok-concierge Evidence Ledger) and you get autonomous research agents that actually improve over time.

## When to Use

Use this skill when:
- You want ongoing or scheduled intelligence from X (e.g., "track sentiment on [topic] and brief me daily").
- You need citable, synthesized summaries rather than raw links.
- You're doing competitive analysis, trend spotting, community monitoring, or research that benefits from "what people are actually saying right now."
- You want to combine X signals with other Hermes tools (web, browser, your own data, cron jobs, messaging bridges).

**Don't use (or use sparingly) when:**
- You need broad web results beyond X discourse (fall back to general web_search).
- The topic is extremely niche or non-English dominant on X.
- You're on strict cost controls (X search via Grok is powerful but part of the premium experience).

## Core Capabilities (Grok Provider)

- Natural language X search with excellent synthesis.
- Direct citations to specific posts (great for verification and follow-up).
- Strong recency and "vibe" understanding.
- Works beautifully with `delegate_task` for parallel angles (e.g., one subagent on bullish posts, one on bearish, one on developer reactions).
- Pairs with Hermes cron for autonomous monitoring.
- Can feed into your Evidence Ledger or other memory systems for long-term project intelligence.

## Strong Workflows

### 1. Persistent Daily/Periodic Briefings
Use Hermes cron + this skill:

Example cron prompt:
"Every morning at 8am: Use grok-x-research to search X for the latest on Hermes Agent + Grok integration. Synthesize top themes, notable posts with citations, and any new use cases or complaints. Record a concise briefing in the Evidence Ledger."

### 2. Multi-Angle Parallel Research (Grok Build style)
Load `grok-build-patterns` + this skill.

Dispatch subagents on Grok:
- One for overall sentiment
- One for technical/developer discussion
- One for comparisons to other agents
- Reviewer subagent synthesizes

This is exactly the Best-of-N / subagent-driven pattern but specialized for X discourse.

### 3. Deep Dive + Follow-up Agents
Start broad with this skill, then use results to spawn targeted browser or web tasks, or save interesting post URLs for later `x_search` or thread fetching.

### 4. Concierge / Bridge Integration
If you're running something like grok-concierge:
- Have Hermes on Grok continuously update the shared Evidence Ledger with X signals relevant to your projects.
- Use the ledger as context so future research is personalized to what *you* care about.

## Practical Examples

**Quick one-shot:**
```
hermes model  # ensure xAI Grok OAuth
hermes chat -q "Using the grok-x-research skill, search X for recent discussion of Hermes Agent with Grok. Summarize the main use cases people are excited about, with 3-5 cited posts."
```

**Using the helper script (from within Hermes on Grok):**
```
bash ${HERMES_SKILL_DIR}/scripts/run-x-research.sh "Hermes Agent Grok integration" 7
```
The script provides a ready template the agent can adapt for terminal invocation or cron setup.

**Cron monitoring setup:**
Use `hermes cron` to schedule recurring research tasks. The skill will guide the agent on good search phrasing, synthesis format, and how to store results persistently.

**Best-of-N research:**
"Run three parallel searches on [topic] from different angles (technical, business, community reaction). Then synthesize the best overall picture with citations."

## Tips for Best Results with Grok on X

- Be specific in queries: "Hermes Agent Grok OAuth integration" beats "Hermes Grok".
- Ask for synthesis + citations explicitly.
- Follow up on promising posts by asking the agent to fetch the full thread if the skill supports it, or note the post IDs.
- Combine with memory: "Considering what we learned last week about [topic]..."
- For visuals: Pair with Grok Imagine if you want charts/summaries turned into shareable images.

## Common Pitfalls

- Treating X search as "ground truth" — always note it's "current discourse on X."
- Overly broad queries that return too much noise (Grok handles it well but tighter is better for agents).
- Forgetting to persist results — use Hermes memory, your ledger, or files so research compounds.
- Running expensive research loops without mixing in cheaper providers where appropriate (see grok-xai-oauth for multi-model tips).

## Verification Checklist

- [ ] Provider is xAI Grok OAuth (confirm with "what model are you?").
- [ ] Results include direct post citations/links.
- [ ] Synthesis feels high-signal and current (not generic).
- [ ] If using cron or persistence: results are actually stored and retrievable later.
- [ ] Subagent parallel research (if used) produced distinct angles that were usefully combined.

## References & Related

- `grok-xai-oauth` — the foundation (setup and general Grok provider advice).
- `grok-build-patterns` — for the orchestration patterns (Best-of-N, reviews) that work great on top of this.
- Hermes docs on cron and memory.
- Your local grok-concierge Evidence Ledger patterns for long-term research memory.
- Recent community examples: Hermes + Grok X search being used for in-depth topic research (e.g., sports predictions, AI agent comparisons) with results turned into content.

---

**Built with Grok Build while shipping contributions back to Hermes and xAI.** 

Improve this by adding more concrete cron examples, integration with specific Hermes memory providers, or advanced subagent research templates. PRs welcome.

This skill shines brightest when you have a real ongoing need for X intelligence that compounds over time.
