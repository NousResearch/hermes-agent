# Adventure: The Agent That Taught Itself About AI Frameworks

**Author:** Community Contributor  
**Date:** February 26, 2026  
**Platform:** Hetzner CX22 · Ubuntu 24.04  
**Model:** claude-sonnet-4-6 (via Nous Portal)  
**Session ID:** `20260226_135939_466b2a`  
**Duration:** ~3 minutes  

---

## The Goal

I gave Hermes a single open-ended research task:

> *"I want you to research the top 5 most starred Python AI agent frameworks on GitHub (like langchain, autogen, crewai, etc). For each one, find their GitHub stars, main features, and biggest weaknesses. Then write a comprehensive skill document called 'ai-agent-frameworks' that I can use next time I need to choose a framework. Save it as a skill when you're done."*

No instructions on how to do it. No toolset hints. Just the goal.

---

## What the Agent Did

Hermes immediately broke the task into two phases on its own:

### Phase 1 — Deep Research (70+ tool calls)

The agent launched an aggressive parallel research campaign without being told to:

- **50+ targeted web searches** across Reddit (`r/LangChain`, `r/LocalLLaMA`, `r/AI_Agents`, `r/MachineLearning`), Hacker News, GitHub issues, and developer blogs
- **20+ page fetches** from specific URLs: the famous Max Woolf "Problem with LangChain" post, HN thread #38987242, multiple Reddit megathreads, and GitHub issue trackers
- It specifically hunted for **real-world complaints** — not marketing copy. Search queries like:
  - `"LangChain" "too complex" OR "over-engineered" site:reddit.com OR site:news.ycombinator.com`
  - `AutoGen "infinite loops" OR "runaway cost" GitHub`
  - `why I stopped using LangChain blog post developer experience`

### Phase 2 — Skill Creation (autonomous)

Without being prompted, after finishing research, the agent:

1. Synthesized findings into a structured internal report
2. Called `skill_manage(action="create")` to write and save a full `SKILL.md`
3. Stored it at `~/.hermes/skills/mlops/ai-agent-frameworks/SKILL.md`

**The agent chose the category `mlops/` on its own.** I never mentioned it.

---

## The Skill It Wrote

The agent produced **a 400+ line SKILL.md** entirely autonomously. Highlights:

### Quick Decision Matrix (agent-generated)

| Use Case | Recommended | Avoid |
|---|---|---|
| Production RAG over private docs | LlamaIndex | MetaGPT |
| Multi-agent research / exploration | AutoGen | CrewAI |
| Rapid PoC / demo with role agents | CrewAI | MetaGPT |
| Widest integration ecosystem needed | LangChain | MetaGPT |
| Cost-sensitive production workload | LlamaIndex / raw SDK | MetaGPT / AutoGen |

### Star Counts (researched live, Feb 2026)

| Framework | Stars | Forks | Company |
|---|---|---|---|
| LangChain | ~97,900 | 15,900 | LangChain Inc |
| MetaGPT | ~47,400 | 5,600 | DeepWisdom (CN) |
| AutoGen | ~40,800 | 5,900 | Microsoft Research |
| LlamaIndex | ~40,000 | 5,700 | LlamaIndex Inc |
| CrewAI | ~28,100 | 3,800 | CrewAI Inc |

### Sharpest Finding Per Framework

The agent didn't just list features — it found the **community consensus criticism** for each:

- **LangChain:** Over-abstraction. "3 lines with raw OpenAI = 20+ lines in LangChain, harder to understand" (Max Woolf, July 2023, HN thread #36645575 referenced)
- **MetaGPT:** $5–$20+ per task in API costs. "A fascinating research demo; not a practical tool."
- **AutoGen:** v0.2 → v0.4 was a near-total rewrite. Community fractured. Infinite loop bugs well documented.
- **LlamaIndex:** RAG is best-in-class but v0.10 reorganization broke everything. "Traumatic" per frequent users.
- **CrewAI:** Agents claim to complete tasks without doing so. Role-playing is just a prompt string — LLMs ignore it.

### Cross-Cutting Insight (agent-synthesized)

The agent identified 6 problems that affect **all** frameworks:

1. LLM non-determinism — none of them solve it
2. Cost opacity — token spend is unpredictable
3. Commercial lock-in risk (LangSmith, Azure AI, crewai.com, LlamaCloud)
4. Debugging is universally hard
5. Demo-to-production gap
6. Rapid API churn across the board

### Decision Flowchart (agent-written)

```
1. Is RAG your primary use case?
   YES → LlamaIndex
   NO  → continue

2. Do you need code generation + execution in a loop?
   YES → AutoGen
   NO  → continue

3. Is this a PoC/demo?
   YES → CrewAI
   NO  → continue

4. Need maximum integration ecosystem?
   YES → LangChain (use LangGraph for agents)
   NO  → use raw SDK
```

---

## What Surprised Me

**The agent went further than asked.** I asked for "top features and biggest weaknesses." It independently:

- Sourced criticisms from primary community discussions (Reddit, HN), not just docs
- Cross-referenced a specific blog post by name (Max Woolf's famous critique)
- Added a "When NOT to Use" section for every framework — I never asked for this
- Added a "Recommended Alternatives" section pointing to `smolagents`, `DSPy`, and `Instructor` — including references to other Hermes skills (`see dspy skill`, `see instructor skill`)
- Categorized the skill under `mlops/` without being told

**The agent also anticipated future sessions.** The skill's `triggers:` frontmatter will cause Hermes to auto-load this skill whenever someone asks about agent frameworks — the agent wrote its own loading conditions.

---

## The Skill File

Saved to: `~/.hermes/skills/mlops/ai-agent-frameworks/SKILL.md`

Compatible with the [agentskills.io](https://agentskills.io) open standard. Can be installed by anyone running Hermes via:

```bash
hermes skills install <path-to-this-skill>
```

---

## Takeaway

This task took **one prompt and ~3 minutes.** The agent:

- Conducted research I would have spent hours on
- Synthesized community consensus from dozens of sources
- Wrote a reusable, structured document that future sessions can load on demand
- Chose appropriate categorization and metadata autonomously

The skill system is the most underrated part of Hermes. The agent didn't just answer a question — it **taught itself something and remembered it.**

---

## Reproduction Steps

```bash
# 1. Install Hermes
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc

# 2. Login
hermes login

# 3. Start chatting
hermes

# 4. Give the agent this prompt:
# "Research the top 5 most starred Python AI agent frameworks on GitHub.
#  For each one, find their GitHub stars, main features, and biggest weaknesses.
#  Write a comprehensive skill document and save it as a skill when done."

# 5. Check the result
cat ~/.hermes/skills/mlops/ai-agent-frameworks/SKILL.md
```

---

*Session log available at: `~/.hermes/sessions/session_20260226_135939_466b2a.json`*
