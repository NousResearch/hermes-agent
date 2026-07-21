# Self-Improvement Protocol

> A systematic self-improvement protocol for Hermes Agent, based on continual learning research.

## Overview

This skill teaches Hermes Agent to **learn from its own mistakes** through a structured protocol derived from the paper "Never Stop Learning: A Survey of Continual Learning and Self-Iteration in LLMs" (Chen 2026).

## Core Concept

```
User Correction → Structured Extraction → Correct Storage → Skill Patch → Active Replay
```

## The OODA-Reflect Loop

```
┌─────────────────────────────────────────────────────┐
│  Observe → Orient → Decide → Act → Reflect → Store  │
│                                        ↑        │    │
│                                        └────────┘    │
└─────────────────────────────────────────────────────┘
```

### When to Reflect

- Complex task completed (5+ tool calls)
- User corrected behavior
- Tool failure followed by recovery
- New edge case discovered

### What to Capture

```yaml
correction_pattern:
  trigger: "<when does this mistake happen>"
  wrong_behavior: "<what I did wrong>"
  correct_behavior: "<what I should do>"
  verifier: "<how to verify it's correct>"
  related_skills: [<affected skills>]
```

## Key Insights from the Paper

| Paper Concept | Agent Implementation |
|---------------|---------------------|
| **Verifier** (§4.1) | User corrections + tool failures |
| **Low-rank updates** (Prop 2) | skill patch > skill rewrite |
| **RAG** (§5.2) | MEMORY + Archive |
| **Replay** (§3.3) | Active replay of corrections |
| **Model merging** (§5.4) | Cross-skill propagation |

## Files

- `SKILL.md` — Main protocol document
- `references/continual-learning-insights.md` — Paper's actionable insights
- `references/correction-patterns.md` — Quick correction templates
- `references/reflection-001-continual-learning.md` — First reflection demo

## Usage

This skill is automatically loaded when:
- User says "复盘 / 反思 / 学到了什么 / post-mortem / lesson learned"
- Complex task completes (5+ tool calls)
- User corrects agent behavior

## Related Skills

- `archived-memory-recall` — Memory retrieval with replay mechanism
- `hermes-agent-skill-authoring` — Skill structure conventions
- `systematic-debugging` — Root cause analysis patterns

## References

- Chen, D. (2026). "Never Stop Learning: A Survey of Continual Learning and Self-Iteration in LLMs"
- Paper: https://victorchen96.github.io/continual_learning_survey.pdf
