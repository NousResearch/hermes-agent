---
name: prompt-crafter
description: Analyze, optimize, and generate AI prompts. Quality scoring across 8 dimensions, template library for common scenarios, variation generation for A/B testing. Improves prompt effectiveness using prompt engineering best practices.
platforms: [linux, macos, windows]
---

# Prompt Crafter

Analyze and optimize prompts for AI models. Uses heuristic-based quality analysis with 8 critical dimensions. **No API keys required.**

## Helper script

This skill includes `scripts/prompt_crafter.py` — a complete CLI tool.

```bash
# Analyze a prompt
python3 SKILL_DIR/scripts/prompt_crafter.py analyze "You are a code reviewer..."

# List available templates
python3 SKILL_DIR/scripts/prompt_crafter.py templates

# Get a specific template
python3 SKILL_DIR/scripts/prompt_crafter.py templates --name code-review

# Generate prompt variations
python3 SKILL_DIR/scripts/prompt_crafter.py variations "Explain quantum computing"
```

## Commands

| Command | What it does | Example |
|---------|-------------|---------|
| `analyze` | Score prompt across 8 quality dimensions | `analyze "Explain X"` |
| `templates` | List or show prompt templates | `templates --name brainstorm` |
| `variations` | Generate improved variations | `variations "Do Y"` |

## Quality Dimensions

1. **Role/Persona** — Clear role assignment (You are a...)
2. **Context** — Background information before instructions
3. **Constraints** — Boundaries (length, format rules, exclusions)
4. **Examples (Few-shot)** — Sample inputs/outputs
5. **Output Format** — Exact structure specified (JSON, markdown, etc.)
6. **Clear Goal** — Specific, measurable objective
7. **Tone/Style** — Communication style defined
8. **Chain of Thought** — Step-by-step reasoning encouraged

## Templates

### code-review
Senior engineer review with security, performance, and quality focus.

### explain-like-im-5
Simplified explanations using analogies and everyday language.

### brainstorm
Creative ideation with diverse perspectives and prioritization.