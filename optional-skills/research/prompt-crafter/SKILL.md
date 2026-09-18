---
name: prompt-crafter
description: Analyze and improve AI prompts.
version: 1.0.0
author: Kewe63
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [prompts, prompt-engineering, llm, quality]
    related_skills: [writing-plans, humanizer, technology-research]
prerequisites:
  commands: [python3]
---

# Prompt Crafter Skill

Analyze, score, and improve AI prompts with a heuristic 8-dimension quality
check, generate variations for A/B testing, and pull reusable templates. No API
key required — everything runs locally with the standard library.

## When to Use

- You are drafting or reviewing a prompt and want an objective quality score
  before sending it to a model.
- You need several reworded variations of one prompt for A/B comparison.
- You want a starting template (code review, ELI5, brainstorm) to adapt.

Prefer `humanizer` when the goal is stripping AI-isms from prose; this skill
targets the *structural* fitness of a prompt, not its voice.

## Prerequisites

- Python 3.10+ (stdlib only — `argparse`, `json`, `sys`).
- No network, no API key.

## How to Run

Drive the bundled helper script through the native `terminal` tool — it is the
supported interaction surface here, not ad-hoc Python pasting. The loader
substitutes `${HERMES_SKILL_DIR}` at scan time, so copy the resolved path or run
from inside the skill directory.

```bash
# Score a prompt across 8 quality dimensions
python3 "${HERMES_SKILL_DIR}/scripts/prompt_crafter.py" analyze "You are a code reviewer..."

# List available templates
python3 "${HERMES_SKILL_DIR}/scripts/prompt_crafter.py" templates

# Show one template
python3 "${HERMES_SKILL_DIR}/scripts/prompt_crafter.py" templates --name code-review

# Generate improved variations of a prompt
python3 "${HERMES_SKILL_DIR}/scripts/prompt_crafter.py" variations "Explain quantum computing"

# Read the prompt from stdin instead of args
echo "Summarize this article" | python3 "${HERMES_SKILL_DIR}/scripts/prompt_crafter.py" analyze
```

## Quick Reference

| Command | Positional | Flags | Returns |
|---------|-----------|-------|---------|
| `analyze` | `<prompt...>` (or stdin) | — | `{word_count, quality_score, checks_passed, details[], suggestions[], verdict}` |
| `templates` | — | `--name <id>` | one template or `available_templates[]` |
| `variations` | `<prompt...>` (or stdin) | — | `{original, variations: [{style, prompt}]}` |

The 8 scored dimensions: Role/Persona, Context, Constraints, Examples (Few-shot),
Output Format, Clear Goal, Tone/Style, Chain of Thought.

## Procedure

1. Run `analyze` on the candidate prompt to get a 0–100 `quality_score` and the
   list of `details[]` checks that failed.
2. Read `suggestions[]` — each names a dimension to add.
3. Run `variations` to auto-generate reworded versions that patch the missing
   dimensions (role, format, constraints). Iterate until the score is ≥80.
4. Use `templates --name <id>` as a structural starting point for a new prompt,
   then re-run `analyze` to confirm the score.

## Pitfalls

- The analyzer is heuristic, not semantic: it keys off surface cues (words like
  "format", "step by step", "you are"). A technically weak prompt can still score
  high if it uses the right vocabulary — treat the score as a checklist, not truth.
- `variations` only adds a dimension if the analysis flagged it missing, so a
  strong prompt returns a single `"Original (strong)"` entry rather than new ideas.
- Template output is plain text, not JSON — pipe `templates --name <id>` through
  `read_file`/`vision_analyze` rather than parsing it as structured data.
- All input is read as-is; very long prompts are scored on word count thresholds
  (context needs >30 words, clear goal >10 words), so tiny prompts auto-fail those
  checks by design.

## Verification

Run the bundled tests (no network required — all logic is pure functions):

```bash
scripts/run_tests.sh tests/skills/test_prompt_crafter_skill.py -q
```

Spot-check a real call before quoting results to the user:

```bash
python3 "${HERMES_SKILL_DIR}/scripts/prompt_crafter.py" analyze "You are a helpful assistant that summarizes text." | head -20
```
