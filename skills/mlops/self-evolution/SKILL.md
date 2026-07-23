---
name: self-evolution
description: Run AI-Scientist or ShinkaEvolve via Hermes LLM keys.
version: 0.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [evolution, research, ai-scientist, shinka]
    category: mlops
    related_skills: [darwinian-evolver]
---

# Self Evolution Skill

Drive **AI-Scientist** (idea → experiment) and **ShinkaEvolve** (code evolution)
through Hermes model tools. Credentials come from the same Hermes pool
(`hermes auth` / `~/.hermes/.env`) — do not invent parallel API-key stores.

## When to Use

- User asks to run AI-Scientist, Sakana research loops, or ShinkaEvolve batches.
- You need autonomous idea generation / code evolution under HERMES_HOME.

Do **not** use when a short manual edit or a single `terminal` command is enough.

## Prerequisites

1. Toolset `self_evolution` enabled (`hermes tools` or `platform_toolsets.<platform>`).
2. Vendor trees present:
   - AI-Scientist: `vendor/openclaw-mirror/AI-Scientist/`
   - ShinkaEvolve: `vendor/shinka-osint/`
3. At least one Hermes LLM credential (Codex OAuth, Nous, NVIDIA, Groq, Gemini,
   Anthropic, OpenRouter, DeepSeek, or local Ollama with
   `auxiliary.ai_scientist.allow_ollama_fallback: true`).
4. Optional deps for AI-Scientist: `uv sync --extra ai-scientist`

## How to Run

Prefer the registered tools (not raw shell):

- `ai_scientist_research` — Sakana AI-Scientist launch
- `shinka_run` — ShinkaEvolve `shinka.cli.run` batch

Both accept `model: "auto"` (default) to route through Hermes credentials.

## Quick Reference

| Tool | Required args | Notes |
|------|---------------|-------|
| `ai_scientist_research` | (optional) `experiment`, `num_ideas` | Results under `HERMES_HOME/evolution/ai_scientist/` |
| `shinka_run` | `task_dir` | Relative to `vendor/shinka-osint` (e.g. `examples/circle_packing`) |

Config (non-secret) under `auxiliary.ai_scientist` / `auxiliary.shinka` in
`config.yaml`. Secrets stay in `~/.hermes/.env`.

## Procedure

1. Confirm `self_evolution` is enabled for the current platform.
2. Call `ai_scientist_research` or `shinka_run` with a concrete task/experiment.
3. Read the JSON result (`success`, `results_dir`, `stderr_tail` on failure).
4. Summarize artifacts for the user; do not dump secrets from env overlays.

## Pitfalls

- Long runs (up to hours) — warn the user before large `num_generations`.
- Shinka tasks need `evaluate.py` + `initial.<ext>` in `task_dir`.
- OSINT / MILSPEC Shinka plugin tools (`shinka_*` under `shinka_evolve_osint`)
  are a separate plugin surface — use those for OSINT evolution, not `shinka_run`.

## Verification

Tool returns `"success": true` and a populated `results_dir`. On credential
failure the error message points at `hermes auth` / `~/.hermes/.env`.
