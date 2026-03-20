---
name: adversarial-council
description: Run multi-perspective adversarial deliberation using the hermes-council MCP server. Five personas (Advocate, Skeptic, Oracle, Contrarian, Arbiter) debate questions through structured rounds, producing calibrated verdicts with confidence scores and evidence links. Use when the user needs rigorous analysis, safety gating, or quality evaluation.
version: 1.0.0
requires: hermes-council MCP server (pip install git+https://github.com/Ridwannurudeen/hermes-council.git)
author: Ridwannurudeen
tags: [council, adversarial, deliberation, safety, evaluation, mcp]
---

# Adversarial Council

Run structured adversarial deliberation through five personas from distinct intellectual traditions.

## Setup (one-time)

### 1. Install the server

    pip install git+https://github.com/Ridwannurudeen/hermes-council.git

### 2. Add to hermes config

In `cli-config.yaml`:

```yaml
mcp_servers:
  council:
    command: hermes-council-server
    env:
      COUNCIL_API_KEY: ""  # Your OpenRouter/OpenAI API key
      COUNCIL_MODEL: "nousresearch/hermes-3-llama-3.1-70b"
    timeout: 180
```

### 3. Set API key

    export OPENROUTER_API_KEY=your-key-here

## Available Tools

| Tool | Use When | Personas Used |
|------|----------|---------------|
| `council_query` | Complex questions needing multi-perspective analysis | All 5 (Advocate, Skeptic, Oracle, Contrarian, Arbiter) |
| `council_evaluate` | Evaluating content quality with configurable criteria | All 5 |
| `council_gate` | Quick safety check before high-stakes actions | Skeptic + Oracle + Arbiter |

## The Five Personas

| Persona | Tradition | What They Do |
|---------|-----------|-------------|
| **Advocate** | Steel-manning | Builds strongest case FOR the position |
| **Skeptic** | Popperian falsification | Finds the observation that kills the claim |
| **Oracle** | Empirical base-rate reasoning | Grounds debate in historical data and statistics |
| **Contrarian** | Kuhnian paradigm critique | Rejects the framing, proposes alternative paradigms |
| **Arbiter** | Bayesian synthesis | Synthesizes all views with explicit prior/posterior updates |

## Usage Patterns

### Deep analysis of a complex question

Use `council_query` with a focused question. The council returns:
- Individual persona arguments with confidence scores
- Evidence links and references
- Arbiter synthesis with Bayesian prior/posterior updates
- Final verdict with calibrated confidence

### Quality evaluation

Use `council_evaluate` with the content to evaluate and optional criteria:
- `accuracy` — factual correctness
- `depth` — analytical thoroughness
- `evidence` — citation and data support
- `falsifiability` — testable predictions

### Safety gating before actions

Use `council_gate` before high-stakes actions. Returns allow/deny with:
- Skeptic's risk assessment
- Oracle's base-rate analysis
- Arbiter's synthesized recommendation

## Response Format

All tools return structured JSON with:
- `verdict` — the synthesized conclusion
- `confidence` — calibrated 0.0-1.0 score
- `persona_results` — individual persona analyses
- `_meta` — token usage and cost estimates

## Tips

- Council makes 3-5 LLM calls per invocation — use `council_gate` for quick checks
- Set timeout to 180s in config (deliberation takes longer than single calls)
- Custom personas can be added via `~/.hermes-council/config.yaml`
- DPO preference pairs are included for RL training use cases
