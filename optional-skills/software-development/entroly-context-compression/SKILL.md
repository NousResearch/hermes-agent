---
name: entroly-context-compression
description: "Compress LLM context locally before API calls. Reduces input tokens by 70-95% on large repos."
version: 0.1.0
author: juyterman1000
license: Apache-2.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Cost-Optimization, Context-Compression, LLM, Token-Reduction, Cache-Alignment]
    related_skills: [code-wiki, codebase-inspection]
---

# Entroly Context Compression Skill

Compress repo context before sending to LLM providers. Reduces input tokens by 70–95% on large repos using knapsack optimization, entropy scoring, and cache alignment. Includes WITNESS hallucination guard.

## When to Use

- Agent is re-reading the same large codebase on every request
- LLM API costs are high due to repetitive context
- Working with repos of 500+ files where context selection matters
- Want to verify that LLM answers are grounded in supplied evidence

Do NOT use for:
- Short prompts under 4K tokens
- Single-file lookups where context is already minimal
- Workflows requiring unmodified raw text sent to the API

## Prerequisites

- Python 3.9+ on PATH
- `pip install entroly`
- No API key required for setup/testing

## How to Run

### Proxy mode (recommended — transparent to all LLM calls)

```bash
entroly proxy --port 9377
```

Then set the LLM base URL to `http://localhost:9377/v1`. All requests are automatically compressed.

### Wrap mode (per-session)

```bash
entroly wrap hermes
```

### MCP server mode

```bash
entroly serve
```

Add as an MCP server in Hermes configuration.

## Verify Installation

```bash
entroly verify-claims
```

Runs a bounded local smoke test: package import, indexing, context optimization, exact recovery. Writes `.entroly_verification.json`. No API key required.

## How It Works

1. **Rank** — scores every repo file by relevance to the query (BM25 + entropy + dependency graph)
2. **Select** — packs optimal files under token budget (knapsack solver)
3. **Compress** — reduces noisy context; originals recoverable via CCR handles
4. **Cache-align** — stabilizes prefix bytes so provider cache discounts apply (Anthropic 90%, OpenAI 50%)
5. **Verify** — WITNESS checks the LLM reply against supplied evidence ($0, ~3ms)

## Expected Results

- 70–95% fewer input tokens on repos with 500+ files
- 100% accuracy retained (NeedleInAHaystack, BFCL benchmarks)
- WITNESS hallucination detection: 0.844 AUROC on HaluEval-QA

Small prompts and tiny repos may show little or no savings. Always measure on your own repo.

## Links

- GitHub: https://github.com/juyterman1000/entroly
- License: Apache-2.0
- Local-first, no outbound analytics by default
