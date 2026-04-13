---
name: deep-research
description: Perform autonomous, multi-hop recursive research. Combines arxiv, Semantic Scholar, and red-teaming validation to build comprehensive, rigorous literature reviews.
version: 1.0.0
author: Hermes Agent
license: MIT
dependencies: []
metadata:
  hermes:
    tags: [Research, Arxiv, Academic, Science, Deep-Research]
    related_skills: [arxiv, ocr-and-documents]
---

# Deep Iterative Research

This skill allows Hermes to perform non-linear, multi-hop research on complex topics. Instead of fetching a single set of papers, it evaluates claims, recursively discovers citations via Semantic Scholar, and synthesizes findings while actively "red-teaming" (falsifying) its own claims.

## Usage

Ask Hermes to deeply research a topic:
- "Run a deep research loop on the impact of Synthetic Data on LLM loss curves."
- "Perform deep research on non-pharmacological interventions for hypertension and validate against conflicting studies."

## Mechanics

1. **Seed Discovery**: Queries arXiv or web searches for 3-5 foundational papers on the topic.
2. **Citation Traversal**: Uses the Semantic Scholar API to locate the most influential papers that cited the seed papers.
3. **Relevance Scoring & Synthesis**: Analyzes the abstracts/claims to gather evidence.
4. **Red-Teaming**: Spawns an internal validation process using your Latticework mental models (specifically the *Hallucination Check* and *Context Window*) to find contradictory literature or logic flaws.
5. **Output**: A comprehensive markdown report saved to `~/Obsidian/Hermes/Research/` containing the synthesis, the source graph, and the debate/falsification logs.

## Helper Script

The underlying recursive engine is executed via Python:

```bash
python scripts/recursive_research.py "Your Topic" --max-depth 3 --output-dir "~/Obsidian/Hermes/Research"
```

*Note: The script bakes in rate limiting and exponential backoff to respect Semantic Scholar's 1 req/sec API limits.*

## Next Steps

After generating the deep research briefing, you can use the `audio-podcast` skill to convert the generated report into a commute-ready audio discussion:
```
Generate an audio podcast of the deep research I just ran.
```
