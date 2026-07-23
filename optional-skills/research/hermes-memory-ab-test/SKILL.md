---
name: memory-enhanced-retrieval
description: >-
  Entity-indexed two-stage retrieval for Hermes. Uses jieba + SQLite as a
  sidecar to build an entity index from conversation history, then performs
  two-stage keyword expansion across retrieval passes. Zero changes to Hermes
  core; removes cleanly.
version: 1.0.0
author: luxuguang-leo
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [memory, retrieval, entity-index, search, mragent]
    related_skills: []
---

# Memory Enhanced Retrieval

Improves `session_search` for multi-hop and cross-session queries using entity
indexing + two-stage retrieval. Based on A/B testing against MRAgent (ICML 2026)
on 10 real-world conversation scenarios.

**Key result**: 100% correct+partial (vs MRAgent 30%), 12s avg (vs 33s), zero
failures (vs MRAgent 50%).

Full report: https://github.com/luxuguang-leo/hermes-memory-ab-test

## How It Works

```
Write: conversation → jieba → extract entities → SQLite sidecar
                                                    ↓
Read:  question → jieba → index lookup → first pass → new entities → second pass → LLM answer
```

1. **Entity Index (sidecar)**: When storing conversations, jieba extracts named
   entities (persons, places, organizations, concepts) and writes them to an
   independent SQLite database.
2. **Stage 1**: On query, entity extraction from the question narrows the search
   scope via the index. Falls back to keyword matching when entity extraction
   yields no results.
3. **Stage 2**: New entities are extracted from stage 1's results and used for
   a second search pass over uncovered turns.
4. **Merge**: Results from both passes are merged with even sampling for
   coverage, then sent to the LLM for a single answer call.

## When to Use

- The user asks a question that requires connecting facts across multiple
  conversation turns
- The user asks a question spanning multiple past sessions
- `session_search` returns sparse or ambiguous results
- The user reports that the agent "couldn't find something they said earlier"

## Prerequisites

```bash
pip install jieba
```

## Installation

```bash
hermes skills install memory-enhanced-retrieval
```

## Rollback

```bash
hermes skills remove memory-enhanced-retrieval
rm -rf ~/.hermes/entity_index/
```

## A/B Test Results

Tested on 10 scenarios (multi-hop, temporal, cross-session) with Gemma4 12B:

| Metric | Proposed | MRAgent | Flat RAG |
|--------|---------|---------|----------|
| Correct | 70% | 20% | 30% |
| Correct+Partial | 100% | 30% | 40% |
| Avg time | 12s | 33s | 4s |
| LLM calls | 1 | 3 | 2 |
| Stable | 10/10 | 5/10 | 5/10 |

## References

- MRAgent: "Memory is Reconstructed, Not Retrieved" — ICML 2026, arXiv:2606.06036
- Hermes Agent: https://hermes-agent.nousresearch.com/
- A/B test repo: https://github.com/luxuguang-leo/hermes-memory-ab-test
