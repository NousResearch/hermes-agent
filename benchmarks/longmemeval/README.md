# LongMemEval Benchmark Adapter

Integrates the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) external benchmark
(ICLR 2025) with the Hermes cognitive memory system.

## Dataset

- **Source**: `xiaowu0162/longmemeval-cleaned` on HuggingFace
- **Split used**: `longmemeval_oracle` (500 questions, ideal haystack)
- **Question types**:
  - `temporal-reasoning` (133) — reasoning about time/sequence across sessions
  - `multi-session` (133) — facts spread across multiple sessions
  - `knowledge-update` (78) — updated/contradicted information
  - `single-session-user` (70) — single session, user-provided facts
  - `single-session-assistant` (56) — single session, assistant-provided facts
  - `single-session-preference` (30) — personal preferences

## How it works

1. For each question, ingest all haystack sessions into a fresh CognitiveMemoryStore
2. Recall against the question query
3. Use HeuristicJudge to score the answer (or LLM judge if available)
4. Report accuracy by question type

## Usage

```bash
cd /workspace/Projects/hermes-agent

# Quick smoke test (5 questions)
python -m benchmarks.longmemeval.runner --sample 5

# Full 500 question run (takes ~10-20 min)
python -m benchmarks.longmemeval.runner

# With LLM judge
python -m benchmarks.longmemeval.runner --judge-model claude-haiku-4-5

# JSON output
python -m benchmarks.longmemeval.runner --json --output benchmarks/results/longmemeval.json
```

## Limitations

- **Ingestion strategy**: All session messages are stored as factual memories.
  The cognitive store is optimised for atomic facts, not full conversation replay.
  Multi-turn conversational context may be partially lost.
- **Abstention**: The benchmark has no "abstain" questions; we never abstain.
- **Large sessions**: Some questions have 10+ sessions with 50+ messages each.
  Performance degrades at scale due to TF-IDF vocabulary growth.
