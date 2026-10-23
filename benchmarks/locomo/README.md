# LoCoMo Benchmark Adapter

Integrates the [LoCoMo](https://github.com/snap-research/LoCoMo) benchmark
(Snap Research) with the Hermes cognitive memory system.

## Citation

Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y.
(2024). **Evaluating Very Long-Term Conversational Memory of LLM Agents.**
arXiv:2402.17753.
https://arxiv.org/abs/2402.17753

```bibtex
@article{maharana2024locomo,
  title   = {Evaluating Very Long-Term Conversational Memory of LLM Agents},
  author=*** Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and
             Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal = {arXiv preprint arXiv:2402.17753},
  year    = {2024}
}
```

## Dataset

- **Source**: `snap-research/locomo` on HuggingFace (or local JSON)
- **Scale**: ~50 long conversations, ~2,000 QA pairs
- **Conversation length**: each conversation spans many sessions, often
  hundreds of turns, making it a genuinely long-context memory benchmark.

### Question Types

| Type | Description |
|------|-------------|
| `single_hop` | Answer lies in a single conversation turn |
| `multi_hop` | Answer requires connecting facts from multiple turns |
| `temporal` | Requires reasoning about time, order, or recency |
| `open_domain` | Open-ended questions without a strict gold answer |

## How it works

For each question in the dataset:

1. A fresh `CognitiveBenchmarkAdapter` (memory store) is created.
2. All messages from the question's conversation are ingested as factual
   memories:
   - User messages: importance `0.6` (likely targets of retrieval)
   - Assistant messages: importance `0.4` (useful context)
   - `simulate_time(0.01)` is called between messages for temporal ordering.
3. The question is answered by recalling the top-K most relevant memories.
4. The context (top-5 recalled memories joined with ` | `) is judged against
   the gold answer using a `HeuristicJudge` or `MemoryJudge`.
5. Full retrieval metrics (Recall@K, MRR, NDCG, MAP, Token F1) are computed
   per question and averaged across the dataset.

Each question is fully isolated — no cross-contamination between questions.

## Usage

```bash
cd /workspace/Projects/hermes-agent

# Quick smoke test (5 questions, heuristic judge, no API calls)
python -m benchmarks.locomo.runner --sample 5

# Full dataset run
python -m benchmarks.locomo.runner

# Only temporal questions
python -m benchmarks.locomo.runner --question-type temporal

# With LLM judge (requires Anthropic API key)
python -m benchmarks.locomo.runner --judge-model claude-haiku-4-5

# Load from a local JSON file (if HF dataset is unavailable)
python -m benchmarks.locomo.runner --local /path/to/locomo.json

# JSON output for programmatic consumption
python -m benchmarks.locomo.runner --json --output benchmarks/results/locomo.json

# Verbose per-question output
python -m benchmarks.locomo.runner --sample 20 --verbose
```

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--sample N` | all | Run only the first N questions |
| `--judge-model` | `heuristic` | `heuristic` or Claude model (e.g. `claude-haiku-4-5`) |
| `--embedding` | `auto` | Embedding model: `auto`, `sentence-transformers`, `tfidf` |
| `--profile` | `balanced` | Cognitive memory profile |
| `--question-type` | all | Filter by type: `single_hop`, `multi_hop`, `temporal`, `open_domain` |
| `--local PATH` | HuggingFace | Load from local JSON instead of HF |
| `--hf-cache PATH` | `/workspace/Projects/.huggingface_cache` | HF cache dir |
| `--output PATH` | `benchmarks/results/locomo.json` | Output file |
| `--json` | off | Print results as JSON to stdout |
| `--verbose` / `-v` | off | Print per-question results |
| `--top-k N` | `10` | Memories to recall per question |

## Expected Baselines

From Table 1 of the LoCoMo paper and the Mem0 benchmark paper:

| System | Overall | single_hop | multi_hop | temporal | open_domain |
|--------|---------|-----------|-----------|----------|-------------|
| GPT-4 (full context) | ~0.83 | ~0.91 | ~0.78 | ~0.79 | ~0.72 |
| GPT-3.5 (full context) | ~0.68 | ~0.77 | ~0.61 | ~0.63 | ~0.58 |
| Mem0 (retrieval-aug) | ~0.26 | ~0.35 | ~0.21 | ~0.22 | ~0.18 |
| Naive RAG | ~0.19 | ~0.27 | ~0.14 | ~0.16 | ~0.13 |

> Note: exact numbers vary by judge type and evaluation protocol.
> The LoCoMo paper primarily measures answer quality using GPT-4 as judge.
> The Hermes heuristic judge uses keyword overlap rather than LLM evaluation,
> so scores are not directly comparable to LLM-judged baselines.

## Local JSON Format

If you have a local copy of the dataset, it should be a JSON file
containing a list of question dicts (or a dict with a `"questions"` key):

```json
[
  {
    "question_id": "q001",
    "question_type": "single_hop",
    "question": "What is Alice's favorite coffee shop?",
    "answer": "Blue Bottle Coffee",
    "conversation": [
      {"role": "user",      "content": "I just discovered Blue Bottle Coffee!"},
      {"role": "assistant", "content": "Blue Bottle is great. What did you order?"},
      {"role": "user",      "content": "A pour-over. It was amazing."}
    ]
  }
]
```

The adapter also understands alternate field names (`query`, `gold_answer`,
`dialog`, `history`, `context`) for compatibility with different dataset
exports.

## Limitations

- **Ingestion granularity**: Each message turn is stored as a single memory.
  Long turns with multiple facts may benefit from sentence-level chunking
  (not yet implemented).
- **Open-domain scoring**: Open-domain questions have no strict gold answer,
  so keyword-based scoring underestimates true performance.  Use an LLM judge
  (`--judge-model claude-haiku-4-5`) for more meaningful open-domain scores.
- **HuggingFace availability**: The `snap-research/locomo` dataset may not be
  publicly available on HuggingFace at all times.  Use `--local` to load from
  a downloaded copy.
