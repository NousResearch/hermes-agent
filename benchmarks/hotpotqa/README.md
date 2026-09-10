# HotpotQA Benchmark

Evaluates Hermes cognitive memory on multi-hop question answering using the
HotpotQA dataset (Yang et al., 2018).

## Citation

```
Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., &
Manning, C. D. (2018). HotpotQA: A Dataset for Diverse, Explainable
Multi-hop Question Answering. EMNLP 2018.
https://arxiv.org/abs/1809.09600
```

## Dataset

HotpotQA contains **113 000 QA pairs** collected from Wikipedia.  Each
question requires chaining facts from at least two paragraphs (multi-hop
reasoning) and comes with gold *supporting facts* — the specific paragraphs
and sentences that are needed to answer it.

We evaluate on the **distractor split** of the validation set.  In this split,
each question is paired with 10 candidate paragraphs: 2 gold supporting
paragraphs and 8 distractors drawn from TF-IDF-similar Wikipedia articles.
The model must both retrieve the right paragraphs and combine the information
to produce the answer.

We draw a **stratified sample of 500 questions** (configurable via `--sample`)
balanced across:

- **Question type** (2 classes): bridge, comparison
- **Difficulty** (3 levels): easy, medium, hard

## Question Types

### Bridge (multi-hop chain)

The answer to the question can only be found by "bridging" two pieces of
information across different paragraphs.  For example:

> "What film was directed by the founder of Pixar?"

Answering this requires first finding who founded Pixar, then finding which
film that person directed.  Neither paragraph alone is sufficient.

### Comparison

The question asks to compare a property of two entities, both of which are
described in separate paragraphs.  For example:

> "Were Scott Derrickson and Ed Wood from the same country?"

Answering this requires retrieving the nationality of each person from their
respective paragraphs and then performing a comparison.

## Supporting Facts

Each HotpotQA question includes a list of *supporting facts* — pairs of
`(paragraph title, sentence index)` that pinpoint the exact evidence needed.

The `supporting_facts_recall` metric measures whether the memory store
retrieved these gold paragraphs.  Specifically:

```
supporting_facts_recall = |gold_titles ∩ retrieved_titles| / |gold_titles|
```

where `gold_titles` is the set of paragraph titles with at least one
supporting fact and `retrieved_titles` is the set of titles present in the
top-K recalled memories.

A high `supporting_facts_recall` means the retrieval layer is surfacing the
right evidence, even if the final answer text is not perfectly formed.

## Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Binary judge verdict: did the retrieved context contain enough information to produce the gold answer? |
| **Exact Match (EM)** | Normalized exact match between predicted answer token and gold answer (Rajpurkar et al., 2016). |
| **Token F1** | Token-level F1 between predicted and gold answer. The primary HotpotQA answer metric. |
| **Supporting Facts Recall** | Fraction of gold supporting-fact paragraphs found in top-K retrieved memories. The key multi-hop retrieval metric. |
| **Recall\@K** | Fraction of gold sentences found at retrieval rank K (K=1,3,5,10). |
| **MRR** | Mean Reciprocal Rank of first relevant retrieved item. |
| **NDCG\@K** | Normalized Discounted Cumulative Gain at rank K. |

## How it Works

For each question the adapter:

1. Creates a fresh, empty memory store (no cross-question contamination).
2. Ingests all 10 distractor context paragraphs sentence-by-sentence.
   Each sentence is stored as `[Title] sentence text` so the paragraph
   title is recoverable from retrieved memories.
3. Issues a recall query for the question text with `top_k=10`.
4. Judges the concatenated recall against the gold answer.
5. Computes `supporting_facts_recall` by checking whether gold paragraph
   titles appear in the retrieved set.
6. Computes the full metric suite (`benchmarks.metrics.compute_metric_suite`).

## Usage

Quick smoke test (5 questions, heuristic judge):

```bash
python -m benchmarks.hotpotqa.runner --sample 5 --verbose
```

Full 500-question evaluation:

```bash
python -m benchmarks.hotpotqa.runner
```

With an LLM judge (more accurate verdicts):

```bash
python -m benchmarks.hotpotqa.runner --judge-model claude-haiku-4-5
```

Only bridge questions:

```bash
python -m benchmarks.hotpotqa.runner --question-type bridge
```

Only easy questions:

```bash
python -m benchmarks.hotpotqa.runner --difficulty easy
```

Load from local JSON (official HotpotQA distractor dev file):

```bash
python -m benchmarks.hotpotqa.runner \
    --local /path/to/hotpot_dev_distractor_v1.json \
    --sample 500
```

Output JSON to stdout:

```bash
python -m benchmarks.hotpotqa.runner --sample 50 --json
```

Custom HuggingFace cache directory:

```bash
python -m benchmarks.hotpotqa.runner \
    --hf-cache /data/hf_cache \
    --sample 500
```

Results are saved to `benchmarks/results/hotpotqa.json` (override with
`--output`).

## Output Format

The JSON results file contains:

```json
{
  "benchmark": "hotpotqa",
  "total": 500,
  "correct": 123,
  "score": 0.246,
  "avg_supporting_facts_recall": 0.612,
  "avg_token_f1": 0.301,
  "avg_exact_match": 0.112,
  "by_type": {
    "bridge":     {"total": 350, "score": 0.24, "avg_supporting_facts_recall": 0.60, ...},
    "comparison": {"total": 150, "score": 0.27, "avg_supporting_facts_recall": 0.65, ...}
  },
  "by_difficulty": {
    "easy":   {"total": 180, "score": 0.31, ...},
    "medium": {"total": 220, "score": 0.22, ...},
    "hard":   {"total": 100, "score": 0.18, ...}
  },
  "results": [ ... per-question records ... ]
}
```

## Baseline Expectations

Multi-hop QA is hard.  A random baseline scores ~15% on HotpotQA.  Strong
retrieval-based systems without a generative model typically achieve:

- Supporting Facts Recall (paragraph-level F1): 0.75–0.85 for specialised models
- Token F1: 0.50–0.65 for joint retriever + reader models
- Exact Match: 0.40–0.55

For a pure memory-retrieval system (no generative step) the key signal is
`supporting_facts_recall`.  If the store reliably surfaces both gold paragraphs
in the top-10, the raw answer text will often partially match the gold, giving
reasonable Token F1 even without a dedicated reader model.
