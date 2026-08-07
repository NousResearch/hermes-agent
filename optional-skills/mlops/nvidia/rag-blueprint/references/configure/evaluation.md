# Evaluation

## When to Use
- User wants to measure RAG pipeline quality
- User asks about accuracy, relevancy, groundedness, or recall metrics

## Process
1. Read `docs/evaluate.md` for full evaluation methodology and setup
2. Choose the appropriate notebook based on metrics needed
3. Run evaluation against the deployed RAG pipeline

## Agent-Specific Notes
- Uses RAGAS framework for all metrics
- Answer Accuracy, Context Relevancy, and Groundedness are covered in one notebook
- Recall is measured separately at top-k cutoffs (1, 3, 5, 10)

## Notebooks
| Notebook | Metrics |
|----------|---------|
| `notebooks/evaluation_01_ragas.ipynb` | Answer Accuracy, Context Relevancy, Groundedness |
| `notebooks/evaluation_02_recall.ipynb` | Recall at top-k cutoffs |

## Source Documentation
- `docs/evaluate.md` -- full evaluation guide and metric definitions
- [RAGAS documentation](https://docs.ragas.io/en/stable/)
- [NVIDIA RAGAS metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/nvidia_metrics/)
