# OpenThoughts-TBLite Evaluation Environment

This environment evaluates terminal agents on the [OpenThoughts-TBLite](https://huggingface.co/datasets/open-thoughts/OpenThoughts-TBLite) benchmark, a difficulty-calibrated subset of [Terminal-Bench 2.0](https://www.tbench.ai/leaderboard/terminal-bench/2.0).

## Source

OpenThoughts-TBLite was created by the [OpenThoughts](https://www.openthoughts.ai/) Agent team in collaboration with [Snorkel AI](https://snorkel.ai/) and [Bespoke Labs](https://bespokelabs.ai/). The original dataset and documentation live at:

- **Dataset (source):** [open-thoughts/OpenThoughts-TBLite](https://huggingface.co/datasets/open-thoughts/OpenThoughts-TBLite)
- **GitHub:** [open-thoughts/OpenThoughts-TBLite](https://github.com/open-thoughts/OpenThoughts-TBLite)
- **Blog post:** [openthoughts.ai/blog/openthoughts-tblite](https://www.openthoughts.ai/blog/openthoughts-tblite)

## Our Dataset

We converted the source into the same schema used by our Terminal-Bench 2.0 environment (pre-built Docker Hub images, base64-encoded test tarballs, etc.) and published it as:

- **Dataset (ours):** [NousResearch/openthoughts-tblite](https://huggingface.co/datasets/NousResearch/openthoughts-tblite)
- **Docker images:** `nousresearch/tblite-<task-name>:latest` on Docker Hub (100 images)

The conversion script is at `scripts/prepare_tblite_dataset.py`.

## Why TBLite?

Terminal-Bench 2.0 is one of the strongest frontier evaluations for terminal agents, but when a model scores near the floor (e.g., Qwen 3 8B at <1%), many changes look identical in aggregate score. TBLite addresses this by calibrating task difficulty using Claude Haiku 4.5 as a reference:

| Difficulty | Pass Rate Range | Tasks |
|------------|----------------|-------|
| Easy       | >= 70%         | 40    |
| Medium     | 40-69%         | 26    |
| Hard       | 10-39%         | 26    |
| Extreme    | < 10%          | 8     |

This gives enough solvable tasks to detect small improvements quickly, while preserving enough hard tasks to avoid saturation. The correlation between TBLite and TB2 scores is **r = 0.911**.

TBLite also runs 2.6-8x faster than the full TB2, making it practical for iteration loops.

## Usage

```bash
# Run the full benchmark
python environments/benchmarks/tblite/tblite_env.py evaluate

# Filter to specific tasks
python environments/benchmarks/tblite/tblite_env.py evaluate \
    --env.task_filter "broken-python,pandas-etl"

# Use a different model
python environments/benchmarks/tblite/tblite_env.py evaluate \
    --server.model_name "qwen/qwen3-30b"
```

## Architecture

`TBLiteEvalEnv` is a thin subclass of `TerminalBench2EvalEnv`. All evaluation logic (agent loop, Docker sandbox management, test verification, metrics) is inherited. Only the defaults differ:

| Setting        | TB2                              | TBLite                                  |
|----------------|----------------------------------|-----------------------------------------|
| Dataset        | `NousResearch/terminal-bench-2`  | `NousResearch/openthoughts-tblite`      |
| Tasks          | 89                               | 100                                     |
| Task timeout   | 1800s (30 min)                   | 1200s (20 min)                          |
| Wandb name     | `terminal-bench-2`               | `openthoughts-tblite`                   |

## Citation

```bibtex
@software{OpenThoughts-TBLite,
  author = {OpenThoughts-Agent team, Snorkel AI, Bespoke Labs},
  month = Feb,
  title = {{OpenThoughts-TBLite: A High-Signal Benchmark for Iterating on Terminal Agents}},
  howpublished = {https://www.openthoughts.ai/blog/openthoughts-tblite},
  year = {2026}
}
```
