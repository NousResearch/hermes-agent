# Autoresearch Examples

This directory contains example research projects using the autoresearch methodology.

## Available Examples

### `lora-rank-study.md`

**Question:** Does LoRA rank affect convergence speed on small datasets?

**Type:** Benchmark optimization, hyperparameter study

**Skills Used:**
- `arxiv` — Literature search
- `mlops` — Model training
- `tensorboard` — Experiment tracking

**Key Takeaway:** Higher rank improves convergence speed up to a point (r=16), then diminishing returns.

---

## Creating Your Own Research

1. Start with `/autoresearch "your question"`
2. Follow the two-loop architecture
3. Commit protocols before running
4. Generate progress reports with `/research-report`

## Tips from Examples

- **Start small:** First experiment should complete in <30 minutes
- **Define metrics upfront:** Know what you're measuring before you start
- **Document surprises:** Negative results are progress too
- **Show your work:** Progress reports help humans follow along
