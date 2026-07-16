---
name: haee-train
description: Train a model on HAEE evaluation data using slime.
version: 0.1.0
platforms: [linux]
author: Hermes
license: MIT
metadata:
  hermes:
    tags: [Training, RL, GRPO, Fine-tuning]
    category: mlops
---

# HAEE Training Bridge

Use when you want to train a model on agent evaluation data — closing the loop between agent usage and model improvement. This skill guides the full cycle: export data → train → evaluate → repeat.

## When to Use

- You have accumulated agent sessions with evaluation scores
- You want to fine-tune a model on your own agent's successes and failures
- You want to run a continuous improvement cycle (evaluate → train → re-evaluate)

## Prerequisites

- A training framework installed (slime, miles, axolotl, or unsloth)
- GPU with sufficient VRAM for your model size
- Agent evaluation data (from `hermes training-data status`)

## How to Run

1. Check available data: `hermes training-data status`
2. Export for training: `hermes training-data export --format sharegpt --min-score 0.7`
3. Configure your training framework with the exported data
4. Run training
5. Evaluate the trained model against the same tasks
6. Compare scores — repeat if improvement is below target

## Procedure

1. **Export data**
   - `hermes training-data status` — see what's available
   - `hermes training-data export --min-score 0.7` — high-quality examples only
   - Export in the format your training framework expects (sharegpt, alpaca, parquet)

2. **Train**
   - Configure your framework with the exported JSONL as the dataset
   - For slime: `slime train --data <export_path> --base_model <model>`
   - For axolotl: add the exported JSONL path to your config's dataset section

3. **Evaluate**
   - Run the trained model against your evaluation tasks
   - Compare scores: `hermes evolution benchmark`
   - Export again to see if improvement is measurable

4. **Repeat**
   - Each cycle generates more training data
   - Higher scores → better training examples
   - The flywheel compounds over time

## Frameworks

| Framework | Install | Use With |
|-----------|---------|----------|
| slime | `pip install slime` | GRPO/PPO training |
| miles | `pip install miles` | Enterprise MoE training |
| axolotl | `pip install axolotl` | General fine-tuning |
| unsloth | `pip install unsloth` | Memory-efficient training |

## Pitfalls

- Training on low-score examples reinforces bad behavior. Use `--min-score 0.7` minimum.
- Each framework has different dataset format requirements. Check which format you need.
- Small datasets (<100 examples) may not produce meaningful improvement. Keep collecting sessions.

## Verification

Run: `hermes training-data status` — should show records available for export.
