---
name: ml-evaluation
description: "ML evaluation and experiment tracking: W&B (logging, sweeps, registry) and lm-eval-harness (benchmarking)."
version: 1.0.0
author: Orchestra Research
license: MIT
dependencies: [wandb, lm-eval]
metadata:
  hermes:
    tags: [MLOps, Evaluation, Experiment Tracking, Benchmarking, Hyperparameter Tuning, Model Registry]

---

# ML Evaluation & Experiment Tracking

This skill provides comprehensive tools for evaluating ML models and tracking experiments. It combines two specialized tools:

## Tools Overview

| Tool | Purpose | Best For |
|------|---------|----------|
| **Weights & Biases (W&B)** | Experiment tracking, logging, sweeps | Real-time monitoring, hyperparameter optimization, team collaboration |
| **lm-evaluation-harness** | LLM benchmarking | Academic benchmarks (MMLU, GSM8K, HumanEval), model comparison |

## Weights & Biases (W&B)

Full-featured MLOps platform for experiment tracking and collaboration.

### When to Use W&B

- **Log experiments**: Automatic metric tracking, visualizations, run comparison
- **Hyperparameter sweeps**: Bayesian optimization, grid search, random search
- **Model registry**: Versioned model storage with lineage tracking
- **Team collaboration**: Shared workspaces, reports, dashboards

### Quick Start

```python
import wandb

wandb.init(project="my-project", config={
    "learning_rate": 0.001,
    "epochs": 10
})

# Training loop
for epoch in range(10):
    train_loss = train()
    wandb.log({"train/loss": train_loss})
```

See [references/weights-and-biases.md](references/weights-and-biases.md) for detailed workflows.

## lm-evaluation-harness

Industry-standard LLM benchmarking toolkit.

### When to Use lm-eval-harness

- **Benchmark model quality**: Standardized academic benchmarks
- **Compare models**: Reproducible evaluation across models
- **Track training progress**: Checkpoint evaluation during training
- **Academic papers**: Standardized metrics for reproducibility

### Quick Start

```bash
pip install lm-eval

lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu,gsm8k,hellaswag \
  --device cuda:0
```

See [references/lm-evaluation-harness.md](references/lm-evaluation-harness.md) for detailed workflows.

## Common Workflows

### 1. Track Training + Evaluate Model Quality

Combine W&B logging with lm-eval benchmarks:

```python
import wandb
import subprocess

wandb.init(project="llm-training")

for checkpoint in checkpoints:
    # Log training metrics
    wandb.log({"step": step, "loss": loss})
    
    # Evaluate with lm-eval
    result = subprocess.run([
        "lm_eval", "--model", "hf",
        "--model_args", f"pretrained={checkpoint}",
        "--tasks", "mmlu,gsm8k"
    ])
    
    # Log evaluation results to W&B
    wandb.log({"mmlu": result.mmlu, "gsm8k": result.gsm8k})
```

### 2. Experiment Tracking + Model Registry

```python
import wandb

# Log experiment
wandb.init(project="experiments", config={
    "model": "Llama-2-7b",
    "learning_rate": 0.001
})

# Log metrics
wandb.log({"accuracy": 0.92})

# Save to model registry
artifact = wandb.Artifact("llama2-7b", type="model")
artifact.add_file("model.pth")
wandb.log_artifact(artifact, aliases=["best"])
```

### 3. Sweep + Benchmark

```python
import wandb

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val/accuracy', 'goal': 'maximize'},
    'parameters': {
        'lr': {'min': 1e-5, 'max': 1e-1},
        'batch_size': {'values': [16, 32, 64]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="sweeps")

def train():
    run = wandb.init()
    # Train model
    # Log metrics
    wandb.log({"val/accuracy": accuracy})

wandb.agent(sweep_id, function=train)
```

## Choosing the Right Tool

| Scenario | Tool |
|----------|------|
| Track training metrics | W&B |
| Hyperparameter optimization | W&B |
| Model registry | W&B |
| Team collaboration | W&B |
| LLM benchmark (MMLU, GSM8K) | lm-eval-harness |
| Model comparison | lm-eval-harness |
| Academic evaluation | lm-eval-harness |

## Resources

- **W&B Documentation**: https://docs.wandb.ai
- **lm-eval-harness GitHub**: https://github.com/EleutherAI/lm-evaluation-harness
- **W&B GitHub**: https://github.com/wandb/wandb