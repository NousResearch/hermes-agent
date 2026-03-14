---
name: tinker
description: Tinker API by Thinking Machines Lab - train and fine-tune LLMs via API. CPU-side training loop, GPU compute handled remotely. Supports LoRA fine-tuning, RL, DPO, RLHF. Perfect for agents without local GPU access.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [training, fine-tuning, LoRA, RL, DPO, RLHF, GPU, API, compute]
    homepage: https://thinkingmachines.ai/tinker/
    docs: https://tinker-docs.thinkingmachines.ai/
---

# Tinker: Training API for LLM Fine-Tuning

Tinker lets you train LLMs without local GPUs. Your training script runs on CPU (here), while GPU compute happens on Tinker's infrastructure. This is ideal for Hermes Agent since the VM is CPU-only.

## Setup

1. Get API key from https://tinker-console.thinkingmachines.ai/
2. Set environment variable: TINKER_API_KEY
3. Install the tinker package (see https://tinker-docs.thinkingmachines.ai/install)

## Quick Start

\`\`\`python
import tinker

service = tinker.ServiceClient()
trainer = service.create_lora_training_client(
    base_model="meta-llama/Llama-3.1-8B",
    rank=32
)

trainer.forward_backward(data, "cross_entropy")
trainer.optim_step(tinker.AdamParams(learning_rate=1e-4))

checkpoint_path = trainer.save_state(name="step_100").result().path
sampler = trainer.save_weights_and_get_sampling_client(name="final")
result = sampler.sample(prompt, tinker.SamplingParams(max_tokens=100))
\`\`\`

## Architecture

\`\`\`
YOUR MACHINE (CPU)              TINKER SERVERS (GPU)
=================              =====================
Training loop logic      -->   Distributed compute
Data loading/prep        -->   forward_backward()
Loss function definition -->   optim_step()
Orchestration/monitoring -->   Checkpoint storage
\`\`\`

## Available Models

| Model | Type | Size |
|-------|------|------|
| Llama-3.1-8B | Base | Small |
| Llama-3.3-70B-Instruct | Instruct | Large |
| Qwen3-8B | Hybrid | Small |
| Qwen3-30B-A3B | MoE | Medium |
| Qwen3-235B-A22B | MoE | Large |
| Qwen3-VL-30B-A3B | Vision MoE | Medium |
| DeepSeek-V3.1 | MoE | Large |
| Kimi-K2-Thinking | Reasoning | Large |

**Cost tip:** MoE models are more cost-effective (pay for active params).

## Core API

### ServiceClient

\`\`\`python
import tinker
service = tinker.ServiceClient()

# List models
for m in service.get_server_capabilities().supported_models:
    print(m.model_name)

# Create training client
trainer = service.create_lora_training_client(
    base_model="meta-llama/Llama-3.1-8B",
    rank=32
)
\`\`\`

### TrainingClient

\`\`\`python
# Forward + backward (accumulates gradients)
future = trainer.forward_backward(data, loss_fn="cross_entropy")
result = future.result()

# Optimizer step
future = trainer.optim_step(tinker.AdamParams(learning_rate=1e-4))

# Checkpointing
path = trainer.save_weights_for_sampler(name="step_100").result().path
full_path = trainer.save_state(name="step_100").result().path

# Resume
trainer.load_state("tinker://<model_id>/step_100")
\`\`\`

## Loss Functions

| Loss FN | Use Case | Inputs |
|---------|----------|--------|
| cross_entropy | Supervised Learning | target_tokens, weights |
| importance_sampling | Policy Gradient | target_tokens, logprobs, advantages |
| ppo | PPO RL | same + config |
| cispo | Clipped IS | same |
| dro | Direct Reward Opt | same + beta |

## Training Patterns

### Supervised Learning

\`\`\`python
service = tinker.ServiceClient()
trainer = service.create_lora_training_client("meta-llama/Llama-3.1-8B")

for step in range(100):
    trainer.forward_backward(batch, "cross_entropy").result()
    trainer.optim_step(types.AdamParams(learning_rate=1e-4)).result()
\`\`\`

### RL Training

\`\`\`python
for iteration in range(15):
    sampler = trainer.save_weights_and_get_sampling_client(f"iter_{iteration}")
    samples = sampler.sample(prompts, params).result()
    
    rewards = compute_rewards(samples, ground_truth)
    advantages = compute_advantages(rewards)
    
    trainer.forward_backward(
        [make_datum(s, a) for s, a in zip(samples, advantages)],
        loss_fn="ppo"
    ).result()
\`\`\`

## Checkpointing

\`\`\`python
# Weights only (fast)
path = trainer.save_weights_for_sampler(name="v1").result().path

# Full state (for resuming)
path = trainer.save_state(name="v1_full").result().path

# Resume
trainer.load_state("tinker://<model_id>/v1_full")
\`\`\`

## Session Duration Strategy

Training runs take hours. Use these strategies:

### 1. Cronjob Monitoring

\`\`\`python
schedule_cronjob(
    prompt="Check training log at ~/training.log. Report step, loss, errors.",
    schedule="every 30m",
    name="training-monitor"
)
\`\`\`

### 2. Checkpoint-Based Resumption

\`\`\`python
for step in range(total_steps):
    train_step(...)
    
    if step % 50 == 0:
        path = trainer.save_state(f"step_{step}").result().path
        with open("training_state.txt", "w") as f:
            f.write(f"{path}\n{step}\n")
\`\`\`

## Autonomous Training Loop

For truly autonomous training where Hermes stays "in session" and makes decisions throughout the run, use the **Self-Chaining Session Pattern**.

### The Problem

Training runs take hours/days. Hermes sessions have duration limits. Cronjobs check periodically but can't make mid-run decisions like:
- Adjust learning rate when loss plateaus
- Stop early if diverging
- Run evaluation at checkpoints and decide whether to continue

### The Solution: Self-Chaining Hermes Sessions

Each Hermes session runs ONE training chunk, evaluates, decides, and spawns the next session.

\`\`\`
Session 1: Train 100 steps -> Checkpoint -> Evaluate -> Decide: continue
    |
    v spawns
Session 2: Resume from checkpoint -> Train 100 steps -> Evaluate -> Decide: reduce LR
    |
    v spawns
Session 3: Resume with lower LR -> Train 100 steps -> Evaluate -> Decide: stop
\`\`\`

### Implementation

Create \`train_chunk.py\` - a self-contained training script:

\`\`\`python
#!/usr/bin/env python3
import argparse
import json
import tinker
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    service = tinker.ServiceClient()
    trainer = service.create_lora_training_client("meta-llama/Llama-3.1-8B")
    
    if args.checkpoint:
        trainer.load_state(args.checkpoint)
    
    losses = []
    for step in range(args.steps):
        result = trainer.forward_backward(data, "cross_entropy").result()
        trainer.optim_step(tinker.AdamParams(learning_rate=args.lr)).result()
        losses.append(result.loss)
    
    checkpoint_path = trainer.save_state(f"step_{args.steps}").result().path
    
    results = {
        "checkpoint_path": checkpoint_path,
        "final_loss": losses[-1],
        "loss_trend": losses[-10:],
        "lr": args.lr,
    }
    
    Path("training_run/chunk_results.json").write_text(json.dumps(results))

if __name__ == "__main__":
    main()
\`\`\`

Then Hermes runs this and decides whether to spawn the next chunk:

\`\`\`python
# 1. Run training chunk
terminal(command="python train_chunk.py --steps 100")

# 2. Read decision
decision = json.loads(Path("training_run/chunk_results.json").read_text())

# 3. Decide whether to continue
if decision["final_loss"] > target:
    # Spawn next Hermes instance
    terminal(command=f"hermes chat -q 'Continue training with checkpoint {decision[\"checkpoint_path\"]}'", background=True)
\`\`\`

## Summary: Three Approaches

| Approach | Best For | Complexity | Autonomy Level |
|----------|----------|------------|----------------|
| Cronjob Monitoring | Set-and-forget, no mid-run decisions | Low | Low |
| Event-Driven Daemon | Real-time control | Medium | Medium |
| Self-Chaining Sessions | Full autonomy, Hermes decides | Medium | High |

The self-chaining pattern is the most autonomous - Hermes runs a chunk, evaluates, decides, and continues without human input.
HERMES_EOF; __hermes_rc=$?; printf '__HERMES_FENCE_a9f7b3__'; exit $__hermes_rc