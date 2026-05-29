---
name: tinker-atropos
description: Bridge Hermes RL environments to Thinking Machines Tinker cloud training. Run Atropos RL environments on cloud GPUs without managing infrastructure.
version: 0.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [atropos, rl, training, tinker, cloud, gpu, reinforcement-learning]
    related_skills: [hermes-atropos-environments, axolotl, grpo-rl-training]
---

# Tinker-Atropos: Cloud RL Training for Hermes Environments

Train Hermes Agent RL environments on Thinking Machines Tinker cloud GPUs.
No infrastructure to manage — just an API key and your Atropos environment.

## When to Use

- You have an Atropos RL environment and want cloud GPU training
- You're building a new Hermes RL environment and want to validate at scale
- You want to avoid managing local GPU clusters for RL training runs
- The `hermes-atropos-environments` skill shows how to build environments —
  this skill shows how to train them on cloud hardware

## Architecture

```
Atropos Environment (your_env.py)
    |
    v
TinkerAtroposTrainer (cloud orchestration — this package)
    |
    v
Tinker API (Thinking Machines cloud GPUs)
    |
    v
LoRA-adapted weights saved to tinker:// paths
```

Any Atropos environment using `managed_server` is compatible. No modifications
needed to the environment code — just pass your Tinker config.

## Quick Start

```bash
# 1. Get an API key: https://tinker-console.thinkingmachines.ai/keys
export TINKER_API_KEY="<your-key>"

# 2. Install in editable mode (one-time)
pip install -e optional-skills/mlops/tinker-atropos/

# 3. Train with built-in GSM8k environment
python optional-skills/mlops/tinker-atropos/scripts/launch_training.py \
  --config optional-skills/mlops/tinker-atropos/templates/default.yaml

# 4. Download trained weights
# The trainer outputs a tinker:// path at end of training.
# Set TINKER_PATH in tinker_atropos/utils/download_weights.py and run:
python optional-skills/mlops/tinker-atropos/tinker_atropos/utils/download_weights.py
```

## Using Your Own Atropos Environment

Any Atropos environment works directly:

```bash
# Terminal 1: Start Atropos API
run-api

# Terminal 2: Start training
export TINKER_API_KEY="<your-key>"
python optional-skills/mlops/tinker-atropos/scripts/launch_training.py \
  --config templates/default.yaml

# Terminal 3: Serve your environment (point to your env + tinker config)
python /path/to/your_env.py serve --config templates/default.yaml
```

The environment uses:
- The `env` section for environment configuration
- The `openai` section for inference server configuration
- The `tinker` section for training parameters (ignored by the environment)

## Configuration

Configs follow the Atropos format with a `tinker` section for Tinker settings:

```yaml
env:
  group_size: 16
  batch_size: 128
  tokenizer_name: "meta-llama/Llama-3.1-8B-Instruct"

openai:
  - model_name: "meta-llama/Llama-3.1-8B-Instruct"
    base_url: "http://localhost:8001/v1"

tinker:
  lora_rank: 32
  learning_rate: 0.00004
  max_token_trainer_length: 2048
  wandb_project: "my-project"
```

See `references/configs.md` for all YAML options. Templates at `templates/`.

## Available Environments

| Environment | File | Description |
|------------|------|-------------|
| GSM8k Tinker | `tinker_atropos/environments/gsm8k_tinker.py` | Math reasoning with GSM8k dataset |

## Programmatic Usage

```python
from tinker_atropos.config import TinkerAtroposConfig
from tinker_atropos.trainer import TinkerAtroposTrainer

config = TinkerAtroposConfig.from_yaml("templates/default.yaml")
trainer = TinkerAtroposTrainer(config=config)
await trainer.run()
```

## Testing

```bash
cd optional-skills/mlops/tinker-atropos
python -m pytest tinker_atropos/tests/ -v
```

## Cost

The Tinker Rate Card and available models:
https://tinker-console.thinkingmachines.ai/rate-card

## Troubleshooting

### "tinker-atropos not installed" from hermes doctor

Known issue — the doctor's `pip list` check may miss editable installs.
The package is functional regardless. Verify with:

```bash
python -c "from tinker_atropos import config; print('OK')"
```

### Import errors after moving the package

Reinstall in editable mode from the new location:

```bash
pip uninstall tinker-atropos -y
pip install -e optional-skills/mlops/tinker-atropos/
```
