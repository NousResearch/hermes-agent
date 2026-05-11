#!/bin/bash
# Run SpotForagingEnv in Atropos process-mode (data collection, no trainer).
# Writes groups to /tmp/spot_foraging_test_N.jsonl (auto-incremented).
#
# Usage: bash scripts/run_atropos_process.sh [total_steps] [group_size]
#
set -a
source ~/.config/litellm/.env 2>/dev/null
source ~/.hermes/.env 2>/dev/null
set +a

TOTAL_STEPS="${1:-2}"
GROUP_SIZE="${2:-2}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export ATROPOS_ALLOW_DUMMY_MANAGED_SERVER=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/nix/store/xm08aqdd7pxcdhm0ak6aqb1v7hw5q6ri-gcc-14.3.0-lib/lib
export PYTHONPATH=/home/olive/Repositories/skypilot-env/spot/rapier-gym/python:/home/olive/Repositories/hermes-agent
cd /home/olive/Repositories/hermes-agent

exec /home/olive/Repositories/hermes-agent/.venv-atropos/bin/python \
  -m environments.spot_foraging_env.env process \
  --env.total_steps "$TOTAL_STEPS" \
  --env.group_size "$GROUP_SIZE" \
  --env.data_path_to_save_groups /tmp/spot_foraging_test.jsonl \
  --env.fall_terminates false \
  --env.ensure_scores_are_not_same false \
  --env.use_wandb false \
  --openai.base_url "https://openrouter.ai/api/v1" \
  --openai.api_key "$OPENROUTER_API_KEY" \
  --openai.model_name "google/gemini-2.5-flash" \
  --openai.server_type openai \
  --openai.health_check false \
  --openai.timeout 60 \
  --env.tokenizer_name gpt2 \
  --env.enable_image false \
  2>&1
