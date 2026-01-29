#!/bin/bash

# Terminal-only evaluation run using Modal sandboxes
# Uses 10 sample tasks from nous-terminal-tasks

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
LOG_FILE="logs/terminal_eval_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Logging output to: $LOG_FILE"
echo "🔧 Using Modal sandboxes (TERMINAL_ENV=modal)"

# Set terminal to use Modal
export TERMINAL_ENV=modal
export TERMINAL_MODAL_IMAGE=nikolaik/python-nodejs:python3.11-nodejs20
export TERMINAL_TIMEOUT=300

python batch_runner.py \
  --dataset_file="nous-terminal-tasks_eval.jsonl" \
  --batch_size=5 \
  --run_name="terminal_eval" \
  --distribution="terminal_only" \
  --model="z-ai/glm-4.7" \
  --base_url="https://openrouter.ai/api/v1" \
  --providers_allowed="gmicloud,siliconflow,atlas-cloud,z-ai,novita" \
  --num_workers=2 \
  --max_turns=30 \
  --ephemeral_system_prompt="You have access to a terminal tool for executing commands. Use it to complete the task. Install any packages you need with apt-get or pip (use --break-system-packages if needed). Do not use interactive tools (vim, nano, python repl). If git output is large, pipe to cat." \
  2>&1 | tee "$LOG_FILE"

echo "✅ Log saved to: $LOG_FILE"
