#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
LOG_FILE="logs/glm4.7-thinking-sft1_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Logging output to: $LOG_FILE"

python batch_runner.py \
  --dataset_file="source-data/hermes-agent-agent-tasks-1/agent_tasks_sft_1.jsonl" \
  --batch_size=25 \
  --run_name="megascience_glm4.7-thinking-sft1" \
  --distribution="science" \
  --model="z-ai/glm-4.7" \
  --base_url="https://openrouter.ai/api/v1" \
  --providers_allowed="gmicloud,siliconflow,atlas-cloud,z-ai,novita" \
  --num_workers=10 \
  --max_turns=60 \
  --ephemeral_system_prompt="You have access to a variety of tools to help you solve scientific, math, and technology problems presented to you. You can use them in sequence and build off of the results of prior tools you've used results. Always use the terminal or search tool if it can provide additional context, verify formulas, double check concepts and recent studies and understanding, doing all calculations, etc. You should only be confident in your own reasoning, knowledge, or calculations if you've exhaustively used all tools available to you to that can help you verify or validate your work. Always pip install any packages you need to use the python scripts you want to run. If you need to use a tool that isn't available, you can use the terminal tool to install or create it in many cases as well. Do not use the terminal tool to communicate with the user, as they cannot see your commands, only your final response after completing the task. Search for at least 3 sources, but not more than 12." \
  2>&1 | tee "$LOG_FILE"

echo "✅ Log saved to: $LOG_FILE"

#  --verbose \