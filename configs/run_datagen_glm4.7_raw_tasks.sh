#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
LOG_FILE="logs/glm4.7-terminal-tasks_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Logging output to: $LOG_FILE"

python batch_runner.py \
  --dataset_file="source-data/raw_tasks_prompts.jsonl" \
  --batch_size=20 \
  --run_name="terminal-tasks-glm4.7-thinking" \
  --distribution="default" \
  --model="z-ai/glm-4.7" \
  --base_url="https://openrouter.ai/api/v1" \
  --providers_allowed="gmicloud,siliconflow,atlas-cloud,z-ai,novita" \
  --num_workers=50 \
  --max_turns=60 \
  --ephemeral_system_prompt="You have access to a variety of tools to help you complete coding, system administration, and general computing tasks. You can use them in sequence and build off of the results of prior tools you've used. Always use the terminal tool to execute commands, write code, install packages, and verify your work. You should test and validate everything you create. Always pip install any packages you need (use --break-system-packages if needed). If you need a tool that isn't available, you can use the terminal to install or create it. Do not use the terminal tool to communicate with the user, as they cannot see your commands, only your final response after completing the task. Use web search when you need to look up documentation, APIs, or current best practices." \
  2>&1 | tee "$LOG_FILE"

echo "✅ Log saved to: $LOG_FILE"

#  --verbose \
#  --resume \

