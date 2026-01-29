#!/bin/bash

# Terminal-focused data generation run
# Uses nous-terminal-tasks.jsonl (597 tasks)
# Distribution: terminal 97%, web 15%, browser 10%, vision 8%, image_gen 3%

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
LOG_FILE="logs/terminal_tasks_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Logging output to: $LOG_FILE"
echo "💻 Running terminal-focused tasks with terminal_tasks distribution"

# Set terminal environment (Modal sandboxes recommended for safety)
export TERMINAL_ENV=modal
export TERMINAL_MODAL_IMAGE=nikolaik/python-nodejs:python3.11-nodejs20
export TERMINAL_TIMEOUT=300

python batch_runner.py \
  --dataset_file="nous-terminal-tasks.jsonl" \
  --batch_size=20 \
  --run_name="terminal_tasks" \
  --distribution="terminal_tasks" \
  --model="z-ai/glm-4.7" \
  --base_url="https://openrouter.ai/api/v1" \
  --providers_allowed="gmicloud,siliconflow,atlas-cloud,z-ai,novita" \
  --num_workers=40 \
  --max_turns=60 \
  --ephemeral_system_prompt="You have access to a terminal tool for executing commands and completing coding, system administration, and computing tasks. Use the terminal to write code, run scripts, install packages (use --break-system-packages with pip if needed), manipulate files, and verify your work. Always test and validate code you create. Do not use interactive tools like vim, nano, or python REPL. If git output is large, pipe to cat. When web search is available, use it to look up documentation, APIs, or best practices. If browser tools are available, use them for web interactions that require page manipulation. Do not use the terminal to communicate with the user - only your final response will be shown to them." \
  2>&1 | tee "$LOG_FILE"

echo "✅ Log saved to: $LOG_FILE"
