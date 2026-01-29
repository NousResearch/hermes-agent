#!/bin/bash

# Mixed browser+terminal data generation run
# Uses mixed-browser-terminal-tasks.jsonl (200 tasks)
# Distribution: browser 92%, terminal 92%, web 35%, vision 15%, image_gen 15%

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
LOG_FILE="logs/mixed_tasks_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Logging output to: $LOG_FILE"
echo "🔀 Running mixed browser+terminal tasks with mixed_tasks distribution"

# Set terminal environment
# SIF images are automatically built/cached by terminal_tool.py
export TERMINAL_ENV=singularity
export TERMINAL_SINGULARITY_IMAGE="docker://nikolaik/python-nodejs:python3.11-nodejs20"
export TERMINAL_TIMEOUT=300

# Set up Apptainer cache directories (use /scratch if available, otherwise /tmp)
if [ -d "/scratch" ] && [ -w "/scratch" ]; then
    CACHE_BASE="/scratch/$USER/.apptainer"
else
    CACHE_BASE="/tmp/$USER/.apptainer"
fi
export APPTAINER_CACHEDIR="$CACHE_BASE"
export APPTAINER_TMPDIR="$CACHE_BASE/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "📁 Apptainer cache: $APPTAINER_CACHEDIR"

python batch_runner.py \
  --dataset_file="mixed-browser-terminal-tasks.jsonl" \
  --batch_size=20 \
  --run_name="mixed_tasks" \
  --distribution="mixed_tasks" \
  --model="moonshotai/kimi-k2.5" \
  --base_url="https://openrouter.ai/api/v1" \
  --num_workers=25 \
  --max_turns=60 \
  --ephemeral_system_prompt="You are an AI assistant capable of both browser automation and terminal operations. Use browser tools to navigate websites, interact with web pages, fill forms, and extract information. Use terminal tools to execute commands, write and run code, install packages (use --break-system-packages with pip if needed), and perform local computations. When web search is available, use it to find URLs, documentation, or current information. If vision is available, use it to analyze images or screenshots. If image generation is available, use it when the task requires creating images. Combine browser and terminal capabilities effectively - for example, you might use the browser to fetch data from a website and terminal to process or analyze it. Always verify your work and handle errors gracefully. Whenever you can do something in a terminal instead of a web browser, you should choose to do so, as it's much cheaper." \
  2>&1 | tee "$LOG_FILE"

echo "✅ Log saved to: $LOG_FILE"
