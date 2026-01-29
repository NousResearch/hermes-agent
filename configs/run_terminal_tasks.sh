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

# Set terminal environment (using Singularity for containerized execution)
export TERMINAL_ENV=singularity
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

# Pre-build SIF image if it doesn't exist (avoids 40 workers all downloading simultaneously)
SIF_IMAGE="$CACHE_BASE/python-nodejs-3.11-20.sif"
DOCKER_IMAGE="docker://nikolaik/python-nodejs:python3.11-nodejs20"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "🔨 Building Singularity image (one-time setup)..."
    echo "   Source: $DOCKER_IMAGE"
    echo "   Target: $SIF_IMAGE"
    apptainer build "$SIF_IMAGE" "$DOCKER_IMAGE"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to build SIF image. Falling back to docker:// URL"
        export TERMINAL_SINGULARITY_IMAGE="$DOCKER_IMAGE"
    else
        echo "✅ SIF image built successfully"
        export TERMINAL_SINGULARITY_IMAGE="$SIF_IMAGE"
    fi
else
    echo "✅ Using pre-built SIF image: $SIF_IMAGE"
    export TERMINAL_SINGULARITY_IMAGE="$SIF_IMAGE"
fi

echo "📁 Apptainer cache: $APPTAINER_CACHEDIR"

python batch_runner.py \
  --dataset_file="nous-terminal-tasks.jsonl" \
  --batch_size=5 \
  --run_name="terminal_tasks-kimi-k2.5" \
  --distribution="terminal_tasks" \
  --model="moonshotai/kimi-k2.5" \
  --verbose \
  --base_url="https://openrouter.ai/api/v1" \
  --num_workers=80 \
  --max_turns=60 \
  --providers_ignored="Novita" \
  --resume \
  --ephemeral_system_prompt="You have access to a terminal tool for executing commands and completing coding, system administration, and computing tasks. Use the terminal to write code, run scripts, install packages (use --break-system-packages with pip if needed), manipulate files, and verify your work. Always test and validate code you create. Do not use interactive tools like vim, nano, or python REPL. If git output is large, pipe to cat. When web search is available, use it to look up documentation, APIs, or best practices. If browser tools are available, use them for web interactions that require page manipulation. Do not use the terminal to communicate with the user - only your final response will be shown to them." \
  2>&1 | tee "$LOG_FILE"

echo "✅ Log saved to: $LOG_FILE"
