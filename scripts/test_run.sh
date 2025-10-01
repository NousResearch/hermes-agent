#!/bin/bash

# Check if a prompt argument was provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a prompt as an argument"
    echo "Usage: $0 \"your prompt here\""
    exit 1
fi

# Get the prompt from the first argument
PROMPT="$1"

# Set debug mode for web tools
export WEB_TOOLS_DEBUG=true

# Resolve repository root relative to this script and run from there
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Prefer local venv if present
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

# Run the agent with the provided prompt
python run_agent.py \
  --query "$PROMPT" \
  --max_turns 30 \
  --model claude-sonnet-4-20250514 \
  --base_url https://api.anthropic.com/v1/ \
  --api_key $ANTHROPIC_API_KEY \
  --save_trajectories \
  --enabled_toolsets=web
  
#  --model claude-sonnet-4-20250514 \
#  
#Possible Toolsets:
#web_tools
#vision_tools
#terminal_tools