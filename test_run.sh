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

# Run the agent with the provided prompt
python run_agent.py \
  --query "$PROMPT" \
  --max_turns 30 \
  --model claude-sonnet-4-5-20250929 \
  --base_url https://api.anthropic.com/v1/ \
  --api_key $ANTHROPIC_API_KEY \
  --save_trajectories