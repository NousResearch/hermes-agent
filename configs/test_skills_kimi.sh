#!/bin/bash

# Test skills tool with Kimi K2.5
# Usage: ./configs/test_skills_kimi.sh "your query here"
# Example: ./configs/test_skills_kimi.sh "List available skills and show me the vllm skill"

# Default query if none provided
QUERY="${1:-List all available skills. Then show me the axolotl skill and view one of its reference files.}"

echo "🎯 Testing Skills Tool with Kimi K2.5"
echo "📝 Query: $QUERY"
echo "=" 

python run_agent.py \
  --enabled_toolsets=skills \
  --model="moonshotai/kimi-k2.5" \
  --base_url="https://openrouter.ai/api/v1" \
  --max_turns=10 \
  --verbose \
  --save_sample \
  --query="$QUERY"
