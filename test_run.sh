python run_agent.py \
  --query "search up the latest docs on huggingface datasets in python 3.13 and write me basic example that's not in their docs. profile its performance" \
  --max_turns 30 \
  --model claude-sonnet-4-20250514 \
  --base_url https://api.anthropic.com/v1/ \
  --api_key $ANTHROPIC_API_KEY