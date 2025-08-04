export WEB_TOOLS_DEBUG=true

python run_agent.py \
  --query "Tell me about this animal pictured: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQi1nkrYXY-ijQv5aCxkwooyg2roNFxj0ewJA&s" \
  --max_turns 30 \
  --model claude-sonnet-4-20250514 \
  --base_url https://api.anthropic.com/v1/ \
  --api_key $ANTHROPIC_API_KEY \
  --enabled_toolsets=vision_tools

#Possible Toolsets:
#web_tools
#vision_tools
#terminal_tools