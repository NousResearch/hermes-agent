## Setup
```
pip install -r requirements.txt
git clone git@github.com:NousResearch/hecate.git
cd hecate
pip install -e .
```

## Run
```
python run_agent.py \
  --query "search up the latest docs on jit in python 3.13 and write me basic example that's not in their docs." \
  --max_turns 20 \
  --model claude-sonnet-4-20250514 \
  --base_url https://api.anthropic.com/v1/ \
  --api_key $ANTHROPIC_API_KEY
```
