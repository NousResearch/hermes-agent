# Usage Patterns — Testing Environments and Evaluating Models

## Pattern 1: Test Your Environment Works (process mode)

Use `process` mode to verify your environment runs end-to-end before
committing. This generates trajectories without needing an Atropos
training server.

**Before running:** Ask the user for their inference setup (see SKILL.md "Inference Setup" section). Replace `<BASE_URL>`, `<MODEL>`, and `<SERVER_TYPE>` below with their chosen values.

### Step 1: Run 1 trajectory

```bash
cd ~/.hermes/hermes-agent
source .venv/bin/activate

python environments/your_env.py process \
  --env.total_steps 1 \
  --env.group_size 1 \
  --env.use_wandb false \
  --env.data_path_to_save_groups /tmp/test_output.jsonl \
  --openai.base_url "<BASE_URL>" \
  --openai.model_name "<MODEL>" \
  --openai.server_type <SERVER_TYPE> \
  --openai.health_check false
```

### Step 2: Verify the output

```python
import json
for line in open("/tmp/test_output.jsonl"):
    data = json.loads(line)
    print(f"Scores: {data.get('scores', [])}")
    print(f"Token sequences: {len(data.get('tokens', []))}")
    # Check messages include tool calls
    for msg_list in data.get("messages", []):
        roles = [m.get("role") for m in msg_list]
        print(f"Roles: {roles}")
        for m in reversed(msg_list):
            if m.get("role") == "assistant" and m.get("content"):
                print(f"Response: {m['content'][:200]}...")
                break
```

### What to check:
- **Scores are not all 0.0** — if so, compute_reward is broken
- **Scores are in [0, 1]** — not negative, not >1
- **Messages include "tool" role entries** — agent used tools
- **Token sequences are non-empty**
- **An HTML visualization is generated** next to the .jsonl

### Common failures:
- `'AgentResult' object has no attribute 'X'` — accessing a field that doesn't exist. See agentresult-fields.md.
- Score always 0.0 — reward function erroring silently
- Score always 1.0 — verification too lenient or not running


## Pattern 2: Evaluate a Model (evaluate mode)

Use `evaluate` mode to benchmark a model on your environment's eval
split. This runs the full agent loop with tools for each eval item.

### Step 1: Run evaluation

```bash
python environments/your_env.py evaluate \
  --env.eval_size 20 \
  --env.use_wandb false \
  --env.data_dir_to_save_evals /tmp/eval_results \
  --openai.base_url "<BASE_URL>" \
  --openai.model_name "<MODEL>" \
  --openai.server_type <SERVER_TYPE> \
  --openai.health_check false
```

### Step 2: Read results

Stdout shows a lighteval-compatible table:

```
Evaluation Results: your-env_eval
|Metric          |  Value|
|mean correctness| 0.850 |
|mean reward     | 0.920 |
|mean tool calls | 4.300 |
|n items         | 20    |
Evaluation completed in 367 seconds
```

JSON results saved to the eval directory:

```python
import json
data = json.load(open("/tmp/eval_results/metrics.json"))
for metric, value in data["results"]["all"].items():
    print(f"{metric}: {value}")
```

### Step 3: Compare models

Run evaluate with different models and compare the metrics.json files.

### What to check:
- **"data_dir_to_save_evals is not set"** — you forgot the flag, results won't be saved
- **Tool usage rate = 0** — evaluate() is using chat_completion instead of HermesAgentLoop
- **All scores identical** — judge failing, falling back to heuristic
- **Very slow** — each item runs a full agent loop (~30-90s). Use `--env.eval_size 5` for quick checks.


## Pattern 3: Generate Training Data (process mode, larger scale)

Generate trajectory data for offline training or analysis:

```bash
python environments/your_env.py process \
  --env.total_steps 50 \
  --env.group_size 4 \
  --env.use_wandb false \
  --env.data_path_to_save_groups data/trajectories.jsonl \
  --openai.base_url "<BASE_URL>" \
  --openai.model_name "<MODEL>" \
  --openai.server_type <SERVER_TYPE> \
  --openai.health_check false
```

### Analyze the distribution:

```python
import json
scores = []
for line in open("data/trajectories.jsonl"):
    data = json.loads(line)
    scores.extend(data.get("scores", []))

print(f"Total: {len(scores)}, Mean: {sum(scores)/len(scores):.3f}")
for bucket in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    count = sum(1 for s in scores if abs(s - bucket) < 0.1)
    print(f"  {bucket:.1f}: {'█' * count} ({count})")
```

### What to check:
- **Score distribution has variance** — RL needs score variance. All-same scores are useless.


## Pattern 4: Full RL Training (serve mode)

For actual RL training with Atropos:

```bash
# Terminal 1: Start Atropos API server
run-api

# Terminal 2: Start your environment
python environments/your_env.py serve \
  --config environments/your_env/default.yaml
```

For Phase 2 with VLLM:

```bash
# Terminal 1: VLLM server
python -m vllm.entrypoints.openai.api_server --model your-model --port 8000

# Terminal 2: Atropos API
run-api

# Terminal 3: Environment
python environments/your_env.py serve \
  --openai.base_url http://localhost:8000/v1 \
  --openai.model_name your-model \
  --openai.server_type vllm
```


## Pattern 5: Quick Smoke Test

Verify imports and config before spending money on API calls:

```python
from environments.your_env import YourEnv
print(f"Name: {YourEnv.name}")
cfg, servers = YourEnv.config_init()
print(f"Toolsets: {cfg.enabled_toolsets}")
print(f"Server: {servers[0].model_name}")
print("All imports OK")
```


## Timing Expectations

| Mode | Items | Time per item | Total |
|------|-------|--------------|-------|
| process (1 item) | 1 | 30-90s | ~1 min |
| evaluate (5 items) | 5 | 30-90s | ~5 min |
| evaluate (20 items) | 20 | 30-90s | ~15-30 min |
| process (50 items) | 50 | 30-90s | ~30-75 min |

Times are for cloud APIs with Claude Sonnet-class models. Local models may be faster or slower depending on hardware.
