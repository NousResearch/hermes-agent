# Atropos BaseEnv Reference

Source: `atroposlib/envs/base.py` (~2124 lines)

## Abstract Methods (MUST implement)

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_next_item()` | `async def get_next_item(self) -> Item` | Return next item for trajectory. Return None to pause. |
| `evaluate()` | `async def evaluate(self, *args, **kwargs)` | Called every steps_per_eval steps. |
| `setup()` | `async def setup(self)` | Called once at start. Load datasets, init models. |
| `collect_trajectory()` | `async def collect_trajectory(self, item) -> Tuple[Optional[ScoredDataItem], List[Item]]` | Single rollout. Or override collect_trajectories instead. |

## Overridable Methods

| Method | Default Behavior | Override When |
|--------|-----------------|---------------|
| `collect_trajectories()` | Runs collect_trajectory group_size times in parallel | Batch generation, MCTS, coupled rollouts |
| `wandb_log()` | Logs completion lengths, rollout table, perf stats | Add custom metrics (always call super) |
| `config_init()` | Returns (env_config_cls(), ServerBaseline()) | Custom defaults + server configs |
| `postprocess_histories()` | Passthrough | Final processing before sending to trainer |
| `save_checkpoint()` | Saves JSON to checkpoint_dir | Custom serialization |
| `cleanup()` | No-op | Release resources after each rollout |

## ScoredDataGroup Structure

```python
ScoredDataGroup = TypedDict with:
    tokens:             List[List[int]]       # Token IDs per rollout
    masks:              List[List[int]]       # -100=prompt, token_id=completion
    scores:             List[float]           # Score per rollout
    advantages:         Optional[...]         # Per-token advantages
    ref_logprobs:       Optional[...]         # Reference model logprobs
    messages:           Optional[...]         # OpenAI-format messages
    inference_logprobs: Optional[...]         # Inference logprobs
```

## BaseEnvConfig Key Fields

| Field | Default | Description |
|-------|---------|-------------|
| `group_size` | 4 | Responses grouped for scoring |
| `steps_per_eval` | 100 | Steps between evaluations |
| `max_token_length` | 2048 | Max token length for generations |
| `total_steps` | 1000 | Total training steps |
| `use_wandb` | True | Enable wandb logging |
| `tokenizer_name` | DeepHermes-3 | Tokenizer for token encoding |
| `ensure_scores_are_not_same` | True | Skip groups with identical scores |
| `worker_timeout` | 600 | Task timeout seconds |

## Data Flow

```
env_manager() → add_train_workers() → handle_env()
    → collect_trajectories() → postprocess_histories()
    → handle_send_to_api() → training server
```

## Atropos Environment Statistics (82 environments analyzed)

- 95% implement setup, collect_trajectories, evaluate, get_next_item
- 76% override wandb_log
- 54% have custom config class
- Most use collect_trajectories (plural), not collect_trajectory (singular)
- Common reward patterns: LLM-judge (~40), regex-extract (~35), code-exec (~12)
