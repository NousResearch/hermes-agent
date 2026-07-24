# Eval Suite Contract

## Suite YAML Format
```yaml
name: suite_name
description: What this suite tests
scenarios:
  - id: S1_unique_id
    description: Human-readable
    user_message: "The exact user prompt to send"
    system_message: "Optional system prompt override"
    config_overrides:
      delegation.max_concurrent_children: 8
      agent.max_iterations: 12
    enabled_toolsets: [terminal, file, delegation]
    skip_memory: true
    skip_context_files: true
    pass_conditions:
      - type: delegate_call_count
        min: 2
      - type: plan_score
        min: 0.8
      - type: no_tool_error
      - type: response_contains
        value: "expected substring"
      - type: no_cache_break  # cost_cache suite
      - type: verify_rate
        min: 0.9
      - type: recall_at_3
        min: 0.85
      - type: custom
        rubric: "module.function_name"
```

## Rubric Format
Each `evals/rubrics/<suite_name>.py` exports:
```python
def grade(scenario: dict, result: dict) -> dict:
    """Return {pass: bool, score: float 0-1, details: dict}"""
    ...
```

## Runner Output Format
```json
{
  "suite": "orchestration",
  "timestamp": "2026-06-30T02:30:00",
  "total": 5,
  "passed": 4,
  "failed": 1,
  "pass_rate": 0.80,
  "scenarios": [
    {
      "id": "S1",
      "pass": true,
      "score": 0.95,
      "details": {...},
      "api_calls": 3,
      "duration_s": 12.5
    }
  ]
}
```

## AIAgent API (from livetest pattern)
```python
from run_agent import AIAgent
agent = AIAgent(
    provider="openrouter",  # or fake for deterministic
    model="...",
    enabled_toolsets=["terminal", "file"],
    quiet_mode=True,
    save_trajectories=False,
    skip_context_files=True,
    skip_memory=True,
    platform="cli",
    max_iterations=12,
)
result = agent.run_conversation(
    user_message="...",
    system_message="...",
)
# result["final_response"], result["messages"]
```
