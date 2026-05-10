---
name: spot-hierarchical-rl
description: Reusable adapter pattern for integrating Spot robotics tasks (like foraging) with Atropos GRPO hierarchical RL training.
category: mlops/training
---

# Spot Hierarchical RL (Atropos Adapter)

This skill documents the adapter pattern used to bridge rapier-gym physics environments (like the Spot foraging task) into the Atropos GRPO hierarchical RL trainer. This pattern transforms an LLM into a high-level skill planner, outputting stochastic skill selections while low-level control policies (PPO) handle movement physics.

## Architecture & Integration Flow

1. **Scenario Generation (`get_next_item`)**: 
   Samples a `ScenarioItem`, which defines the environmental setup (e.g., random spawn location and battery layout).
2. **Rollout Collection (`collect_trajectories`)**: 
   Runs `group_size` rollouts on the same scenario in parallel. This produces a clean GRPO-style contrastive group where the only differential between rollouts is the LLM's stochastic skill picks.
3. **Execution Context**: 
   Each rollout executes within a single `ManagedServer` context, enabling multi-turn token accumulation with proper masking.
4. **Environment Loop**:
   - **Chat Setup**: Initializes with the system prompt and the initial user message (state JSON + top-down rendered PNG).
   - **Decision Iteration** (up to `max_skill_picks` times):
     - `chat_completion(tools=[select_skill], tool_choice="auto")` -> The LLM acts as the planner, selecting a skill.
     - `SkillExecutor.execute(skill)` -> Steps the rapier-gym physics engine (e.g., 50 ticks) utilizing the low-level policy (like the walk PPO policy) combined with the skill's command vector.
     - Message Appending: Assistant message (skill pick), tool-result message (execution outcome), next user message (new state).
     - Break conditions: e.g., fall or success (all items collected).
5. **Reward & Scoring**:
   The entire sequence of LLM skill picks is evaluated using a composite reward function.
   Example: `+10 * collected + 20 * success - 0.05 * picks - 5 * fell + 0.5 * distance_decrement`
6. **Data Assembly**:
   Concatenates per-turn `node.tokens` and `node.masked_tokens` from the `ManagedServer` into one contiguous sequence. The result is pushed as a `ScoredDataGroup`.

## Skills Definition Pattern

High-level movement primitives share a single underlying low-level policy (e.g., a walk PPO). They differ only in target command vectors (vx, vy, ω) and execution burst duration. 
Adding a new skill is minimal (a simple 4-line `Skill(...)` append).

## Validation Pipeline (No Policies Required)

You can validate the LLM-loop -> tool-parse -> skill-execute -> reward pipeline end-to-end without a fully trained policy using `--process` mode.

```bash
cd /home/olive/Repositories/hermes-agent

# Run scenarios in process mode against any multimodal endpoint
python -m environments.spot_foraging_env.env process \
  --config environments/spot_foraging_env/configs/default.yaml \
  --env.data_path_to_save_groups /tmp/spot_foraging.jsonl \
  --env.total_steps 4 \
  --openai.base_url "https://openrouter.ai/api/v1" \
  --openai.api_key "$OPENROUTER_API_KEY" \
  --openai.model_name "anthropic/claude-sonnet-4.5" \
  --openai.server_type openai \
  --openai.health_check false
```

*Output:* Validates the loop and writes JSONL (e.g., `/tmp/spot_foraging.jsonl`) with `ScoredDataGroups` and static HTML visualizer output.

## Known Gaps & Future Work

- **Walk Policy Integration**: Drop a trained ONNX PPO model into `~/.cache/spot_rapier/walk_policy.onnx`. Without it, `load_walk_policy` defaults to zero-actions (episodes complete, but Spot stays stationary).
- **In-sim State Enforcement**: rapier-gym's PyO3 layer needs to expose methods (e.g., `set_battery_positions` or `get_body_quaternion`) to enforce scenario determinism in the physics engine.
- **Packaging**: Transition `rapier-gym` from `sys.path` injection to a standalone installed wheel for reliability.
