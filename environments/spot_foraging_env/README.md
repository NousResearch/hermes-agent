# Spot Foraging Env (LLM-as-skill-selector)

Hierarchical RL env that bridges:

- **Atropos** (LLM RL gym) on top — the LLM is the trainee, picking high-level skills.
- **rapier-gym** (Rust + PyO3 Spot quadruped sim) underneath — physics + a low-level walk PPO policy execute the chosen skill.

The LLM gets a structured JSON state + a top-down rendered frame at ~1 Hz, picks one of 8 movement skills via tool-calling, and the underlying policy runs that skill at 50 Hz against the rapier-gym sim. Reward is dominated by batteries collected, with secondary penalties for inefficiency and falling.

## Architecture

```
                                   ┌────────────────────────────┐
                                   │ Atropos API server          │
                                   │ (run-api)                   │
                                   └──────────────▲──────────────┘
                                                  │ POST /scored_data
                                                  │
┌──────────────────────────────────────────────────┴─────────────────────┐
│ SpotForagingEnv  (this package)                                         │
│ ─────────────────────────────────────────────────                       │
│  Atropos BaseEnv subclass.                                              │
│  Each rollout = one full foraging episode driven by an LLM:             │
│                                                                         │
│  for pick in 0..max_skill_picks:                                        │
│      LLM.chat_completion(state_json + image, tools=[select_skill])      │
│         └► picks Skill via tool call                                    │
│      SkillExecutor.execute(skill)                                       │
│         └► steps rapier-gym for skill.duration_steps with the           │
│            walk PPO policy (joint targets at 50 Hz)                     │
│      observe new state, append to messages                              │
│      if collected_all or fell: break                                    │
│                                                                         │
│  score = +per_battery × collected                                       │
│        + success_bonus  if all collected                                │
│        - skill_pick_cost × n_picks                                      │
│        - fall_penalty   if fell                                         │
│        + distance_shaping_term                                          │
└──────────────┬──────────────────────────────────────────────────────────┘
               │ chat_completion (multimodal)
               ▼
┌──────────────────────────────────┐
│ Inference fleet (vLLM / SGLang / │
│ OpenAI / OpenRouter ...)         │
│ ─ multimodal-capable              │
└──────────────────────────────────┘
```

## Files

| File | Purpose |
|---|---|
| `env.py` | `SpotForagingEnv` — the Atropos `BaseEnv` subclass. |
| `skills.py` | `Skill` registry (8 movement primitives) + `SkillExecutor`. |
| `policy.py` | Walk PPO policy loader. ONNX preferred; zero-action fallback. |
| `renderer.py` | Top-down matplotlib snapshot → PNG → OpenAI multimodal data-URL. |
| `prompts.py` | System prompt + `select_skill` tool spec + state JSON format. |
| `reward.py` | Per-episode reward function. |
| `configs/default.yaml` | Reference config; copy and edit. |
| `__init__.py` | Public surface. |

## Skills

The LLM picks from these (defined in `skills.py`):

| Skill | Command (vx, vy, ω) | Duration | Use |
|---|---|---|---|
| `walk_forward` | (0.6, 0, 0) | 1.0 s | Standard locomotion. |
| `walk_backward` | (-0.4, 0, 0) | 1.0 s | Backing out. |
| `strafe_left` | (0, 0.4, 0) | 1.0 s | Lateral move. |
| `strafe_right` | (0, -0.4, 0) | 1.0 s | Lateral move. |
| `turn_left` | (0, 0, 0.8) | 1.0 s | Rotate CCW. |
| `turn_right` | (0, 0, -0.8) | 1.0 s | Rotate CW. |
| `walk_forward_fast` | (1.0, 0, 0) | 2.0 s | Cover ground fast. |
| `stop` | (0, 0, 0) | 0.5 s | Stabilize. |

All skills delegate to the same walk PPO policy with different `command` vectors — no per-skill policy training. Adding new skills is a one-entry append to `SKILL_REGISTRY`.

## Dependencies

- **`atroposlib`** — `pip install atroposlib`. The Atropos framework.
- **`spot_rapier`** — the rapier-gym Python module. Build via `cd /home/olive/Repositories/skypilot-env/spot/rapier-gym && maturin develop --release` (Rust toolchain required) or use the prebuilt `.so` already present.
- **`onnxruntime`** — optional. Required only when loading a trained walk policy. Install: `pip install onnxruntime`.
- **`matplotlib`** — for the top-down renderer. Already in `hermes-agent` deps.



## Walk policy

The env loads the walk PPO policy via `policy.load_walk_policy()`. Resolution order:

1. `walk_policy_onnx` config field (explicit override).
2. `SPOT_WALK_POLICY_ONNX` env var.
3. `~/.cache/spot_rapier/walk_policy.onnx` (conventional location).
4. Zero-action fallback (logs a warning).

The fallback is deliberate — the env runs end-to-end even with no trained walk policy. Spot just stands still; the LLM's skill picks are recorded but have no physical effect. Useful for testing the Atropos plumbing before walk training has converged. Once you have a trained policy, export it via `train_rapier.py`'s ONNX export path (the `export_onnx` helper) and either drop it at `~/.cache/spot_rapier/walk_policy.onnx` or set `walk_policy_onnx` in the config.

## Running

### Quick local test (no trainer, no API — `process` mode)

```bash
cd /home/olive/Repositories/hermes-agent
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

This runs 4 scenarios end-to-end against any OpenAI-compatible multimodal endpoint, dumps `ScoredDataGroup`s to `/tmp/spot_foraging.jsonl`, and writes a static HTML viz alongside it. Good first sanity check.

### Live training loop (`serve` mode)

Three terminals:

```bash
# A: API server
run-api &

# B: Env server (this package)
cd /home/olive/Repositories/hermes-agent
python -m environments.spot_foraging_env.env serve \
    --config environments/spot_foraging_env/configs/default.yaml \
    --slurm False

# C: Trainer (whatever you use — example_trainer/grpo.py, Axolotl, custom)
python /tmp/atropos/example_trainer/grpo.py
```

### Inference setup

The env is multimodal-capable (sends rendered PNGs as `image_url` parts when `enable_image=true`). Pick a multimodal endpoint:

- **OpenRouter**: `--openai.base_url https://openrouter.ai/api/v1 --openai.model_name anthropic/claude-sonnet-4.5 --openai.health_check false` — cheapest path to test.
- **vLLM with a multimodal trainee** (e.g. Llava-Next, InternVL): `--openai.base_url http://localhost:8001/v1 --openai.server_type vllm`.
- **Text-only model** (no images): set `--env.enable_image False` and the env falls back to JSON-only state.

## Reward shape

```
score = +10.0  per battery collected
      + 20.0  if ALL collected before timeout (success bonus)
      -  0.05 × n_skill_picks   (efficiency penalty)
      -  5.0  if Spot fell      (terminal failure)
      +  0.5  × cumulative-best-distance-decrement   (dense shaping)

clipped to ±200.0
```

A successful 5-battery episode in ~15 picks scores ~ 10×5 + 20 - 0.05×15 + shaping ≈ 70-80. A faller after collecting 1 scores ~10 - 5 - some penalty ≈ 0-5. A timeout-no-collect scores ~0 plus whatever shaping accumulated.

Tunable in `RewardConfig` (per-episode) or `SpotForagingEnvConfig` (CLI/YAML).

## Group construction

Each `collect_trajectories` call runs `group_size` rollouts on the **same scenario** (same seed, same battery layout, same spawn). The differential is the LLM's stochastic skill picks. This gives Atropos a clean GRPO contrastive group: same starting state, varying scores driven by varying skill sequences.

For PPO-style training without contrastive grouping, set `group_size=1` and the same env works.

## Observability

Per-train-step (logged via `wandb_log`):
- `train/avg_collected` — mean batteries collected over recent episodes.
- `train/avg_skill_picks` — mean episode length in picks.
- `train/fall_rate` — fraction of recent episodes that ended in a fall.
- `train/success_rate` — fraction that collected all batteries.

Per-eval (logged when `evaluate()` runs every `steps_per_eval` steps):
- `eval/success_rate`, `eval/avg_collected`, `eval/avg_skill_picks`, `eval/fall_rate`, `eval/n_episodes`.

Plus the base-class rollout-table (set `num_rollouts_to_keep` in the config to control table size).

## Known limitations / TODOs

- **`set_battery_positions`** isn't yet exposed by rapier-gym's PyO3 layer. Currently the env relies on `spawn_targets=True` letting the rapier sim place batteries via its own random sampling — the explicit `battery_positions` we sample in `ScenarioItem` is used for shaping/scoring but not enforced in-sim. Will land properly once the Rust side exposes the setter.
- **Yaw heading in the renderer** is estimated from recent displacement (atan2 of trail). When Spot is stationary it shows whatever the last motion direction was. Cleanup: expose body quaternion via the rapier-gym Python API and read it directly.
- **The walk policy fallback is zero-action.** Episodes still run, but Spot stands still. Until walk training (v23) ships an ONNX checkpoint, the env exercises only the LLM-loop, not the physical execution.

- **Image encoding cost**: rendering a 256×256 PNG every skill pick adds ~10-20ms latency. For high-throughput training, consider lowering `image_pixels` or disabling images entirely.

## Future directions

- **Multi-skill policy training**: train per-skill policies (sit, stand, climb, etc.) instead of reusing the walk policy with command overrides. Each skill becomes a separate ONNX checkpoint.
- **WASM render integration**: replace the matplotlib top-down with rendered frames from the existing in-cluster Rerun viewer pods or the WASM browser sim — the LLM gets a third-person rendering matching what humans see.
- **Curriculum**: start with `n_batteries=1` in a small arena, scale up as the trainee succeeds. Atropos doesn't ship curriculum support; implement via overriding `get_next_item`.
- **Federated rollouts**: this env is stateless across episodes. Multiple env-server replicas pointed at one Atropos API gives free horizontal scale of rollout collection.
