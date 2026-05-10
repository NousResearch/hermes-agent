"""SpotForagingEnv — the Atropos BaseEnv for LLM-as-skill-selector foraging.

Architecture (one rollout):
    1. Reset rapier-gym to a scenario (random spawn, batteries scattered).
    2. Build initial chat: system prompt + first state JSON + rendered frame.
    3. Loop until terminal:
       a. LLM chat_completion (under ManagedServer for token tracking).
       b. Parse the `select_skill` tool call → Skill name.
       c. SkillExecutor runs the skill against the gym (50 physics ticks
          driven by the walk PPO policy + skill's command vector).
       d. Append assistant message; append new user message with updated
          state JSON + new frame.
    4. Score the LLM's full skill-pick trajectory (battery collection,
       efficiency, fall penalty, distance shaping).
    5. Return ScoredDataGroup with concatenated multi-turn tokens/masks.

Group construction: `group_size` parallel rollouts from the SAME scenario
(seed shared) — gives GRPO contrastive groups where the differential
across rollouts is the LLM's skill-pick noise.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Atropos imports — fail loudly with a useful message if not installed.
try:
    from atroposlib.envs.base import (
        BaseEnv,
        BaseEnvConfig,
        Item,
        ScoredDataGroup,
    )
    from atroposlib.envs.server_handling.server_manager import (
        ServerBaseline,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "atroposlib not installed. Run `pip install atroposlib` or install "
        "from the cloned repo with `pip install -e .[all]`."
    ) from e

from .policy import PolicyFn, load_walk_policy
from .prompts import (
    SYSTEM_PROMPT,
    build_tool_spec,
    build_user_message_content,
    format_state_json,
)
from .renderer import (
    SceneSnapshot,
    make_image_content_part,
    snapshot_from_gym,
)
from .reward import EpisodeMetrics, RewardConfig, compute_reward
from .skills import (
    SKILL_REGISTRY,
    Skill,
    SkillExecutor,
    SkillResult,
    get_skill,
)


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Config


class SpotForagingEnvConfig(BaseEnvConfig):
    """Foraging-specific knobs on top of BaseEnvConfig."""

    # Scenario params
    n_batteries: int = 5
    arena_half_size_m: float = 4.0
    spawn_radius_m: float = 0.5

    # Episode shape
    max_skill_picks: int = 30                 # episode timeout in skill-picks
    fall_terminates: bool = True

    # Image rendering
    enable_image: bool = True
    image_pixels: int = 256

    # Walk policy
    walk_policy_onnx: Optional[str] = None    # explicit override; else env-resolution

    # LLM control
    skill_pick_temperature: float = 1.0
    skill_pick_max_tokens: int = 256

    # Reward
    reward_per_battery: float = 10.0
    reward_success_bonus: float = 20.0
    reward_skill_pick_cost: float = 0.05
    reward_fall_penalty: float = 5.0
    reward_distance_decrement_weight: float = 0.5

    # Override BaseEnv defaults — sensible for this env
    group_size: int = 4                       # 4 LLM rollouts per scenario
    max_token_length: int = 8192              # multi-turn chat is long
    steps_per_eval: int = 50
    use_wandb: bool = True
    include_messages: bool = True             # multimodal — keep messages on the wire


# ──────────────────────────────────────────────────────────────────────────
# Scenario item


@dataclass
class ScenarioItem:
    """One foraging scenario: where to spawn Spot, where to put batteries."""

    seed: int
    spot_start_xy: Tuple[float, float]
    battery_positions: List[Tuple[float, float]]


def _sample_scenario(
    cfg: SpotForagingEnvConfig, rng: np.random.Generator
) -> ScenarioItem:
    """Produce a fresh scenario respecting arena/spawn config."""
    half = cfg.arena_half_size_m
    spot_xy = tuple(
        rng.uniform(-cfg.spawn_radius_m, cfg.spawn_radius_m, size=2).astype(float)
    )

    # Sample battery positions, rejection-sampling to avoid spawn-overlap.
    batteries: List[Tuple[float, float]] = []
    attempts = 0
    while len(batteries) < cfg.n_batteries and attempts < 200:
        b = tuple(rng.uniform(-half, half, size=2).astype(float))
        if all(math.hypot(b[0] - x, b[1] - y) > 0.6 for x, y in batteries + [spot_xy]):
            batteries.append(b)
        attempts += 1

    return ScenarioItem(
        seed=int(rng.integers(0, 2**31 - 1)),
        spot_start_xy=spot_xy,  # type: ignore[arg-type]
        battery_positions=batteries,
    )


# ──────────────────────────────────────────────────────────────────────────
# Env


class SpotForagingEnv(BaseEnv):
    """LLM-as-skill-selector foraging env, Atropos style."""

    name = "spot_foraging"
    env_config_cls = SpotForagingEnvConfig

    @classmethod
    def config_init(cls) -> Tuple[SpotForagingEnvConfig, ServerBaseline]:
        return cls.env_config_cls(), ServerBaseline()

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def setup(self) -> None:
        # Lazy import: SpotEnvRapier needs the .so, which we may have just
        # added to sys.path. Doing it here avoids hard-failing if someone
        # imports SpotForagingEnv on a box without rapier-gym.
        from spot_rapier import SpotEnvRapier

        self._SpotEnvRapier = SpotEnvRapier
        self._policy: PolicyFn = load_walk_policy(
            self.config.walk_policy_onnx
        )
        self._reward_cfg = RewardConfig(
            per_battery=self.config.reward_per_battery,
            success_bonus=self.config.reward_success_bonus,
            skill_pick_cost=self.config.reward_skill_pick_cost,
            fall_penalty=self.config.reward_fall_penalty,
            distance_decrement_weight=self.config.reward_distance_decrement_weight,
        )

        # Train / eval scenario RNGs.
        self._train_rng = np.random.default_rng(seed=42)
        self._eval_scenarios = [
            _sample_scenario(self.config, np.random.default_rng(seed=1000 + i))
            for i in range(16)
        ]

        # Episode metrics for wandb summary (last N episodes).
        self._recent_metrics: List[EpisodeMetrics] = []

        logger.info(
            "SpotForagingEnv setup complete. n_batteries=%d max_picks=%d "
            "image_enabled=%s policy=%s",
            self.config.n_batteries,
            self.config.max_skill_picks,
            self.config.enable_image,
            "onnx" if self.config.walk_policy_onnx else "auto",
        )

    async def get_next_item(self) -> Optional[Item]:
        return _sample_scenario(self.config, self._train_rng)

    # ── main rollout ──────────────────────────────────────────────────────

    async def collect_trajectories(
        self, item: ScenarioItem
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        """Run `group_size` rollouts on the same scenario; build group."""
        rollouts = await asyncio.gather(
            *[
                self._run_episode(item)
                for _ in range(self.config.group_size)
            ]
        )

        # Drop any rollouts that errored.
        rollouts = [r for r in rollouts if r is not None]
        if not rollouts:
            return None, []

        scores = [r["score"] for r in rollouts]
        if (
            self.config.ensure_scores_are_not_same
            and all(s == scores[0] for s in scores)
        ):
            # Atropos convention — drop unprobeable groups.
            return None, []

        group: ScoredDataGroup = {
            "tokens": [r["tokens"] for r in rollouts],
            "masks": [r["masks"] for r in rollouts],
            "scores": scores,
        }
        if self.config.include_messages:
            group["messages"] = [r["messages"] for r in rollouts]

        # Stash metrics for wandb_log.
        for r in rollouts:
            self._recent_metrics.append(r["metrics"])
        self._recent_metrics = self._recent_metrics[-100:]

        return group, []

    async def _run_episode(self, item: ScenarioItem) -> Optional[Dict[str, Any]]:
        """One rollout: full LLM-driven foraging episode → tokens/masks/score."""
        try:
            return await asyncio.wait_for(self._run_episode_inner(item), timeout=600)
        except Exception as e:
            logger.warning("Episode failed: %s", e, exc_info=True)
            return None

    async def _run_episode_inner(self, item: ScenarioItem) -> Dict[str, Any]:
        # Build a fresh gym env per rollout (state isolation).
        gym_env = self._SpotEnvRapier(
            config={
                "behavior": "forage",
                "spawn_targets": True,
                "n_targets": self.config.n_batteries,
                "seed": item.seed,
            }
        )
        # Reset to spawn at the scenario's start position.
        gym_env.reset(seed=item.seed)
        # Override battery positions if rapier-gym supports explicit placement.
        # Falls back to whatever the env spawned via its own random sampling
        # if direct placement isn't exposed.
        if hasattr(gym_env.sim, "set_battery_positions"):
            try:
                gym_env.sim.set_battery_positions([
                    [b[0], 0.0, b[1]] for b in item.battery_positions
                ])
            except Exception as e:
                logger.debug("set_battery_positions not supported: %s", e)

        executor = SkillExecutor(gym_env, self._policy)

        # State for distance shaping.
        trail: List[Tuple[float, float]] = []
        nearest_dist_history: List[float] = []
        skill_history: List[str] = []
        terminated = False

        # Build initial chat state.
        initial_snapshot = snapshot_from_gym(gym_env, trail)
        initial_state_json = format_state_json(
            spot_xy=initial_snapshot.spot_xy,
            spot_yaw_deg=math.degrees(initial_snapshot.spot_yaw),
            batteries=initial_snapshot.batteries,
            collected=initial_snapshot.collected_count,
            skill_history=skill_history,
            steps_remaining=self.config.max_skill_picks,
            energy=initial_snapshot.energy,
        )

        image_part = (
            make_image_content_part(initial_snapshot, pixels=self.config.image_pixels)
            if self.config.enable_image
            else None
        )

        messages: List[Dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_message_content(initial_state_json, image_part),
            },
        ]

        tools = build_tool_spec()

        # Run the episode under a single ManagedServer context — multi-turn
        # tokens accumulate correctly with proper masking across all picks.
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            for pick_idx in range(self.config.max_skill_picks):
                chat = await managed.chat_completion(
                    messages=messages,
                    n=1,
                    max_tokens=self.config.skill_pick_max_tokens,
                    temperature=self.config.skill_pick_temperature,
                    tools=tools,
                    tool_choice="auto",
                )

                assistant_msg = chat.choices[0].message
                messages.append(_assistant_message_to_dict(assistant_msg))

                skill = _parse_skill_from_message(assistant_msg)
                if skill is None:
                    # LLM whiffed: nudge it back on track and let it retry.
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "I couldn't parse a `select_skill` tool call. "
                                "Try again: call `select_skill` with one of "
                                f"{sorted(SKILL_REGISTRY.keys())}."
                            ),
                        }
                    )
                    continue

                skill_history.append(skill.name)
                result: SkillResult = executor.execute(skill)

                # Update post-skill state.
                trail.append(
                    (
                        float(gym_env.sim.get_base_position()[0]),
                        float(gym_env.sim.get_base_position()[2]),
                    )
                )

                # Track nearest-battery distance for shaping.
                snap = snapshot_from_gym(gym_env, trail)
                if snap.batteries:
                    sx, sz = snap.spot_xy
                    nearest = min(
                        math.hypot(sx - bx, sz - bz) for bx, bz in snap.batteries
                    )
                else:
                    nearest = 0.0
                nearest_dist_history.append(nearest)

                # Tool-result message (compact).
                tool_summary = {
                    "skill_executed": skill.name,
                    "displacement_xy": [round(d, 2) for d in result.displacement],
                    "collected_in_burst": result.collected_during_burst,
                    "total_collected": int(gym_env.collected_targets),
                    "fell": bool(result.fell),
                }
                tool_call_id = _first_tool_call_id(assistant_msg)
                if tool_call_id is not None:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps(tool_summary),
                        }
                    )

                # Terminal conditions.
                if result.fell and self.config.fall_terminates:
                    terminated = True
                    break
                if int(gym_env.collected_targets) >= len(item.battery_positions):
                    terminated = True
                    break

                # Append next state for the LLM.
                state_json = format_state_json(
                    spot_xy=snap.spot_xy,
                    spot_yaw_deg=math.degrees(snap.spot_yaw),
                    batteries=snap.batteries,
                    collected=snap.collected_count,
                    skill_history=skill_history,
                    steps_remaining=self.config.max_skill_picks - (pick_idx + 1),
                    energy=snap.energy,
                )
                next_image_part = (
                    make_image_content_part(snap, pixels=self.config.image_pixels)
                    if self.config.enable_image
                    else None
                )
                messages.append(
                    {
                        "role": "user",
                        "content": build_user_message_content(
                            state_json, next_image_part
                        ),
                    }
                )

            nodes = managed.get_state()["nodes"]

        # Concat multi-turn tokens/masks into one sequence.
        all_tokens = []
        all_masks = []
        for node in nodes:
            all_tokens.extend(node.tokens)
            all_masks.extend(node.masked_tokens)

        # Compute episode score.
        metrics = EpisodeMetrics(
            batteries_collected=int(gym_env.collected_targets),
            total_batteries=len(item.battery_positions),
            skill_picks=len(skill_history),
            fell=terminated and any(
                m.get("fell")
                for m in messages
                if isinstance(m.get("content"), str)
            ),
            timed_out=not terminated,
            nearest_battery_distance_history=nearest_dist_history,
        )
        score = compute_reward(metrics, self._reward_cfg)

        return {
            "tokens": all_tokens,
            "masks": all_masks,
            "messages": messages,
            "score": score,
            "metrics": metrics,
        }

    # ── eval ──────────────────────────────────────────────────────────────

    async def evaluate(self, *args, **kwargs) -> None:
        """Run held-out scenarios; log eval/* metrics to wandb."""
        eval_metrics: List[EpisodeMetrics] = []
        for scenario in self._eval_scenarios:
            try:
                rollout = await self._run_episode_inner(scenario)
                eval_metrics.append(rollout["metrics"])
            except Exception as e:
                logger.warning("Eval episode failed: %s", e)

        if not eval_metrics:
            return

        n = len(eval_metrics)
        success_rate = sum(
            1 for m in eval_metrics
            if m.batteries_collected == m.total_batteries
        ) / n
        avg_collected = sum(m.batteries_collected for m in eval_metrics) / n
        avg_picks = sum(m.skill_picks for m in eval_metrics) / n
        fall_rate = sum(1 for m in eval_metrics if m.fell) / n

        # Stash for wandb_log to pick up.
        self._last_eval_metrics = {
            "eval/success_rate": success_rate,
            "eval/avg_collected": avg_collected,
            "eval/avg_skill_picks": avg_picks,
            "eval/fall_rate": fall_rate,
            "eval/n_episodes": n,
        }

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None) -> None:
        if wandb_metrics is None:
            wandb_metrics = {}

        if self._recent_metrics:
            collected = [m.batteries_collected for m in self._recent_metrics]
            picks = [m.skill_picks for m in self._recent_metrics]
            falls = [m.fell for m in self._recent_metrics]
            wandb_metrics["train/avg_collected"] = sum(collected) / len(collected)
            wandb_metrics["train/avg_skill_picks"] = sum(picks) / len(picks)
            wandb_metrics["train/fall_rate"] = sum(falls) / len(falls)
            wandb_metrics["train/success_rate"] = sum(
                1 for m in self._recent_metrics
                if m.batteries_collected == m.total_batteries
            ) / len(self._recent_metrics)

        if hasattr(self, "_last_eval_metrics"):
            wandb_metrics.update(self._last_eval_metrics)

        await super().wandb_log(wandb_metrics)


# ──────────────────────────────────────────────────────────────────────────
# Tool-call parsing helpers


def _assistant_message_to_dict(msg: Any) -> Dict:
    """Coerce an OpenAI-format assistant message (with tool_calls) to a dict."""
    out: Dict[str, Any] = {"role": "assistant"}
    if getattr(msg, "content", None):
        out["content"] = msg.content
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return out


def _first_tool_call_id(msg: Any) -> Optional[str]:
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        return tcs[0].id
    return None


def _parse_skill_from_message(msg: Any) -> Optional[Skill]:
    """Extract a Skill from a `select_skill` tool call. Returns None on miss."""
    tcs = getattr(msg, "tool_calls", None) or []
    for tc in tcs:
        if getattr(tc, "type", "") != "function":
            continue
        fn = getattr(tc, "function", None)
        if fn is None or fn.name != "select_skill":
            continue
        try:
            args = json.loads(fn.arguments or "{}")
        except json.JSONDecodeError:
            continue
        skill_name = args.get("skill")
        if not isinstance(skill_name, str):
            continue
        skill = get_skill(skill_name)
        if skill is not None:
            return skill
    return None


# ──────────────────────────────────────────────────────────────────────────
# CLI entrypoint


if __name__ == "__main__":
    SpotForagingEnv.cli()
