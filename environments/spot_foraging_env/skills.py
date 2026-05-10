"""Skill library for the LLM-as-skill-selector Spot foraging env.

Skills are short bursts of low-level locomotion executed by the walk PPO
policy with a fixed command vector. Each skill maps to:
    (command_velocity, duration_steps, semantic_label)

The LLM picks skills by name; the SkillExecutor runs them against the
rapier-gym SpotSim by stepping the underlying gym env with the policy
producing 12-D joint targets at every physics tick.

This file is intentionally agnostic to the policy implementation — pass
any callable that maps `obs -> action` (numpy 12-D float32). See
`policy.py` for the loader that pulls a trained ONNX policy when one is
available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# Control-rate context: the underlying SpotSim runs physics at 200 Hz with a
# decimation of 4, giving 50 Hz policy steps. One LLM-skill burst is the
# duration_steps below × 1/50s = 1.0s for the default 50-step skills.
PHYSICS_HZ = 200
DECIMATION = 4
POLICY_HZ = PHYSICS_HZ // DECIMATION  # 50


@dataclass(frozen=True)
class Skill:
    """One movement primitive available to the LLM.

    The command vector is what gets fed to the walk PPO policy:
      command[0] = forward velocity   (m/s, body-frame X+)
      command[1] = lateral velocity   (m/s, body-frame Z+)  // strafe
      command[2] = yaw rate           (rad/s, world-up)
    """

    name: str
    command: Tuple[float, float, float]  # (v_x, v_y_lat, w_z)
    duration_steps: int = POLICY_HZ      # default: 1 second
    description: str = ""


SKILL_REGISTRY: Dict[str, Skill] = {
    s.name: s
    for s in [
        Skill(
            name="walk_forward",
            command=(0.6, 0.0, 0.0),
            duration_steps=POLICY_HZ,
            description="Walk forward at moderate pace for ~1 second.",
        ),
        Skill(
            name="walk_backward",
            command=(-0.4, 0.0, 0.0),
            duration_steps=POLICY_HZ,
            description="Walk backward (slower, since reverse gait is weaker).",
        ),
        Skill(
            name="strafe_left",
            command=(0.0, 0.4, 0.0),
            duration_steps=POLICY_HZ,
            description="Side-step to the left.",
        ),
        Skill(
            name="strafe_right",
            command=(0.0, -0.4, 0.0),
            duration_steps=POLICY_HZ,
            description="Side-step to the right.",
        ),
        Skill(
            name="turn_left",
            command=(0.0, 0.0, 0.8),
            duration_steps=POLICY_HZ,
            description="Rotate counter-clockwise in place.",
        ),
        Skill(
            name="turn_right",
            command=(0.0, 0.0, -0.8),
            duration_steps=POLICY_HZ,
            description="Rotate clockwise in place.",
        ),
        Skill(
            name="walk_forward_fast",
            command=(1.0, 0.0, 0.0),
            duration_steps=POLICY_HZ * 2,
            description="Walk forward briskly for 2 seconds (covers more ground).",
        ),
        Skill(
            name="stop",
            command=(0.0, 0.0, 0.0),
            duration_steps=POLICY_HZ // 2,
            description="Stand still briefly; useful for stabilizing or observing.",
        ),
    ]
}


# Action-space type hint for the policy callable.
PolicyFn = Callable[[np.ndarray], np.ndarray]
"""obs -> 12-D joint-target action (float32). The walk PPO trainee's signature."""


class SkillExecutor:
    """Runs a chosen skill against rapier-gym's SpotEnvRapier.

    Holds the gym env and the low-level PPO policy. Doesn't own the
    high-level loop — that lives in SpotForagingEnv.collect_trajectories.

    Why not just step the SpotSim directly? Because the env wraps useful
    plumbing: command-frame observation construction, foot-contact tracking,
    fall detection, battery state, energy bookkeeping. The skill executor
    overrides `command` per skill and lets the existing env produce
    observations that match what the policy was trained on.
    """

    def __init__(self, gym_env, policy: PolicyFn):
        self.gym_env = gym_env
        self.policy = policy

    def execute(self, skill: Skill) -> "SkillResult":
        """Run one skill burst. Returns telemetry from the burst."""
        gym = self.gym_env
        # Pin the command vector to this skill's value for the burst.
        # SpotEnvRapier's _sample_command runs only on reset(), so writing
        # to .command directly is the override path.
        gym.command = np.array(skill.command, dtype=np.float32)

        steps_taken = 0
        total_reward = 0.0
        terminated = False
        truncated = False
        observation = None  # last observation
        info: Dict = {}

        # Capture pre-skill state for delta accounting.
        pre_xy = np.array(gym.sim.get_base_position(), dtype=np.float32)[[0, 2]]
        pre_targets = list(gym.sim.get_battery_positions() or [])
        pre_collected = int(gym.collected_targets)

        for _ in range(skill.duration_steps):
            obs = gym._compute_observation() if hasattr(gym, "_compute_observation") else gym.sim.get_observation()
            obs_np = np.array(obs, dtype=np.float32)

            action = self.policy(obs_np)
            assert action.shape == (12,), f"policy must return 12-D action, got {action.shape}"

            observation, step_reward, terminated, truncated, info = gym.step(action)
            steps_taken += 1
            total_reward += float(step_reward)

            if terminated or truncated:
                break

        post_xy = np.array(gym.sim.get_base_position(), dtype=np.float32)[[0, 2]]
        post_targets = list(gym.sim.get_battery_positions() or [])
        post_collected = int(gym.collected_targets)

        return SkillResult(
            skill=skill,
            steps_taken=steps_taken,
            total_low_level_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            collected_during_burst=post_collected - pre_collected,
            displacement=tuple(map(float, (post_xy - pre_xy))),
            pre_xy=tuple(map(float, pre_xy)),
            post_xy=tuple(map(float, post_xy)),
            pre_battery_count=len(pre_targets),
            post_battery_count=len(post_targets),
            info=info,
            last_observation=observation,
        )


@dataclass
class SkillResult:
    skill: Skill
    steps_taken: int
    total_low_level_reward: float
    terminated: bool
    truncated: bool
    collected_during_burst: int
    displacement: Tuple[float, float]
    pre_xy: Tuple[float, float]
    post_xy: Tuple[float, float]
    pre_battery_count: int
    post_battery_count: int
    info: Dict
    last_observation: Optional[np.ndarray]

    @property
    def fell(self) -> bool:
        """True if the burst ended in a fall (vs natural completion)."""
        return self.terminated and not self.truncated


def list_skill_names() -> List[str]:
    """Sorted list of skill names — used to render the LLM's tool spec."""
    return sorted(SKILL_REGISTRY.keys())


def get_skill(name: str) -> Optional[Skill]:
    """Lookup by name; returns None if unknown (LLM hallucinated a skill)."""
    return SKILL_REGISTRY.get(name)
