"""LLM-as-skill-selector hierarchical RL env for Spot foraging.

The LLM picks a movement skill at ~1 Hz; the underlying walk PPO policy
executes the chosen skill at 50 Hz against the rapier-gym physics sim.
The Atropos training signal is the LLM's success at directing Spot to
collect scattered batteries.

See README.md for architecture, conventions, and run instructions.
"""

from .env import SpotForagingEnv, SpotForagingEnvConfig
from .skills import SKILL_REGISTRY, Skill, SkillExecutor

__all__ = [
    "SpotForagingEnv",
    "SpotForagingEnvConfig",
    "SKILL_REGISTRY",
    "Skill",
    "SkillExecutor",
]
