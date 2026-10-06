"""Project verification subsystem.

Scoped port of superagent-ai/grok-cli's verify subsystem (``src/verify/recipes.ts``,
``src/verify/environment.ts``): static run-recipe detection, a persisted
environment manifest, and a smoke-test runner used by ``hermes verify``.
"""

from agent.verify.environment import load_manifest, load_or_detect, manifest_path, save_manifest
from agent.verify.recipes import Recipe, detect_package_manager, detect_recipe
from agent.verify.runner import PhaseResult, ReadinessResult, VerifyResult, run_verify

__all__ = [
    "Recipe", "detect_recipe", "detect_package_manager", "load_manifest", "save_manifest",
    "load_or_detect", "manifest_path", "run_verify", "PhaseResult", "ReadinessResult", "VerifyResult",
]
