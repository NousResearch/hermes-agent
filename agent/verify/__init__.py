"""Project verification subsystem.

Ported from superagent-ai/grok-cli's verify subsystem (scoped):
static run-recipe detection, a persisted environment manifest, and a
smoke-test runner used by the ``hermes verify`` CLI command.

Sources:
- https://github.com/superagent-ai/grok-cli/blob/main/src/verify/recipes.ts
- https://github.com/superagent-ai/grok-cli/blob/main/src/verify/environment.ts
"""

from agent.verify.environment import (
    load_manifest,
    load_or_detect,
    manifest_path,
    save_manifest,
)
from agent.verify.recipes import Recipe, detect_package_manager, detect_recipe
from agent.verify.runner import (
    PhaseResult,
    ReadinessResult,
    VerifyResult,
    run_verify,
)

__all__ = [
    "Recipe",
    "detect_recipe",
    "detect_package_manager",
    "load_manifest",
    "save_manifest",
    "load_or_detect",
    "manifest_path",
    "run_verify",
    "PhaseResult",
    "ReadinessResult",
    "VerifyResult",
]
