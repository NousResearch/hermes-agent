"""Log setup-pm's restore results and write policy without traversing cache trees."""
from __future__ import annotations

import json
import os


def log_caches(inputs: dict, steps: dict, scope: str) -> None:
    for family, name, producer, output in [
        ("tools", "tools", "prepare", "store"),
        ("uv", "python", "install", "uv-cache-path"),
        ("npm", "node", "install", "npm-cache-path"),
    ]:
        candidates = [steps.get(f"{name}-cache", {}), steps.get(f"{name}-cache-restore", {})]
        selected = next((step for step in candidates if step.get("outcome") not in (None, "skipped")), {})
        outcome = selected.get("outcome", "skipped")
        hit = selected.get("outputs", {}).get("cache-hit", "")
        policy = "disabled" if inputs.get(f"save-{name}-cache") == "false" else "on-success"
        if not selected:
            policy = "not-registered"
        path = steps.get(producer, {}).get("outputs", {}).get(output, "")
        # This is policy, not a save receipt: the native post action reports
        # exact-hit skips, successful archive sizes and upload failures later.
        print(f"[setup-pm cache] {family}: restore={outcome}, cache-hit={hit!r}, "
              f"save={policy}, path={path!r}, scope={scope}", flush=True)


if __name__ == "__main__":
    log_caches(json.loads(os.environ["_PM_CACHE_INPUTS"]), json.loads(os.environ["_PM_CACHE_STEPS"]),
               os.environ.get("GITHUB_REF", "unknown"))
