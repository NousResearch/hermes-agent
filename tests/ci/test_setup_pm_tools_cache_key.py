"""setup-pm's tools-cache key must not hash bytecode the prepare step writes."""

import re
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]


def test_tools_cache_key_ignores_pycache_and_matches_restore_key():
    """prepare imports pm, so pm/__pycache__ exists by the time hashFiles runs.

    Its .pyc files embed per-checkout source mtimes, so hashing them gives every
    job a new key. Restore-only jobs must look up the key saving jobs write.
    """
    action = YAML(typ="base").load(
        (ROOT / ".github/actions/setup-pm/action.yml").read_text(encoding="utf-8")
    )
    steps = {step.get("id"): step for step in action["runs"]["steps"]}
    save = steps["tools-cache"]["with"]["key"]
    restore = steps["tools-cache-restore"]["with"]["key"]
    assert save == restore
    hashed = re.search(r"hashFiles\(([^)]*)\)", save)
    assert hashed, save
    assert "'!**/__pycache__/**'" in hashed.group(1), save
