"""Bind CI imports to the verified worktree, independent of worker launch state."""
import os
from pathlib import Path


def ci_environment(worktree: Path, additions=None):
    root = worktree.resolve()
    roots = [root]
    scripts = (root / "scripts").resolve()
    if scripts.is_dir() and scripts.is_relative_to(root):
        roots.append(scripts)
    env = dict(os.environ)
    env.update(additions or {})
    # Replace inherited paths from the controller or another worktree. This
    # explicitly admits only the checked repository and its CLI helper folder;
    # PYTHONSAFEPATH remains in force for every other implicit import location.
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in roots)
    return env
