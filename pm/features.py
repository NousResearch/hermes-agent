"""The enabled-features file: the exact extras the install carries.

The bundle is built with `uv sync --all-extras`, but WHICH extras
actually resolve differs per platform (markers gate some off). The
shipped default is the EXACT list that installed — written at bundle
time beside the payload, read at sync time:

- lazy installs OFF (security.allow_lazy_installs: false): the feature
  list is FROZEN to this file; pm sync never deviates from it and never
  installs a plugin dep.
- lazy installs ON: the file is the baseline; extras union on top.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

FEATURES_FILENAME = "enabled-features.json"


def features_path(base_dir: Optional[Path] = None) -> Path:
    """Where the enabled-features file lives. In a bundle, the payload
    root (beside manifest.json — bundle-written, sealed-shipped). At
    runtime, the runtime dir (beside the byte store — per-install,
    writable on every install kind). Same relative location on both:
    a sealed install's store_root() is <payload>/tools, and the file
    sits at the payload root = store_root().parent."""
    if base_dir is not None:
        return base_dir / FEATURES_FILENAME
    from pm.paths import store_root

    return store_root().parent / FEATURES_FILENAME


def write_features(extras: list[str], base_dir: Optional[Path] = None) -> Path:
    """Record the exact extras an install carries. Sorted, deduped."""
    path = features_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema": 1, "extras": sorted(set(extras))}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def read_features() -> Optional[list[str]]:
    """The frozen/baseline feature list; None when no file exists (source
    installs without a bundle — the baseline is the recorded venv
    state)."""
    try:
        data = json.loads(features_path().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    extras = data.get("extras")
    if not isinstance(extras, list):
        return None
    return sorted({str(e) for e in extras})


def declared_extras(repo_dir: Path) -> list[str]:
    """Every extra the repo's pyproject declares."""
    import tomllib

    with (repo_dir / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    return sorted(data.get("project", {}).get("optional-dependencies", {}))


def installed_extras(repo_dir: Path, venv_dir: Path) -> list[str]:
    """The extras that ACTUALLY installed on this target: every declared
    extra whose pm anchor import resolves in the venv. Markers-gated
    extras show up as missing anchors — the honest per-target record of
    what `uv sync --all-extras` put on THIS machine."""
    from pm.extras import ANCHORS

    installed: list[str] = []
    for extra in declared_extras(repo_dir):
        anchor = ANCHORS.get(extra, extra.replace("-", "_"))
        if isinstance(anchor, str):
            anchor = (anchor,)
        if all(_importable_in(anchor_member, venv_dir) for anchor_member in anchor):
            installed.append(extra)
    return sorted(installed)


def _importable_in(anchor: str, venv_dir: Path) -> bool:
    """Does the anchor import resolve inside venv_dir's site-packages?"""
    import importlib.util
    import sys
    import sysconfig
    from pathlib import Path as _P

    sites = set()
    pure = sysconfig.get_paths(vars={"base": str(venv_dir)}).get("purelib")
    if pure:
        sites.add(_P(pure))
    plat = sysconfig.get_paths(vars={"base": str(venv_dir)}).get("platlib")
    if plat:
        sites.add(_P(plat))

    saved = list(sys.path)
    try:
        sys.path = [str(s) for s in sites]
        try:
            return importlib.util.find_spec(anchor) is not None
        except (ImportError, ValueError):
            return False
    finally:
        sys.path = saved
