"""A checkout's in-tree venv as a main-era installer or a developer built it, without building one.

Shared by the PM and update suites that test the pre-PM venv carry: PM reads that venv's
``pyvenv.cfg`` and site-packages, never imports it, so a directory tree with the right files is
the real input.
"""
import json
from pathlib import Path

_TOP_LEVEL = ("agent", "hermes_cli", "tools")


def site_packages(root: Path, name: str = "venv") -> Path:
    return root / name / "lib" / "python3.11" / "site-packages"


def main_era_venv(root: Path, *anchors: str, name: str = "venv", reinstalled_from_pm_tree: bool = False) -> Path:
    """``root/<name>`` as a main-era installer left it: Hermes installed on a uv-managed Python.

    ``reinstalled_from_pm_tree``: a pre-PM updater then ran ``uv pip install -e .[all]`` from the
    tree it had just pulled, so the dist-info is the PM tree's (its top level lists ``pm``).
    """
    home = root.parent / "uv-python" / "cpython-3.11.9-linux-x86_64-gnu" / "bin"
    top_level = (*_TOP_LEVEL, "pm") if reinstalled_from_pm_tree else _TOP_LEVEL
    return _hermes_installed(root, name, anchors, home=home, top_level=top_level)


def pm_built_venv(root: Path, *anchors: str, name: str = ".venv") -> Path:
    """``root/<name>`` as ``pm.build_env --source . --out .venv`` built it before PM marked its
    environments: on PM's pinned Python, from a tool store beside *root*."""
    store = root.parent / "pm-tools"
    entry = store / "python-3.14.9+20261021-linux-x64"
    (entry / "bin").mkdir(parents=True, exist_ok=True)
    (store / "facts.json").write_text(json.dumps({"schema": 1, "packages": {"python": {
        "entry": entry.name, "target": "linux-x64", "version": "3.14.9+20261021"}}}), encoding="utf-8")
    return _hermes_installed(root, name, anchors, home=entry / "bin", top_level=(*_TOP_LEVEL, "pm"))


def _hermes_installed(root: Path, name: str, anchors, *, home: Path, top_level: tuple[str, ...]) -> Path:
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "pyvenv.cfg").write_text(f"home = {home}\ninclude-system-site-packages = false\n",
                                            encoding="utf-8")
    site = site_packages(root, name)
    dist = site / "hermes_agent-0.0.0.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "top_level.txt").write_text("\n".join(top_level) + "\n", encoding="utf-8")
    for anchor in anchors:
        (site / anchor).mkdir(parents=True, exist_ok=True)
    return site
