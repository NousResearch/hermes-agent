"""A checkout's in-tree venv as a main-era installer or a developer built it, without building one.

Shared by the PM and update suites that test the pre-PM venv carry: PM reads that venv's
site-packages, never imports it, so a directory tree with the right files is the real input.
"""
from pathlib import Path


def site_packages(root: Path, name: str = "venv") -> Path:
    return root / name / "lib" / "python3.11" / "site-packages"


def main_era_venv(root: Path, *anchors: str, name: str = "venv") -> Path:
    """``root/<name>`` as a main-era installer left it: Hermes installed from a tree without ``pm``."""
    site = _hermes_installed(root, name, top_level=("agent", "hermes_cli", "tools"))
    for anchor in anchors:
        (site / anchor).mkdir(parents=True, exist_ok=True)
    return site


def pm_built_venv(root: Path, *anchors: str, name: str = ".venv") -> Path:
    """``root/<name>`` as ``pm.build_env --source . --out .venv`` builds it from a PM tree."""
    site = _hermes_installed(root, name, top_level=("agent", "hermes_cli", "pm", "tools"))
    for anchor in anchors:
        (site / anchor).mkdir(parents=True, exist_ok=True)
    return site


def _hermes_installed(root: Path, name: str, *, top_level: tuple[str, ...]) -> Path:
    site = site_packages(root, name)
    dist = site / "hermes_agent-0.0.0.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "top_level.txt").write_text("\n".join(top_level) + "\n", encoding="utf-8")
    return site
