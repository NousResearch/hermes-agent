"""The Termux .deb job assembles the TUI on the runner host, not in the payload."""
from pathlib import Path

from ruamel.yaml import YAML

from scripts.ci.setup_toolchain import packages

ROOT = Path(__file__).resolve().parents[2]


def test_termux_deb_assembly_runs_on_the_locked_node():
    job = YAML(typ="base").load(
        (ROOT / ".github/workflows/desktop-bundled-release.yml").read_text(encoding="utf-8")
    )["jobs"]["termux-deb"]
    steps = job["steps"]
    (toolchain,) = [step for step in steps if step.get("uses") == "./.github/actions/setup-pm"]
    assert "node" in packages(toolchain["with"]["toolchain"]), (
        "scripts/termux/build.py prepares the locked ui-tui dependencies on the runner host and "
        "scripts/build/node-deps.mjs enforces package.json engines.node there; the payload Node "
        "is bionic and cannot run on the glibc runner, so the job must install the locked Node"
    )
    (assembly,) = [index for index, step in enumerate(steps) if step.get("name") == "Assemble the .deb"]
    assert steps.index(toolchain) < assembly
