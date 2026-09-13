"""Launch-time dependency selection survives a change after launcher minting."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_cli import _launchers
from hermes_cli.runtime_paths import install_state_dir, site_packages


@pytest.mark.platforms("windows")
def test_minted_launcher_reads_current_selection_and_editable_members(tmp_path, monkeypatch):
    from hermes_cli import runtime_paths, runtime_state
    import hermes_constants

    root = tmp_path / "repo"
    package = root / "hermes_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    # Real selection code, with a fixture entry point rather than a live CLI.
    (package / "runtime_paths.py").write_bytes(Path(runtime_paths.__file__).read_bytes())
    (package / "runtime_state.py").write_bytes(Path(runtime_state.__file__).read_bytes())
    (root / "hermes_constants.py").write_bytes(Path(hermes_constants.__file__).read_bytes())
    (root / "hermes_bootstrap.py").write_text(
        "from pathlib import Path\nfrom hermes_cli.runtime_paths import activate_dependencies\n"
        "activate_dependencies(Path(__file__).resolve().parent)\n"
    )
    (package / "main.py").write_text(
        "import selection_probe, editable_probe\n"
        "def main():\n    print(selection_probe.VALUE, editable_probe.VALUE)\n    return 0\n"
    )
    base = site_packages(root / "venv")
    base.mkdir(parents=True)
    (base / "selection_probe.py").write_text("VALUE = 'base'\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    state = install_state_dir(root)
    out = tmp_path / "bin"
    out.mkdir()
    launcher = _launchers.mint_launcher("hermes", root, out, Path(sys.executable), base)
    assert launcher is not None

    selected = state / "environments" / "selected" / "venv"
    site = site_packages(selected)
    site.mkdir(parents=True)
    (selected / "pyvenv.cfg").write_text("home = fixture\n")
    (site / "selection_probe.py").write_text("VALUE = 'selected'\n")
    editable = tmp_path / "editable"
    editable.mkdir()
    (editable / "editable_probe.py").write_text("VALUE = 'editable'\n")
    (site / "member.pth").write_text(str(editable) + "\n")
    (state / "facts.json").write_text(json.dumps({"packages": {"venv": {"environment": str(selected)}}}))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    result = subprocess.run([str(launcher)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "selected editable"
