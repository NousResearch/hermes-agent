"""Hosted site processing is a code load: a ``.pth`` file executes import-time code.

``site.addsitedir``/``site.addpackage`` are not path bookkeeping — processing a site
directory executes every ``.pth`` file in it, and a ``.pth`` line can read and ``exec``
any file on the disk. Two real loaders feed those calls from inputs the hosted fence
treats as untrusted: PM's dependency activation selects the directory from install
state under the deployment's data root (during ``import gateway.run``, BEFORE the start
command's fence arms), and the Windows gateway overlay takes its first candidate from
``VIRTUAL_ENV``. Each case here plants a data-side environment whose ``.pth`` writes a
marker file, then drives the real code path in a ``python -I`` child: the marker must
not exist afterwards, and the child's output must name the provenance refusal.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The child's platform-native data root moves off the account's: PM scopes dependency
# state to the data root a process runs under, and a HERMES_HOME nested under the
# native root collapses back onto it, so a probe that reuses the account's native root
# would read and write the real install's dependency state. The suffix keeps this
# test's planted state inside its own temporary home (see pm.dependency_home_root).
_CHILD_DATA_SUFFIX = "-hosted-site-probe"


def _child_env(home: Path) -> dict[str, str]:
    return {**os.environ,
            "HERMES_HOME": str(home),
            "HERMES_DATA_DIR_SUFFIX": _CHILD_DATA_SUFFIX,
            "UV_OFFLINE": "1"}


def _run_child(script: str, home: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-I", "-c", script], cwd=str(REPO_ROOT),
                          env=_child_env(home), capture_output=True, text=True,
                          timeout=240, check=False)


def _planted_state_script(home: Path, marker: Path) -> str:
    """A child script that plants a committed PM environment whose site-packages holds
    a ``.pth`` executing ``payload.py`` (which writes *marker*). The ``.pth`` also
    re-extends ``sys.path`` with the child's original entries so the planted environment
    alone cannot starve a later import of its image dependencies — the marker write is
    the behaviour under test, not the environment's usefulness."""
    return f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(REPO_ROOT)!r})
from pm import environments as e
root = Path({str(REPO_ROOT)!r})
state = e.install_state_dir(root)
env = state / "environments" / "probe"
site = e.site_packages(env)
site.mkdir(parents=True)
(env / "pyvenv.cfg").write_text("version = " + ".".join(map(str, sys.version_info[:3])) + "\\n")
payload = Path({str(home)!r}) / "payload.py"
payload.write_text("from pathlib import Path\\nPath(" + repr({str(marker)!r}) + ').write_text("executed")\\n')
(site / "probe.pth").write_text(
    "import sys; exec(open(" + repr(str(payload)) + ").read()); sys.path.extend(" + repr(list(sys.path)) + ")\\n")
(state / "facts.json").write_text(json.dumps({{"packages": {{"venv": {{"environment": str(env)}}}}}}))
"""


@pytest.mark.parametrize("variant", ["direct", "bootstrap", "start_command"])
def test_a_data_side_dependency_environment_never_runs_its_pth(tmp_path, variant):
    """PM's dependency activation must judge the selected site directory against the
    deployment image before processing it. ``direct`` calls the activation with the fence
    armed and the hosted verdict frozen; ``bootstrap`` performs a real ``import
    gateway.run`` with the verdict frozen (the fence inside that import arms AFTER the
    bootstrap's activation); ``start_command`` is the real boot order — the hosted flag
    alone, no recorded verdict, no armed fence before the activation runs."""
    home = tmp_path / "home"
    home.mkdir()
    marker = home / "executed"

    script = _planted_state_script(home, marker)
    if variant == "direct":
        script += """
sys.argv = ["gateway.run", "--hosted"]
from hermes_cli.hosted_code_fence import arm_from_start_argv, hosted_code_refusal
arm_from_start_argv()
from hermes_cli import hosted_mode
hosted_mode.record_boot_config({"hosted": {"enabled": True}})
assert hosted_code_refusal(site)  # the control: the predicate refuses this directory
e.activate_dependencies(root)
"""
    elif variant == "bootstrap":
        script += """
from hermes_cli import hosted_mode
hosted_mode.record_boot_config({"hosted": {"enabled": True}})
sys.argv = ["gateway.run", "--hosted"]
import gateway.run
"""
    else:
        script += """
sys.argv = ["gateway.run", "--hosted"]
import gateway.run
"""

    result = _run_child(script, home)
    output = result.stdout + result.stderr
    assert not marker.exists(), (
        f"a data-side dependency environment's .pth executed during {variant}:\n{output}")
    assert "resolves outside that image" in output, (
        f"the activation must refuse with the provenance error, not another failure:\n{output}")


def test_the_windows_venv_overlay_never_processes_a_data_side_site(tmp_path):
    """The Windows gateway overlay's first candidate comes from ``VIRTUAL_ENV`` — an
    environment variable, not the deployment interpreter's own venv. A data-side
    ``Lib/site-packages`` with a ``.pth`` that executes a payload file must be refused
    before ``addsitedir`` processes it (the branch is exercised on this host by
    switching ``sys.platform``; the directory layout and the guard are the real ones)."""
    home = tmp_path / "home"
    home.mkdir()
    marker = home / "executed"

    script = f"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, {str(REPO_ROOT)!r})
payload = Path({str(home)!r}) / "payload.py"
payload.write_text("from pathlib import Path\\nPath(" + repr({str(marker)!r}) + ').write_text("executed")\\n')
sys.argv = ["gateway.run", "--hosted"]
import gateway.run as gateway
from hermes_cli.hosted_code_fence import arm_from_start_argv, hosted_fence_installed
arm_from_start_argv()
from hermes_cli import hosted_mode
hosted_mode.record_boot_config({{"hosted": {{"enabled": True}}}})
assert hosted_fence_installed()
win = Path({str(home)!r}) / "winvenv"
site_dir = win / "Lib" / "site-packages"
site_dir.mkdir(parents=True)
(win / "pyvenv.cfg").write_text("version = " + ".".join(map(str, sys.version_info[:3])) + "\\n")
(site_dir / "probe.pth").write_text("import os; exec(open(" + repr(str(payload)) + ").read())\\n")
os.environ["VIRTUAL_ENV"] = str(win)
sys.platform = "win32"
gateway._ensure_windows_gateway_venv_imports()
print("OVERLAY_COMPLETED", flush=True)
"""

    result = _run_child(script, home)
    output = result.stdout + result.stderr
    assert "OVERLAY_COMPLETED" in output, f"the overlay probe must run to completion:\n{output}"
    assert not marker.exists(), f"a VIRTUAL_ENV-selected site's .pth executed:\n{output}"
