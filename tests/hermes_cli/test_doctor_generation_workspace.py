"""A generation workspace resolves launch contracts through its owning install.

Reproduced live on a dual-layout host: PM materializes a generation as
``installs/<install-key>/environments/<gen>/{venv,workspace}``, where the
install key is a hash of the install root — a path that appears nowhere in
the generation path. A PATH shim inside ``venv/bin`` imports from
``workspace/``, so ``get_project_root()`` lands on the workspace. The
workspace has no facts.json, so the doctor check wanted ``workspace/hermes``
("Hermes entry point not found") and a launcher minted against the orphan
workspace boots to "no dependency environment is committed; run
`hermes pm repair`" — which itself cannot run inside the workspace (no
install stamp, no lockfile).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pm.environments import install_state_dir, owning_generation, owning_install_root

REPO_ROOT = Path(__file__).resolve().parents[2]


def _generation_tree(tmp_path, monkeypatch):
    """A real install plus one committed generation, under a temp home."""
    home = (tmp_path / "data").resolve()
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    install = tmp_path / "source"
    install.mkdir()
    (install / ".install_method").write_text("git", encoding="utf-8")

    gen_root = install_state_dir(install) / "environments" / "gen-a"
    venv = gen_root / "venv"
    workspace = gen_root / "workspace"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(Path(sys.executable).resolve())
    (venv / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    workspace.mkdir()
    (install_state_dir(install) / "facts.json").write_text(
        json.dumps({"packages": {"venv": {"environment": str(venv)}}}), encoding="utf-8"
    )
    return home, install, venv, workspace


def _store_python(home):
    """PM's committed store interpreter (what resolve_store_python reads)."""
    interpreter = home / "tools" / "python-fixture" / "bin" / "python3"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(sys._base_executable).resolve())
    (home / "tools" / "facts.json").write_text(
        json.dumps({"packages": {"python": {"entry": "python-fixture"}}}), encoding="utf-8"
    )
    return interpreter


def _materialized_workspace(install):
    """A workspace as PM's commit leaves it: build-input snapshots plus the
    console scripts its venv installed. Resolution runs from this shape."""
    gen = install_state_dir(install) / "environments" / "gen-a"
    workspace = gen / "workspace"
    for name in ("hermes", "hermes-agent"):
        (gen / "venv" / "bin" / name).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for relative in ("pm/__init__.py", "pm/paths.py", "hermes_bootstrap.py",
                     "hermes_constants.py", "hermes_cli/__init__.py"):
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, target)
    return workspace


_BODY_RESOLVE = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from pm.environments import owning_generation, owning_install_root\n"
    "w = Path(sys.argv[2]).resolve()\n"
    "print(json.dumps({'generation': str(owning_generation(w)),"
    " 'install': str(owning_install_root(w))}))\n"
)

_BODY_DOCTOR = (
    "import sys\n"
    "from argparse import Namespace\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from hermes_cli import doctor, doctor_platform\n"
    "doctor.PROJECT_ROOT = Path(sys.argv[2])\n"
    "doctor.HERMES_HOME = Path(sys.argv[3])\n"
    "doctor.DOCTOR_CHECKS = ((None, doctor_platform._check_command_installation),)\n"
    "doctor.run_doctor(Namespace(fix=False))\n"
)

_BODY_STAGE = (
    "import sys\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from hermes_cli import _launchers\n"
    "workspace = Path(sys.argv[2])\n"
    "out = Path(sys.argv[3])\n"
    "staged = _launchers.stage_launcher('hermes', workspace, out)\n"
    "print('STAGED', staged)\n"
    "if staged is not None:\n"
    "    print('SOURCE', (out / 'hermes').read_text(encoding='utf-8'))\n"
)


def _boot_from_generation(home, install, args, body):
    """Run *body* the way a process launched from inside the generation
    boots: cwd inside the generation, fixture HERMES_HOME, and an explicit
    HERMES_INSTALL_ROOT — the contract a real launch command carries when
    it boots a generation (the store interpreter itself names no tree)."""
    boot = home.parent / f"boot-{abs(hash(body))}.py"
    boot.write_text(body, encoding="utf-8")
    gen_bin = install_state_dir(install) / "environments" / "gen-a" / "venv" / "bin"
    env = {k: v for k, v in os.environ.items()
           if k not in ("HERMES_HOME", "HERMES_RUNTIME_DIR", "PREFIX")}
    env["HERMES_HOME"] = str(home)
    env["HERMES_INSTALL_ROOT"] = str(install)
    result = subprocess.run([sys.executable, str(boot), *args],
                            capture_output=True, text=True, timeout=120,
                            cwd=str(gen_bin), env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_workspace_recognizes_its_owning_generation(tmp_path, monkeypatch):
    _home, install, _venv, workspace = _generation_tree(tmp_path, monkeypatch)

    assert owning_generation(workspace) == workspace.parent
    # The install itself is its own contract owner; a plain checkout, a
    # stray tree, or a generation venv are never generation workspaces.
    assert owning_generation(install) is None
    assert owning_install_root(install) == install
    stray = tmp_path / "checkout"
    stray.mkdir()
    assert owning_generation(stray) is None
    assert owning_install_root(stray) is None


def test_the_generation_venv_is_not_mistaken_for_a_workspace(tmp_path, monkeypatch):
    """The environment beside the code copy is not the code copy."""
    _home, _install, venv, _workspace = _generation_tree(tmp_path, monkeypatch)
    assert owning_generation(venv) is None
    assert owning_install_root(venv) is None


def test_orphan_workspace_has_no_owner(tmp_path, monkeypatch):
    _home, install, _venv, workspace = _generation_tree(tmp_path, monkeypatch)
    (install_state_dir(install) / "facts.json").unlink()

    # Recognized as a generation, but nothing to delegate to.
    assert owning_generation(workspace) is not None
    assert owning_install_root(workspace) is None


@pytest.mark.platforms("posix")
def test_workspace_boots_to_its_owning_install(tmp_path, monkeypatch):
    """Resolution from a process launched from inside the generation — the
    real boot shape — finds the install that owns the workspace code copy,
    even though the workspace carries no install markers."""
    home, install, _venv, workspace = _generation_tree(tmp_path, monkeypatch)
    _materialized_workspace(install)
    result = _boot_from_generation(home, install,
                                   [str(REPO_ROOT), str(workspace)], _BODY_RESOLVE)
    data = json.loads(result.stdout)
    assert data["install"] == str(install)
    assert data["generation"] == str(workspace.parent)


@pytest.mark.platforms("posix")
def test_doctor_repoints_entry_point_check_to_the_owning_install(tmp_path, monkeypatch):
    """The symptom of the live bug: doctor booted from a generation
    workspace console script flags 'Hermes entry point not found' against
    the workspace. With the fix it resolves through the owning install."""
    home, install, _venv, workspace = _generation_tree(tmp_path, monkeypatch)
    _store_python(home)  # PM committed store -> pm_launcher contract
    _materialized_workspace(install)
    entry = install / "hermes"
    entry.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    entry.chmod(0o755)

    result = _boot_from_generation(home, install,
                                   [str(REPO_ROOT), str(workspace), str(home)],
                                   _BODY_DOCTOR)
    assert "Hermes entry point not found" not in result.stdout
    assert "Hermes entry point exists" in result.stdout
    assert str(install / "hermes") in result.stdout


@pytest.mark.platforms("posix")
def test_stage_launcher_from_workspace_uses_the_owning_store_python(tmp_path, monkeypatch):
    """Without the fix this minted a launcher booting to
    'no dependency environment is committed' — same symptom as the live bug."""
    home, install, _venv, workspace = _generation_tree(tmp_path, monkeypatch)
    _store_python(home)
    _materialized_workspace(install)

    link_dir = tmp_path / ".local" / "bin"
    link_dir.mkdir(parents=True)

    result = _boot_from_generation(home, install,
                                   [str(REPO_ROOT), str(workspace), str(link_dir)],
                                   _BODY_STAGE)
    source = result.stdout
    # Minted against the owning install's committed store interpreter...
    assert str(home / "tools" / "python-fixture" / "bin" / "python3") in source
    # ...while still executing the workspace code copy it was asked to bind.
    assert str(workspace.resolve()) in source
