"""Fresh takeover bootstrap prepares actual dependencies before completion."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from pm.lock import Facts, Lockfile
from pm.store import current_target, tree_digest
from tests.pm._fixtures import _wheel


@pytest.mark.platforms("linux")
def test_fresh_takeover_prepares_generation_and_runs_selected_python(tmp_path):
    source = Path(__file__).resolve().parents[2]
    root = tmp_path / "source with spaces"
    root.mkdir()
    for name in ("pm", "hermes_cli"):
        shutil.copytree(source / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("hermes_constants.py", "hermes_yaml.py", "utils.py", "hermes_bootstrap.py"):
        shutil.copy2(source / name, root / name)
    home, store, wheels = (tmp_path / name for name in ("home", "tools", "wheels"))
    for directory in (home, store, wheels):
        directory.mkdir()
    _wheel(wheels, "takeover_dep", "1.0")
    (root / "pyproject.toml").write_text(
        '[project]\nname="takeover-fixture"\nversion="1"\nrequires-python=">=3.14"\n'
        'dependencies=["takeover-dep==1.0"]\n[project.optional-dependencies]\nall=[]\n'
        '[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8")
    (root / ".git").mkdir()
    (root / "install-stamp.json").write_text('{"updateMechanism":"self"}', encoding="utf-8")
    uv = shutil.which("uv")
    assert uv
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HERMES_", "PYTHON", "UV_"))}
    env.update(HOME=str(home), HERMES_HOME=str(home), HERMES_RUNTIME_DIR=str(store), UV_PYTHON_DOWNLOADS="never")
    subprocess.run([uv, "lock", "--offline", "--python", sys.executable], cwd=root, env=env, check=True, capture_output=True)
    # Only the interpreter and private manager tool are needed by this tiny
    # application. They are real host tools; PM's worker and resolver stay real.
    (root / "pm/lock.json").write_text('{"schema":1,"packages":{}}', encoding="utf-8")
    lock = Lockfile(root / "pm/lock.json")
    target = current_target()
    for name, executable, rel in (("python", sys._base_executable, "bin/python3"), ("uv", uv, "uv")):
        entry = store / name
        binary = entry / rel
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.symlink_to(Path(executable).resolve())
        if name == "uv":
            (entry / "uvx").symlink_to(Path(uv).resolve().with_name("uvx"))
        lock.set_pin(name, "fixture", {})
        Facts(store / "facts.json").record(name, "fixture", name, {"PATH": [str(binary.parent)]}, store,
                                          target=target, digest=tree_digest(entry))
    lock.save()
    # Probe the selected-interpreter boundary; app completion orchestration has
    # its own contract tests. This script cannot import the dep from the parent.
    (root / "hermes_cli/update_finish.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "request=json.loads(Path(sys.argv[1]).read_text())\n"
        "sys.path.insert(0,request['root'])\n"
        "from hermes_cli.runtime_paths import activate_dependencies\n"
        "activate_dependencies(Path(request['root']))\nimport takeover_dep\n"
        "assert request['pm_receipt']['update_id']==request['update_id']\n"
        "Path(sys.argv[2]).write_text(json.dumps({'python':sys.executable,'dep':takeover_dep.__file__}))\n",
        encoding="utf-8")
    context, result = tmp_path / "context.json", tmp_path / "result.json"
    context.write_text(json.dumps({"root": str(root), "home": str(home)}), encoding="utf-8")
    completed = subprocess.run([sys.executable, "-I", "-S", "-B", str(root / "hermes_cli/_update_takeover.py"), str(context), str(result)],
                               env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = json.loads(result.read_text())
    assert Path(output['python']).is_relative_to(store)
    assert Path(output['dep']).is_relative_to(home / "installs")
    assert (root / ".hermes/bin/hermes").is_file()
    assert not (root / "venv").exists()

    # Repair restores the recorded graph; a checkout update must also advance
    # that graph to the new source inputs before normal bootstrap checks it.
    facts = next((home / "installs").glob("*/facts.json"))
    first_stamp = json.loads(facts.read_text())["packages"]["venv"]["stamp"]
    (facts.parent / ".repair-incomplete").write_text("{}", encoding="utf-8")
    with (root / "uv.lock").open("a", encoding="utf-8") as changed:
        changed.write("\n# changed source inputs\n")
    repaired = subprocess.run([sys.executable, "-I", "-S", "-B", str(root / "hermes_cli/_update_takeover.py"), str(context), str(result)],
                              env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert repaired.returncode == 0, repaired.stdout + repaired.stderr
    assert json.loads(facts.read_text())["packages"]["venv"]["stamp"] != first_stamp
    assert not (facts.parent / ".repair-incomplete").exists()

    # A child which dies before acknowledging receipt ownership must not
    # leave the parent reporting success from its pre-handoff receipt.
    (root / "hermes_cli/update_finish.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    result.unlink()
    crashed = subprocess.run([sys.executable, "-I", "-S", "-B", str(root / "hermes_cli/_update_takeover.py"), str(context), str(result)],
                             env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert crashed.returncode == 7
    assert json.loads(result.read_text())["receipt_handled"]
    crash_receipt = json.loads((home / "logs/update_receipts/latest.json").read_text())
    assert crash_receipt["exit_code"] == 7 and crash_receipt["outcome"] == "failed"

    # A failed preparation must not delegate to application completion or
    # replace the working generation, and must leave a truthful receipt.
    facts = next((home / "installs").glob("*/facts.json"))
    before = facts.read_bytes()
    (root / "uv.lock").write_text("not valid TOML [", encoding="utf-8")
    result.unlink()
    failed = subprocess.run([sys.executable, "-I", "-S", "-B", str(root / "hermes_cli/_update_takeover.py"), str(context), str(result)],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert failed.returncode != 0
    assert facts.read_bytes() == before
    failure = json.loads(result.read_text())
    assert failure["receipt_handled"] is True
    latest = json.loads((home / "logs/update_receipts/latest.json").read_text())
    assert latest["outcome"] == "failed"
