"""Process death cannot leave new plugin code paired with the old dependency selection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("committed", [False, True])
def test_boot_recovers_plugin_publication_after_process_death(tmp_path, committed):
    home = tmp_path / "home"
    target = home / "plugins" / "example"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("old code", encoding="utf-8")
    metadata = target.parent / ".install-metadata.json"
    metadata.write_text('{"example":{"revision":"old"}}\n', encoding="utf-8")
    previous = metadata.read_bytes()
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "__init__.py").write_text("new code", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    program = '''
from pathlib import Path
import os,sys
from hermes_cli.plugins_transaction import PluginPublication
from hermes_cli.runtime_paths import runtime_facts_path
from pm.lock import Facts
project,staged,target = map(Path,sys.argv[1:4])
pub = PluginPublication(project,staged,target,{"example":{"revision":"new"}})
if sys.argv[4] == "True":
    Facts(runtime_facts_path(project)).record_state("venv","new",[])
os._exit(17)
'''
    child = subprocess.run([sys.executable, "-c", program, str(project), str(staged), str(target), str(committed)],
                           env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert child.returncode == 17, child.stderr
    recovery = '''
from pathlib import Path
import sys
from hermes_cli.runtime_state import runtime_lock,recover_publication
project=Path(sys.argv[1])
with runtime_lock(project):
    recover_publication(project)
    recover_publication(project)
'''
    result = subprocess.run([sys.executable, "-c", recovery, str(project)], env=env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert (target / "__init__.py").read_text(encoding="utf-8") == ("new code" if committed else "old code")
    assert json.loads(metadata.read_text(encoding="utf-8"))["example"]["revision"] == ("new" if committed else "old")
    if not committed:
        assert metadata.read_bytes() == previous
    assert not list(target.parent.glob(".previous-*"))
