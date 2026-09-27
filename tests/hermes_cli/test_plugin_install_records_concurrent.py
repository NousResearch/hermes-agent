"""Plugins installed a second apart both keep their install record.

The Desktop install card runs its rows a second apart. Each install reads
.install-metadata.json before its clone; by the time the second one publishes,
the first has already added its row. Publication applies only the installing
plugin's record to the CURRENT sidecar, so neither record is lost and the
second install is not refused.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

PROGRAM = '''
import json, sys
from pathlib import Path
from pm.publication import StagedPlugin
project, staged, target, old, new = sys.argv[1:6]
StagedPlugin({"staged": staged, "target": target, "target_digest": None,
              "old_metadata": json.loads(old), "new_metadata": json.loads(new)}).publish(Path(project))
'''


def _publish(tmp_path: Path, home: Path, name: str, old: dict, new: dict) -> subprocess.CompletedProcess:
    staged = tmp_path / f"staged-{name}"
    staged.mkdir()
    (staged / "plugin.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    return subprocess.run(
        [sys.executable, "-c", PROGRAM, str(project), str(staged), str(home / "plugins" / name),
         json.dumps(old), json.dumps(new)],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60)


def test_second_install_keeps_the_first_install_record(tmp_path):
    home = tmp_path / "home"
    (home / "plugins").mkdir(parents=True)
    metadata = home / "plugins" / ".install-metadata.json"
    first = {"source": "https://example.test/nvidia-app.git", "revision": "a" * 40, "pinned": False}
    second = {"source": "https://example.test/nvidia-broadcast.git", "revision": "b" * 40, "pinned": False}

    # Both installs read the sidecar before either published: it was empty for both.
    done = _publish(tmp_path, home, "nvidia-app", {}, {"nvidia-app": first})
    assert done.returncode == 0, done.stderr
    done = _publish(tmp_path, home, "nvidia-broadcast", {}, {"nvidia-broadcast": second})
    assert done.returncode == 0, done.stderr

    assert json.loads(metadata.read_text(encoding="utf-8")) == {"nvidia-app": first, "nvidia-broadcast": second}


def test_a_concurrent_change_to_the_same_plugin_record_is_still_refused(tmp_path):
    home = tmp_path / "home"
    (home / "plugins").mkdir(parents=True)
    metadata = home / "plugins" / ".install-metadata.json"
    metadata.write_text(json.dumps({"example": {"revision": "moved"}}) + "\n", encoding="utf-8")

    done = _publish(tmp_path, home, "example", {"example": {"revision": "old"}}, {"example": {"revision": "new"}})

    assert done.returncode != 0
    assert "metadata changed" in done.stderr
    assert json.loads(metadata.read_text(encoding="utf-8")) == {"example": {"revision": "moved"}}
