"""The canonical runner can use explicitly granted scratch without changing isolation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_tests_parallel as runner


@pytest.mark.parametrize("kind", ["relative", "missing", "file"])
def test_scratch_root_rejects_invalid_directory(tmp_path, kind):
    path = tmp_path / "not-directory"
    if kind == "file":
        path.write_text("not a directory")
    value = "relative-directory" if kind == "relative" else str(path)
    with pytest.raises(argparse.ArgumentTypeError, match="scratch root"):
        runner._scratch_root_arg(value)


def test_scratch_root_accepts_existing_directory_without_creating_children(tmp_path):
    before = set(tmp_path.iterdir())
    assert runner._scratch_root_arg(str(tmp_path)) == str(tmp_path.resolve())
    assert set(tmp_path.iterdir()) == before


def test_cli_scratch_root_reaches_each_isolated_child_and_is_cleaned(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    probes = tmp_path / "probes"
    probes.mkdir()
    config = probes / "pytest.ini"
    config.write_text("[pytest]\n")
    observations = []
    for number in range(2):
        observation = tmp_path / f"child-{number}.json"
        observations.append(observation)
        (probes / f"test_child_{number}.py").write_text(
            "import json,os,tempfile\nfrom pathlib import Path\n"
            "def test_observe():\n"
            f"    Path({str(observation)!r}).write_text(json.dumps("
            "{'root':os.environ['PYTEST_DEBUG_TEMPROOT'],'tmp':tempfile.gettempdir()}))\n"
        )
    script = Path(runner.__file__).resolve()
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    result = subprocess.run(
        [sys.executable, str(script), "--scratch-root", str(scratch), "-j", "2",
         "--file-retries", "0", str(probes), "--", "-q", "-c", str(config)],
        cwd=script.parents[1], env=env, capture_output=True, text=True,
        encoding="utf-8", timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = [json.loads(p.read_text()) for p in observations]
    assert data[0]["root"] != data[1]["root"]
    for child in data:
        assert Path(child["root"]).parent == scratch
        assert child["tmp"] == child["root"]
        assert not Path(child["root"]).exists()
    assert {p.name for p in scratch.iterdir()} == {runner._DURATIONS_FILE}
    assert len(json.loads((scratch / runner._DURATIONS_FILE).read_text())) == 2
    assert "2 tests passed" in result.stdout
