"""Build scripts still run main() when the checkout is reached through a symlink (#157761).

Node resolves symlinks in ``import.meta.url`` but not in ``process.argv[1]``,
so a plain URL comparison made every entry point a silent no-op that exited 0.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

BUILD = Path(__file__).resolve().parents[2] / "scripts" / "build"


@pytest.fixture
def node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    return node


@pytest.fixture
def linked_build(tmp_path):
    link = tmp_path / "linked-build"
    link.symlink_to(BUILD, target_is_directory=True)
    return link


def _run(node, args, cwd, env=None):
    return subprocess.run([node, *args], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=60, check=False)


def test_frontend_common_is_main_through_symlink(node, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    common = (BUILD / "frontend-common.mjs").as_uri()
    (real / "probe.mjs").write_text(
        f"import {{ isMain }} from {json.dumps(common)}\n"
        "console.log(JSON.stringify(isMain(import.meta.url)))\n", encoding="utf-8")
    via_link = _run(node, [str(tmp_path / "link" / "probe.mjs")], tmp_path)
    direct = _run(node, [str(real / "probe.mjs")], tmp_path)
    assert via_link.stdout.strip() == "true", via_link.stdout + via_link.stderr
    assert direct.stdout.strip() == "true", direct.stdout + direct.stderr


def test_frontend_common_is_main_false_when_imported(node, tmp_path):
    common = (BUILD / "frontend-common.mjs").as_uri()
    code = (f"import {{ isMain }} from {json.dumps(common)}\n"
            "console.log(JSON.stringify(isMain(import.meta.url)))\n")
    result = _run(node, ["--input-type=module", "-e", code, "not-a-file"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "false"


@pytest.mark.parametrize("script", ["node-deps.mjs", "electron-runtime.mjs"])
def test_entry_points_run_main_through_symlink(node, linked_build, tmp_path, script):
    # Without --source, main() refuses; before the fix it never ran and exited 0.
    result = _run(node, [str(linked_build / script)], tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "--source is required" in result.stderr


def test_python_entry_point_runs_through_symlink(node, linked_build, tmp_path):
    marker = tmp_path / "ran"
    env = {"HERMES_PYTHON": sys.executable, "PATH": str(Path(node).parent)}
    result = _run(node, [str(linked_build / "python.mjs"), "-c",
                         f"open({str(marker)!r}, 'w').close()"], tmp_path, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.exists()
