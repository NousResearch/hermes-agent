"""scripts/ci/check_retired_shims_manifest.py keeps the tested shim manifest complete."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/ci/check_retired_shims_manifest.py"


def _run(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args, str(root)], capture_output=True,
                          text=True, timeout=60, check=False, stdin=subprocess.DEVNULL)


def _tree(root: Path, shims: list) -> None:
    (root / "tools").mkdir(parents=True)
    (root / "tools/old.py").write_text("def gone():\n    stop_historical_updater('gone')\n"
                                       "def fine():\n    return 1\n", encoding="utf-8")
    (root / "tests/fixtures").mkdir(parents=True)
    (root / "tests/fixtures/retired_updater_shims.json").write_text(json.dumps(shims), encoding="utf-8")


def test_a_new_shim_missing_from_the_manifest_fails(tmp_path):
    _tree(tmp_path, [["hermes_cli.main", "_write_update_incomplete_marker"]])
    proc = _run(tmp_path)
    assert proc.returncode == 1 and "tools.old.gone is not in" in proc.stdout, proc.stdout


def test_a_complete_manifest_passes(tmp_path):
    _tree(tmp_path, [["hermes_cli.main", "_write_update_incomplete_marker"], ["tools.old", "gone"]])
    proc = _run(tmp_path)
    assert proc.returncode == 0, proc.stdout


def test_an_empty_scan_never_passes(tmp_path):
    proc = _run(tmp_path)
    assert proc.returncode == 2, proc.stdout
