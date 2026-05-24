from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WATCHDOG = REPO_ROOT / "contrib" / "plugins" / "memorymunch" / "watchdog.py"
PLUGIN_FILES = ("__init__.py", "original_bridge.py", "readonly_recall.py", "plugin.yaml")


def _copy_vendor(dst: Path) -> None:
    src = REPO_ROOT / "contrib" / "plugins" / "memorymunch"
    dst.mkdir(parents=True, exist_ok=True)
    for name in PLUGIN_FILES:
        (dst / name).write_bytes((src / name).read_bytes())


def test_memorymunch_watchdog_passes_when_runtime_matches_repo(tmp_path):
    runtime = tmp_path / "runtime" / "memorymunch"
    _copy_vendor(runtime)
    proc = subprocess.run(
        [sys.executable, str(WATCHDOG), "--runtime-plugin-dir", str(runtime), "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["after"]["status"] == "PASS"
    assert payload["after"]["drift"] == []


def test_memorymunch_watchdog_detects_and_repairs_runtime_drift(tmp_path):
    runtime = tmp_path / "runtime" / "memorymunch"
    _copy_vendor(runtime)
    (runtime / "plugin.yaml").write_text("name: broken\n", encoding="utf-8")

    drift_proc = subprocess.run(
        [sys.executable, str(WATCHDOG), "--runtime-plugin-dir", str(runtime), "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert drift_proc.returncode == 2
    drift_payload = json.loads(drift_proc.stdout)
    assert "plugin.yaml" in drift_payload["before"]["drift"]

    repair_proc = subprocess.run(
        [sys.executable, str(WATCHDOG), "--runtime-plugin-dir", str(runtime), "--repair", "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert repair_proc.returncode == 0, repair_proc.stderr or repair_proc.stdout
    repair_payload = json.loads(repair_proc.stdout)
    assert repair_payload["repair_copied"] == list(PLUGIN_FILES)
    assert repair_payload["after"]["status"] == "PASS"
    assert repair_payload["after"]["drift"] == []
