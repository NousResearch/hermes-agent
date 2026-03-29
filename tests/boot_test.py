import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))

BOOT_CLI = os.path.join(ROOT, "tools", "boot_orchestrator.py")


def run_check(runtime_dir):
    proc = subprocess.run([sys.executable, BOOT_CLI, "--check", "--path", runtime_dir], capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def write_yaml(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def test_no_files(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    rc, out, err = run_check(str(runtime))
    assert rc != 0
    data = json.loads(out)
    assert data["valid"] is False
    assert "no boot order files found" in data["errors"]


def test_good_and_bad_files(tmp_path):
    runtime = tmp_path / "runtime"
    boot_dir = runtime / "boot_order.d"
    boot_dir.mkdir(parents=True)
    good = boot_dir / "01_good.yaml"
    bad = boot_dir / "02_bad.yaml"
    write_yaml(str(good), "name: good\nsteps:\n  - a\n  - b\n")
    write_yaml(str(bad), "not: valid\n")
    rc, out, err = run_check(str(runtime))
    assert rc != 0
    data = json.loads(out)
    assert data["valid"] is False
    assert "02_bad.yaml" in data["files_report"]
    assert data["files_report"]["02_bad.yaml"]["valid"] is False
    assert data["files_report"]["01_good.yaml"]["valid"] is True


def test_parse_error(tmp_path):
    runtime = tmp_path / "runtime"
    boot_dir = runtime / "boot_order.d"
    boot_dir.mkdir(parents=True)
    bad = boot_dir / "01_bad.yaml"
    write_yaml(str(bad), "name: broken: :\nsteps: [a, b]\n")
    rc, out, err = run_check(str(runtime))
    assert rc != 0
    data = json.loads(out)
    assert data["valid"] is False
    assert "01_bad.yaml" in data["files_report"]
    assert data["files_report"]["01_bad.yaml"]["valid"] is False
    assert "parse error" in " ".join(data["errors"]).lower()
