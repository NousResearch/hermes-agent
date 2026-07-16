"""Tests for the ULW evidence ledger scaffold helper."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "scaffold_ulw_ledger.py"


def _load_scaffold():
    assert SCRIPT_PATH.exists(), "scripts/scaffold_ulw_ledger.py should exist"
    spec = importlib.util.spec_from_file_location("scaffold_ulw_ledger", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _symlink_or_skip(target: Path, link: Path, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable on this platform: {exc}")


def _windows_junction_or_skip(target: Path, link: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows junction coverage only applies on Windows")
    target_from_parent = os.path.relpath(target, link.parent)
    command = f"mklink /J {link.name} {target_from_parent}"
    proc = subprocess.run(
        ["cmd.exe", "/d", "/c", command],
        cwd=link.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).decode("cp949", errors="replace")
        pytest.skip(f"junction creation unavailable: {detail}")


def test_default_root_uses_hermes_home_without_runtime_config(monkeypatch, tmp_path):
    mod = _load_scaffold()
    hermes_home = tmp_path / "hermes-home"

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    assert mod.default_root() == hermes_home / "reports" / "ulw-loop"


def test_default_root_falls_back_without_hermes_runtime_helpers(monkeypatch, tmp_path):
    mod = _load_scaffold()
    monkeypatch.delenv("HERMES_HOME", raising=False)

    if os.name == "nt":
        local_app_data = tmp_path / "LocalAppData"
        monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
        expected = local_app_data / "hermes" / "reports" / "ulw-loop"
    else:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        expected = Path.home() / ".hermes" / "reports" / "ulw-loop"

    assert mod.default_root() == expected


def test_dry_run_reports_paths_without_writing(tmp_path):
    mod = _load_scaffold()

    result = mod.create_scaffold(
        root=tmp_path,
        run_id="demo-run",
        goal="demo goal",
        owner="HQ",
        dry_run=True,
    )

    assert result["run_id"] == "demo-run"
    assert result["dry_run"] is True
    assert result["files"]["brief"].endswith("brief.md")
    assert not (tmp_path / "demo-run").exists()


def test_normal_run_creates_parseable_scaffold(tmp_path):
    mod = _load_scaffold()

    result = mod.create_scaffold(
        root=tmp_path,
        run_id="demo-run",
        goal="demo goal",
        owner="HQ",
    )

    run_dir = tmp_path / "demo-run"
    assert result["created"] is True
    assert (run_dir / "brief.md").exists()
    assert (run_dir / "goals.json").exists()
    assert (run_dir / "ledger.jsonl").exists()
    assert (run_dir / "evidence" / "README.md").exists()

    goals = json.loads((run_dir / "goals.json").read_text(encoding="utf-8"))
    assert goals["run_id"] == "demo-run"
    assert goals["goal"] == "demo goal"
    assert goals["status"] == "planned"

    ledger_lines = (run_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger_lines) == 1
    event = json.loads(ledger_lines[0])
    assert event["event"] == "created"
    assert event["run_id"] == "demo-run"


def test_existing_run_id_refuses_to_overwrite_without_force(tmp_path):
    mod = _load_scaffold()
    mod.create_scaffold(root=tmp_path, run_id="demo-run", goal="demo goal")

    with pytest.raises(FileExistsError):
        mod.create_scaffold(root=tmp_path, run_id="demo-run", goal="demo goal")


def test_run_directory_symlink_is_rejected_before_resolve(tmp_path, monkeypatch):
    mod = _load_scaffold()
    root = tmp_path.resolve()
    requested_run_dir = root / "demo-run"
    path_type = type(root)
    original_is_symlink = path_type.is_symlink
    original_resolve = path_type.resolve

    def fake_is_symlink(self):
        if self == requested_run_dir:
            return True
        return original_is_symlink(self)

    def fail_if_requested_run_dir_resolves_first(self, *args, **kwargs):
        if self == requested_run_dir:
            raise AssertionError("requested run directory resolved before symlink check")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(path_type, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(path_type, "resolve", fail_if_requested_run_dir_resolves_first)

    with pytest.raises(ValueError, match="symlink"):
        mod.create_scaffold(root=root, run_id="demo-run", goal="demo goal", force=True)


def test_force_refuses_symlinked_run_directory(tmp_path):
    mod = _load_scaffold()
    target_dir = tmp_path / "target-run"
    target_dir.mkdir()
    _symlink_or_skip(target_dir, tmp_path / "demo-run", target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        mod.create_scaffold(root=tmp_path, run_id="demo-run", goal="demo goal", force=True)

    assert not (target_dir / "brief.md").exists()


def test_force_refuses_symlinked_scaffold_file(tmp_path):
    mod = _load_scaffold()
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    _symlink_or_skip(outside, run_dir / "brief.md")

    with pytest.raises(ValueError, match="symlink"):
        mod.create_scaffold(root=tmp_path, run_id="demo-run", goal="demo goal", force=True)

    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_force_refuses_symlinked_evidence_directory(tmp_path):
    mod = _load_scaffold()
    run_dir = tmp_path / "demo-run"
    outside_dir = tmp_path / "outside-evidence"
    run_dir.mkdir()
    outside_dir.mkdir()
    _symlink_or_skip(outside_dir, run_dir / "evidence", target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        mod.create_scaffold(root=tmp_path, run_id="demo-run", goal="demo goal", force=True)

    assert not (outside_dir / "README.md").exists()


def test_force_refuses_windows_junctioned_evidence_directory(tmp_path):
    mod = _load_scaffold()
    run_dir = tmp_path / "demo-run"
    outside_dir = tmp_path / "outside-evidence"
    run_dir.mkdir()
    outside_dir.mkdir()
    junction = run_dir / "evidence"
    _windows_junction_or_skip(outside_dir, junction)

    try:
        assert junction.is_symlink() is False
        with pytest.raises(ValueError, match="reparse|junction"):
            mod.create_scaffold(root=tmp_path, run_id="demo-run", goal="demo goal", force=True)
        assert not (outside_dir / "README.md").exists()
    finally:
        if junction.exists():
            subprocess.run(
                ["cmd.exe", "/d", "/c", f'rmdir "{junction.name}"'],
                cwd=junction.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )


def test_force_refuses_hardlinked_scaffold_file(tmp_path):
    mod = _load_scaffold()
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    hardlink = run_dir / "brief.md"
    try:
        os.link(outside, hardlink)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"hardlink creation unavailable on this platform: {exc}")

    assert hardlink.is_symlink() is False
    assert hardlink.stat().st_nlink > 1
    with pytest.raises(ValueError, match="hardlink"):
        mod.create_scaffold(root=tmp_path, run_id="demo-run", goal="demo goal", force=True)

    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_run_id_rejects_path_traversal_and_empty_values():
    mod = _load_scaffold()

    for bad in ["", ".", "..", "../bad", "bad/name", r"bad\\name"]:
        with pytest.raises(ValueError):
            mod.validate_run_id(bad)


def test_cli_dry_run_json_is_parseable_and_does_not_write(tmp_path):
    assert SCRIPT_PATH.exists(), "scripts/scaffold_ulw_ledger.py should exist"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--root",
            str(tmp_path),
            "--run-id",
            "demo-cli",
            "--goal",
            "demo goal",
            "--dry-run",
            "--format",
            "json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    data = json.loads(proc.stdout)
    assert data["run_id"] == "demo-cli"
    assert data["dry_run"] is True
    assert not (tmp_path / "demo-cli").exists()
