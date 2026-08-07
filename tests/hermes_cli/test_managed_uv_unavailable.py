from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


_ISSUE_REPRODUCTION_FALLBACK = {
    "platform": "Windows 11",
    "command": "hermes update",
    "python_version": "3.11.15",
    "sqlite_version": "3.50.4",
    "journal_mode": "delete",
    "quick_check": "ok",
    "uv_catalog": [
        "cpython-3.11.15-windows-x86_64-none",
        "cpython-3.11.14-windows-x86_64-none",
    ],
}


def _issue_reproduction() -> dict:
    fixture_name = os.environ.get("HERMES_76106_REPRO")
    if fixture_name:
        return json.loads(Path(fixture_name).read_text(encoding="utf-8"))
    return _ISSUE_REPRODUCTION_FALLBACK


def _runtime_info(executable: Path, sqlite=(3, 50, 4), *, source="vulnerable"):
    from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo

    return SQLiteRuntimeInfo(
        executable=executable,
        base_prefix=executable.parent.parent,
        python_version=(3, 11, 15),
        sqlite_version=sqlite,
        sqlite_version_string=".".join(str(part) for part in sqlite),
        sqlite_source_id=source,
    )


def _runtime_install(tmp_path: Path):
    root = tmp_path / "checkout"
    (root / "venv" / "bin").mkdir(parents=True)
    (root / "venv" / "Scripts").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    live_python = root / "venv" / "bin" / "python"
    live_python.write_text("live", encoding="utf-8")
    (root / "venv" / "Scripts" / "python.exe").write_text("live", encoding="utf-8")
    (root / "venv" / "sentinel").write_text("live", encoding="utf-8")
    return root, live_python


def _uv_process(monkeypatch, *, uv_version, candidate_sqlite):
    import hermes_cli.managed_uv as managed_uv

    calls = []
    state = {"generation": None}

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1:] == ["--version"]:
            return SimpleNamespace(returncode=0, stdout=f"uv {uv_version}\n", stderr="")
        if "install" in command:
            state["generation"] = Path(kwargs["env"]["UV_PYTHON_INSTALL_DIR"])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "find" in command:
            python = state["generation"] / "cpython" / "bin" / "python3"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text(command[3], encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout=str(python), stderr="")
        if "list" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([
                    {
                        "implementation": "cpython",
                        "variant": "default",
                        "version_parts": {"major": 3, "minor": 11, "patch": 15},
                    }
                ]),
                stderr="",
            )
        raise AssertionError(f"unexpected uv command: {command}")

    def fake_probe(executable, **_kwargs):
        executable = Path(executable)
        if executable.name in ("python", "python.exe") and executable.parent.parent.name == "venv":
            return _runtime_info(executable)
        return _runtime_info(
            executable,
            candidate_sqlite,
            source="fixed" if candidate_sqlite != (3, 50, 4) else "vulnerable",
        )

    monkeypatch.setattr(managed_uv.subprocess, "run", fake_run)
    monkeypatch.setattr(managed_uv, "probe_sqlite_runtime", fake_probe)
    return fake_run, calls


def _windows_hermetic(monkeypatch):
    import hermes_cli.managed_uv as managed_uv

    monkeypatch.setattr(managed_uv.platform, "system", lambda: "Windows")
    monkeypatch.setattr(managed_uv.platform, "machine", lambda: "AMD64")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.main",
        SimpleNamespace(_detect_venv_python_processes=lambda: []),
    )


def test_issue_76106_repeated_update_skips_same_vulnerable_artifact(tmp_path, monkeypatch, capsys):
    """The issue artifact drives two real repair/reporting invocations."""
    import hermes_cli.managed_uv as managed_uv

    fixture = _issue_reproduction()
    assert fixture["platform"] == "Windows 11"
    assert fixture["python_version"] == "3.11.15"
    assert fixture["sqlite_version"] == "3.50.4"
    _windows_hermetic(monkeypatch)
    root, live_python = _runtime_install(tmp_path)
    _, calls = _uv_process(monkeypatch, uv_version="0.8.4", candidate_sqlite=(3, 50, 4))
    refresh = Mock(return_value=False)
    monkeypatch.setattr(managed_uv, "_refresh_managed_uv_catalog", refresh)

    first = managed_uv.repair_vulnerable_runtime("uv.exe", project_root=root)
    managed_uv._report_runtime_repair_failure(first)
    first_output = capsys.readouterr().out

    second = managed_uv.repair_vulnerable_runtime("uv.exe", project_root=root)
    managed_uv._report_runtime_repair_failure(second)
    second_output = capsys.readouterr().out

    installs = [call for call in calls if "install" in call]
    assert first.status == "unavailable"
    assert second.status == "unavailable"
    assert len(installs) == 1
    assert refresh.call_count == 1
    assert "Python 3.11.15" in first_output
    assert "SQLite 3.50.4" in first_output
    assert "current uv selection has no fixed candidate" in second_output
    assert "Provisioning a private Python" not in second_output
    assert live_python.read_text(encoding="utf-8") == "live"
    assert (root / ".hermes-runtime" / "python" / "unavailable-artifact.json").is_file()


def test_changed_uv_accepts_fixed_same_version_artifact(tmp_path, monkeypatch):
    import hermes_cli.managed_uv as managed_uv

    _windows_hermetic(monkeypatch)
    root, _ = _runtime_install(tmp_path)
    _, calls = _uv_process(monkeypatch, uv_version="0.8.4", candidate_sqlite=(3, 50, 4))
    monkeypatch.setattr(managed_uv, "_refresh_managed_uv_catalog", lambda *_: False)
    first = managed_uv.repair_vulnerable_runtime("uv.exe", project_root=root)
    assert first.status == "unavailable"

    candidate = root / ".hermes-runtime" / "venv-candidate"
    _uv_process(
        monkeypatch, uv_version="0.8.5", candidate_sqlite=(3, 53, 1)
    )
    staged = Mock(return_value=candidate)
    monkeypatch.setattr(managed_uv, "_stage_candidate_venv", staged)
    monkeypatch.setattr(
        managed_uv,
        "_cut_over_candidate",
        Mock(return_value=(True, None, _runtime_info(candidate / "bin" / "python", (3, 53, 1), source="fixed"), "")),
    )

    second = managed_uv.repair_vulnerable_runtime("uv.exe", project_root=root)
    assert second.status == "repaired"
    staged.assert_called_once()
    assert not (root / ".hermes-runtime" / "python" / "unavailable-artifact.json").exists()


def test_uncertain_bare_candidate_does_not_block_fixed_patch_retry(tmp_path, monkeypatch):
    import hermes_cli.managed_uv as managed_uv
    from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo

    root, _ = _runtime_install(tmp_path)
    current = _runtime_info(root / "venv" / "bin" / "python")
    state = {"generation": None, "requests": []}

    def fake_run(command, **kwargs):
        if "install" in command:
            state["generation"] = Path(kwargs["env"]["UV_PYTHON_INSTALL_DIR"])
            state["requests"].append(command[3])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "find" in command:
            python = state["generation"] / "cpython" / "bin" / "python3"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text(command[3], encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout=str(python), stderr="")
        if "list" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([
                    {
                        "implementation": "cpython",
                        "variant": "default",
                        "version_parts": {"major": 3, "minor": 11, "patch": 16},
                    }
                ]),
                stderr="",
            )
        raise AssertionError(command)

    def fake_probe(executable, **_kwargs):
        executable = Path(executable)
        requested = executable.read_text(encoding="utf-8")
        if requested == "3.11":
            return SQLiteRuntimeInfo(
                executable=executable,
                base_prefix=executable.parent.parent,
                python_version=(3, 11, 14),
                sqlite_version=(3, 50, 4),
                sqlite_version_string="3.50.4",
                sqlite_source_id="vulnerable",
            )
        return SQLiteRuntimeInfo(
            executable=executable,
            base_prefix=executable.parent.parent,
            python_version=(3, 11, 16),
            sqlite_version=(3, 53, 1),
            sqlite_version_string="3.53.1",
            sqlite_source_id="fixed",
        )

    monkeypatch.setattr(managed_uv.subprocess, "run", fake_run)
    monkeypatch.setattr(managed_uv, "probe_sqlite_runtime", fake_probe)

    result = managed_uv._install_safe_python_generation(
        "uv.exe", project_root=root, current=current
    )

    assert result.status == "ready"
    assert state["requests"] == ["3.11", "3.11.16"]


@pytest.mark.parametrize(
    "failure", ["install", "lookup", "probe", "probe_exception", "catalog"]
)
def test_uncertain_candidate_failure_is_not_cached(tmp_path, monkeypatch, failure):
    import hermes_cli.managed_uv as managed_uv

    _windows_hermetic(monkeypatch)
    root, _ = _runtime_install(tmp_path)
    calls = []
    state = {}

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1:] == ["--version"]:
            return SimpleNamespace(returncode=0, stdout="uv 0.8.4\n", stderr="")
        if "install" in command:
            if failure == "install":
                return SimpleNamespace(returncode=1, stdout="", stderr="offline")
            state["generation"] = Path(kwargs["env"]["UV_PYTHON_INSTALL_DIR"])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "find" in command:
            if failure == "lookup":
                return SimpleNamespace(returncode=1, stdout="", stderr="missing")
            python = state["generation"] / "cpython" / "bin" / "python3"
            python.parent.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, stdout=str(python), stderr="")
        if "list" in command:
            if failure == "catalog":
                return SimpleNamespace(returncode=1, stdout="", stderr="offline")
            return SimpleNamespace(returncode=0, stdout="[]", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(managed_uv.subprocess, "run", fake_run)
    live_python = root / "venv" / "Scripts" / "python.exe"
    current = _runtime_info(live_python)
    def fake_probe(executable):
        if Path(executable) == live_python:
            return current
        if failure == "probe_exception":
            raise RuntimeError("probe interrupted")
        if failure == "catalog":
            return current
        return None

    monkeypatch.setattr(managed_uv, "probe_sqlite_runtime", fake_probe)
    monkeypatch.setattr(managed_uv, "_refresh_managed_uv_catalog", Mock(return_value=False))

    first = managed_uv.repair_vulnerable_runtime("uv.exe", project_root=root)
    second = managed_uv.repair_vulnerable_runtime("uv.exe", project_root=root)
    marker = root / ".hermes-runtime" / "python" / "unavailable-artifact.json"
    assert first.status == "failed"
    assert second.status == "failed"
    assert not marker.exists()
    assert len([call for call in calls if "install" in call]) == 2


def test_structured_provisioning_failure_refreshes_and_retries(tmp_path, monkeypatch):
    import hermes_cli.managed_uv as managed_uv

    _windows_hermetic(monkeypatch)
    root, live_python = _runtime_install(tmp_path)
    current = _runtime_info(live_python)
    monkeypatch.setattr(managed_uv, "probe_sqlite_runtime", lambda path: current)
    monkeypatch.setattr(managed_uv, "_uv_version_string", lambda _path: "0.8.4")
    provision = Mock(
        side_effect=[
            managed_uv._ProvisioningResult.failed("probe failed"),
            managed_uv._ProvisioningResult.failed("probe failed again"),
        ]
    )
    refresh = Mock(return_value=True)
    monkeypatch.setattr(managed_uv, "_install_safe_python_generation", provision)
    monkeypatch.setattr(managed_uv, "_refresh_managed_uv_catalog", refresh)

    result = managed_uv.repair_vulnerable_runtime("uv.exe", project_root=root)

    assert result.status == "failed"
    assert provision.call_count == 2
    refresh.assert_called_once_with("uv.exe")
    assert not (root / ".hermes-runtime" / "python" / "unavailable-artifact.json").exists()


def test_malformed_unavailable_marker_fails_open(tmp_path, monkeypatch):
    import hermes_cli.managed_uv as managed_uv

    _windows_hermetic(monkeypatch)
    root, _ = _runtime_install(tmp_path)
    marker = root / ".hermes-runtime" / "python" / "unavailable-artifact.json"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"\xff")
    _, calls = _uv_process(monkeypatch, uv_version="0.8.4", candidate_sqlite=(3, 50, 4))
    monkeypatch.setattr(managed_uv, "_refresh_managed_uv_catalog", lambda *_: False)

    result = managed_uv.repair_vulnerable_runtime("uv.exe", project_root=root)
    assert result.status == "unavailable"
    assert len([call for call in calls if "install" in call]) == 1
    assert marker.read_bytes() != b"\xff"


def test_deeply_malformed_unavailable_marker_fails_open(tmp_path, monkeypatch):
    import hermes_cli.managed_uv as managed_uv

    _windows_hermetic(monkeypatch)
    root, _ = _runtime_install(tmp_path)
    marker = root / ".hermes-runtime" / "python" / "unavailable-artifact.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("[" * 2000 + "0" + "]" * 2000, encoding="utf-8")
    _, calls = _uv_process(monkeypatch, uv_version="0.8.4", candidate_sqlite=(3, 50, 4))
    monkeypatch.setattr(managed_uv, "_refresh_managed_uv_catalog", lambda *_: False)

    result = managed_uv.repair_vulnerable_runtime("uv.exe", project_root=root)

    assert result.status == "unavailable"
    assert len([call for call in calls if "install" in call]) == 1
    assert marker.read_bytes() != b"[" * 2000 + b"0" + b"]" * 2000


def test_malformed_catalog_is_not_cached(tmp_path, monkeypatch):
    import hermes_cli.managed_uv as managed_uv

    _windows_hermetic(monkeypatch)
    root, _ = _runtime_install(tmp_path)
    original_run, calls = _uv_process(
        monkeypatch, uv_version="0.8.4", candidate_sqlite=(3, 50, 4)
    )

    def malformed_catalog(command, **kwargs):
        if "list" in command:
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout="{}", stderr="")
        return original_run(command, **kwargs)

    monkeypatch.setattr(managed_uv.subprocess, "run", malformed_catalog)
    monkeypatch.setattr(managed_uv, "_refresh_managed_uv_catalog", Mock(return_value=False))

    first = managed_uv.repair_vulnerable_runtime("uv.exe", project_root=root)
    second = managed_uv.repair_vulnerable_runtime("uv.exe", project_root=root)

    marker = root / ".hermes-runtime" / "python" / "unavailable-artifact.json"
    assert first.status == "failed"
    assert second.status == "failed"
    assert not marker.exists()
    assert len([call for call in calls if "install" in call]) == 2


def test_semantically_malformed_marker_fails_open(tmp_path, monkeypatch):
    import hermes_cli.managed_uv as managed_uv

    _windows_hermetic(monkeypatch)
    root, _ = _runtime_install(tmp_path)
    marker = root / ".hermes-runtime" / "python" / "unavailable-artifact.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "uv_path": str(Path("uv.exe").resolve()),
                "uv_version": "uv 0.8.4",
                "platform": "Windows",
                "machine": "AMD64",
                "requested_python": "3.11",
                "live_runtime": {
                    "python_version": [3, 11, 15],
                    "sqlite_version": [3, 50, 4],
                    "sqlite_version_string": "3.50.4",
                    "sqlite_source_id": "vulnerable",
                },
                "rejected_candidates": [
                    {
                        "python_version": [3, 11, 15],
                        "sqlite_version": [3, 50, 4],
                        "sqlite_version_string": None,
                        "sqlite_source_id": "vulnerable",
                        "executable": "candidate",
                        "base_prefix": "candidate",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _, calls = _uv_process(monkeypatch, uv_version="0.8.4", candidate_sqlite=(3, 50, 4))
    monkeypatch.setattr(managed_uv, "_refresh_managed_uv_catalog", lambda *_: False)

    result = managed_uv.repair_vulnerable_runtime("uv.exe", project_root=root)

    assert result.status == "unavailable"
    assert len([call for call in calls if "install" in call]) == 1


def test_safe_runtime_clears_unavailable_marker(tmp_path, monkeypatch):
    import hermes_cli.managed_uv as managed_uv

    _windows_hermetic(monkeypatch)
    root, live_python = _runtime_install(tmp_path)
    marker = root / ".hermes-runtime" / "python" / "unavailable-artifact.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    current = _runtime_info(live_python, (3, 53, 1), source="fixed")
    monkeypatch.setattr(managed_uv, "probe_sqlite_runtime", lambda *_: current)
    install = Mock()
    monkeypatch.setattr(managed_uv, "_install_safe_python_generation", install)

    result = managed_uv.repair_vulnerable_runtime("uv.exe", project_root=root)
    assert result.status == "safe"
    assert not marker.exists()
    install.assert_not_called()


def test_foreign_uv_is_never_refreshed(tmp_path, monkeypatch):
    import hermes_cli.managed_uv as managed_uv

    managed = tmp_path / "managed" / "uv.exe"
    foreign = tmp_path / "foreign" / "uv.exe"
    monkeypatch.setattr(managed_uv, "managed_uv_path", lambda: managed)
    monkeypatch.setattr(managed_uv, "_uv_version_string", lambda *_: "uv 0.8.4")
    installer = Mock()
    monkeypatch.setattr(managed_uv, "_install_uv", installer)

    assert managed_uv._refresh_managed_uv_catalog(str(foreign)) is False
    installer.assert_not_called()
