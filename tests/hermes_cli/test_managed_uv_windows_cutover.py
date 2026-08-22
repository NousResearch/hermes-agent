"""Native-Windows coverage for managed runtime cutover."""

from __future__ import annotations

import subprocess
import venv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _python(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe"


@pytest.mark.windows_only
def test_runtime_config_cutover_does_not_rename_running_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import managed_uv

    live = tmp_path / "venv"
    candidate = tmp_path / "candidate"
    venv.EnvBuilder(with_pip=False).create(live)
    venv.EnvBuilder(with_pip=False).create(candidate)

    candidate_config = candidate / "pyvenv.cfg"
    replacement = candidate_config.read_bytes() + b"runtime-cutover = candidate\n"
    candidate_config.write_bytes(replacement)
    info = SimpleNamespace(sqlite_version_string="3.53.1")
    monkeypatch.setattr(
        managed_uv,
        "_smoke_candidate_venv",
        lambda target: (True, "", info),
    )

    process = subprocess.Popen(
        [str(_python(live)), "-c", "import time; time.sleep(30)"],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        ok, runtime_in_use, final_info, detail = (
            managed_uv._cut_over_windows_runtime_config(candidate, live=live)
        )

        assert process.poll() is None
        assert ok is True
        assert runtime_in_use is True
        assert final_info is info
        assert detail == ""
        assert (live / "pyvenv.cfg").read_bytes() == replacement
        assert live.is_dir()
        probe = subprocess.run(
            [str(_python(live)), "-I", "-c", "print('repointed')"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert probe.returncode == 0
        assert probe.stdout.strip() == "repointed"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.windows_only
def test_runtime_config_cutover_rolls_back_failed_live_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import managed_uv

    live = tmp_path / "venv"
    candidate = tmp_path / "candidate"
    live.mkdir()
    candidate.mkdir()
    original = b"home = old-runtime\n"
    (live / "pyvenv.cfg").write_bytes(original)
    (candidate / "pyvenv.cfg").write_bytes(b"home = new-runtime\n")
    monkeypatch.setattr(
        managed_uv,
        "_smoke_candidate_venv",
        lambda target: (False, "core import smoke failed", None),
    )

    ok, runtime_in_use, info, detail = managed_uv._cut_over_windows_runtime_config(
        candidate, live=live
    )

    assert ok is False
    assert runtime_in_use is False
    assert info is None
    assert detail == "post-cutover smoke failed: core import smoke failed"
    assert (live / "pyvenv.cfg").read_bytes() == original


@pytest.mark.windows_only
def test_runtime_repair_uses_config_cutover_on_windows(tmp_path: Path) -> None:
    from hermes_cli.managed_uv import repair_vulnerable_runtime
    from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo

    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    live = root / "venv"
    _python(live).parent.mkdir(parents=True)
    _python(live).write_bytes(b"live")
    (live / "pyvenv.cfg").write_text("home = old-runtime\n", encoding="utf-8")

    generation = root / ".hermes-runtime" / "python" / "generation-test"
    candidate_python = generation / "python.exe"
    candidate_python.parent.mkdir(parents=True)
    candidate_python.write_bytes(b"candidate")
    candidate = root / ".hermes-runtime" / "venv-candidate-test"
    _python(candidate).parent.mkdir(parents=True)
    _python(candidate).write_bytes(b"candidate")
    (candidate / "pyvenv.cfg").write_text(
        f"home = {generation}\n",
        encoding="utf-8",
    )

    current = SQLiteRuntimeInfo(
        executable=_python(live),
        base_prefix=live,
        python_version=(3, 11, 15),
        sqlite_version=(3, 50, 4),
        sqlite_version_string="3.50.4",
        sqlite_source_id="old",
    )
    fixed = SQLiteRuntimeInfo(
        executable=candidate_python,
        base_prefix=generation,
        python_version=(3, 11, 15),
        sqlite_version=(3, 53, 1),
        sqlite_version_string="3.53.1",
        sqlite_source_id="new",
    )

    with (
        patch(
            "hermes_cli.managed_uv.probe_sqlite_runtime",
            side_effect=[current, current],
        ),
        patch(
            "hermes_cli.managed_uv._windows_runtime_holders",
            return_value=(False, ""),
        ),
        patch(
            "hermes_cli.managed_uv._install_safe_python_generation",
            return_value=(generation, candidate_python, fixed),
        ),
        patch(
            "hermes_cli.managed_uv._stage_candidate_venv",
            return_value=candidate,
        ),
        patch(
            "hermes_cli.managed_uv._cut_over_windows_runtime_config",
            return_value=(True, True, fixed, ""),
        ) as windows_cutover,
        patch("hermes_cli.managed_uv._cut_over_candidate") as directory_cutover,
    ):
        result = repair_vulnerable_runtime("uv", project_root=root)

    assert result.status == "repaired"
    assert result.sqlite_before == "3.50.4"
    assert result.sqlite_after == "3.53.1"
    windows_cutover.assert_called_once_with(candidate, live=live)
    directory_cutover.assert_not_called()
    assert not candidate.exists()
    assert generation.exists()
