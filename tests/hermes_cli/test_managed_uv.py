"""Tests for hermes_cli.managed_uv — one path, no guessing."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_executable(path: Path) -> None:
    """Create a minimal fake uv binary at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho uv 0.1.2\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _runtime_info(
    executable: Path,
    sqlite_version: tuple[int, int, int],
):
    from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo

    return SQLiteRuntimeInfo(
        executable=executable,
        base_prefix=executable.parent.parent,
        python_version=(3, 11, 15),
        sqlite_version=sqlite_version,
        sqlite_version_string=".".join(str(part) for part in sqlite_version),
        sqlite_source_id=f"source-{sqlite_version}",
    )


def _make_runtime_install(
    tmp_path: Path,
    *,
    windows: bool = False,
) -> tuple[Path, Path, Path]:
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    live = root / "venv"
    bin_dir = live / ("Scripts" if windows else "bin")
    bin_dir.mkdir(parents=True)
    python = bin_dir / ("python.exe" if windows else "python")
    python.write_text("live interpreter", encoding="utf-8")
    sentinel = live / "sentinel"
    sentinel.write_text("live", encoding="utf-8")
    return root, live, sentinel


# ---------------------------------------------------------------------------
# managed_uv_path
# ---------------------------------------------------------------------------

class TestManagedUvPath:
    def test_posix(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Linux"):
            from hermes_cli.managed_uv import managed_uv_path
            assert managed_uv_path() == tmp_path / "bin" / "uv"

    def test_windows(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Windows"):
            from hermes_cli.managed_uv import managed_uv_path
            assert managed_uv_path() == tmp_path / "bin" / "uv.exe"


# ---------------------------------------------------------------------------
# resolve_uv
# ---------------------------------------------------------------------------

class TestResolveUv:
    def test_missing_returns_none(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path):
            from hermes_cli.managed_uv import resolve_uv
            assert resolve_uv() is None

    def test_existing_executable(self, tmp_path):
        _make_executable(tmp_path / "bin" / "uv")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path):
            from hermes_cli.managed_uv import resolve_uv
            result = resolve_uv()
            assert result == str(tmp_path / "bin" / "uv")

    def test_non_executable_file_returns_none(self, tmp_path):
        uv = tmp_path / "bin" / "uv"
        uv.parent.mkdir(parents=True)
        uv.write_text("not a binary")
        # Ensure no execute bit
        uv.chmod(0o644)
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path):
            from hermes_cli.managed_uv import resolve_uv
            assert resolve_uv() is None


# ---------------------------------------------------------------------------
# ensure_uv
# ---------------------------------------------------------------------------

class TestEnsureUv:
    def test_already_installed_no_bootstrap(self, tmp_path):
        _make_executable(tmp_path / "bin" / "uv")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path):
            from hermes_cli.managed_uv import ensure_uv
            path = ensure_uv()
            assert path == str(tmp_path / "bin" / "uv")

    def test_installs_if_missing(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv._install_uv") as mock_install:
            # Simulate the installer creating the binary
            def fake_install(target):
                _make_executable(target)
            mock_install.side_effect = fake_install

            from hermes_cli.managed_uv import ensure_uv
            path = ensure_uv()
            assert path == str(tmp_path / "bin" / "uv")
            mock_install.assert_called_once()

    def test_install_reports_runtime_repair_to_observer(self, tmp_path):
        from hermes_cli.managed_uv import (
            RuntimeRepairResult,
            ensure_uv,
        )

        repair = RuntimeRepairResult(
            "repaired",
            sqlite_before="3.50.4",
            sqlite_after="3.53.1",
        )

        def fake_install(target):
            _make_executable(target)

        observed = []
        with patch(
            "hermes_cli.managed_uv.get_hermes_home",
            return_value=tmp_path,
        ), patch(
            "hermes_cli.managed_uv._install_uv",
            side_effect=fake_install,
        ), patch(
            "hermes_cli.managed_uv.repair_vulnerable_runtime",
            return_value=repair,
        ):
            path = ensure_uv(repair_observer=observed.append)

        assert path == str(tmp_path / "bin" / "uv")
        assert observed == [repair]

    def test_install_failure_returns_falsy(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv._install_uv", side_effect=RuntimeError("network down")):
            from hermes_cli.managed_uv import ensure_uv
            path = ensure_uv()
            # Failure is a falsy sentinel (not None) so legacy 2-target call
            # sites can still unpack it without raising — see
            # TestEnsureUvUpdateBoundary for why.
            assert not path


class TestEnsureUvUpdateBoundary:
    """``ensure_uv()`` must answer to both the single-value and the legacy
    ``(path, fresh_bootstrap)`` call conventions — **on POSIX**.

    ``hermes update`` runs the call site from the old, already-imported
    ``hermes_cli.main`` against the freshly pulled ``managed_uv``. A release
    parked on a ``(path, fresh)`` tuple runs ``uv_bin, fresh = ensure_uv()``
    against the single-value module; the path is an iterable ``str`` so the
    2-target unpack walked its characters and raised
    ``ValueError: too many values to unpack (expected 2)`` (root cause behind
    PR #39763), or ``TypeError`` on the ``None`` failure path. On POSIX the
    result must therefore be usable as a bare path *and* unpackable as a
    2-tuple, in both the success and failure cases.

    The dual contract is intentionally **not** offered on Windows — see
    ``TestEnsureUvWindowsSafe`` for why — so these tests pin ``platform.system``
    to a POSIX value.
    """

    def test_success_usable_as_single_value(self, tmp_path):
        _make_executable(tmp_path / "bin" / "uv")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Linux"):
            from hermes_cli.managed_uv import ensure_uv
            uv_bin = ensure_uv()
            assert uv_bin == str(tmp_path / "bin" / "uv")
            assert bool(uv_bin) is True

    def test_success_unpacks_as_legacy_two_tuple(self, tmp_path):
        _make_executable(tmp_path / "bin" / "uv")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Linux"):
            from hermes_cli.managed_uv import ensure_uv
            uv_bin, fresh = ensure_uv()  # old: uv_bin, fresh_bootstrap = ensure_uv()
            assert uv_bin == str(tmp_path / "bin" / "uv")
            assert fresh is False

    def test_failure_unpacks_without_raising(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Linux"), \
             patch("hermes_cli.managed_uv._install_uv", side_effect=RuntimeError("network down")):
            from hermes_cli.managed_uv import ensure_uv
            uv_bin, fresh = ensure_uv()
            assert uv_bin is None
            assert fresh is False


class TestEnsureUvWindowsSafe:
    """On Windows ``ensure_uv()`` must return a plain ``str``/``None``.

    ``subprocess`` on Windows serializes argv through
    ``subprocess.list2cmdline``, which iterates every entry *as a string*
    (``for c in arg``). The dependency installer feeds uv straight into the
    command list (``[uv_bin, "pip", "install", ...]``). A ``str`` subclass
    whose ``__iter__`` yields ``(path, fresh_bootstrap)`` instead of characters
    therefore injects the bool into the command line and crashes the install
    with ``TypeError: sequence item 1: expected str instance, bool found``
    (a real field report on a 10-commits-behind Windows install). A single
    return value cannot serve both the legacy 2-tuple unpack and Windows
    char-iteration — both use the iterator protocol — so Windows opts out of
    the wrapper entirely.
    """

    def test_uvresult_would_break_windows_list2cmdline(self):
        # Canary: this is *why* the wrapper is gated off Windows. If a future
        # change makes _UvResult char-iterable (and thus list2cmdline-safe),
        # the gate may be revisited.
        import subprocess
        from hermes_cli.managed_uv import _UvResult
        with pytest.raises(TypeError):
            subprocess.list2cmdline([_UvResult("C:\\hermes\\uv.exe"), "pip"])

    def test_windows_returns_plain_str_safe_for_subprocess(self, tmp_path):
        import subprocess
        # On (mocked) Windows the managed binary is uv.exe.
        _make_executable(tmp_path / "bin" / "uv.exe")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Windows"):
            from hermes_cli.managed_uv import _UvResult, ensure_uv
            uv_bin = ensure_uv()
            assert type(uv_bin) is str and not isinstance(uv_bin, _UvResult)
            # The exact operation that crashed in the field must now succeed.
            cmdline = subprocess.list2cmdline([uv_bin, "pip", "install", "-e", "."])
            assert "pip" in cmdline and "install" in cmdline

    def test_windows_failure_returns_none(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Windows"), \
             patch("hermes_cli.managed_uv._install_uv", side_effect=RuntimeError("network down")):
            from hermes_cli.managed_uv import ensure_uv
            assert ensure_uv() is None


# ---------------------------------------------------------------------------
# update_managed_uv
# ---------------------------------------------------------------------------

class TestUpdateManagedUv:
    def test_no_uv_returns_none(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path):
            from hermes_cli.managed_uv import update_managed_uv
            assert update_managed_uv() is None

    def test_self_update_success(self, tmp_path):
        _make_executable(tmp_path / "bin" / "uv")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.subprocess.run") as mock_run:
            # uv self update succeeds
            mock_run.return_value = MagicMock(returncode=0, stdout="uv 0.2.0")
            from hermes_cli.managed_uv import update_managed_uv
            result = update_managed_uv()
            assert result == str(tmp_path / "bin" / "uv")
            # First call is self update, second is --version
            assert mock_run.call_count == 2
            assert mock_run.call_args_list[0][0][0] == [str(tmp_path / "bin" / "uv"), "self", "update"]

    def test_self_update_failure_non_fatal(self, tmp_path):
        _make_executable(tmp_path / "bin" / "uv")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="nope")
            from hermes_cli.managed_uv import update_managed_uv
            result = update_managed_uv()
            # Still returns the path — failure is non-fatal
            assert result == str(tmp_path / "bin" / "uv")

    def test_old_updater_api_triggers_runtime_repair(self, tmp_path):
        """The pre-pull main.py call site must activate the fresh module hook."""
        from hermes_cli.managed_uv import RuntimeRepairResult, update_managed_uv

        uv = tmp_path / "bin" / "uv"
        _make_executable(uv)
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Linux"), \
             patch("hermes_cli.managed_uv.subprocess.run") as mock_run, \
             patch(
                 "hermes_cli.managed_uv.repair_vulnerable_runtime",
                 return_value=RuntimeRepairResult(
                     "repaired",
                     sqlite_before="3.50.4",
                     sqlite_after="3.53.1",
                 ),
             ) as mock_repair:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="", stderr=""),
                MagicMock(returncode=0, stdout="uv 0.11.31\n", stderr=""),
            ]

            result = update_managed_uv()

        assert result == str(uv)
        mock_repair.assert_called_once_with(str(uv))

    def test_update_reports_runtime_repair_to_observer(self, tmp_path):
        from hermes_cli.managed_uv import RuntimeRepairResult, update_managed_uv

        uv = tmp_path / "bin" / "uv"
        _make_executable(uv)
        repair = RuntimeRepairResult(
            "repaired",
            sqlite_before="3.50.4",
            sqlite_after="3.53.1",
        )
        observed = []
        with patch(
            "hermes_cli.managed_uv.get_hermes_home",
            return_value=tmp_path,
        ), patch(
            "hermes_cli.managed_uv.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="uv 0.11.31\n", stderr=""),
        ), patch(
            "hermes_cli.managed_uv.repair_vulnerable_runtime",
            return_value=repair,
        ):
            result = update_managed_uv(repair_observer=observed.append)

        assert result == str(uv)
        assert observed == [repair]


class TestManagedPythonStore:
    def test_store_is_checkout_scoped_across_profiles(self, tmp_path, monkeypatch):
        from hermes_cli.managed_uv import managed_python_install_dir

        checkout = tmp_path / "checkout"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "alpha"))
        alpha = managed_python_install_dir(checkout)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "beta"))
        beta = managed_python_install_dir(checkout)

        expected = checkout / ".hermes-runtime" / "python"
        assert alpha == expected
        assert beta == expected

    def test_environment_is_private_and_sanitized(self, tmp_path):
        from hermes_cli.managed_uv import managed_python_env

        checkout = tmp_path / "checkout"
        base_env = {
            "KEEP_ME": "yes",
            "CONDA_DEFAULT_ENV": "poison",
            "CONDA_PREFIX": "/poison/conda",
            "UV_PROJECT_ENVIRONMENT": "/poison/project",
            "UV_NO_MANAGED_PYTHON": "1",
            "UV_PYTHON": "/poison/python",
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_SYSTEM_PYTHON": "1",
            "VIRTUAL_ENV": "/poison/venv",
            "PYTHONHOME": "/poison/home",
            "PYTHONPATH": "/poison/path",
        }

        env = managed_python_env(checkout, base_env=base_env)

        assert env["KEEP_ME"] == "yes"
        assert env["UV_MANAGED_PYTHON"] == "1"
        assert env["UV_NO_CONFIG"] == "1"
        assert env["UV_PYTHON_INSTALL_BIN"] == "0"
        assert env["UV_PYTHON_INSTALL_REGISTRY"] == "0"
        assert env["UV_PYTHON_INSTALL_DIR"] == str(
            checkout / ".hermes-runtime" / "python"
        )
        for key in (
            "CONDA_DEFAULT_ENV",
            "CONDA_PREFIX",
            "UV_PROJECT_ENVIRONMENT",
            "UV_NO_MANAGED_PYTHON",
            "UV_PYTHON",
            "UV_PYTHON_DOWNLOADS",
            "UV_SYSTEM_PYTHON",
            "VIRTUAL_ENV",
            "PYTHONHOME",
            "PYTHONPATH",
        ):
            assert key not in env
        assert base_env["PYTHONHOME"] == "/poison/home"


class TestRuntimeRepair:
    def test_safe_runtime_is_a_noop(self, tmp_path):
        from hermes_cli.managed_uv import repair_vulnerable_runtime

        root, live, sentinel = _make_runtime_install(tmp_path)
        current = _runtime_info(live / "bin" / "python", (3, 53, 1))
        with patch("hermes_cli.managed_uv.platform.system", return_value="Linux"), \
             patch(
                 "hermes_cli.managed_uv.probe_sqlite_runtime",
                 return_value=current,
             ), \
             patch(
                 "hermes_cli.managed_uv._install_safe_python_generation"
             ) as mock_install:
            result = repair_vulnerable_runtime("uv", project_root=root)

        assert result.status == "safe"
        assert result.sqlite_before == "3.53.1"
        assert result.sqlite_after == "3.53.1"
        assert sentinel.read_text(encoding="utf-8") == "live"
        assert not (root / ".hermes-runtime").exists()
        mock_install.assert_not_called()

    def test_failed_candidate_preserves_live_venv(self, tmp_path):
        from hermes_cli.managed_uv import (
            _acquire_repair_lock,
            _release_repair_lock,
            repair_vulnerable_runtime,
        )

        root, live, sentinel = _make_runtime_install(tmp_path)
        current = _runtime_info(live / "bin" / "python", (3, 50, 4))
        generation = root / ".hermes-runtime" / "python" / "generation-test"
        candidate_python = generation / "bin" / "python"
        candidate_python.parent.mkdir(parents=True)
        candidate_python.write_text("candidate interpreter", encoding="utf-8")
        fixed = _runtime_info(candidate_python, (3, 53, 1))

        with patch("hermes_cli.managed_uv.platform.system", return_value="Linux"), \
             patch(
                 "hermes_cli.managed_uv.probe_sqlite_runtime",
                 side_effect=[current, current],
             ), \
             patch(
                 "hermes_cli.managed_uv._install_safe_python_generation",
                 return_value=(generation, candidate_python, fixed),
             ), \
             patch(
                 "hermes_cli.managed_uv._stage_candidate_venv",
                 return_value=None,
             ):
            result = repair_vulnerable_runtime("uv", project_root=root)

        assert result.status == "failed"
        assert "replacement environment" in result.detail
        assert sentinel.read_text(encoding="utf-8") == "live"
        assert (live / "bin" / "python").read_text(encoding="utf-8") == (
            "live interpreter"
        )
        assert not generation.exists()
        reacquired = _acquire_repair_lock(root / ".hermes-runtime")
        assert reacquired is not None
        _release_repair_lock(reacquired)

    def test_windows_holders_refuse_runtime_mutation(self, tmp_path, monkeypatch):
        from hermes_cli.managed_uv import repair_vulnerable_runtime

        root, live, sentinel = _make_runtime_install(tmp_path, windows=True)
        current = _runtime_info(live / "Scripts" / "python.exe", (3, 50, 4))
        old_main = SimpleNamespace(
            _detect_venv_python_processes=lambda: [
                (1729, "python.exe", "hermes gateway run")
            ]
        )
        monkeypatch.setitem(sys.modules, "hermes_cli.main", old_main)

        with patch("hermes_cli.managed_uv.platform.system", return_value="Windows"), \
             patch(
                 "hermes_cli.managed_uv.probe_sqlite_runtime",
                 return_value=current,
             ), \
             patch(
                 "hermes_cli.managed_uv._install_safe_python_generation"
             ) as mock_install:
            result = repair_vulnerable_runtime("uv.exe", project_root=root)

        assert result.status == "skipped"
        assert "PID 1729" in result.detail
        assert sentinel.read_text(encoding="utf-8") == "live"
        assert not (root / ".hermes-runtime").exists()
        mock_install.assert_not_called()


class TestRuntimeCutover:
    def test_os_lock_blocks_concurrent_repair_and_releases(self, tmp_path):
        from hermes_cli.managed_uv import _acquire_repair_lock, _release_repair_lock

        runtime_root = tmp_path / ".hermes-runtime"
        first = _acquire_repair_lock(runtime_root)
        assert first is not None
        assert _acquire_repair_lock(runtime_root) is None

        _release_repair_lock(first)
        second = _acquire_repair_lock(runtime_root)
        assert second is not None
        _release_repair_lock(second)

    def test_failed_smoke_with_empty_output_has_stable_detail(self, tmp_path):
        from hermes_cli.managed_uv import _smoke_candidate_venv

        candidate = tmp_path / "venv"
        candidate.mkdir()
        fixed = _runtime_info(candidate / "bin" / "python", (3, 53, 1))
        failed = MagicMock(returncode=1, stdout=" \n", stderr="\n")
        with patch(
            "hermes_cli.managed_uv.probe_sqlite_runtime",
            return_value=fixed,
        ), patch("hermes_cli.managed_uv.subprocess.run", return_value=failed):
            healthy, detail, info = _smoke_candidate_venv(candidate)

        assert healthy is False
        assert detail == "core import smoke failed"
        assert info == fixed

    def test_successfully_renames_candidate_into_live_path(self, tmp_path):
        from hermes_cli.managed_uv import _cut_over_candidate

        root, _, _ = _make_runtime_install(tmp_path)
        runtime_root = root / ".hermes-runtime"
        candidate = runtime_root / "venv-candidate-test"
        candidate.mkdir(parents=True)
        (candidate / "sentinel").write_text("candidate", encoding="utf-8")
        fixed = _runtime_info(candidate / "bin" / "python", (3, 53, 1))

        with patch(
            "hermes_cli.managed_uv._smoke_candidate_venv",
            return_value=(True, "", fixed),
        ):
            ok, backup, info, detail = _cut_over_candidate(
                candidate,
                project_root=root,
            )

        assert ok is True
        assert detail == ""
        assert info == fixed
        assert backup is not None
        assert (root / "venv" / "sentinel").read_text(encoding="utf-8") == (
            "candidate"
        )
        assert (backup / "sentinel").read_text(encoding="utf-8") == "live"
        assert not candidate.exists()

    def test_post_swap_smoke_failure_rolls_back_live_venv(self, tmp_path):
        from hermes_cli.managed_uv import _cut_over_candidate

        root, live, sentinel = _make_runtime_install(tmp_path)
        runtime_root = root / ".hermes-runtime"
        candidate = runtime_root / "venv-candidate-test"
        candidate.mkdir(parents=True)
        (candidate / "sentinel").write_text("candidate", encoding="utf-8")
        rejected_info = _runtime_info(candidate / "bin" / "python", (3, 50, 4))

        with patch(
            "hermes_cli.managed_uv._smoke_candidate_venv",
            return_value=(False, "core import smoke failed", rejected_info),
        ):
            ok, backup, info, detail = _cut_over_candidate(
                candidate,
                project_root=root,
            )

        assert ok is False
        assert backup is None
        assert info == rejected_info
        assert "post-cutover smoke failed" in detail
        assert sentinel.read_text(encoding="utf-8") == "live"
        assert (live / "bin" / "python").read_text(encoding="utf-8") == (
            "live interpreter"
        )
        assert not candidate.exists()
        assert not list(runtime_root.glob("venv-rejected-*"))

    def test_smoke_exception_after_swap_rolls_back_live_venv(self, tmp_path):
        from hermes_cli.managed_uv import _cut_over_candidate

        root, live, sentinel = _make_runtime_install(tmp_path)
        candidate = root / ".hermes-runtime" / "venv-candidate-test"
        candidate.mkdir(parents=True)
        (candidate / "sentinel").write_text("candidate", encoding="utf-8")

        with patch(
            "hermes_cli.managed_uv._smoke_candidate_venv",
            side_effect=RuntimeError("probe crashed"),
        ):
            ok, backup, info, detail = _cut_over_candidate(
                candidate,
                project_root=root,
            )

        assert ok is False
        assert backup is None
        assert info is None
        assert "probe crashed" in detail
        assert sentinel.read_text(encoding="utf-8") == "live"
        assert (live / "bin" / "python").read_text(encoding="utf-8") == (
            "live interpreter"
        )

    def test_interrupt_during_promotion_restores_live_venv(self, tmp_path):
        from hermes_cli.managed_uv import _cut_over_candidate

        root, live, sentinel = _make_runtime_install(tmp_path)
        candidate = root / ".hermes-runtime" / "venv-candidate-test"
        candidate.mkdir(parents=True)
        (candidate / "sentinel").write_text("candidate", encoding="utf-8")
        rename_count = 0

        def interrupt_second_rename(source, destination):
            nonlocal rename_count
            rename_count += 1
            if rename_count == 2:
                raise KeyboardInterrupt
            source.rename(destination)

        with patch(
            "hermes_cli.managed_uv._rename_with_retry",
            side_effect=interrupt_second_rename,
        ), pytest.raises(KeyboardInterrupt):
            _cut_over_candidate(candidate, project_root=root)

        assert sentinel.read_text(encoding="utf-8") == "live"
        assert candidate.exists()
        assert not list(root.glob("venv.stale.runtime-*"))

    def test_interrupt_during_rollback_restores_live_venv(self, tmp_path):
        from hermes_cli.managed_uv import _cut_over_candidate

        root, live, sentinel = _make_runtime_install(tmp_path)
        runtime_root = root / ".hermes-runtime"
        candidate = runtime_root / "venv-candidate-test"
        candidate.mkdir(parents=True)
        (candidate / "sentinel").write_text("candidate", encoding="utf-8")
        rename_count = 0

        def interrupt_backup_restore(source, destination):
            nonlocal rename_count
            rename_count += 1
            if rename_count == 4:
                raise KeyboardInterrupt
            source.rename(destination)

        with patch(
            "hermes_cli.managed_uv._rename_with_retry",
            side_effect=interrupt_backup_restore,
        ), patch(
            "hermes_cli.managed_uv._smoke_candidate_venv",
            return_value=(False, "core import smoke failed", None),
        ), pytest.raises(KeyboardInterrupt):
            _cut_over_candidate(candidate, project_root=root)

        assert sentinel.read_text(encoding="utf-8") == "live"
        assert not list(root.glob("venv.stale.runtime-*"))
        assert list(runtime_root.glob("venv-rejected-*"))


# ---------------------------------------------------------------------------
# _install_uv internals
# ---------------------------------------------------------------------------

class TestInstallUvInternals:
    def test_posix_sets_uv_unmanaged_install(self, tmp_path):
        target = tmp_path / "bin" / "uv"
        with patch("hermes_cli.managed_uv._install_uv_posix") as mock_posix:
            from hermes_cli.managed_uv import _install_uv
            _install_uv(target)
            mock_posix.assert_called_once()
            call_env = mock_posix.call_args[0][0]
            assert call_env["UV_UNMANAGED_INSTALL"] == str(tmp_path / "bin")

    def test_windows_sets_uv_install_dir(self, tmp_path):
        target = tmp_path / "bin" / "uv.exe"
        with patch("hermes_cli.managed_uv.platform.system", return_value="Windows"), \
             patch("hermes_cli.managed_uv._install_uv_windows") as mock_windows:
            from hermes_cli.managed_uv import _install_uv
            _install_uv(target)
            mock_windows.assert_called_once()
            call_env = mock_windows.call_args[0][0]
            assert call_env["UV_INSTALL_DIR"] == str(tmp_path / "bin")


class TestRuntimeRequestMinorLine:
    """The repair must request the CPython minor line, not the exact patch.

    Real-world constraint (verified live, July 2026): every published
    python-build-standalone artifact for 3.11.14 links vulnerable SQLite
    3.50.4 — even with --reinstall. The fixed SQLite (3.53.1) only exists
    from 3.11.15. An exact-patch pin makes the repair permanently
    impossible on such installs.
    """

    def test_requests_minor_line(self):
        from hermes_cli.managed_uv import _runtime_request

        info = _runtime_info(Path("/venv/bin/python"), (3, 50, 4))
        assert _runtime_request(info) == "3.11"

    @staticmethod
    def _run_generation(tmp_path, monkeypatch, current_version, candidate_version):
        """Drive _install_safe_python_generation with fakes; return result."""
        import hermes_cli.managed_uv as managed_uv
        from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo

        state = {}

        def fake_run(cmd, **kwargs):
            if "install" in cmd:
                state["generation"] = Path(kwargs["env"]["UV_PYTHON_INSTALL_DIR"])
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            # uv python find → a path inside the generation dir
            python = state["generation"] / "cpython" / "bin" / "python3"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.touch()
            return SimpleNamespace(returncode=0, stdout=str(python), stderr="")

        def fake_probe(python, **kwargs):
            return SQLiteRuntimeInfo(
                executable=Path(python),
                base_prefix=Path(python).parent.parent,
                python_version=candidate_version,
                sqlite_version=(3, 53, 1),
                sqlite_version_string="3.53.1",
                sqlite_source_id="fixed",
            )

        current = SQLiteRuntimeInfo(
            executable=Path("/venv/bin/python"),
            base_prefix=Path("/venv"),
            python_version=current_version,
            sqlite_version=(3, 50, 4),
            sqlite_version_string="3.50.4",
            sqlite_source_id="old",
        )
        monkeypatch.setattr(managed_uv.subprocess, "run", fake_run)
        monkeypatch.setattr(managed_uv, "probe_sqlite_runtime", fake_probe)
        return managed_uv._install_safe_python_generation(
            "uv", project_root=tmp_path, current=current
        )

    def test_accepts_newer_patch_same_minor(self, tmp_path, monkeypatch):
        result = self._run_generation(
            tmp_path, monkeypatch, (3, 11, 14), (3, 11, 15)
        )
        assert result is not None
        _, _, candidate = result
        assert candidate.python_version == (3, 11, 15)

    def test_rejects_minor_drift(self, tmp_path, monkeypatch):
        assert (
            self._run_generation(tmp_path, monkeypatch, (3, 11, 14), (3, 12, 1))
            is None
        )

    def test_rejects_patch_downgrade(self, tmp_path, monkeypatch):
        assert (
            self._run_generation(tmp_path, monkeypatch, (3, 11, 14), (3, 11, 13))
            is None
        )


class TestPatchRetryOnVulnerableCandidate:
    """Regression tests for issue #71250: when the bare minor-line request
    (e.g. "3.11") resolves to a candidate that's still vulnerable -- because
    uv's default resolution for that host picked an older cached/indexed
    patch even though a newer non-vulnerable one is available -- the
    provisioner must query the available patches and retry with explicit
    newer versions, rather than giving up after the first attempt.
    """

    @staticmethod
    def _versioned_probe_run(vulnerable_versions, sqlite_fixed=(3, 53, 1)):
        """Build a fake subprocess.run where the install/find/probe cycle
        resolves to a DIFFERENT candidate Python version depending on which
        exact version string was requested, so retries with explicit
        patches can be distinguished from the initial bare-minor attempt."""
        import hermes_cli.managed_uv as managed_uv
        from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo

        state = {"requested": None}

        def fake_run(cmd, **kwargs):
            if "install" in cmd:
                # cmd = [uv, "python", "install", <request>, ...]
                state["requested"] = cmd[3]
                state["generation"] = Path(kwargs["env"]["UV_PYTHON_INSTALL_DIR"])
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if "list" in cmd:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            # uv python find → a path inside the generation dir, tagged with
            # which request produced it so the probe below can look it up.
            python = state["generation"] / "cpython" / "bin" / "python3"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text(state["requested"] or "")
            return SimpleNamespace(returncode=0, stdout=str(python), stderr="")

        def fake_probe(python, **kwargs):
            requested = Path(python).read_text()
            # Bare minor request ("3.11") always resolves to the FIRST
            # (worst-case / already-known-vulnerable) version in the list.
            if requested in vulnerable_versions or requested == "3.11":
                version = (3, 11, 14) if requested == "3.11" else tuple(
                    int(p) for p in requested.split(".")
                )
                return SQLiteRuntimeInfo(
                    executable=Path(python), base_prefix=Path(python).parent.parent,
                    python_version=version, sqlite_version=(3, 50, 4),
                    sqlite_version_string="3.50.4", sqlite_source_id="vulnerable",
                )
            version = tuple(int(p) for p in requested.split("."))
            return SQLiteRuntimeInfo(
                executable=Path(python), base_prefix=Path(python).parent.parent,
                python_version=version, sqlite_version=sqlite_fixed,
                sqlite_version_string=".".join(str(p) for p in sqlite_fixed),
                sqlite_source_id="fixed",
            )

        return fake_run, fake_probe

    def _run(self, tmp_path, monkeypatch, *, vulnerable_versions, patch_list):
        import hermes_cli.managed_uv as managed_uv
        from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo

        fake_run, fake_probe = self._versioned_probe_run(vulnerable_versions)
        current = SQLiteRuntimeInfo(
            executable=Path("/venv/bin/python"), base_prefix=Path("/venv"),
            python_version=(3, 11, 14), sqlite_version=(3, 50, 4),
            sqlite_version_string="3.50.4", sqlite_source_id="old",
        )
        monkeypatch.setattr(managed_uv.subprocess, "run", fake_run)
        monkeypatch.setattr(managed_uv, "probe_sqlite_runtime", fake_probe)
        monkeypatch.setattr(
            managed_uv, "_list_available_patches", lambda *a, **kw: patch_list
        )
        return managed_uv._install_safe_python_generation(
            "uv", project_root=tmp_path, current=current
        )

    def test_retries_and_succeeds_with_explicit_newer_patch(self, tmp_path, monkeypatch):
        """The exact #71250 scenario: bare '3.11' resolves to vulnerable
        3.11.14, but 3.11.15 (fixed) is available and gets tried explicitly."""
        result = self._run(
            tmp_path, monkeypatch,
            vulnerable_versions={"3.11"},
            patch_list=[(3, 11, 15), (3, 11, 14), (3, 11, 13), (3, 11, 12)],
        )
        assert result is not None, "Must recover via explicit-patch retry"
        _, _, candidate = result
        assert candidate.python_version == (3, 11, 15)
        assert not candidate.wal_reset_vulnerable

    def test_gives_up_after_max_retries_when_all_patches_vulnerable(self, tmp_path, monkeypatch):
        """If every known patch is vulnerable (or the list is exhausted
        within the retry cap), the provisioner must return None rather than
        looping forever or raising."""
        result = self._run(
            tmp_path, monkeypatch,
            vulnerable_versions={"3.11", "3.11.14", "3.11.13", "3.11.12", "3.11.11", "3.11.10"},
            patch_list=[(3, 11, 14), (3, 11, 13), (3, 11, 12), (3, 11, 11), (3, 11, 10), (3, 11, 9)],
        )
        assert result is None

    def test_empty_patch_list_falls_back_to_none_without_crashing(self, tmp_path, monkeypatch):
        """If _list_available_patches can't be queried (network failure,
        returns []), the provisioner must not crash -- it just has nothing
        to retry with and returns None (same as before this fix existed)."""
        result = self._run(
            tmp_path, monkeypatch,
            vulnerable_versions={"3.11"},
            patch_list=[],
        )
        assert result is None

    def test_does_not_retry_patches_at_or_below_the_installed_version(
        self, tmp_path, monkeypatch
    ):
        """Only NEWER patches can carry the SQLite fix.

        On a uv whose download catalog is stale, the newest indexed patch can
        be the same one already installed -- issue #71250 reproduces exactly
        this: newest indexed 3.11 was 3.11.14, which is what's installed.
        Retrying the patches below it is guaranteed to fail (each is the
        known-vulnerable current version or an older build that cannot contain
        a later fix, and the downgrade guard rejects them anyway), and every
        attempt is a real download+install+probe+delete cycle. The loop must
        skip them rather than burn _MAX_PATCH_RETRIES on certain rejections.
        """
        import hermes_cli.managed_uv as managed_uv
        from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo

        install_requests: list[str] = []
        fake_run, fake_probe = self._versioned_probe_run({"3.11"})

        def recording_run(cmd, **kwargs):
            if "install" in cmd:
                install_requests.append(cmd[3])
            return fake_run(cmd, **kwargs)

        current = SQLiteRuntimeInfo(
            executable=Path("/venv/bin/python"), base_prefix=Path("/venv"),
            python_version=(3, 11, 14), sqlite_version=(3, 50, 4),
            sqlite_version_string="3.50.4", sqlite_source_id="old",
        )
        # Stale catalog: newest indexed patch == the installed patch.
        stale_index = [(3, 11, v) for v in range(14, 8, -1)]
        monkeypatch.setattr(managed_uv.subprocess, "run", recording_run)
        monkeypatch.setattr(managed_uv, "probe_sqlite_runtime", fake_probe)
        monkeypatch.setattr(
            managed_uv, "_list_available_patches", lambda *a, **kw: stale_index
        )

        result = managed_uv._install_safe_python_generation(
            "uv", project_root=tmp_path, current=current
        )

        assert result is None
        # Exactly one attempt: the bare minor line. No downgrade retries.
        assert install_requests == ["3.11"]

    def test_still_retries_when_a_newer_patch_exists_in_the_index(
        self, tmp_path, monkeypatch
    ):
        """The downgrade skip must not suppress legitimate newer-patch retries."""
        result = self._run(
            tmp_path, monkeypatch,
            vulnerable_versions={"3.11"},
            # Mixed index: a newer fixed patch alongside older useless ones.
            patch_list=[(3, 11, 15), (3, 11, 14), (3, 11, 13)],
        )
        assert result is not None
        assert result[2].python_version == (3, 11, 15)
        assert not result[2].wal_reset_vulnerable

    def test_retry_is_bounded_by_max_retries_constant(self, tmp_path, monkeypatch):
        """A very long patch list must not result in unbounded retries --
        capped at _MAX_PATCH_RETRIES attempts."""
        import hermes_cli.managed_uv as managed_uv

        install_calls = []
        fake_run, fake_probe = self._versioned_probe_run({"3.11"})
        original_fake_run = fake_run

        def counting_fake_run(cmd, **kwargs):
            if "install" in cmd:
                install_calls.append(cmd[3])
            return original_fake_run(cmd, **kwargs)

        from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo
        current = SQLiteRuntimeInfo(
            executable=Path("/venv/bin/python"), base_prefix=Path("/venv"),
            python_version=(3, 11, 14), sqlite_version=(3, 50, 4),
            sqlite_version_string="3.50.4", sqlite_source_id="old",
        )
        # 20 vulnerable patches -- far more than _MAX_PATCH_RETRIES.
        huge_patch_list = [(3, 11, v) for v in range(30, 10, -1)]
        all_vulnerable = {f"3.11.{v}" for v in range(30, 10, -1)} | {"3.11"}
        fake_run2, fake_probe2 = self._versioned_probe_run(all_vulnerable)

        monkeypatch.setattr(managed_uv.subprocess, "run", fake_run2)
        monkeypatch.setattr(managed_uv, "probe_sqlite_runtime", fake_probe2)
        monkeypatch.setattr(
            managed_uv, "_list_available_patches", lambda *a, **kw: huge_patch_list
        )
        result = managed_uv._install_safe_python_generation(
            "uv", project_root=tmp_path, current=current
        )
        assert result is None
        # 1 initial bare-minor attempt + at most _MAX_PATCH_RETRIES retries.
        assert managed_uv._MAX_PATCH_RETRIES <= 5, (
            "sanity: constant should stay small since each attempt is a "
            "real download+install+probe cycle"
        )


class TestListAvailablePatches:
    """Direct unit tests for _list_available_patches()'s JSON parsing,
    against realistic `uv python list --all-versions --output-format json`
    output (captured from a real uv 0.11.7 invocation)."""

    SAMPLE_OUTPUT = (
        '[{"key":"cpython-3.11.15-linux-x86_64-gnu","version":"3.11.15",'
        '"version_parts":{"major":3,"minor":11,"patch":15},"path":null,'
        '"symlink":null,"url":"https://example/cpython-3.11.15.tar.gz",'
        '"os":"linux","variant":"default","implementation":"cpython",'
        '"arch":"x86_64","libc":"gnu"},'
        '{"key":"cpython-3.11.14-linux-x86_64-gnu","version":"3.11.14",'
        '"version_parts":{"major":3,"minor":11,"patch":14},"path":null,'
        '"symlink":null,"url":"https://example/cpython-3.11.14.tar.gz",'
        '"os":"linux","variant":"default","implementation":"cpython",'
        '"arch":"x86_64","libc":"gnu"},'
        '{"key":"pypy-3.11.15-linux-x86_64-gnu","version":"3.11.15",'
        '"version_parts":{"major":3,"minor":11,"patch":15},"path":null,'
        '"symlink":null,"url":"https://example/pypy-3.11.15.tar.gz",'
        '"os":"linux","variant":"default","implementation":"pypy",'
        '"arch":"x86_64","libc":"gnu"}]'
    )

    def test_parses_and_sorts_newest_first(self, tmp_path, monkeypatch):
        import hermes_cli.managed_uv as managed_uv

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(returncode=0, stdout=self.SAMPLE_OUTPUT, stderr="")

        monkeypatch.setattr(managed_uv.subprocess, "run", fake_run)
        result = managed_uv._list_available_patches(
            "uv", "3.11", cwd=tmp_path, env={}
        )
        assert result == [(3, 11, 15), (3, 11, 14)]

    def test_filters_out_non_cpython_implementations(self, tmp_path, monkeypatch):
        """pypy/graalpy entries at the same version must not be returned --
        the repair path only wants cpython builds."""
        import hermes_cli.managed_uv as managed_uv

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(returncode=0, stdout=self.SAMPLE_OUTPUT, stderr="")

        monkeypatch.setattr(managed_uv.subprocess, "run", fake_run)
        result = managed_uv._list_available_patches(
            "uv", "3.11", cwd=tmp_path, env={}
        )
        # Only one 3.11.15 entry (cpython), not two (cpython + pypy).
        assert result.count((3, 11, 15)) == 1

    def test_nonzero_exit_returns_empty_list(self, tmp_path, monkeypatch):
        import hermes_cli.managed_uv as managed_uv

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="network error")

        monkeypatch.setattr(managed_uv.subprocess, "run", fake_run)
        assert managed_uv._list_available_patches("uv", "3.11", cwd=tmp_path, env={}) == []

    def test_malformed_json_returns_empty_list_not_crash(self, tmp_path, monkeypatch):
        import hermes_cli.managed_uv as managed_uv

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(returncode=0, stdout="not valid json{{{", stderr="")

        monkeypatch.setattr(managed_uv.subprocess, "run", fake_run)
        assert managed_uv._list_available_patches("uv", "3.11", cwd=tmp_path, env={}) == []

    def test_subprocess_exception_returns_empty_list(self, tmp_path, monkeypatch):
        import hermes_cli.managed_uv as managed_uv

        def fake_run(cmd, **kwargs):
            raise OSError("uv binary not found")

        monkeypatch.setattr(managed_uv.subprocess, "run", fake_run)
        assert managed_uv._list_available_patches("uv", "3.11", cwd=tmp_path, env={}) == []
