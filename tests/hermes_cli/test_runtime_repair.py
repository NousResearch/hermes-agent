"""Tests for hermes_cli.runtime_repair — the venv/SQLite repair machinery."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _pinned_uv_resolves(monkeypatch):
    """repair_vulnerable_runtime() resolves uv itself via pm.ensure.uv;
    pin it so no test realizes the pm store for real.

    NOTE: the module object must be reached through sys.modules — both the
    string form "pm.ensure.uv" and `import pm.ensure as x` resolve through
    the `ensure` FUNCTION re-exported by pm/__init__, which shadows the
    submodule attribute of the same name on the package.
    """
    import importlib

    pm_ensure = importlib.import_module("pm.ensure")
    monkeypatch.setattr(pm_ensure, "uv", lambda *a, **k: ("uv", {}))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


class TestManagedPythonStore:
    def test_store_is_checkout_scoped_across_profiles(self, tmp_path, monkeypatch):
        from hermes_cli.runtime_repair import managed_python_install_dir

        checkout = tmp_path / "checkout"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "alpha"))
        alpha = managed_python_install_dir(checkout)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "beta"))
        beta = managed_python_install_dir(checkout)

        expected = checkout / ".hermes-runtime" / "python"
        assert alpha == expected
        assert beta == expected

    def test_environment_is_private_and_sanitized(self, tmp_path):
        from hermes_cli.runtime_repair import managed_python_env

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


@pytest.mark.skipif(sys.platform == "win32",
                    reason="POSIX-only: fixtures build the bin/ (not Scripts/) venv layout")
class TestRuntimeRepair:
    def test_safe_runtime_is_a_noop(self, tmp_path):
        from hermes_cli.runtime_repair import repair_vulnerable_runtime

        root, live, sentinel = _make_runtime_install(tmp_path)
        current = _runtime_info(live / "bin" / "python", (3, 53, 1))
        with patch(
                 "hermes_cli.runtime_repair.probe_sqlite_runtime",
                 return_value=current,
             ), \
             patch(
                 "hermes_cli.runtime_repair._install_safe_python_generation"
             ) as mock_install:
            result = repair_vulnerable_runtime(project_root=root)

        assert result.status == "safe"
        assert result.sqlite_before == "3.53.1"
        assert result.sqlite_after == "3.53.1"
        assert sentinel.read_text(encoding="utf-8") == "live"
        assert not (root / ".hermes-runtime").exists()
        mock_install.assert_not_called()

    def test_stage_candidate_sync_keeps_uv_project_config(self, tmp_path):
        from hermes_cli.runtime_repair import _stage_candidate_venv

        root = tmp_path / "checkout"
        root.mkdir()
        (root / "uv.lock").write_text("# lock\n", encoding="utf-8")
        generation = root / ".hermes-runtime" / "python" / "gen"
        python = generation / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("py", encoding="utf-8")

        calls = []

        def fake_run(argv, **kwargs):
            calls.append((list(argv), kwargs.get("env")))
            return MagicMock(returncode=0)

        with patch("hermes_cli.runtime_repair.subprocess.run", side_effect=fake_run), \
             patch(
                 "hermes_cli.runtime_repair._smoke_candidate_venv",
                 return_value=(True, "", None),
             ):
            candidate = _stage_candidate_venv(
                "uv",
                project_root=root,
                generation=generation,
                python=python,
            )

        assert candidate is not None
        assert len(calls) == 2
        venv_argv, venv_env = calls[0]
        sync_argv, sync_env = calls[1]
        assert venv_argv[:2] == ["uv", "venv"]
        assert "--no-config" in venv_argv
        assert venv_env.get("UV_NO_CONFIG") == "1"
        assert sync_argv[:2] == ["uv", "sync"]
        assert "--locked" in sync_argv
        assert "--no-config" not in sync_argv
        assert "UV_NO_CONFIG" not in sync_env

    def test_failed_candidate_preserves_live_venv(self, tmp_path):
        from hermes_cli.runtime_repair import (
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

        with patch(
                 "hermes_cli.runtime_repair.probe_sqlite_runtime",
                 side_effect=[current, current],
             ), \
             patch(
                 "hermes_cli.runtime_repair._install_safe_python_generation",
                 return_value=(generation, candidate_python, fixed),
             ), \
             patch(
                 "hermes_cli.runtime_repair._stage_candidate_venv",
                 return_value=None,
             ):
            result = repair_vulnerable_runtime(project_root=root)

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

    def test_safe_runtime_sweeps_old_stale_backups(self, tmp_path):
        """A fixed runtime reclaims aged venv.stale.runtime-* leftovers
        (issue #73109) but leaves fresh ones (possible in-flight repair)."""
        import os
        import time as _time

        from hermes_cli.runtime_repair import repair_vulnerable_runtime

        root, live, sentinel = _make_runtime_install(tmp_path)
        old_backup = root / f"{live.name}.stale.runtime-1-2-aaaa"
        (old_backup / "bin").mkdir(parents=True)
        (old_backup / "bin" / "python").write_text("old", encoding="utf-8")
        stale_mtime = _time.time() - 7200
        os.utime(old_backup, (stale_mtime, stale_mtime))

        fresh_backup = root / f"{live.name}.stale.runtime-9-9-bbbb"
        (fresh_backup / "bin").mkdir(parents=True)

        current = _runtime_info(live / "bin" / "python", (3, 53, 1))
        with patch(
                 "hermes_cli.runtime_repair.probe_sqlite_runtime",
                 return_value=current,
             ):
            result = repair_vulnerable_runtime(project_root=root)

        assert result.status == "safe"
        assert not old_backup.exists(), "aged stale backup must be reclaimed"
        assert fresh_backup.exists(), "fresh backup may be an in-flight repair"
        assert sentinel.read_text(encoding="utf-8") == "live"

    def test_successful_repair_removes_parked_backup(self, tmp_path):
        """After a successful cutover the parked venv is removed instead of
        leaking ~1 GB at the project root forever (issue #73109)."""
        from hermes_cli.runtime_repair import repair_vulnerable_runtime

        root, live, sentinel = _make_runtime_install(tmp_path)
        current = _runtime_info(live / "bin" / "python", (3, 50, 4))
        generation = root / ".hermes-runtime" / "python" / "generation-test"
        candidate_python = generation / "bin" / "python"
        candidate_python.parent.mkdir(parents=True)
        candidate_python.write_text("candidate interpreter", encoding="utf-8")
        fixed = _runtime_info(candidate_python, (3, 53, 1))
        candidate_venv = root / ".hermes-runtime" / "venv-candidate"
        (candidate_venv / "bin").mkdir(parents=True)
        (candidate_venv / "bin" / "python").write_text(
            "candidate venv interpreter", encoding="utf-8"
        )

        with patch(
                 "hermes_cli.runtime_repair.probe_sqlite_runtime",
                 side_effect=[current, current],
             ), \
             patch(
                 "hermes_cli.runtime_repair._install_safe_python_generation",
                 return_value=(generation, candidate_python, fixed),
             ), \
             patch(
                 "hermes_cli.runtime_repair._stage_candidate_venv",
                 return_value=candidate_venv,
             ), \
             patch(
                 "hermes_cli.runtime_repair._smoke_candidate_venv",
                 return_value=(True, "", fixed),
             ):
            result = repair_vulnerable_runtime(project_root=root)

        assert result.status == "repaired"
        assert result.backup_venv is not None
        assert not result.backup_venv.exists(), (
            "parked venv must be removed after a successful repair"
        )
        leftovers = list(root.glob(f"{live.name}.stale.runtime-*"))
        assert leftovers == [], f"no stale markers may remain: {leftovers}"


class TestRuntimeCutover:
    def test_os_lock_blocks_concurrent_repair_and_releases(self, tmp_path):
        from hermes_cli.runtime_repair import _acquire_repair_lock, _release_repair_lock

        runtime_root = tmp_path / ".hermes-runtime"
        first = _acquire_repair_lock(runtime_root)
        assert first is not None
        assert _acquire_repair_lock(runtime_root) is None

        _release_repair_lock(first)
        second = _acquire_repair_lock(runtime_root)
        assert second is not None
        _release_repair_lock(second)

    def test_post_swap_smoke_failure_rolls_back_live_venv(self, tmp_path):
        from hermes_cli.runtime_repair import _cut_over_candidate

        root, live, sentinel = _make_runtime_install(tmp_path)
        runtime_root = root / ".hermes-runtime"
        candidate = runtime_root / "venv-candidate-test"
        candidate.mkdir(parents=True)
        (candidate / "sentinel").write_text("candidate", encoding="utf-8")
        rejected_info = _runtime_info(candidate / "bin" / "python", (3, 50, 4))

        with patch(
            "hermes_cli.runtime_repair._smoke_candidate_venv",
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
        assert (live / "bin" / "python").exists() or (
            live / "Scripts" / "python.exe"
        ).exists()
        assert not candidate.exists()
        assert not list(runtime_root.glob("venv-rejected-*"))


class TestRuntimeRequestMinorLine:
    """The repair must request the CPython minor line, not the exact patch.

    Real-world constraint (verified live, July 2026): every published
    python-build-standalone artifact for 3.11.14 links vulnerable SQLite
    3.50.4 — even with --reinstall. The fixed SQLite (3.53.1) only exists
    from 3.11.15. An exact-patch pin makes the repair permanently
    impossible on such installs.
    """

    def test_requests_minor_line(self):
        from hermes_cli.runtime_repair import _runtime_request

        info = _runtime_info(Path("/venv/bin/python"), (3, 50, 4))
        assert _runtime_request(info) == "3.11"

    @staticmethod
    def _run_generation(tmp_path, monkeypatch, current_version, candidate_version):
        """Drive _install_safe_python_generation with fakes; return result."""
        import hermes_cli.runtime_repair as runtime_repair
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
        monkeypatch.setattr(runtime_repair.subprocess, "run", fake_run)
        monkeypatch.setattr(runtime_repair, "probe_sqlite_runtime", fake_probe)
        return runtime_repair._install_safe_python_generation(
            "uv", project_root=tmp_path, current=current
        )

    def test_accepts_newer_patch_same_minor(self, tmp_path, monkeypatch):
        result = self._run_generation(
            tmp_path, monkeypatch, (3, 11, 14), (3, 11, 15)
        )
        assert result is not None
        _, _, candidate = result
        assert candidate.python_version == (3, 11, 15)


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
        import hermes_cli.runtime_repair as runtime_repair
        from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo

        fake_run, fake_probe = self._versioned_probe_run(vulnerable_versions)
        current = SQLiteRuntimeInfo(
            executable=Path("/venv/bin/python"), base_prefix=Path("/venv"),
            python_version=(3, 11, 14), sqlite_version=(3, 50, 4),
            sqlite_version_string="3.50.4", sqlite_source_id="old",
        )
        monkeypatch.setattr(runtime_repair.subprocess, "run", fake_run)
        monkeypatch.setattr(runtime_repair, "probe_sqlite_runtime", fake_probe)
        monkeypatch.setattr(
            runtime_repair, "_list_available_patches", lambda *a, **kw: patch_list
        )
        return runtime_repair._install_safe_python_generation(
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

    def test_retry_is_bounded_by_max_retries_constant(self, tmp_path, monkeypatch):
        """A very long patch list must not result in unbounded retries -- capped at
        _MAX_PATCH_RETRIES attempts.  After exhausting same-minor retries the
        fallback tries the next minor line, which may succeed."""
        import hermes_cli.runtime_repair as runtime_repair

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

        install_calls = []

        def counting_fake_run(cmd, **kwargs):
            if "install" in cmd:
                install_calls.append(cmd[3])
            return fake_run2(cmd, **kwargs)

        monkeypatch.setattr(runtime_repair.subprocess, "run", counting_fake_run)
        monkeypatch.setattr(runtime_repair, "probe_sqlite_runtime", fake_probe2)
        monkeypatch.setattr(
            runtime_repair, "_list_available_patches", lambda *a, **kw: huge_patch_list
        )
        result = runtime_repair._install_safe_python_generation(
            "uv", project_root=tmp_path, current=current
        )
        # The same-minor retries are bounded, but the minor-line fallback
        # (3.11 → 3.12) succeeds because the mock returns a fixed build.
        assert result is not None, (
            "Minor-line fallback should find a fixed 3.12 build"
        )
        # 1 initial bare-minor attempt + at most _MAX_PATCH_RETRIES retries.
        assert runtime_repair._MAX_PATCH_RETRIES <= 5, (
            "sanity: constant should stay small since each attempt is a "
            "real download+install+probe cycle"
        )
        same_minor_explicit = [
            call for call in install_calls if call.startswith("3.11.")
        ]
        assert len(same_minor_explicit) <= runtime_repair._MAX_PATCH_RETRIES, (
            f"same-minor explicit retries must be capped: {same_minor_explicit}"
        )
        assert install_calls[0] == "3.11"
        # The run ends the moment the bare next-minor fallback succeeds.
        assert install_calls[-1] == "3.12"
        assert install_calls.count("3.12") == 1


class TestMinorLineFallForward:
    """Regression tests for issue #76106: when EVERY build on the current
    minor line (e.g. all of 3.11 on Windows) links a vulnerable SQLite,
    the provisioner must fall forward to the next supported minor line
    (3.12, then 3.13) -- first via a bare minor request, then via explicit
    patches on that line -- instead of leaving the user stuck on every
    `hermes update` with no path to a fixed runtime.
    """

    @staticmethod
    def _mapped_run(resolutions, fixed_versions, install_calls):
        """Fake subprocess.run/probe pair driven by explicit tables:

        - *resolutions*: request string -> python_version tuple the probe
          reports for that request (bare minors resolve like uv would).
        - *fixed_versions*: set of version tuples that link FIXED SQLite;
          everything else probes as vulnerable 3.50.4.
        - *install_calls*: list collecting each `uv python install` request,
          in order, so tests can assert the actual request sequence.
        """
        from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo

        state: dict = {"requested": None}

        def fake_run(cmd, **kwargs):
            if "install" in cmd:
                # cmd = [uv, "python", "install", <request>, ...]
                state["requested"] = cmd[3]
                state["generation"] = Path(kwargs["env"]["UV_PYTHON_INSTALL_DIR"])
                install_calls.append(cmd[3])
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if "list" in cmd:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            # uv python find → a path inside the generation dir, tagged with
            # the request that produced it so the probe can look it up.
            python = state["generation"] / "cpython" / "bin" / "python3"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text(state["requested"] or "")
            return SimpleNamespace(returncode=0, stdout=str(python), stderr="")

        def fake_probe(python, **kwargs):
            requested = Path(python).read_text()
            version = resolutions[requested]
            if version in fixed_versions:
                return SQLiteRuntimeInfo(
                    executable=Path(python),
                    base_prefix=Path(python).parent.parent,
                    python_version=version, sqlite_version=(3, 53, 1),
                    sqlite_version_string="3.53.1", sqlite_source_id="fixed",
                )
            return SQLiteRuntimeInfo(
                executable=Path(python),
                base_prefix=Path(python).parent.parent,
                python_version=version, sqlite_version=(3, 50, 4),
                sqlite_version_string="3.50.4", sqlite_source_id="vulnerable",
            )

        return fake_run, fake_probe

    @staticmethod
    def _current_3_11_14():
        from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo

        return SQLiteRuntimeInfo(
            executable=Path("/venv/bin/python"), base_prefix=Path("/venv"),
            python_version=(3, 11, 14), sqlite_version=(3, 50, 4),
            sqlite_version_string="3.50.4", sqlite_source_id="old",
        )

    def test_explicit_patch_fallback_when_bare_next_minor_is_vulnerable(
        self, tmp_path, monkeypatch
    ):
        """The review-gap scenario from #76252: the bare '3.12' request
        resolves to a VULNERABLE 3.12 build, but an explicit 3.12 patch
        links fixed SQLite -- the `_list_available_patches(..., '3.12', ...)`
        fallback branch must run, skip the already-tried bare resolution,
        and succeed via the explicit patch."""
        import hermes_cli.runtime_repair as runtime_repair

        install_calls = []
        fake_run, fake_probe = self._mapped_run(
            resolutions={
                "3.11": (3, 11, 14),      # bare current minor: vulnerable
                "3.12": (3, 12, 11),      # bare next minor: ALSO vulnerable
                "3.12.10": (3, 12, 10),   # explicit patch: fixed
            },
            fixed_versions={(3, 12, 10)},
            install_calls=install_calls,
        )
        patch_lists = {
            # No newer 3.11 patch exists (the Windows #76106 reality).
            "3.11": [(3, 11, 14), (3, 11, 13)],
            # Newest 3.12 is the same build the bare request resolved to.
            "3.12": [(3, 12, 11), (3, 12, 10)],
        }
        monkeypatch.setattr(runtime_repair.subprocess, "run", fake_run)
        monkeypatch.setattr(runtime_repair, "probe_sqlite_runtime", fake_probe)
        monkeypatch.setattr(
            runtime_repair, "_list_available_patches",
            lambda uv_bin, minor, **kw: patch_lists[minor],
        )

        result = runtime_repair._install_safe_python_generation(
            "uv", project_root=tmp_path, current=self._current_3_11_14()
        )
        assert result is not None, (
            "Explicit-patch fallback on the next minor line must recover"
        )
        _, _, candidate = result
        assert candidate.python_version == (3, 12, 10)
        assert not candidate.wal_reset_vulnerable
        # The actual uv-install request sequence: bare current minor, then
        # bare next minor, then STRAIGHT to the fixed explicit patch --
        # 3.12.11 must NOT be re-requested explicitly, because the bare
        # '3.12' attempt already resolved to (and rejected) that build.
        assert install_calls == ["3.11", "3.12", "3.12.10"]

    def test_returns_none_with_bounded_attempts_when_all_minors_exhausted(
        self, tmp_path, monkeypatch
    ):
        """When every build on every supported minor line (3.11-3.13) is
        vulnerable, the provisioner must give up with None -- and the total
        install workload must stay bounded by _MAX_PATCH_RETRIES per line."""
        import hermes_cli.runtime_repair as runtime_repair

        install_calls = []
        resolutions = {"3.11": (3, 11, 14), "3.12": (3, 12, 30), "3.13": (3, 13, 30)}
        patch_lists = {}
        for minor in (11, 12, 13):
            versions = [(3, minor, v) for v in range(30, 10, -1)]  # 20 patches
            patch_lists[f"3.{minor}"] = versions
            for version in versions:
                resolutions[".".join(str(p) for p in version)] = version

        fake_run, fake_probe = self._mapped_run(
            resolutions=resolutions, fixed_versions=set(),
            install_calls=install_calls,
        )
        monkeypatch.setattr(runtime_repair.subprocess, "run", fake_run)
        monkeypatch.setattr(runtime_repair, "probe_sqlite_runtime", fake_probe)
        monkeypatch.setattr(
            runtime_repair, "_list_available_patches",
            lambda uv_bin, minor, **kw: patch_lists[minor],
        )

        result = runtime_repair._install_safe_python_generation(
            "uv", project_root=tmp_path, current=self._current_3_11_14()
        )
        assert result is None, "Nothing fixed anywhere: must give up cleanly"

        cap = runtime_repair._MAX_PATCH_RETRIES
        # Per line: one bare request + at most _MAX_PATCH_RETRIES explicit
        # patches; three lines total (3.11, 3.12, 3.13) and nothing beyond
        # 3.13 (requires-python is <3.14).
        assert install_calls.count("3.11") == 1
        assert install_calls.count("3.12") == 1
        assert install_calls.count("3.13") == 1
        assert not any(call.startswith("3.14") for call in install_calls)
        for minor in (11, 12, 13):
            explicit = [
                call for call in install_calls
                if call.startswith(f"3.{minor}.")
            ]
            assert len(explicit) <= cap, (
                f"3.{minor} explicit retries must be capped at {cap}: {explicit}"
            )
        assert len(install_calls) <= 3 * (1 + cap)


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
        import hermes_cli.runtime_repair as runtime_repair

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(returncode=0, stdout=self.SAMPLE_OUTPUT, stderr="")

        monkeypatch.setattr(runtime_repair.subprocess, "run", fake_run)
        result = runtime_repair._list_available_patches(
            "uv", "3.11", cwd=tmp_path, env={}
        )
        assert result == [(3, 11, 15), (3, 11, 14)]

    def test_subprocess_exception_returns_empty_list(self, tmp_path, monkeypatch):
        import hermes_cli.runtime_repair as runtime_repair

        def fake_run(cmd, **kwargs):
            raise OSError("uv binary not found")

        monkeypatch.setattr(runtime_repair.subprocess, "run", fake_run)
        assert runtime_repair._list_available_patches("uv", "3.11", cwd=tmp_path, env={}) == []


class TestPmUvResolution:
    """The repair resolves uv through pm — never a foreign uv, never its
    own acquisition (the retired managed_uv half)."""

    def test_pm_uv_unavailable_skips_without_touching_the_venv(
        self, tmp_path, monkeypatch
    ):
        from hermes_cli.runtime_repair import repair_vulnerable_runtime

        root, live, sentinel = _make_runtime_install(
            tmp_path, windows=sys.platform == "win32"
        )
        import importlib

        pm_ensure = importlib.import_module("pm.ensure")
        monkeypatch.setattr(pm_ensure, "uv", lambda *a, **k: (None, {}))

        result = repair_vulnerable_runtime(project_root=root)

        assert result.status == "skipped"
        assert "pinned uv unavailable" in result.detail
        assert sentinel.read_text(encoding="utf-8") == "live"


class TestDefaultLiveVenv:
    """_default_live_venv() must cover BOTH install layouts (venv/ and .venv/).

    Historically repair hardcoded venv/, so uv-default/.venv checkouts got
    'not-applicable' on every hermes update and stayed on journal_mode=DELETE
    (2,600x slower state.db appends) while the WAL warning promised repair.
    """

    def _checkout(self, tmp_path, *dirs):
        windows = sys.platform == "win32"
        root = tmp_path / "checkout"
        root.mkdir()
        (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        for d in dirs:
            bin_dir = root / d / ("Scripts" if windows else "bin")
            bin_dir.mkdir(parents=True)
            (bin_dir / ("python.exe" if windows else "python")).write_text(
                "py", encoding="utf-8"
            )
        return root

    @pytest.mark.platforms("linux")
    def test_dot_venv_only_is_targeted(self, tmp_path):
        from hermes_cli.runtime_repair import _default_live_venv

        root = self._checkout(tmp_path, ".venv")
        assert _default_live_venv(root) == root / ".venv"

    def test_managed_venv_takes_precedence(self, tmp_path):
        from hermes_cli.runtime_repair import _default_live_venv

        root = self._checkout(tmp_path, "venv", ".venv")
        assert _default_live_venv(root) == root / "venv"

    def test_neither_layout_keeps_not_applicable(self, tmp_path):
        from hermes_cli.runtime_repair import (
            _default_live_venv,
            repair_vulnerable_runtime,
        )

        root = self._checkout(tmp_path)
        # Neither venv nor .venv has an interpreter -> repair is not applicable.
        assert _default_live_venv(root) == root / "venv"
        result = repair_vulnerable_runtime(project_root=root)
        assert result.status == "not-applicable"


class TestVenvPythonUpdateBoundary:
    """``_venv_python`` must survive a hermes_constants predating its symbol.

    ``hermes update`` imports hermes_constants from the OLD checkout, ``git
    pull`` replaces that file, and the freshly-pulled runtime_repair then runs its
    lazy ``from hermes_constants import venv_python_path`` against the module
    object already cached in ``sys.modules``. That cached module has no such
    symbol, so the import raises — while naming the NEW file on disk, which
    plainly contains it, which is what made the error so confusing:

        cannot import name 'venv_python_path' from 'hermes_constants'
        (~/.hermes/hermes-agent/hermes_constants.py)

    It aborted the managed-Python runtime repair on the first update from any
    release older than the symbol. Same class as the ``ensure_uv()`` arity skew
    documented on ``_UvResult``.
    """

    def test_recovers_when_the_cached_module_predates_the_symbol(self, monkeypatch):
        import hermes_constants

        from hermes_cli.runtime_repair import _venv_python

        # The stale in-memory module: the symbol the new code wants is absent,
        # exactly as on an install that booted the pre-upgrade checkout. The
        # file on disk is the current one, so a reload recovers the real helper.
        monkeypatch.delattr(hermes_constants, "venv_python_path", raising=False)

        # Host-native: the subject is the reload-recovery seam, not the
        # bin/Scripts mapping — assert whatever layout the real host resolves.
        expected = Path("/opt/hermes/venv/Scripts/python.exe") \
            if sys.platform == "win32" else Path("/opt/hermes/venv/bin/python")
        assert _venv_python(Path("/opt/hermes/venv")) == expected

    def test_recovery_uses_the_shared_helper_not_a_second_copy(self, monkeypatch):
        """The reload must resolve through hermes_constants, not open-code it.

        Hand-rolling `Scripts`/`bin` here is what #76105 deduped away and what
        `test_no_open_coded_venv_layout_remains_in_hermes_cli` bans.
        """
        import hermes_constants

        from hermes_cli.runtime_repair import _venv_python

        monkeypatch.delattr(hermes_constants, "venv_python_path", raising=False)

        sentinel = Path("/sentinel/from/shared/helper")
        real_reload = __import__("importlib").reload

        def _reload_with_marker(module):
            fresh = real_reload(module)
            monkeypatch.setattr(
                fresh, "venv_python_path", lambda *a, **k: sentinel, raising=False
            )
            return fresh

        monkeypatch.setattr("importlib.reload", _reload_with_marker)
        assert _venv_python(Path("/opt/hermes/venv")) == sentinel

    def test_uses_the_real_helper_when_it_is_importable(self, monkeypatch):
        """The normal path never reloads — recovery stays a fallback."""
        from hermes_cli.runtime_repair import _venv_python

        def _no_reload(module):  # pragma: no cover - must not run
            raise AssertionError("reload must not run when the import succeeds")

        monkeypatch.setattr("importlib.reload", _no_reload)

        expected = Path("/opt/hermes/venv/Scripts/python.exe") \
            if sys.platform == "win32" else Path("/opt/hermes/venv/bin/python")
        assert _venv_python(Path("/opt/hermes/venv")) == expected



class TestWindowsRuntimeSelfLock:
    """The repair pre-flight must see the ONE holder the generic scan hides:
    the updater itself (#93032).

    A CLI ``hermes update`` runs from the venv's own python, and
    ``_detect_venv_python_processes`` excludes the calling process and its
    ancestors on purpose (correct for the dependency-sync path).  For the
    whole-venv park rename that exemption is fatal on Windows: a directory
    containing an executable mapped by a running process cannot be renamed,
    so the cutover retries burn out against a lock that cannot be released
    while the updater lives.  The repair must detect the self-lock and defer
    with honest guidance instead of provisioning a candidate for a doomed
    rename.
    """

    def _checkout(self, tmp_path):
        root, live, sentinel = _make_runtime_install(tmp_path)
        # Windows-layout interpreter so sys.executable can point inside the
        # live venv on any host (the detector only string-compares paths).
        scripts_python = live / "Scripts" / "python.exe"
        scripts_python.parent.mkdir(parents=True, exist_ok=True)
        scripts_python.write_text("live interpreter", encoding="utf-8")
        return root, live, sentinel, scripts_python

    def test_self_lock_defers_repair_before_provisioning(
        self, tmp_path, monkeypatch, capsys
    ):
        """Regression for #93032: pre-fix, the repair walks straight into the
        doomed rename (provisioning + cutover) whenever the updater itself
        maps the live venv; the park then fails with WinError 5 and the user
        gets the misleading 'next update will retry' message forever."""
        from hermes_cli import runtime_repair as repair_mod
        from hermes_cli.runtime_repair import repair_vulnerable_runtime

        root, live, sentinel, scripts_python = self._checkout(tmp_path)
        current = _runtime_info(scripts_python, (3, 50, 4))
        monkeypatch.setattr(repair_mod.platform, "system", lambda: "Windows")
        monkeypatch.setattr(sys, "executable", str(scripts_python))

        with patch(
                 "hermes_cli.runtime_repair._windows_runtime_holders",
                 return_value=(False, ""),
             ), \
             patch(
                 "hermes_cli.runtime_repair.probe_sqlite_runtime",
                 return_value=current,
             ), \
             patch(
                 "hermes_cli.runtime_repair._install_safe_python_generation"
             ) as mock_install:
            result = repair_vulnerable_runtime(project_root=root)

        assert result.status == "skipped"
        assert "live venv" in result.detail
        assert mock_install.call_count == 0, (
            "a self-locked updater must not provision a candidate it can "
            "never cut over"
        )
        assert sentinel.read_text(encoding="utf-8") == "live"
        assert not (root / ".hermes-runtime").exists()

        out = capsys.readouterr().out
        assert "SQLite runtime repair deferred" in out
        assert "will retry" not in out, (
            "the structural self-lock must not promise that retrying helps"
        )
        assert "outside" in out, "the deferral must point at an escape hatch"

    def test_non_self_locked_repair_proceeds(self, tmp_path, monkeypatch):
        """The guard must fail OPEN when the updater runs from outside the
        venv — an always-firing deferral would recreate the never-converging
        loop this fix removes (#86735 class)."""
        from hermes_cli import runtime_repair as repair_mod
        from hermes_cli.runtime_repair import repair_vulnerable_runtime

        root, live, sentinel, scripts_python = self._checkout(tmp_path)
        current = _runtime_info(scripts_python, (3, 50, 4))
        monkeypatch.setattr(repair_mod.platform, "system", lambda: "Windows")
        monkeypatch.setattr(
            sys, "executable", str(tmp_path / "outside" / "python.exe")
        )

        with patch(
                 "hermes_cli.runtime_repair._windows_runtime_holders",
                 return_value=(False, ""),
             ), \
             patch(
                 "hermes_cli.runtime_repair.probe_sqlite_runtime",
                 return_value=current,
             ), \
             patch(
                 "hermes_cli.runtime_repair._install_safe_python_generation",
                 return_value=None,
             ) as mock_install:
            result = repair_vulnerable_runtime(project_root=root)

        assert result.status == "failed"
        assert "provision" in result.detail
        mock_install.assert_called_once()
        assert sentinel.read_text(encoding="utf-8") == "live"

    def test_self_lock_is_a_noop_off_windows(self, tmp_path, monkeypatch):
        """POSIX renames work while the updater maps the venv, so the guard
        must stay Windows-only."""
        from hermes_cli import runtime_repair as repair_mod

        root, live, sentinel, scripts_python = self._checkout(tmp_path)
        monkeypatch.setattr(repair_mod.platform, "system", lambda: "Linux")
        monkeypatch.setattr(sys, "executable", str(scripts_python))

        locked, detail = repair_mod._windows_runtime_self_lock(live)
        assert (locked, detail) == (False, "")

    def test_venv_launcher_ancestor_is_a_self_lock(self, tmp_path, monkeypatch):
        r"""The venv\Scripts\hermes.exe shim stays mapped while it waits for
        this child — an ancestor running from the venv blocks the rename too."""
        from hermes_cli import runtime_repair as repair_mod

        root, live, sentinel, scripts_python = self._checkout(tmp_path)
        monkeypatch.setattr(repair_mod.platform, "system", lambda: "Windows")
        monkeypatch.setattr(
            sys, "executable", str(tmp_path / "outside" / "python.exe")
        )

        class _FakeProc:
            def __init__(self, pid, exe):
                self.pid = pid
                self._exe = exe

            def exe(self):
                return self._exe

        fake_psutil = SimpleNamespace(
            Process=lambda: SimpleNamespace(
                parents=lambda: [_FakeProc(999, str(scripts_python))],
            ),
        )
        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            locked, detail = repair_mod._windows_runtime_self_lock(live)

        assert locked
        assert "999" in detail
