"""Scratch-root validation and runner cleanup contracts; no legacy nested runner invocation."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts import run_tests_parallel as runner


@pytest.mark.platforms("posix")
def test_custom_root_validated_without_pruning_siblings(tmp_path, monkeypatch):
    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    sentinel = root / "unrelated"
    sentinel.mkdir()
    (sentinel / "keep").write_text("unchanged")
    monkeypatch.setattr(runner, "_sweep_killed_run_roots", lambda *_: pytest.fail("custom root pruned"))
    assert runner._prepare_runner_scratch_root(str(root)) == str(root)
    assert (sentinel / "keep").read_text() == "unchanged"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("kind", ["relative", "missing", "file", "writable", "symlink", "parent_symlink", "guarded"])
def test_unsafe_custom_root_rejected_without_repair(tmp_path, monkeypatch, kind):
    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    candidate = root
    if kind == "relative":
        candidate = Path("relative")
    if kind == "missing":
        candidate = tmp_path / "missing"
    if kind == "file":
        candidate = tmp_path / "file"
        candidate.write_text("not a directory")
    if kind == "writable":
        root.chmod(0o777)
    if kind in ("symlink", "parent_symlink"):
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        candidate = alias
        if kind == "parent_symlink":
            (root / "nested").mkdir(mode=0o700)
            candidate = alias / "nested"
    if kind == "guarded":
        monkeypatch.setenv("HERMES_HOME", str(root))
    before = root.stat().st_mode
    with pytest.raises((ValueError, OSError)):
        runner._validate_scratch_root(str(candidate))
    assert root.stat().st_mode == before
    assert not (tmp_path / "missing").exists()


@pytest.mark.platforms("windows")
def test_custom_root_fails_closed_without_native_ownership_contract(tmp_path):
    with pytest.raises(ValueError, match="POSIX"):
        runner._validate_scratch_root(str(tmp_path.resolve()))


@pytest.mark.platforms("posix")
def test_spawn_failure_cleans_only_current_attempt(tmp_path, monkeypatch):
    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    sentinel = root / "unrelated"
    sentinel.mkdir()
    monkeypatch.setattr(runner.subprocess, "Popen", MagicMock(side_effect=OSError("spawn failed")))
    with pytest.raises(OSError, match="spawn failed"):
        runner._run_one_file_once(tmp_path / "test_example.py", [], tmp_path, 10, scratch_root=str(root))
    assert list(root.iterdir()) == [sentinel]


@pytest.mark.platforms("posix")
def test_retry_propagates_same_root_but_owns_unique_attempts(tmp_path, monkeypatch):
    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    attempts = []
    procs, killed = [], []

    def spawn(*args, **kwargs):
        attempts.append(kwargs["env"]["TMPDIR"])
        assert kwargs["env"]["TMP"] == kwargs["env"]["TEMP"] == attempts[-1]
        assert Path(attempts[-1]).parent == root
        proc = MagicMock()
        procs.append(proc)
        proc.pid = 99999999
        proc.returncode = 1 if len(attempts) == 1 else 0
        proc.communicate.return_value = ("1 failed" if proc.returncode else "1 passed", "")
        return proc

    monkeypatch.setattr(runner.subprocess, "Popen", spawn)
    monkeypatch.setattr(runner, "_kill_tree", lambda proc, **_kw: killed.append(proc))
    monkeypatch.setattr(runner, "_kill_detached_leftovers", lambda *_: pytest.fail("broad kill"))
    result = runner._run_one_file(tmp_path / "test_example.py", [], tmp_path, 10,
                                  retries=1, scratch_root=str(root))
    assert result[1] == 0
    assert len(set(attempts)) == 2
    assert killed == procs
    assert not list(root.iterdir())


def test_default_prepare_preserves_legacy_pruner_contract(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "_runner_scratch_root", lambda: "legacy")
    monkeypatch.setattr(runner, "_sweep_killed_run_roots", calls.append)
    assert runner._prepare_runner_scratch_root(None) is None
    assert calls == ["legacy"]


@pytest.mark.platforms("posix")
def test_custom_root_rejects_wrong_owner(tmp_path, monkeypatch):
    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    owner = root.stat().st_uid
    monkeypatch.setattr(runner.os, "getuid", lambda: owner + 1)
    with pytest.raises(ValueError, match="actor-owned"):
        runner._validate_scratch_root(str(root))


@pytest.mark.platforms("posix")
def test_custom_cleanup_never_scans_or_chmods_caller_root(tmp_path, monkeypatch):
    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    attempt = root / "attempt"
    attempt.mkdir()
    sibling = root / "sibling"
    sibling.mkdir()
    monkeypatch.setattr(runner, "_kill_detached_leftovers", lambda *_: pytest.fail("broad kill"))
    monkeypatch.setattr(runner, "_rmtree_force", lambda *_: pytest.fail("parent chmod"))
    runner._cleanup_finished_attempt(str(attempt), str(root))
    assert list(root.iterdir()) == [sibling]
    assert root.stat().st_mode & 0o777 == 0o700


def test_custom_durations_never_write_source(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_save_durations", lambda *_: pytest.fail("source write"))
    runner._cache_clean_durations([(tmp_path / "test_example.py", 1.0)], tmp_path, str(tmp_path))


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("joined", [False, True])
def test_wrapper_cli_forwards_scratch_and_no_bytecode_to_real_worker(tmp_path, joined):
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    (root / "sentinel").write_text("keep")
    home = tmp_path / "home"
    home.mkdir()
    prefix = tmp_path / "bytecode"
    probe = tmp_path / "test_worker_probe.py"
    probe.write_text(
        "import importlib.util, os, sys\nfrom pathlib import Path\n"
        + "forwarded = (os.environ.get('PYTHONPYCACHEPREFIX'), sys.pycache_prefix, sys.dont_write_bytecode)\n"
        + f"spec = importlib.util.spec_from_file_location('probe_repo_conftest', {str(repo / 'tests/conftest.py')!r})\n"
        + "module = importlib.util.module_from_spec(spec)\nsys.modules[spec.name] = module\nspec.loader.exec_module(module)\n"
        + "pytest_plugins = ['probe_repo_conftest']\n"
        + "def test_worker_environment(tmp_path, pytestconfig, _isolate_hermes_home):\n"
        + f"    root = Path({str(root)!r})\n"
        + "    attempt = Path(os.environ['PYTEST_DEBUG_TEMPROOT'])\n"
        + "    assert attempt.parent == root\n"
        + "    assert Path(os.environ['TMPDIR']) == attempt\n"
        + "    assert Path(os.environ['TMP']) == Path(os.environ['TEMP']) == attempt\n"
        + "    assert sys.dont_write_bytecode\n"
        + f"    assert forwarded == ({str(prefix)!r}, {str(prefix)!r}, True)\n"
        + "    assert 'PYTHONPYCACHEPREFIX' not in os.environ and sys.pycache_prefix is None\n"
        + "    assert pytestconfig.pluginmanager.is_registered(module)\n"
        + "    assert any(marker.startswith('allow_real_home_io:') for marker in pytestconfig.getini('markers'))\n"
        + "    assert pytestconfig._tmp_path_factory.getbasetemp().is_relative_to(attempt)\n"
        + "    assert tmp_path.is_relative_to(attempt)\n"
        + "    assert Path(os.environ['HERMES_HOME']).is_relative_to(tmp_path)\n"
        + "    assert not any(key.endswith(('_TOKEN', '_API_KEY', '_SECRET')) for key in os.environ)\n",
        encoding="utf-8",
    )
    flags = ["--scratch-root=" + str(root)] if joined else ["--scratch-root", str(root)]
    duration_file = repo / "test_durations.json"
    before = duration_file.read_bytes() if duration_file.exists() else None
    result = subprocess.run(
        ["bash", str(repo / "scripts/run_tests.sh"), *flags, str(probe), "--file-retries", "0", "-q"],
        cwd=repo, env={"PATH": os.environ["PATH"], "HOME": str(home), "HERMES_PYTHON": sys.executable,
                       "TMPDIR": str(root), "TMP": str(root), "TEMP": str(root),
                       "PYTHONDONTWRITEBYTECODE": "0", "PYTHONPYCACHEPREFIX": str(prefix)},
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "bytecode warmup disabled" in result.stdout
    assert "1 tests passed" in result.stdout
    assert list(root.iterdir()) == [root / "sentinel"]
    assert (root / "sentinel").read_text() == "keep"
    assert not prefix.exists()
    assert (duration_file.read_bytes() if duration_file.exists() else None) == before


@pytest.mark.platforms("posix")
def test_custom_cleanup_permission_error_propagates_without_touching_siblings(tmp_path, monkeypatch):
    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    attempt, sibling = root / "attempt", root / "sibling"
    attempt.mkdir()
    sibling.mkdir()
    before = root.stat().st_mode
    monkeypatch.setattr(runner.shutil, "rmtree", MagicMock(side_effect=PermissionError("cleanup denied")))
    monkeypatch.setattr(runner.os, "chmod", lambda *_: pytest.fail("chmod attempted"))
    monkeypatch.setattr(runner, "_sweep_killed_run_roots", lambda *_: pytest.fail("pruner attempted"))
    monkeypatch.setattr(runner, "_kill_detached_leftovers", lambda *_: pytest.fail("broad kill attempted"))
    with pytest.raises(PermissionError, match="cleanup denied"):
        runner._cleanup_finished_attempt(str(attempt), str(root))
    assert set(root.iterdir()) == {attempt, sibling}
    assert root.stat().st_mode == before
