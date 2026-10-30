"""A corrupt entry in PM's uv cache must not fail every later install.

A machine stop during an install can leave an unpacked wheel in uv's cache
whose files read back as NUL bytes: uv renames its unpack directory into
``archive-v0`` without flushing the data first. uv then reuses that entry on
every sync and fails with "The wheel is invalid", so retrying never helps. PM
purges the damaged package(s) through ``uv cache clean`` and retries, and
gives up with a clear message when a fresh copy is broken too.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from pm.environment import PythonEnvironment
from pm.package import InstallError
from pm.uv_cache_health import corrupt_cached_packages, failed_cached_packages
from tests.pm._fixtures import _run, _wheel

LIVE_FAILURE = """\
Building hermes-agent @ file:///opt/data/installs/351f/environments/g/workspace
      Built hermes-agent @ file:///opt/data/installs/351f/environments/g/workspace
Prepared 1 package in 2.82s
error: Failed to install: lxml-6.1.3-cp314-cp314-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl (lxml==6.1.3)
  cause: The wheel is invalid: Invalid Wheel-Version in `WHEEL` file: None
"""


@pytest.mark.parametrize(("output", "expected"), [
    (LIVE_FAILURE, {"lxml"}),
    ("error: Failed to install: Foo_Bar-1.0-py3-none-any.whl (foo-bar==1.0)\n"
     "  cause: `RECORD` file is invalid\n  cause: CSV deserialize error", {"foo-bar"}),
    ("error: Failed to install: alpha-1.0-py3-none-any.whl (alpha==1.0)\n"
     "  cause: failed to read directory `/c/archive-v0/waZDcAKYSZH8lwFY`: No such file or directory (os error 2)",
     {"alpha"}),
    ("error: Failed to install: alpha-1.0-py3-none-any.whl (alpha==1.0)\n"
     "  cause: failed to read directory `C:\\Users\\a\\cache\\uv\\archive-v0\\waZD`: "
     "The system cannot find the path specified. (os error 3)", {"alpha"}),
])
def test_cache_corruption_names_the_package(output, expected):
    assert failed_cached_packages(output) == expected


@pytest.mark.parametrize("output", [
    "error: No solution found when resolving dependencies:\n  Because alpha==2 was not found ...",
    "error: Failed to fetch: `https://pypi.org/simple/alpha/`\n  Caused by: Request failed after 3 retries",
    "error: Failed to install: alpha-1.0-py3-none-any.whl (alpha==1.0)\n"
    "  cause: failed to hardlink file from /c/archive-v0/abc/alpha/__init__.py to /v/lib/alpha/__init__.py: "
    "No space left on device (os error 28)",
    "error: Failed to install: alpha-1.0-py3-none-any.whl (alpha==1.0)\n"
    "  cause: failed to create directory `/v/lib/alpha`: Permission denied (os error 13)",
    "  Caused by: The wheel is invalid: Invalid Wheel-Version",  # no install step to name a package
    "error: Failed to install: --force-1.0-py3-none-any.whl (x==1)\n  cause: The wheel is invalid: x",
])
def test_other_failures_are_not_cache_corruption(output):
    assert failed_cached_packages(output) == set()


@pytest.fixture
def project(tmp_path):
    uv = shutil.which("uv")
    assert uv, "cache recovery tests require uv on PATH"
    source = tmp_path / "source"
    source.mkdir()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name in ("alpha_dep", "beta_dep"):
        _wheel(wheels, name)
    (source / "pyproject.toml").write_text(
        '[project]\nname="cache-root"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["alpha-dep==1.0", "beta-dep==1.0"]\n'
        '[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("UV_", "PYTHON")) and key != "VIRTUAL_ENV"}
    env.update(HOME=str(home), USERPROFILE=str(home), XDG_CONFIG_HOME=str(home), XDG_CONFIG_DIRS=str(home))
    environment = _environment(Path(uv), tmp_path, "first", env)
    environment.lock(source, timeout=120)
    environment.create()
    environment.sync(source, timeout=120)  # populates the cache
    return source, Path(uv), env, wheels


def _environment(uv: Path, root: Path, name: str, env, output=None) -> PythonEnvironment:
    return PythonEnvironment(uv=uv, python=Path(sys.executable), destination=root / name / "venv",
                             cache=root / "cache", env=env, offline=True, output=output)


def _cached(cache: Path, name: str) -> Path:
    matches = list((cache / "archive-v0").glob(f"*/{name}-1.0.dist-info"))
    assert len(matches) == 1, matches
    return matches[0]


def _nul_fill(path: Path) -> None:
    """What a crash leaves: the file keeps its size, its data never reached the disk."""
    size = max(path.stat().st_size, 1)
    path.write_bytes(b"\0" * size)


def _record_uv_calls(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []
    real = PythonEnvironment._run

    def recording(self, args, *, cwd, timeout):
        calls.append(list(args))
        return real(self, args, cwd=cwd, timeout=timeout)

    monkeypatch.setattr(PythonEnvironment, "_run", recording)
    return calls


def test_sync_recovers_from_nul_filled_cache_entries(project, tmp_path, monkeypatch):
    """The live shape: two packages damaged by one crash, the first a NUL WHEEL file."""
    source, uv, env, _ = project
    cache = tmp_path / "cache"
    _nul_fill(_cached(cache, "alpha_dep") / "WHEEL")
    for part in _cached(cache, "beta_dep").iterdir():
        _nul_fill(part)
    assert corrupt_cached_packages(cache) == {"alpha-dep", "beta-dep"}
    calls = _record_uv_calls(monkeypatch)
    notes = io.StringIO()

    second = _environment(uv, tmp_path, "second", env, output=notes)
    second.create()
    second.sync(source, timeout=120)

    assert _run([str(second.executable), "-I", "-c", "import alpha_dep, beta_dep; print(alpha_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.0"
    # One purge covered both packages, then exactly one retry.
    assert [c[:2] for c in calls].count(["cache", "clean"]) == 1
    assert ["cache", "prune"] not in [c[:2] for c in calls]  # prune breaks PM's seeded cache
    assert sorted(next(c for c in calls if c[:2] == ["cache", "clean"])[2:]) == ["alpha-dep", "beta-dep"]
    assert sum(c[0] == "sync" for c in calls) == 2
    assert corrupt_cached_packages(cache) == set()
    assert "alpha-dep, beta-dep is corrupt" in notes.getvalue()


def test_a_wheel_broken_at_the_source_fails_once_with_a_clear_message(project, tmp_path, monkeypatch):
    """Purging cannot fix a source that serves a broken wheel: one retry, then a named error."""
    source, uv, env, wheels = project
    cache = tmp_path / "cache"
    (wheels / "alpha_dep-1.0-py3-none-any.whl").unlink()
    _wheel(wheels, "alpha_dep", extra={"alpha_dep-1.0.dist-info/WHEEL": "\0" * 40})
    # The lock's hash would reject the new bytes before install; forget it so uv takes the wheel.
    (source / "uv.lock").unlink()
    _environment(uv, tmp_path, "relock", env).lock(source, timeout=120)
    shutil.rmtree(cache)
    calls = _record_uv_calls(monkeypatch)

    broken = _environment(uv, tmp_path, "broken", env)
    broken.create()
    with pytest.raises(InstallError) as caught:
        broken.sync(source, timeout=120)

    message = str(caught.value)
    assert "alpha-dep" in message and str(cache) in message and "still invalid" in message
    assert "broken wheel" in caught.value.remedy
    assert [c[:2] for c in calls].count(["cache", "clean"]) == 1
    assert sum(c[0] == "sync" for c in calls) == 2


def test_a_failure_that_is_not_cache_corruption_is_not_retried(project, tmp_path, monkeypatch):
    """A missing distribution (offline, nothing cached) fails once and never touches the cache."""
    source, uv, env, wheels = project
    cache = tmp_path / "cache"
    (wheels / "alpha_dep-1.0-py3-none-any.whl").unlink()
    shutil.rmtree(cache)
    calls = _record_uv_calls(monkeypatch)

    missing = _environment(uv, tmp_path, "missing", env)
    missing.create()
    with pytest.raises(InstallError):
        missing.sync(source, timeout=120)

    assert not any(c[:1] == ["cache"] for c in calls)
    assert sum(c[0] == "sync" for c in calls) == 1


def test_doctor_reports_then_repairs_corrupt_cache_entries(project, tmp_path, monkeypatch, capsys):
    from pm import cli
    import pm.packages

    _, uv, env, _ = project
    cache = tmp_path / "cache"
    _nul_fill(_cached(cache, "alpha_dep") / "METADATA")
    monkeypatch.setattr(pm.packages, "uv_cache_path", lambda: cache)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (uv, Path(sys.executable)))
    monkeypatch.setattr("pm.environment._base_environment", lambda base=None: dict(env))

    assert cli._doctor_uv_cache(repair=False) == 1
    assert "alpha-dep" in capsys.readouterr().out
    assert corrupt_cached_packages(cache) == {"alpha-dep"}  # read-only without --repair

    assert cli._doctor_uv_cache(repair=True) == 0
    assert "removed corrupt entries for alpha-dep" in capsys.readouterr().out
    assert corrupt_cached_packages(cache) == set()
    assert cli._doctor_uv_cache(repair=False) == 0


def _bucket(cache: Path, name: str = "alpha_dep", files=None) -> Path:
    """An archive entry by hand: ``files`` maps dist-info file name -> bytes."""
    info = cache / "archive-v0" / "bucket" / f"{name}-1.0.dist-info"
    info.mkdir(parents=True)
    for file, body in (files or {"WHEEL": b"\0" * 20, "METADATA": b"Name: x\n", "RECORD": b"a,,\n"}).items():
        (info / file).write_bytes(body)
    return info.parent


@pytest.mark.parametrize("files", [
    {"WHEEL": b"wheel-version: 1.0\n", "METADATA": b"metadata-version: 2.1\nname: alpha-dep\n", "RECORD": b"a,,\n"},
    {"WHEEL": b"Wheel-Version: 1.0\n", "METADATA": b"Metadata-Version: 2.1\n" + b"X: y\n" * 20000 + b"Name: a\n",
     "RECORD": b"a,,\n"},
])
def test_valid_entries_uv_accepts_are_never_flagged(tmp_path, files):
    """Header names are case-insensitive and METADATA may be long; only torn writes count."""
    _bucket(tmp_path, files=files)
    assert corrupt_cached_packages(tmp_path) == set()


@pytest.mark.parametrize("damage", ["WHEEL", "METADATA", "RECORD", "missing-RECORD", "WHEEL-without-version"])
def test_torn_writes_are_flagged(tmp_path, damage):
    files = {"WHEEL": b"Wheel-Version: 1.0\n", "METADATA": b"Name: alpha-dep\n", "RECORD": b"a,,\n"}
    if damage == "missing-RECORD":
        del files["RECORD"]
    elif damage == "WHEEL-without-version":
        files["WHEEL"] = b"Root-Is-Purelib: true\n"
    else:
        files[damage] = b"\0" * 152
    _bucket(tmp_path, files=files)
    assert corrupt_cached_packages(tmp_path) == {"alpha-dep"}


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("link", ["archive", "bucket", "dist-info"])
def test_sweep_never_deletes_through_a_symlink(tmp_path, link):
    from pm.uv_cache_health import remove_corrupt_entries

    cache = tmp_path / "cache"
    outside = tmp_path / "outside"
    victim = _bucket(outside)  # a corrupt-looking tree outside the cache
    (victim / "user-data").write_text("must survive", encoding="utf-8")
    archive = cache / "archive-v0"
    if link == "archive":
        cache.mkdir()
        archive.symlink_to(outside / "archive-v0", target_is_directory=True)
    elif link == "bucket":
        archive.mkdir(parents=True)
        (archive / "bucket").symlink_to(victim, target_is_directory=True)
    else:
        (archive / "bucket").mkdir(parents=True)
        info = next(victim.glob("*.dist-info"))
        (archive / "bucket" / info.name).symlink_to(info, target_is_directory=True)

    assert corrupt_cached_packages(cache) == set()
    assert remove_corrupt_entries(cache, {"alpha-dep"})
    assert (victim / "user-data").is_file() and next(victim.glob("*.dist-info")).is_dir()


@pytest.mark.platforms("linux", "macos")
def test_sweep_waits_for_uv_and_leaves_entries_while_uv_holds_the_cache(tmp_path, monkeypatch):
    """uv holds a shared flock on <cache>/.lock while it runs; the sweep must not delete under it."""
    import fcntl

    import pm.uv_cache_health as health

    monkeypatch.setattr(health, "_LOCK_WAIT_SECONDS", 0.2)
    bucket = _bucket(tmp_path)
    with open(tmp_path / ".lock", "w", encoding="utf-8") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_SH)
        assert health.remove_corrupt_entries(tmp_path, {"alpha-dep"}) is False
        assert bucket.is_dir()
    assert health.remove_corrupt_entries(tmp_path, {"alpha-dep"}) is True
    assert not bucket.exists()


def test_a_failed_purge_keeps_the_original_install_error(project, tmp_path, monkeypatch):
    source, uv, env, _ = project
    _nul_fill(_cached(tmp_path / "cache", "alpha_dep") / "WHEEL")
    real = PythonEnvironment._run

    def refuse_clean(self, args, *, cwd, timeout):
        if args[:2] == ["cache", "clean"]:
            return subprocess.CompletedProcess(args, 2, "", "error: cache is read-only")
        return real(self, args, cwd=cwd, timeout=timeout)

    monkeypatch.setattr(PythonEnvironment, "_run", refuse_clean)
    second = _environment(uv, tmp_path, "second", env)
    second.create()
    with pytest.raises(InstallError) as caught:
        second.sync(source, timeout=120)
    assert "cache is read-only" in str(caught.value)
    assert "The wheel is invalid" in str(caught.value)
