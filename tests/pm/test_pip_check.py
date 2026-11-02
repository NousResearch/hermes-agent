"""PM's `uv pip check` accepts exactly the mismatches the lock's overrides resolved.

A `[tool.uv] override-dependencies` entry (Pillow 13 over browser-harness's
`pillow==12.3.0`) installs a version the dependent's own metadata rejects, so
`uv pip check` reports it. uv.lock's `[manifest] overrides` plus the locked
version are what make that line expected; nothing else may pass.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from pm.package import InstallError
from pm.pip_check import override_requirements, overridden_versions, unexpected_incompatibilities
from tests.pm._fixtures import _run, _wheel

REPO_LOCK = Path(__file__).resolve().parents[2] / "uv.lock"

_LOCK = """\
version = 1
requires-python = ">=3.11"

[manifest]
overrides = [
    { name = "Shared_Dep", specifier = "==2.0" },
    { name = "unused", marker = "sys_platform == 'never'" },
]

[[package]]
name = "pinned-parent"
version = "1.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "shared-dep"
version = "2.0"
source = { registry = "https://pypi.org/simple" }
"""

OVERRIDDEN = "The package `pinned-parent` requires `shared-dep==1.0`, but `2.0` is installed"


def _report(*lines: str) -> str:
    plural = "incompatibility" if len(lines) == 1 else "incompatibilities"
    return f"Checked 3 packages in 1ms\nFound {len(lines)} {plural}\n" + "\n".join(lines) + "\n"


@pytest.fixture
def lock(tmp_path):
    path = tmp_path / "uv.lock"
    path.write_text(_LOCK, encoding="utf-8")
    return path


def test_override_resolved_mismatch_is_expected(lock):
    assert unexpected_incompatibilities(_report(OVERRIDDEN), lock) == []


@pytest.mark.parametrize("line", [
    # Not overridden.
    "The package `pinned-parent` requires `other-dep==1.0`, but `2.0` is installed",
    # Overridden, but not installed at the version the lock resolved.
    "The package `pinned-parent` requires `shared-dep==1.0`, but `3.0` is installed",
    # Missing entirely: an override never explains an absent package.
    "The package `pinned-parent` requires `shared-dep==1.0`, but it's not installed",
    # Overridden only by a never-true marker, so nothing about its version is expected.
    "The package `pinned-parent` requires `unused>=1`, but `1.0` is installed",
])
def test_lines_the_lock_does_not_explain_stay_failures(lock, line):
    assert unexpected_incompatibilities(_report(OVERRIDDEN, line), lock) == [line]


def test_requirement_markers_and_name_spelling_do_not_hide_the_override(lock):
    line = ("The package `pinned-parent` requires `Shared.Dep==1.0 ; python_full_version >= '3.14'`, "
            "but `2.0` is installed")
    assert unexpected_incompatibilities(_report(line), lock) == []


def test_without_a_lock_every_line_fails(tmp_path):
    assert unexpected_incompatibilities(_report(OVERRIDDEN), None) == [OVERRIDDEN]
    assert unexpected_incompatibilities(_report(OVERRIDDEN), tmp_path / "missing.lock") == [OVERRIDDEN]


@pytest.mark.parametrize("report", [
    "error: failed to inspect the environment\n",
    # uv counted more incompatibilities than the parser recognised.
    "Found 2 incompatibilities\n" + OVERRIDDEN + "\nThe package `x` has an unreadable requirement\n",
    # No count: the head of the report may be missing, taking unrelated failures with it.
    OVERRIDDEN + "\n",
])
def test_unrecognised_report_is_never_read_as_clean(lock, report):
    assert unexpected_incompatibilities(report, lock) is None


def test_overrides_round_trip_as_requirement_lines(lock):
    assert override_requirements(lock) == ["Shared_Dep==2.0", "unused ; sys_platform == 'never'"]


def test_stdlib_fallback_reads_the_real_lock_like_tomllib(monkeypatch):
    """The bootstrap runner may lack tomllib; its parser must agree on Hermes's own lock."""
    with_tomllib = overridden_versions(REPO_LOCK)
    requirements = override_requirements(REPO_LOCK)
    monkeypatch.setitem(sys.modules, "tomllib", None)
    assert overridden_versions(REPO_LOCK) == with_tomllib
    assert override_requirements(REPO_LOCK) == requirements
    assert with_tomllib, "Hermes's lock declares overrides; an empty read means the parser missed them"


@pytest.fixture
def overridden_project(tmp_path):
    """A real project whose dependency pins shared-dep==1.0, overridden to 2.0."""
    return _overridden_project(tmp_path, ["shared_dep"])


def _overridden_project(tmp_path, overridden, extra_wheels=(), extra_dependencies=()):
    uv = shutil.which("uv")
    assert uv, "environment construction tests require uv on PATH"
    source = tmp_path / "source"
    source.mkdir()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _wheel(wheels, "pinned_parent", requirements=[f"{name.replace('_', '-')}==1.0" for name in overridden])
    for name in overridden:
        _wheel(wheels, name, "1.0")
        _wheel(wheels, name, "2.0")
    for name, requirements in extra_wheels:
        _wheel(wheels, name, requirements=requirements)
    dependencies = json.dumps(["pinned-parent==1.0", *extra_dependencies])
    overrides = json.dumps([f"{name.replace('_', '-')}==2.0" for name in overridden])
    (source / "pyproject.toml").write_text(
        '[project]\nname="override-root"\nversion="1"\nrequires-python=">=3.11"\n'
        f'dependencies={dependencies}\n'
        f'[tool.uv]\npackage=false\nno-index=true\noverride-dependencies={overrides}\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8",
    )
    home = tmp_path / "isolated-home"
    home.mkdir()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("UV_", "PYTHON")) and key != "VIRTUAL_ENV"}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(home / ".hermes"),
               UV_CACHE_DIR=str(tmp_path / "cache"), UV_PYTHON=sys.executable, UV_OFFLINE="1")
    _run([uv, "lock", "--python", sys.executable], cwd=source, env=env)
    return source, env


@pytest.mark.parametrize("color", [{}, {"FORCE_COLOR": "1", "CLICOLOR_FORCE": "1"}], ids=["plain", "forced-color"])
def test_build_accepts_the_override_and_check_still_rejects_what_it_does_not_cover(overridden_project, tmp_path, color):
    from pm import build_environment
    from pm.environment import managed_environment

    source, env = overridden_project
    # Forced colour must not turn uv's report into ANSI the parser cannot read.
    env = {**env, **color}
    out = tmp_path / "environment"
    executable = build_environment(explicit=True, source=source, python=Path(sys.executable), out=out,
                                   cache=tmp_path / "cache", env=env, offline=True, frozen=True)
    assert _run([str(executable), "-I", "-c", "import shared_dep; print(shared_dep.__version__)"],
                cwd=tmp_path, env=env) == "2.0"

    environment = managed_environment(out, python=Path(sys.executable), cache=tmp_path / "cache",
                                      env=env, offline=True, explicit=True)
    environment.check(source / "uv.lock")
    # The same environment, judged without the lock that explains it, is invalid.
    with pytest.raises(InstallError, match="requires `shared-dep==1.0`, but `2.0` is installed"):
        environment.check()
    # A lock resolving a different version does not cover what is installed.
    stale = tmp_path / "stale.lock"
    stale.write_text((source / "uv.lock").read_text(encoding="utf-8").replace(
        'name = "shared-dep"\nversion = "2.0"', 'name = "shared-dep"\nversion = "1.0"'), encoding="utf-8")
    assert overridden_versions(stale) == {"shared-dep": {"1.0"}}
    with pytest.raises(InstallError, match="dependency validation failed"):
        environment.check(stale)


def test_requirements_builds_from_an_overridden_lock_apply_its_overrides(overridden_project, tmp_path):
    """Termux and the PM runtime install the lock's export with `uv pip install`, which
    re-resolves: without the lock's overrides it rejects the graph the lock resolved."""
    from pm import build_requirements_environment, export_requirements
    from pm.environment import ResolutionConflict, managed_environment

    source, env = overridden_project
    wheels = tmp_path / "wheels"
    exported = tmp_path / "requirements.txt"
    export_requirements(source, exported, cache=tmp_path / "cache", env=env, explicit=True)
    requirements = exported.read_text(encoding="utf-8").splitlines()
    assert "shared-dep==2.0" in requirements

    def build(out, **kwargs):
        return build_requirements_environment(requirements, out=tmp_path / out, python=Path(sys.executable),
                                              wheelhouse=wheels, cache=tmp_path / "cache", env=env,
                                              offline=True, explicit=True, **kwargs)

    with pytest.raises(ResolutionConflict, match="shared-dep"):
        build("without-lock")
    executable = build("with-lock", lock=source / "uv.lock")
    assert _run([str(executable), "-I", "-c", "import shared_dep; print(shared_dep.__version__)"],
                cwd=tmp_path, env=env) == "2.0"

    environment = managed_environment(tmp_path / "wheelhouse-install", python=Path(sys.executable),
                                      cache=tmp_path / "cache", env=env, offline=True, explicit=True)
    environment.create()
    environment.install_wheelhouse(source, wheels)


def test_streamed_check_still_sees_failures_ahead_of_a_long_expected_report(tmp_path):
    """Interactive builds stream uv output and keep only its tail. An unrelated failure
    reported before 2,000+ characters of override-covered lines must still fail the check."""
    from pm import build_environment
    from pm.environment import managed_environment
    from pm.environments import site_packages

    overridden = [f"shared_dep_{index:02d}" for index in range(30)]
    source, env = _overridden_project(tmp_path, overridden,
                                      extra_wheels=[("aa_parent", ["aa-dep==1.0"]), ("aa_dep", [])],
                                      extra_dependencies=["aa-parent==1.0"])
    out = tmp_path / "environment"
    build_environment(explicit=True, source=source, python=Path(sys.executable), out=out,
                      cache=tmp_path / "cache", env=env, offline=True, frozen=True)
    # Break the environment in a way no override explains.
    shutil.rmtree(next(site_packages(out).glob("aa_dep-1.0.dist-info")))

    environment = managed_environment(out, python=Path(sys.executable), cache=tmp_path / "cache",
                                      env=env, offline=True, explicit=True, output=io.StringIO())
    with pytest.raises(InstallError, match="requires `aa-dep==1.0`, but it's not installed"):
        environment.check(source / "uv.lock")
