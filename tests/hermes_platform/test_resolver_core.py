from __future__ import annotations

import os
import stat
import sys

import pytest

from hermes_platform.resolver import ABSENT, CheckState, LookupContext, Observation, locate_command


def _make_exe(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable bits")
def test_path_hit_comes_first_and_known_dirs_are_still_recorded(tmp_path):
    on_path = _make_exe(tmp_path / "pathbin" / "tool")
    in_known = _make_exe(tmp_path / "known" / "tool")
    res = locate_command("tool", LookupContext(path=str(on_path.parent)), known_dirs=(str(in_known.parent),))
    assert res.kind == "path_executable"
    assert res.command == (str(on_path),)
    assert [c.present for c in res.candidates] == [True, True]
    assert [c.value for c in res.present] == [str(on_path), str(in_known)]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable bits")
def test_known_dir_hit_when_path_misses(tmp_path):
    in_known = _make_exe(tmp_path / "known" / "tool")
    res = locate_command("tool", LookupContext(path=""), known_dirs=(str(in_known.parent),))
    assert res.kind == "known_path"
    assert res.source == f"known_dir:{in_known.parent}"
    assert res.candidates[0].present is False


def test_empty_path_is_a_miss_not_ambient(tmp_path):
    res = locate_command("python3", LookupContext(path=""))
    assert res.kind == "missing"
    assert res.command == ()
    assert res.present == ()


def test_absent_and_none_both_mean_ambient():
    assert LookupContext().path is ABSENT
    assert LookupContext().effective_path() is None
    assert LookupContext(path=None).effective_path() is None
    assert LookupContext(path="/x").effective_path() == "/x"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable bits")
def test_explicit_path_bypasses_search(tmp_path):
    exe = _make_exe(tmp_path / "bin" / "tool")
    res = locate_command(str(exe), LookupContext(path=""))
    assert res.kind == "explicit_path"
    assert res.command == (str(exe),)
    missing = locate_command(str(tmp_path / "nope" / "tool"), LookupContext(path=""))
    assert missing.kind == "missing" and missing.candidates[0].source == "explicit"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable bits")
def test_locate_never_searches_the_working_directory(tmp_path, monkeypatch):
    _make_exe(tmp_path / "tool")
    monkeypatch.chdir(tmp_path)
    assert locate_command("tool", LookupContext(path="")).kind == "missing"
    relative = locate_command("./tool", LookupContext(path=""))
    assert relative.kind == "missing" and relative.candidates[0].present is False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable bits")
def test_known_dir_expands_home_and_env(tmp_path, monkeypatch):
    exe = _make_exe(tmp_path / "home" / ".local" / "bin" / "tool")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TOOLROOT", str(tmp_path / "home" / ".local"))
    for spec in ("~/.local/bin", "$TOOLROOT/bin"):
        res = locate_command("tool", LookupContext(path=""), known_dirs=(spec,))
        assert res.command == (str(exe),), spec


@pytest.mark.windows_only
def test_windows_pathext_is_honored_without_mutating_environ(tmp_path, monkeypatch):
    exe = tmp_path / "bin" / "tool.cmd"
    exe.parent.mkdir()
    exe.write_text("@echo off\r\n", encoding="utf-8")
    before = os.environ.get("PATHEXT")
    res = locate_command("tool", LookupContext(path=str(exe.parent), pathext=".CMD"))
    assert res.kind == "path_executable"
    assert res.command[0].lower() == str(exe).lower()
    assert os.environ.get("PATHEXT") == before


def test_observation_not_checked_is_distinct_from_absent():
    assert Observation.not_checked().state is CheckState.NOT_CHECKED
    assert Observation(CheckState.ABSENT, False).state is not CheckState.NOT_CHECKED
    assert Observation(CheckState.ABSENT, False).value is False


def test_one_candidate_per_known_dir_regardless_of_pathext(tmp_path):
    res = locate_command("nothing-here", LookupContext(path=""), known_dirs=(str(tmp_path / "a"), str(tmp_path / "b")))
    assert [c.source for c in res.candidates] == ["PATH", f"known_dir:{tmp_path / 'a'}", f"known_dir:{tmp_path / 'b'}"]


def test_known_dir_tables_match_the_host_os():
    from hermes_platform.resolver import known_dirs as kd

    every = (*kd.homebrew_dirs(), *kd.user_local_bin(), *kd.rust_tool_dirs(),
             *kd.node_tool_dirs(), *kd.hermes_vendored_dirs(), *kd.windows_user_program_dirs())
    assert every, "at least one table applies on every host"
    if sys.platform == "win32":
        assert not any(d.startswith("/") for d in every)
    else:
        assert not any("%" in d for d in every)
        assert bool(kd.homebrew_dirs()) == (sys.platform == "darwin")
