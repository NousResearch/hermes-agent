"""Regression tests for the worker-spawn argv extraction (godfile wave 1, s5 c2).

``kanban_db``'s worker-spawn argv resolution functions were moved VERBATIM
into ``hermes_cli/worker_spawn_mixin.py`` (agreement: move=32) and are
re-exported from ``kanban_db`` so the public API is unchanged. These tests
pin the moved bodies' behavior: path normalization, PATH resolution without
implicit current-dir search, Windows batch-shim avoidance, and the
interpreter-bound fallback.
"""

from __future__ import annotations

import os
import shutil
import sys

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import worker_spawn_mixin as mixin
from hermes_cli.worker_spawn_mixin import (
    _absolute_hermes_path,
    _hermes_path_argv,
    _is_windows_batch_shim,
    _looks_like_path,
    _module_hermes_argv,
    _path_search_names,
    _resolve_hermes_argv,
    _safe_which_no_cwd,
)


def test_module_hermes_argv():
    assert _module_hermes_argv() == [sys.executable, "-m", "hermes_cli.main"]


def test_absolute_hermes_path():
    assert os.path.isabs(_absolute_hermes_path("relative/path"))
    assert _absolute_hermes_path("relative/path").endswith(os.path.join("relative", "path"))
    assert _absolute_hermes_path("~") == os.path.expanduser("~")


def test_looks_like_path():
    assert _looks_like_path("~/bin/hermes")
    assert _looks_like_path(os.path.abspath("hermes"))
    assert _looks_like_path("bin/hermes")
    assert _looks_like_path("bin\\hermes.exe")
    assert not _looks_like_path("hermes")


def test_is_windows_batch_shim():
    assert _is_windows_batch_shim("hermes.cmd")
    assert _is_windows_batch_shim("HERMES.BAT")
    assert not _is_windows_batch_shim("hermes.exe")
    assert not _is_windows_batch_shim("hermes")


def test_path_search_names(monkeypatch):
    monkeypatch.setattr(mixin, "_IS_WINDOWS", False)
    assert _path_search_names("hermes") == ["hermes"]
    monkeypatch.setattr(mixin, "_IS_WINDOWS", True)
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT")
    assert _path_search_names("hermes") == ["hermes.COM", "hermes.EXE", "hermes.BAT"]
    assert _path_search_names("hermes.exe") == ["hermes.exe"]


def test_safe_which_no_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr(mixin, "_IS_WINDOWS", True)
    exe = tmp_path / "hermes.EXE"
    exe.write_text("", encoding="utf-8")
    monkeypatch.setenv("PATHEXT", ".EXE;.CMD")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + ".")
    assert _safe_which_no_cwd("hermes") == str(exe)
    assert _safe_which_no_cwd("does-not-exist") is None
    # empty and '.' entries are skipped, later entries still found
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", str(tmp_path)]))
    assert _safe_which_no_cwd("hermes") == str(exe)


def test_hermes_path_argv(tmp_path, monkeypatch):
    monkeypatch.setattr(mixin, "_IS_WINDOWS", False)
    plain = tmp_path / "hermes"
    assert _hermes_path_argv(str(plain)) == [str(plain)]
    monkeypatch.setattr(mixin, "_IS_WINDOWS", True)
    shim = tmp_path / "hermes.CMD"
    shim.write_text("@echo off\n", encoding="utf-8")
    assert _hermes_path_argv(str(shim)) == [sys.executable, "-m", "hermes_cli.main"]


def test_resolve_hermes_argv_explicit_path(monkeypatch):
    monkeypatch.setenv("HERMES_BIN", os.path.abspath("somewhere/hermes"))
    assert _resolve_hermes_argv() == [os.path.abspath("somewhere/hermes")]


def test_resolve_hermes_argv_bare_name_on_path(tmp_path, monkeypatch):
    monkeypatch.setattr(mixin, "_IS_WINDOWS", True)
    exe = tmp_path / "hermes-custom.EXE"
    exe.write_text("", encoding="utf-8")
    monkeypatch.setenv("PATHEXT", ".EXE;.CMD")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HERMES_BIN", "hermes-custom")
    assert _resolve_hermes_argv() == [str(exe)]


def test_resolve_hermes_argv_windows_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(mixin, "_IS_WINDOWS", True)
    monkeypatch.setenv("PATH", str(tmp_path))  # empty dir: no hermes found
    monkeypatch.delenv("HERMES_BIN", raising=False)
    assert _resolve_hermes_argv() == [sys.executable, "-m", "hermes_cli.main"]


def test_resolve_hermes_argv_posix_which(monkeypatch):
    monkeypatch.setattr(mixin, "_IS_WINDOWS", False)
    monkeypatch.delenv("HERMES_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/hermes")
    assert _resolve_hermes_argv() == ["/usr/bin/hermes"]
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert _resolve_hermes_argv() == [sys.executable, "-m", "hermes_cli.main"]


def test_public_api_reexported_from_kanban_db():
    for name in (
        "_module_hermes_argv", "_absolute_hermes_path", "_looks_like_path",
        "_is_windows_batch_shim", "_path_search_names", "_safe_which_no_cwd",
        "_hermes_path_argv", "_resolve_hermes_argv",
    ):
        assert callable(getattr(kb, name)), name
    assert kb._resolve_hermes_argv is _resolve_hermes_argv
    assert kb._module_hermes_argv is _module_hermes_argv
