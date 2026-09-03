"""Behavioral tests for hermes_cli._startup_fast — the pre-import fast-path helpers.

The existing tests cover import-weight and version parity. These pin the actual
behavior of the pure probe/dispatch functions so a subtle env/path regression
(e.g. the drift that killed the Termux fast path) can't slip through.
"""

import io

import pytest

import hermes_cli._startup_fast as sf


# ── is_termux_env ─────────────────────────────────────────────────────────────

def test_is_termux_env_via_TERMUX_VERSION(monkeypatch):
    monkeypatch.setenv("TERMUX_VERSION", "0.118.1")
    monkeypatch.delenv("PREFIX", raising=False)
    assert sf.is_termux_env() is True


def test_is_termux_env_via_com_termux_prefix(monkeypatch):
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    assert sf.is_termux_env() is True


def test_is_termux_env_false_when_neither(monkeypatch):
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    assert sf.is_termux_env() is False


# ── version argv dispatch ─────────────────────────────────────────────────────

def test_fast_version_argv_accepts_version_and_v():
    assert sf.is_termux_fast_version_argv(["--version"]) is True
    assert sf.is_termux_fast_version_argv(["-V"]) is True
    assert sf.is_termux_fast_version_argv([]) is False
    assert sf.is_termux_fast_version_argv(["--version", "--foo"]) is False


def test_global_fast_version_argv_matches_plain_version():
    assert sf.is_global_fast_version_argv(["--version"]) is True
    assert sf.is_global_fast_version_argv(["-V"]) is True
    assert sf.is_global_fast_version_argv(["serve"]) is False


# ── container detection ───────────────────────────────────────────────────────

def test_is_container_startup_environment_dockerenv(monkeypatch, tmp_path):
    monkeypatch.setattr(sf.os.path, "exists", lambda p: p == "/.dockerenv")
    assert sf.is_container_startup_environment() is True


def test_is_container_startup_environment_cgroup(monkeypatch):
    import builtins

    def fake_exists(p):
        return False

    def fake_open(p, **kw):
        return io.StringIO("5:cpu:/docker/abc\n")

    monkeypatch.setattr(sf.os.path, "exists", fake_exists)
    monkeypatch.setattr(builtins, "open", fake_open)
    assert sf.is_container_startup_environment() is True


def test_is_container_startup_environment_false_when_no_marker(monkeypatch):
    import builtins

    monkeypatch.setattr(sf.os.path, "exists", lambda p: False)

    def fake_open(p, **kw):
        raise OSError("no cgroup")

    monkeypatch.setattr(builtins, "open", fake_open)
    assert sf.is_container_startup_environment() is False


# ── active profile override ───────────────────────────────────────────────────

def test_active_profile_may_override_home_non_default(monkeypatch, tmp_path):
    active = tmp_path / "active_profile"
    active.write_text("coder", encoding="utf-8")
    assert sf.active_profile_may_override_home(str(tmp_path)) is True


def test_active_profile_may_override_home_default(monkeypatch, tmp_path):
    active = tmp_path / "active_profile"
    active.write_text("default", encoding="utf-8")
    assert sf.active_profile_may_override_home(str(tmp_path)) is False


def test_active_profile_may_override_home_missing(monkeypatch, tmp_path):
    assert sf.active_profile_may_override_home(str(tmp_path)) is False


# ── read_openai_version / read_install_method ────────────────────────────────

def test_read_openai_version_parses_value(monkeypatch, tmp_path):
    pkg = tmp_path / "openai"
    pkg.mkdir()
    (pkg / "_version.py").write_text('__version__ = "1.52.0"\n', encoding="utf-8")
    monkeypatch.setattr(sf.sys, "path", [str(tmp_path)])
    assert sf.read_openai_version() == "1.52.0"


def test_read_openai_version_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(sf.sys, "path", [str(tmp_path)])
    assert sf.read_openai_version() is None


def test_read_install_method_reads_stamp(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".install_method").write_text("Git\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert sf.read_install_method() == "git"


def test_read_install_method_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "nope"))
    assert sf.read_install_method() is None


# ── try_fast_version dispatch ─────────────────────────────────────────────────

def test_try_fast_version_returns_false_for_non_version(monkeypatch):
    assert sf.try_fast_version(["serve"]) is False


def test_try_fast_version_dispatches_global_version(monkeypatch):
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.delenv("HERMES_DEV", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    printed = []
    monkeypatch.setattr(sf, "print_fast_version_info", lambda *a, **k: printed.append(1))
    monkeypatch.setattr(sf, "container_mode_may_be_active", lambda: False)
    assert sf.try_fast_version(["--version"]) is True
    assert printed == [1]


def test_try_fast_version_skips_when_container_may_be_active(monkeypatch):
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.delenv("HERMES_DEV", raising=False)
    monkeypatch.setattr(sf, "container_mode_may_be_active", lambda: True)
    monkeypatch.setattr(sf, "print_fast_version_info", lambda *a, **k: pytest.fail("should not print"))
    assert sf.try_fast_version(["--version"]) is False


def test_try_fast_version_termux_escape_hatch(monkeypatch):
    monkeypatch.setenv("TERMUX_VERSION", "0.118.1")
    monkeypatch.setenv("HERMES_TERMUX_DISABLE_FAST_CLI", "1")
    monkeypatch.setattr(sf, "print_fast_version_info", lambda *a, **k: pytest.fail("should not print"))
    assert sf.try_fast_version(["--version"]) is False


def test_project_root_is_repo_root():
    import pathlib

    expected = str(pathlib.Path(sf.__file__).resolve().parent.parent)
    assert sf.project_root_str() == expected
