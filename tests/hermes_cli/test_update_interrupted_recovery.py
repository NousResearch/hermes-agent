"""Tests for interrupted-install self-heal (the ``.update-incomplete`` marker).

Covers the breadcrumb lifecycle and the launch-time recovery guard added so a
``hermes update`` killed mid-install (Ctrl-C, terminal close, WSL OOM) gets
finished automatically on the next launch instead of leaving a half-built venv.
"""

from __future__ import annotations

from pathlib import Path

import hermes_cli.main as m


def test_marker_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    marker = m._update_marker_path()
    assert marker == tmp_path / ".update-incomplete"
    assert not marker.exists()

    m._write_update_incomplete_marker()
    assert marker.exists()
    body = marker.read_text()
    assert "started=" in body
    assert "pid=" in body

    m._clear_update_incomplete_marker()
    assert not marker.exists()


def test_clear_when_absent_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    # Must not raise when the marker was never written.
    m._clear_update_incomplete_marker()
    assert not m._update_marker_path().exists()


def test_recovery_noop_without_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    called = {"install": False}
    monkeypatch.setattr(
        m,
        "_install_python_dependencies_with_optional_fallback",
        lambda *a, **k: called.__setitem__("install", True),
    )
    m._recover_from_interrupted_install()
    assert called["install"] is False, "recovery must not install when no marker"


def test_recovery_clears_stray_marker_without_pyproject(tmp_path, monkeypatch):
    # No pyproject.toml (PyPI/Docker install) — a stray marker is not ours to
    # act on; recovery should just clear it without trying to install.
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    m._write_update_incomplete_marker()
    called = {"install": False}
    monkeypatch.setattr(
        m,
        "_install_python_dependencies_with_optional_fallback",
        lambda *a, **k: called.__setitem__("install", True),
    )
    m._recover_from_interrupted_install()
    assert called["install"] is False
    assert not m._update_marker_path().exists()


def test_recovery_runs_install_and_clears_marker(tmp_path, monkeypatch):
    # Source-tree install (pyproject present) with marker set → recovery should
    # run the dep install and clear the marker on success.
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    m._write_update_incomplete_marker()

    seen = {"ensurepip": False, "install": False}

    def fake_run(cmd, *a, **k):
        if "ensurepip" in cmd:
            seen["ensurepip"] = True

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    monkeypatch.setattr(m, "_is_termux_env", lambda *a, **k: False)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: None)
    monkeypatch.setattr(
        m,
        "_install_python_dependencies_with_optional_fallback",
        lambda *a, **k: seen.__setitem__("install", True),
    )

    m._recover_from_interrupted_install()

    assert seen["ensurepip"] is True, "ensurepip must run unconditionally first"
    assert seen["install"] is True, "dep install must run"
    assert not m._update_marker_path().exists(), "marker cleared on success"


def test_recovery_uv_install_uses_active_sidecar_virtualenv(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    m._write_update_incomplete_marker()

    sidecar = tmp_path / ".venv-bma-sidecar"
    sidecar.mkdir()
    monkeypatch.setattr(m.sys, "prefix", str(sidecar))
    monkeypatch.setattr(m.sys, "base_prefix", "/usr")
    monkeypatch.setattr(m, "_is_termux_env", lambda *a, **k: False)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: "/usr/bin/uv")

    class R:
        returncode = 0

    seen = {}
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(
        m,
        "_install_python_dependencies_with_optional_fallback",
        lambda *a, **k: seen.update({"cmd": a[0], "env": k.get("env")}),
    )

    m._recover_from_interrupted_install()

    assert seen["cmd"] == ["/usr/bin/uv", "pip"]
    assert seen["env"]["VIRTUAL_ENV"] == str(sidecar)
    assert not m._update_marker_path().exists()


def test_recovery_keeps_marker_on_failure(tmp_path, monkeypatch):
    # If the install itself blows up, the marker must survive so the next
    # launch retries — and recovery must not raise.
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    m._write_update_incomplete_marker()

    class R:
        returncode = 0

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(m, "_is_termux_env", lambda *a, **k: False)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: None)

    def boom(*a, **k):
        raise RuntimeError("install died")

    monkeypatch.setattr(
        m, "_install_python_dependencies_with_optional_fallback", boom
    )

    # Must not raise.
    m._recover_from_interrupted_install()
    assert m._update_marker_path().exists(), "marker preserved for retry on failure"


def _stub_install_env(monkeypatch, m, seen):
    """Common stubs so recovery's install path is inert and observable."""

    class R:
        returncode = 0

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(m, "_is_termux_env", lambda *a, **k: False)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: None)
    monkeypatch.setattr(
        m,
        "_install_python_dependencies_with_optional_fallback",
        lambda *a, **k: seen.__setitem__("install", True),
    )


def test_recovery_skips_when_lock_held(tmp_path, monkeypatch):
    # Another process is mid-recovery (fresh lockfile) — this launch must skip
    # the install entirely and leave both marker and lock untouched.
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    m._write_update_incomplete_marker()
    lock = tmp_path / ".update-incomplete.lock"
    lock.write_text("12345\n")

    seen = {"install": False}
    _stub_install_env(monkeypatch, m, seen)

    m._recover_from_interrupted_install()

    assert seen["install"] is False, "must not install while another holds the lock"
    assert m._update_marker_path().exists(), "marker left for the lock holder"
    assert lock.exists(), "fresh lock must not be broken"


def test_recovery_breaks_stale_lock(tmp_path, monkeypatch):
    # A lock older than an hour is from a crashed holder — it gets removed so
    # the NEXT launch can recover (this launch still skips).
    import os as _os

    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    m._write_update_incomplete_marker()
    lock = tmp_path / ".update-incomplete.lock"
    lock.write_text("12345\n")
    stale = m._time.time() - 7200
    _os.utime(lock, (stale, stale))

    seen = {"install": False}
    _stub_install_env(monkeypatch, m, seen)

    m._recover_from_interrupted_install()

    assert not lock.exists(), "stale lock must be broken"
    assert m._update_marker_path().exists()

    # Next launch proceeds normally.
    m._recover_from_interrupted_install()
    assert seen["install"] is True
    assert not m._update_marker_path().exists()
    assert not lock.exists(), "lock released after recovery"


def test_recovery_releases_lock_after_run(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    m._write_update_incomplete_marker()

    seen = {"install": False}
    _stub_install_env(monkeypatch, m, seen)

    m._recover_from_interrupted_install()

    assert seen["install"] is True
    assert not (tmp_path / ".update-incomplete.lock").exists()


def test_recovery_output_goes_to_stderr(tmp_path, monkeypatch, capfd):
    # ACP speaks JSON-RPC on stdout — recovery output (including the streamed
    # install, which inherits fd 1) must land on stderr only.
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    m._write_update_incomplete_marker()

    seen = {"install": False}
    _stub_install_env(monkeypatch, m, seen)

    m._recover_from_interrupted_install()

    out, err = capfd.readouterr()
    assert "interrupted mid-install" not in out
    assert "interrupted mid-install" in err
    assert "recovered" in err
