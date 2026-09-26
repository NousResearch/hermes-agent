"""Gateway boot interpreter guard (#123185).

A standalone system Python newer than the checkout's pinned venv can win the PATH race
when a launcher is generated, so the detached Windows gateway spawns under it and every
compiled wheel from the venv fails to import (pydantic_core._pydantic_core) — a
crash-loop nothing inside gateway.run can recover from. These tests pin the two halves
of the contract: generated launchers embed the checkout's own interpreter, and a boot
under any other interpreter re-execs before the heavy gateway.run imports bind.
"""

from pathlib import Path

import pytest

import hermes_cli.gateway as gateway


def test_get_python_path_posix_keeps_running_interpreter_over_stale_venv(monkeypatch, tmp_path):
    """No PM store Python committed, POSIX: the running interpreter stays the launcher's owner.

    The in-tree venv fallback is Windows-only (#123185's PATH race is a Windows shape). A
    developer installing systemd/launchd units from a Nix/developer runtime must not have the
    unit silently rewritten onto a stale checkout venv (_launchers.py's ownership rule).
    """
    project = tmp_path / "project"
    venv = project / "venv"
    venv_bin = venv / "bin"
    venv_bin.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("version = 3.11.16\n", encoding="utf-8")
    (venv_bin / "python").write_text("", encoding="utf-8")

    import hermes_cli._launchers as _launchers

    monkeypatch.setattr(gateway, "PROJECT_ROOT", project)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda root: None)
    monkeypatch.setattr(gateway.sys, "executable", "/nix/store/xyz-python3/bin/python3")

    assert gateway.get_python_path() == "/nix/store/xyz-python3/bin/python3"


@pytest.mark.platforms("windows")
def test_get_python_path_windows_falls_back_to_checkout_venv(monkeypatch, tmp_path):
    """No PM store Python committed, Windows: the venv's own interpreter wins over
    sys.executable — the generator must never bake a PATH-race system Python into a
    regenerated launcher."""
    project = tmp_path / "project"
    venv = project / "venv"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("version = 3.11.16\n", encoding="utf-8")
    venv_python = scripts / "python.exe"
    venv_python.write_text("", encoding="utf-8")

    import hermes_cli._launchers as _launchers

    monkeypatch.setattr(gateway, "PROJECT_ROOT", project)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda root: None)

    assert gateway.get_python_path() == str(venv_python)


def test_get_python_path_still_prefers_committed_store_python(monkeypatch, tmp_path):
    """A committed PM store Python remains the answer regardless of the venv (PM installs
    must not be hijacked back onto a legacy in-tree venv, #122736's other half)."""
    project = tmp_path / "project"
    venv = project / "venv"
    scripts = venv / ("Scripts" if gateway.sys.platform == "win32" else "bin")
    scripts.mkdir(parents=True)
    venv_python = scripts / ("python.exe" if scripts.name == "Scripts" else "python")
    venv_python.write_text("", encoding="utf-8")

    import hermes_cli._launchers as _launchers

    store = tmp_path / "store" / "python" / "cpython-3.14" / "python.exe"
    store.parent.mkdir(parents=True)
    store.write_text("", encoding="utf-8")

    monkeypatch.setattr(gateway, "PROJECT_ROOT", project)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda root: store)

    assert Path(gateway.get_python_path()) == store


def test_reexec_plan_replays_minus_m_argv_under_committed_interpreter(tmp_path):
    """A launcher-shaped boot under the wrong interpreter re-execs -m hermes_cli.main under
    the committed one, replacing only the interpreter — flags/profile args pass through."""
    target = tmp_path / "venv" / "Scripts" / "python.exe"
    plan = gateway._gateway_reexec_plan(
        target,
        str(tmp_path / "Python314" / "python.exe"),
        ["C:\\\\Python314\\\\python.exe", "-m", "hermes_cli.main", "-p", "work", "gateway", "run"],
    )
    assert plan == [str(target), "-m", "hermes_cli.main", "-p", "work", "gateway", "run"]


def test_reexec_plan_noop_when_already_on_target(tmp_path):
    target = tmp_path / "venv" / "Scripts" / "python.exe"
    plan = gateway._gateway_reexec_plan(target, str(target), ["python", "-m", "hermes_cli.main", "gateway", "run"])
    assert plan is None


def test_reexec_plan_noop_for_console_script_trampolines(tmp_path):
    """A console-script trampoline (no -m) already embeds the right interpreter; the guard
    leaves it alone instead of guessing how to replay it."""
    target = tmp_path / "venv" / "Scripts" / "python.exe"
    plan = gateway._gateway_reexec_plan(
        target,
        str(tmp_path / "Python314" / "python.exe"),
        ["C:\\\\hermes.exe", "gateway", "run"],
    )
    assert plan is None


@pytest.mark.platforms("windows")
def test_run_gateway_reexecs_before_heavy_imports(monkeypatch, tmp_path):
    """The guard runs before gateway.run is imported and re-execs (SystemExit via the
    Windows spawn-and-exit contract) under the committed interpreter with the reexec
    marker set, so the child cannot loop."""
    project = tmp_path / "project"
    venv = project / "venv"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    venv_python = scripts / "python.exe"
    venv_python.write_text("", encoding="utf-8")
    (venv / "pyvenv.cfg").write_text("version = 3.11.16\n", encoding="utf-8")

    import hermes_cli._launchers as _launchers

    monkeypatch.setattr(gateway, "PROJECT_ROOT", project)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda root: None)
    monkeypatch.delenv(gateway._GATEWAY_PYTHON_REEXEC_ENV, raising=False)
    monkeypatch.setattr(gateway.sys, "executable", str(tmp_path / "Python314" / "python.exe"))
    monkeypatch.setattr(
        gateway.sys, "orig_argv",
        ["C:\\\\Python314\\\\python.exe", "-m", "hermes_cli.main", "gateway", "run"])

    calls = {}

    def fake_exec(argv, env=None):
        calls["argv"] = argv
        calls["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr(gateway.subprocess, "call", fake_exec)
    # neutralize the pre-guard lifecycle guards so we isolate the re-exec path
    for name in ("_guard_official_docker_root_gateway", "_attach_to_host_gateway_or_guard",
                 "_guard_supervised_gateway_conflict", "_guard_existing_gateway_process_conflict"):
        monkeypatch.setattr(gateway, name, lambda *a, **k: None)

    with pytest.raises(SystemExit) as exc:
        gateway._reexec_gateway_under_committed_python()

    assert exc.value.code == 0
    assert calls["argv"][0] == str(venv_python)
    assert calls["argv"][1:3] == ["-m", "hermes_cli.main"]
    assert calls["env"][gateway._GATEWAY_PYTHON_REEXEC_ENV] == "1"


def test_reexec_disarms_the_parent_startup_watchdog(monkeypatch, tmp_path):
    """hermes_cli.main arms the watchdog before dispatch; the re-exec parent waits on the
    long-lived child without ever reaching the gateway's own disarm point, so a healthy
    boot would be os._exit(75)ed at the configured timeout. The parent must disarm its
    handle before spawning the child."""
    project = tmp_path / "project"
    venv = project / "venv"
    # Layout the host's venv_python_path() resolves (bin/ on POSIX hosts); the is_windows
    # flag under test gates the guard only. Windows-layout resolution is covered by the
    # windows-marked tests above on a real runner.
    venv_bin = venv / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").write_text("", encoding="utf-8")
    (venv / "pyvenv.cfg").write_text("version = 3.11.16\n", encoding="utf-8")

    import hermes_cli._launchers as _launchers
    import hermes_startup_watchdog as watchdog

    monkeypatch.setattr(gateway, "PROJECT_ROOT", project)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda root: None)
    monkeypatch.delenv(gateway._GATEWAY_PYTHON_REEXEC_ENV, raising=False)
    monkeypatch.setattr(gateway.sys, "executable", str(tmp_path / "Python314" / "python.exe"))
    monkeypatch.setattr(
        gateway.sys, "orig_argv",
        ["C:\\\\Python314\\\\python.exe", "-m", "hermes_cli.main", "gateway", "run"])
    monkeypatch.setattr(gateway.subprocess, "call", lambda argv, **kw: 0)

    handle = watchdog.arm_startup_watchdog(timeout_s=3600)
    assert handle is not None and not handle.disarmed
    try:
        with pytest.raises(SystemExit):
            gateway._reexec_gateway_under_committed_python(is_windows=True)
        assert handle.disarmed, "parent watchdog still armed across the re-exec wait"
    finally:
        watchdog._reset_for_tests()


def test_reexec_flush_tolerates_console_less_streams():
    """Legacy pythonw.exe launchers reach the repair with sys.stdout/sys.stderr None; the
    flush helper must not raise before the child is spawned (#71671 class). A stream that
    raises on flush must be survived too — flushing is best-effort, the child matters."""

    class _Broken:
        def flush(self):
            raise OSError("console-less stream exploded")

    gateway._flush_reexec_streams((None, None))  # must not raise
    gateway._flush_reexec_streams((None, _Broken()))


def test_reexec_spawn_kwargs_hide_console_on_windows_only():
    """The recovery child is the console python.exe; on Windows it must carry the
    hidden-console flag (the _resolve_detached_python contract) without detach bits —
    this call waits on the child and forwards its status."""
    kwargs = gateway._reexec_spawn_kwargs({"A": "1"}, is_windows=True)
    from hermes_cli._subprocess_compat import _CREATE_NO_WINDOW

    assert kwargs["creationflags"] == _CREATE_NO_WINDOW
    assert "HERMES_GATEWAY_DETACHED" not in kwargs["env"]  # env passes through untouched

    assert gateway._reexec_spawn_kwargs({"A": "1"}, is_windows=False) == {"env": {"A": "1"}}
