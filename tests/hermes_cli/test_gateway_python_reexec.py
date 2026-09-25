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


def test_get_python_path_prefers_checkout_venv_over_running_executable(monkeypatch, tmp_path):
    """No PM store Python committed: the venv's own interpreter wins over sys.executable —
    the generator must never bake a PATH-race system Python into a regenerated launcher."""
    project = tmp_path / "project"
    venv = project / "venv"
    scripts = venv / ("Scripts" if gateway.sys.platform == "win32" else "bin")
    scripts.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("version = 3.11.16\n", encoding="utf-8")
    venv_python = scripts / ("python.exe" if scripts.name == "Scripts" else "python")
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
