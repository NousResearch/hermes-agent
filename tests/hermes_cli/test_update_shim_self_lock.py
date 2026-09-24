"""The Windows console-shim update self-lock (#88838, #89599, #86093, #79542).

``venv\\Scripts\\hermes.exe`` is a launcher that runs the interpreter with the
shim itself as its script, keeping the file open without FILE_SHARE_DELETE for
the whole command. An update started that way must therefore replace a file it
is holding, which Windows refuses — so the DEPENDENCY SYNC re-runs itself under
``venv\\Scripts\\python.exe``.

The detection and hand-off tests are ``windows_only``: the shim matcher is
gated on the real host, so they run on the Windows lane. The pending-rename
filter and the venv-layout lookup are host-independent and run everywhere.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from hermes_cli import main as cli_main
from hermes_cli import main_install_repair
from hermes_constants import venv_bin_dir

SHIM_NAMES = ["hermes.exe", "hermes-agent.exe", "hermes-acp.exe", "hermes-gateway.exe"]


@pytest.fixture
def venv(tmp_path, monkeypatch):
    """A project venv Scripts dir with a python.exe, wired into the shim matcher."""
    scripts = tmp_path / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_bytes(b"")
    # update_cmd reads these off hermes_cli.main (frozen ``_m()`` surface); the
    # install-repair helpers read their own module globals — patch both.
    for target in (cli_main, main_install_repair):
        monkeypatch.setattr(target, "_venv_scripts_dir", lambda: scripts)
    monkeypatch.setattr(sys, "argv", ["hermes", "update"])
    monkeypatch.delenv(cli_main._UPDATE_REEXEC_ENV, raising=False)
    _fake_psutil(monkeypatch, [])
    return scripts


def _fake_psutil(monkeypatch, ancestor_exes: list[str]):
    """Stand in for psutil with a fixed self+ancestor executable chain."""

    class _Proc:
        def __init__(self, exe=None, pid=os.getpid()):
            self._exe, self.pid = exe, pid

        def exe(self):
            if self._exe is None:
                raise OSError("exe unavailable")
            return self._exe

        def parents(self):
            return [_Proc(exe, 1000 + i) for i, exe in enumerate(ancestor_exes)]

    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(Process=_Proc))


def _capture_popen(monkeypatch, raises: Exception | None = None):
    calls = []

    def fake_popen(cmd, env=None, **kwargs):
        if raises is not None:
            raise raises
        calls.append((list(cmd), dict(env or {}), kwargs))
        return object()

    monkeypatch.setattr(cli_main.subprocess, "Popen", fake_popen)
    return calls


# ---------------------------------------------------------------------------
# Shim detection
# ---------------------------------------------------------------------------


@pytest.mark.windows_only
@pytest.mark.parametrize("shim_name", SHIM_NAMES)
def test_detects_shim_as_argv0(venv, monkeypatch, shim_name):
    monkeypatch.setattr(sys, "argv", [str(venv / shim_name), "update"])
    assert main_install_repair._windows_shim_in_process_chain() == venv / shim_name


@pytest.mark.windows_only
def test_detects_shim_from_zipapp_main_py(venv, monkeypatch):
    """runpy/zipapp launches put ``<shim>\\__main__.py`` in argv[0]."""
    monkeypatch.setattr(sys, "argv", [str(venv / "hermes.exe" / "__main__.py")])
    assert main_install_repair._windows_shim_in_process_chain() == venv / "hermes.exe"


@pytest.mark.windows_only
def test_detects_shim_in_ancestor_chain(venv, monkeypatch):
    """The launcher is usually a separate parent process, not argv[0]."""
    _fake_psutil(monkeypatch, [str(venv / "hermes.exe")])
    assert main_install_repair._windows_shim_in_process_chain() == venv / "hermes.exe"
    # ...and it is that launcher's pid, not ours, a detached child must outwait (#101600).
    assert main_install_repair._windows_shim_holder_pid() == 1000


@pytest.mark.windows_only
def test_ignores_hermes_exe_outside_the_project_venv(venv, monkeypatch, tmp_path):
    """A shim from some other install must never trigger a re-exec."""
    other = tmp_path / "other" / "Scripts"
    other.mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", [str(other / "hermes.exe"), "update"])
    _fake_psutil(monkeypatch, [str(other / "hermes.exe")])
    assert main_install_repair._windows_shim_in_process_chain() is None


# ---------------------------------------------------------------------------
# Re-exec hand-off
# ---------------------------------------------------------------------------


@pytest.mark.windows_only
def test_reexec_runs_same_args_under_venv_python(venv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(venv / "hermes.exe"), "update", "--yes"])
    calls = _capture_popen(monkeypatch)
    token = {"resume_needed": True, "profiles": {"default": 4}, "unmapped": []}

    assert cli_main._reexec_dependency_sync_off_windows_shim(token) is True
    cmd, env, kwargs = calls[0]
    assert cmd == [
        str(venv / "python.exe"), "-m", "hermes_cli.main", "update", "--yes",
    ]
    assert env[cli_main._UPDATE_REEXEC_ENV] == "1"
    # The parent exits, so a prompt in the child could never be answered.
    assert kwargs["stdin"] is cli_main.subprocess.DEVNULL
    # #101600: the child waits for THIS pid and resumes exactly the paused fleet; the parent's
    # copy is disarmed so it exits instead of relaunching gateways while it still holds the shim.
    from hermes_cli import update_handoff
    assert env[update_handoff.SHIM_PARENT_PID_ENV] == str(os.getpid())
    assert token["resume_needed"] is False
    monkeypatch.setenv(update_handoff.GATEWAY_RESUME_ENV, env[update_handoff.GATEWAY_RESUME_ENV])
    assert update_handoff.adopt_handed_off_gateway_resume() == {
        "resume_needed": True, "profiles": {"default": 4}, "unmapped": []}
    assert update_handoff.GATEWAY_RESUME_ENV not in os.environ


@pytest.mark.windows_only
def test_reexec_does_not_recurse(venv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(venv / "hermes.exe"), "update"])
    monkeypatch.setenv(cli_main._UPDATE_REEXEC_ENV, "1")
    calls = _capture_popen(monkeypatch)

    assert cli_main._reexec_dependency_sync_off_windows_shim() is False
    assert calls == []


@pytest.mark.windows_only
def test_reexec_falls_through_when_spawn_fails(venv, monkeypatch):
    """A failed spawn keeps the sync in-process (it then fails closed on the lock)."""
    monkeypatch.setattr(sys, "argv", [str(venv / "hermes.exe"), "update"])
    _capture_popen(monkeypatch, raises=OSError("no exec"))

    assert cli_main._reexec_dependency_sync_off_windows_shim() is False


# ---------------------------------------------------------------------------
# Hand-off placement: the dependency-sync boundary
# ---------------------------------------------------------------------------


@pytest.mark.windows_only
def test_sync_guard_hands_off_when_only_the_shim_is_held(venv, monkeypatch):
    """No native module mapped, but we ARE the shim: hand off and exit 0 WITHOUT resuming the
    paused fleet here — the child owns the token (#101600)."""
    monkeypatch.setattr(sys, "argv", [str(venv / "hermes.exe"), "update"])
    monkeypatch.setattr(cli_main, "_detect_self_loaded_native_modules", lambda: [])
    resumed = []
    monkeypatch.setattr(cli_main, "_resume_windows_gateways_after_update", resumed.append)
    calls = _capture_popen(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        cli_main._abort_dependency_sync_if_self_locked({"resume_needed": True, "profiles": {}})

    assert excinfo.value.code == 0
    assert calls, "expected the dependency sync to be handed to the venv python"
    assert resumed == [], "the shim parent must exit at once, not relaunch gateways"


@pytest.mark.windows_only
def test_sync_guard_defers_native_lock_before_considering_the_shim(venv, monkeypatch):
    """A mapped .pyd still exits 2 — the marker recovery owns that case."""
    monkeypatch.setattr(sys, "argv", [str(venv / "hermes.exe"), "update"])
    monkeypatch.setattr(
        cli_main, "_detect_self_loaded_native_modules", lambda: ["PyYAML (_yaml.pyd)"]
    )
    monkeypatch.setattr(cli_main, "_defer_update_for_self_lock", lambda loaded: None)
    calls = _capture_popen(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        cli_main._abort_dependency_sync_if_self_locked()

    assert excinfo.value.code == 2
    assert calls == [], "a native-module deferral must not also spawn a child"


# ---------------------------------------------------------------------------
# Reboot-deferred renames
# ---------------------------------------------------------------------------


def test_pending_rename_filter_drops_only_our_shim_pairs():
    shims = [Path(r"C:\hermes\venv\Scripts\hermes.exe")]
    entries = [
        r"\??\C:\other\thing.dll", r"!\??\C:\other\thing.dll.bak",
        r"\??\C:\hermes\venv\Scripts\hermes.exe",
        r"!\??\C:\hermes\venv\Scripts\hermes.exe.old.1755624735000",
    ]
    kept, removed = main_install_repair._filter_pending_shim_renames(entries, shims)
    assert removed == 1
    assert kept == entries[:2]


def test_pending_rename_filter_keeps_a_shim_pair_with_a_foreign_target():
    shims = [Path(r"C:\hermes\venv\Scripts\hermes.exe")]
    entries = [
        r"\??\C:\hermes\venv\Scripts\hermes.exe", r"!\??\C:\somewhere\else.exe",
    ]
    kept, removed = main_install_repair._filter_pending_shim_renames(entries, shims)
    assert removed == 0
    assert kept == entries


def test_pending_rename_filter_preserves_a_trailing_delete_entry():
    """A bare source with an empty target is a scheduled delete, not a pair."""
    entries = [r"\??\C:\other\thing.dll", "", r"\??\C:\other\orphan.dll"]
    kept, removed = main_install_repair._filter_pending_shim_renames(entries, [])
    assert removed == 0
    assert kept == entries


# ---------------------------------------------------------------------------
# venv layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("venv_name", ["venv", ".venv"])
def test_venv_scripts_dir_finds_both_layouts(tmp_path, monkeypatch, venv_name):
    """uv writes .venv; our installers write venv. Both must resolve (#79542).

    A ``venv``-only lookup silently returned None on a ``.venv`` install, so the
    whole Windows shim-lock preflight skipped itself. Uses the host's real bin
    dir name (``Scripts``/``bin``), so no OS is faked.
    """
    scripts = venv_bin_dir(tmp_path / venv_name, windows=main_install_repair._is_windows())
    scripts.mkdir(parents=True)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
    assert main_install_repair._venv_scripts_dir() == scripts
