"""Windows facts for ``hermes gateway migrate --backends``: which Task Scheduler task or Startup-folder
entry starts an old backend, the ``hermes.exe`` launcher collapsed into its backend child, and printed
steps that run as typed in PowerShell and Command Prompt. Pure inputs only (a ``schtasks /Query /XML``
dump, a process table, files in a temporary Startup folder); nothing here needs Windows."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli import gateway_migrate_backends_windows as gw
from hermes_cli import gateway_migrate_backends_plan as plan_mod
from hermes_cli.gateway_migrate_backends import BackendItem
from tests.hermes_cli.test_gateway_migrate_backends_apply import _PASSWORD, FakeHost, _args, _files


@pytest.fixture(autouse=True)
def _standard_root(monkeypatch):
    """The homes here stand for the user's standard Hermes root, so steps say `hermes -p NAME`; the root
    pin for any other root is tested in test_gateway_migrate_backends_root_pin.py."""
    monkeypatch.setattr("hermes_cli.gateway_migrate_backends_plan._is_standard_root", lambda root: True)

_PY = r"C:\Users\u\AppData\Local\hermes\tools\python-3.14.9-win32-x64\python.exe"
_EXE = r"C:\Users\u\AppData\Local\hermes\bin\hermes.exe"
_BACKEND = [_PY, "-I", _EXE, "dashboard", "--no-open", "--host", "0.0.0.0", "--port", "9119"]


def _task(path: str, command: str, arguments: str, *, enabled: bool = True) -> str:
    settings = "" if enabled else "<Enabled>false</Enabled>"
    return (f"\r\n<!-- {path} -->\r\n<?xml version=\"1.0\" encoding=\"UTF-16\"?>\r\n"
            "<Task version=\"1.3\" xmlns=\"http://schemas.microsoft.com/windows/2004/02/mit/task\">\r\n"
            f"  <Settings>{settings}<ExecutionTimeLimit>PT0S</ExecutionTimeLimit></Settings>\r\n"
            "  <Actions Context=\"Author\"><Exec>\r\n"
            f"    <Command>{command}</Command>\r\n    <Arguments>{arguments}</Arguments>\r\n"
            "  </Exec></Actions>\r\n</Task>\r\n")


def _dump(*tasks: str) -> str:
    return "\r\n<Tasks>\r\n" + "".join(tasks) + "</Tasks>\r\n"


def test_the_task_that_runs_the_backends_arguments_is_named_with_its_folder():
    xml = _dump(_task(r"\Hermes_Gateway", "wscript.exe", r'"C:\x\gateway.vbs"'),
                _task(r"\Hermes Dashboard", _EXE, "dashboard --no-open --host 0.0.0.0 --port 9119"),
                _task(r"\Me\Other", _EXE, "dashboard --port 9200"),
                _task(r"\Microsoft\Windows\Defrag", "defrag.exe", "-c"))
    assert gw.tasks_running(xml, _BACKEND) == [("Hermes Dashboard", True)]
    assert gw.tasks_running(xml, [_PY, "-I", _EXE, "dashboard", "--port", "9200"]) == [(r"Me\Other", True)]


def test_a_task_whose_action_quotes_the_command_and_is_disabled_still_matches():
    xml = _dump(_task(r"\Dash", f'"{_EXE}"', '"dashboard" --no-open --host 0.0.0.0 --port 9119', enabled=False))
    assert gw.tasks_running(xml, _BACKEND) == [("Dash", False)]


def test_no_task_matches_a_wrapper_script_or_an_unreadable_dump():
    xml = _dump(_task(r"\Dash", "powershell.exe", "-File C:\\x\\start-dashboard.ps1"))
    assert gw.tasks_running(xml, _BACKEND) == []
    assert gw.tasks_running("<!-- \\Broken -->\r\n<Task><Actions>", _BACKEND) == []
    assert gw.tasks_running(xml, ["python.exe", "server.py"]) == []


def test_the_hermes_exe_launcher_is_dropped_for_the_backend_child_it_runs():
    rows = [(10, f'"{_EXE}" dashboard --port 9119'), (11, " ".join(_BACKEND)), (30, f'"{_EXE}" serve')]
    parents = {10: 4, 11: 10, 30: 2}
    assert gw.drop_launchers(rows, ppid_of=parents.get) == [(11, " ".join(_BACKEND)), (30, f'"{_EXE}" serve')]


def _lnk(target: str, arguments: str) -> bytes:
    """A minimal Shell Link: header + RELATIVE_PATH and COMMAND_LINE_ARGUMENTS string data (Unicode)."""
    import struct
    clsid = bytes.fromhex("0114020000000000c000000000000046")
    header = struct.pack("<I16sIIQQQIIIHHII", 0x4C, clsid, 0x08 | 0x20 | 0x80, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)
    strings = b"".join(struct.pack("<H", len(s)) + s.encode("utf-16-le") for s in (target, arguments))
    return header + strings + b"\x00\x00\x00\x00"


_ARGS = "dashboard --no-open --host 0.0.0.0 --port 9119"


def test_startup_folder_entries_are_found_by_the_exact_command_they_start(tmp_path):
    lnk = tmp_path / "Hermes dashboard.lnk"
    lnk.write_bytes(_lnk(_EXE, _ARGS))
    (tmp_path / "other port.lnk").write_bytes(_lnk(_EXE, "dashboard --no-open --host 0.0.0.0 --port 9200"))
    (tmp_path / "fewer args.lnk").write_bytes(_lnk(_EXE, "dashboard"))
    (tmp_path / "gateway.vbs").write_text('CreateObject("WScript.Shell").Run "hermes gateway run", 0')
    (tmp_path / "notes.txt").write_text(f"hermes {_ARGS}")
    assert gw.lnk_command(lnk.read_bytes()) == [_EXE, *_ARGS.split()]
    assert gw.startup_entries_running(_BACKEND, dirs=[tmp_path, tmp_path / "missing"]) == [lnk]


def test_a_script_in_the_startup_folder_is_matched_by_its_whole_command(tmp_path):
    vbs = tmp_path / "dash.vbs"
    vbs.write_text(f'CreateObject("WScript.Shell").Run "hermes {_ARGS}", 0')
    (tmp_path / "dash9200.cmd").write_text(f'@"{_EXE}" dashboard --no-open --host 0.0.0.0 --port 9200\r\n')
    assert gw.startup_entries_running(_BACKEND, dirs=[tmp_path]) == [vbs]


def test_two_matching_startup_entries_name_none_and_the_steps_ask_the_operator(tmp_path, monkeypatch):
    for name in ("a.lnk", "b.lnk"):
        (tmp_path / name).write_bytes(_lnk(_EXE, _ARGS))
    monkeypatch.setattr(gw, "startup_dirs", lambda: [tmp_path])
    monkeypatch.setattr(gw, "_schedule_service_pid", lambda: None)

    sup, unit, path = gw.supervisor(42, list(_BACKEND))
    steps = gw.manual_steps("0.0.0.0", 9119, _item(supervisor=sup, unit=unit, unit_path=path, pid=42))

    assert (sup, unit, path) == ("startup-folder", None, None)
    assert any("more than one Startup-folder entry starts this exact command" in s for s in steps)
    assert not any(s.startswith("move ") for s in steps)


_H = "hermes -p default"  # a bare `hermes` would follow a sticky `hermes profile use` choice
_TAIL = [f"{_H} gateway status", f"# then by what it prints: 'Gateway is not running' (no service installed) -> "
         f"{_H} gateway install; 'No gateway process detected' -> {_H} gateway start; 'Gateway process running' or "
         f"'Running manually' -> {_H} gateway restart", "# the gateway takes a few seconds to serve the address: if the next line prints 'The gateway is "
         "not running', wait 5 seconds and run it again", f"{_H} webapp --status"]


def test_windows_steps_name_the_profile_and_copy_an_accepted_public_url(tmp_path):
    profile = tmp_path / "hermes" / "profiles" / "work"
    steps = gw.manual_steps("127.0.0.1", 9120, _item(pid=42), home=str(profile),
                            public_url=("https://dash.example", None))
    assert not any("dash.example" in s for s in steps)
    assert any("into webapp.public_url: open config.yaml with hermes -p work config edit" in s for s in steps)
    assert all(s.startswith(("hermes -p work ", "taskkill", "#")) for s in steps)
    refused = gw.manual_steps("127.0.0.1", 9120, _item(pid=42), public_url=(None, "The old backend's public URL"))
    assert any("not a URL the gateway accepts" in s for s in refused)


@pytest.mark.parametrize("url", ["https://dash.example/$tenant", "https://dash.example/$(Get-Date)",
                                 "https://dash.example/o'brien"])
def test_windows_steps_never_put_a_public_url_on_a_command_line(url):
    """PowerShell expands $name and $(...) inside double quotes, and a ' ends a single-quoted literal."""
    steps = gw.manual_steps("127.0.0.1", 9120, _item(pid=42), public_url=(url, None))

    assert not any("dash.example" in s for s in steps)
    assert any(s.startswith("# copy the old backend's public URL") for s in steps)


def _item(**kw):
    return SimpleNamespace(**{"supervisor": "manual", "unit": None, "unit_path": None, "pid": None, **kw})


def test_steps_for_a_scheduled_task_end_and_disable_it_before_the_gateway_is_installed():
    steps = gw.manual_steps("0.0.0.0", 9119, _item(supervisor="scheduled-task", unit="Hermes Dashboard", pid=11))
    assert steps == [f"{_H} webapp --build-only", f"{_H} config set webapp.enabled true",
                     f"{_H} config set webapp.host 0.0.0.0", f"{_H} config set webapp.port 9119",
                     'schtasks /End /TN "Hermes Dashboard"', 'schtasks /Change /TN "Hermes Dashboard" /DISABLE',
                     *_TAIL]


def test_steps_never_use_posix_only_syntax_and_name_env_keys_without_values(tmp_path):
    entry = tmp_path / "Hermes dashboard.lnk"
    for item in (_item(supervisor="startup-folder", unit=entry.name, unit_path=str(entry), pid=42),
                 _item(supervisor="scheduled-task"), _item(pid=42), _item()):
        steps = gw.manual_steps("127.0.0.1", 9120, item, env_keys=["HERMES_DASHBOARD_BASIC_AUTH_PASSWORD"],
                                home=str(tmp_path))
        joined = "\n".join(steps)
        assert "&&" not in joined and "chmod" not in joined and "sudo" not in joined
        assert steps[0].startswith("# these lines set HERMES_HOME")  # a home that is no profile: pinned per line
        pinned = plan_mod.pin_hermes_home(str(tmp_path), "hermes", powershell=True)
        assert steps[-len(_TAIL):] == [s.replace(_H, pinned) for s in _TAIL]
        assert f"HERMES_DASHBOARD_BASIC_AUTH_PASSWORD from the old backend's environment into {tmp_path / '.env'}" \
            in joined
    steps = gw.manual_steps("127.0.0.1", 9120, _item(supervisor="startup-folder", unit_path=str(entry), pid=42))
    assert steps[4] == "taskkill /PID 42 /T /F"
    assert steps[6] == "# move it to a folder outside shell:startup, and move it back to undo"  # home unknown
    assert steps[5].startswith("# this Startup-folder entry starts the same command; if it is what starts")
    steps = gw.manual_steps("127.0.0.1", 9120, _item(supervisor="scheduled-task", pid=42))  # task not named
    assert steps[4:6] == ["taskkill /PID 42 /T /F", "# and Disable that task in Task Scheduler"]


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes-home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    (h / "config.yaml").write_text("model:\n  default: fake-model\n", encoding="utf-8")
    (h / ".env").write_text("OPENAI_API_KEY=t2-fake\n", encoding="utf-8")
    os.chmod(h / ".env", 0o600)
    return h


class WindowsHost(FakeHost):
    """The dry-run tests' fake host with a Task Scheduler backend on Windows (credentials in its process env)."""

    def os_family(self):
        return "win32"

    def backends(self):
        return [BackendItem(supervisor="scheduled-task", running=True, pid=4242, argv=list(_BACKEND),
                            unit="Hermes Dashboard", hermes_home=str(self.home), this_home=True)]


def test_windows_dry_run_exits_6_changes_nothing_and_prints_the_task_steps(home, capsys):
    host = WindowsHost(home)
    before = _files(home)

    code = engine.cmd_apply(host, _args(dry_run=True))

    assert code == engine.EXIT_UNSUPPORTED and _files(home) == before
    assert not (home / "backend_migration.json").exists()
    out = capsys.readouterr().out
    lines = [line.strip() for line in out.splitlines()]
    steps = lines[lines.index("Steps, in this order (one command per line):") + 1:]
    assert steps[4].startswith("# copy HERMES_DASHBOARD_BASIC_AUTH_PASSWORD, HERMES_DASHBOARD_BASIC_AUTH_SECRET")
    assert steps[5:] == ['schtasks /End /TN "Hermes Dashboard"', 'schtasks /Change /TN "Hermes Dashboard" /DISABLE',
                         *_TAIL]
    assert _PASSWORD not in out and "chmod" not in out and "&&" not in out


def test_windows_steps_warn_when_a_turn_is_running_on_the_old_backend(home, capsys):
    host = WindowsHost(home)
    host.turns = 1

    assert engine.cmd_apply(host, _args(dry_run=True)) == engine.EXIT_UNSUPPORTED
    assert "1 turn(s) are running on it now: stopping it interrupts them" in capsys.readouterr().out


def test_where_wmic_works_the_windows_report_reads_and_prints_what_main_does(monkeypatch):
    from hermes_cli import gateway_migrate_backends as gmb
    monkeypatch.setattr(gmb, "os_family", lambda: "win32")
    monkeypatch.setattr(gw, "wmic_missing", lambda: False)
    monkeypatch.setattr("hermes_cli.dashboard_procs._iter_process_table",
                        lambda: [(10, f'"{_EXE}" dashboard --port 9119'), (11, " ".join(_BACKEND))])
    for name in ("process_rows", "drop_launchers", "argv_of", "supervisor"):
        monkeypatch.setattr(gw, name, lambda *a, _n=name, **k: pytest.fail(f"{_n} used where wmic works"))

    rows = gmb._running_backends()
    item = gmb.BackendItem(supervisor="unknown", running=True, pid=11, argv=list(_BACKEND))

    assert [pid for pid, _ in rows] == [10, 11]  # main lists the hermes.exe launcher too
    assert gmb.format_report([item])[3] == f"    runs:    {shlex.join(_BACKEND)}"


def test_wmic_missing_follows_the_path_through_the_shared_resolver(tmp_path, monkeypatch):
    """The lookup goes through hermes_platform's resolver (no bare shutil.which): an empty PATH has no
    wmic, a PATH holding one does."""
    tool = tmp_path / ("wmic.exe" if os.name == "nt" else "wmic")
    tool.write_text("@echo off\n" if os.name == "nt" else "#!/bin/sh\n", encoding="utf-8")
    tool.chmod(0o755)
    for path, missing in (("", True), (str(tmp_path), False)):
        monkeypatch.setenv("PATH", path)
        gw.wmic_missing.cache_clear()
        assert gw.wmic_missing() is missing
    gw.wmic_missing.cache_clear()


def test_a_task_of_another_install_with_the_same_arguments_is_not_named():
    """InstallB's task runs the same arguments, but stopping it leaves InstallA's backend starting at sign-in."""
    other = r"C:\InstallB\hermes\bin\hermes.exe"
    xml = _dump(_task(r"\Hermes Dashboard", other, "dashboard --no-open --host 0.0.0.0 --port 9119"))
    assert gw.tasks_running(xml, _BACKEND) == []
    # A task that names a bare `hermes` cannot be tied to an install: it is not named either.
    assert gw.tasks_running(_dump(_task(r"\Bare", "hermes", "dashboard --no-open --host 0.0.0.0 --port 9119")),
                            _BACKEND) == []


def test_a_backend_found_without_wmic_gets_its_profile_home_from_its_exact_argv(tmp_path, monkeypatch):
    """The no-wmic scan reads exact args with psutil, but the shared home resolver re-read argv with the
    dashboard reader, which has no Windows implementation: `hermes -p work serve` under a root-pinned
    environment resolved to the root, not profiles/work."""
    from hermes_cli import dashboard_procs, main_dashboard
    from hermes_cli import gateway_migrate_backends as gmb
    root = tmp_path / "hermes"
    (root / "profiles" / "work").mkdir(parents=True)
    exact = [_EXE, "-p", "work", "serve", "--port", "9119"]
    monkeypatch.setattr(gmb, "os_family", lambda: "win32")
    monkeypatch.setattr(gw, "wmic_missing", lambda: True)
    monkeypatch.setattr(gw, "argv_of", lambda pid: list(exact))
    monkeypatch.setattr(gw, "supervisor", lambda pid, argv: ("manual", None, None))
    monkeypatch.setattr(main_dashboard, "_dashboard_cmdline_for_pid", lambda pid: None)
    monkeypatch.setattr(main_dashboard, "_systemd_owner_for_pid", lambda pid: None)
    monkeypatch.setattr(gmb, "_s6_slot", lambda pid: None)
    monkeypatch.setattr(gmb, "_container_cgroup", lambda pid: False)
    monkeypatch.setattr(gmb, "_in_container", lambda: False)
    monkeypatch.setattr(dashboard_procs, "_pid_environ", lambda pid: {"HERMES_HOME": str(root)})

    item = gmb._classify_running(4242, " ".join(exact), launchd_jobs=None, plists={}, ssh_pids=set(),
                                 desktop_pids=set(), seen_units=set())

    assert Path(item.hermes_home) == root / "profiles" / "work"


def test_a_startup_shortcut_is_moved_out_of_the_startup_folder_with_its_undo():
    """A renamed `.lnk.disabled` left in Startup makes Windows ask which app opens it at every sign-in."""
    startup = r"C:\Users\u\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup"
    entry, home = startup + r"\Hermes dashboard.lnk", r"C:\Users\u\.hermes"
    item = _item(supervisor="startup-folder", unit="Hermes dashboard.lnk", unit_path=entry, pid=42)

    steps = gw.manual_steps("127.0.0.1", 9120, item, home=home)

    backup = home + r"\backend_migration\Hermes dashboard.lnk"
    i = steps.index(f'move "{entry}" "{backup}"')
    assert steps[i - 2] == f'mkdir "{home}\\backend_migration"'
    assert steps[i + 1] == f'# to undo later: move "{backup}" "{entry}"'
    assert not any(".disabled" in s for s in steps)
    assert not any(s.startswith("move ") and startup in s.split('" "')[-1] for s in steps), steps
