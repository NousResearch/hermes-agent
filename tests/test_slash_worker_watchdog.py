import inspect
import io
import subprocess
import sys
import types
from pathlib import Path

from tui_gateway import slash_worker


def test_module_import_does_not_load_cli_before_watchdog_can_start():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; assert 'cli' not in sys.modules; "
            "import tui_gateway.slash_worker; assert 'cli' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        timeout=10,
    )


def test_is_orphaned_true_when_ppid_changes():
    # Our parent went away and we were reparented to a subreaper/init.
    assert slash_worker._is_orphaned(1234, getppid=lambda: 999999) is True


def test_is_orphaned_false_when_direct_parent_is_unchanged():
    original_ppid = 1234
    assert slash_worker._is_orphaned(original_ppid, getppid=lambda: original_ppid) is False


def test_parent_death_watchdog_contract_has_no_create_time_plumbing():
    assert list(inspect.signature(slash_worker._is_orphaned).parameters) == [
        "original_ppid",
        "getppid",
    ]
    assert list(inspect.signature(slash_worker._start_parent_death_watchdog).parameters) == [
        "original_ppid",
    ]


def test_main_arms_watchdog_for_spawn_parent_before_cli_startup(monkeypatch):
    parent_pid = 424242
    events = []

    class FakeHermesCLI:
        def __init__(self, **kwargs):
            events.append(("cli", kwargs["resume"]))

    cli_module = types.ModuleType("cli")
    setattr(cli_module, "HermesCLI", FakeHermesCLI)
    monkeypatch.setitem(sys.modules, "cli", cli_module)
    monkeypatch.setattr(
        slash_worker,
        "_start_parent_death_watchdog",
        lambda pid: events.append(("watchdog", pid)),
    )
    monkeypatch.setattr(
        slash_worker,
        "_prepare_slash_worker_runtime",
        lambda: events.append(("runtime",)),
    )

    def unexpected_getppid():
        raise AssertionError("spawn parent PID must come from argv")

    monkeypatch.setattr(slash_worker.os, "getppid", unexpected_getppid)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slash_worker",
            "--session-key",
            "session-1",
            "--parent-pid",
            str(parent_pid),
        ],
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    slash_worker.main()

    assert events == [
        ("watchdog", parent_pid),
        ("runtime",),
        ("cli", "session-1"),
    ]
